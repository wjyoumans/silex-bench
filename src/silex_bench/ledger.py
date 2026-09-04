"""Transactional SQLite run ledger."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    CAMPAIGN_SCHEMA_VERSION,
    AgreementResult,
    AgreementStatus,
    Case,
    EngineInfo,
    Observation,
    ObservationStatus,
    validate_campaign_manifest,
)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS run (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS engines (
    backend TEXT PRIMARY KEY,
    available INTEGER NOT NULL,
    info_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    case_key TEXT PRIMARY KEY,
    workload TEXT NOT NULL,
    case_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    case_key TEXT NOT NULL REFERENCES cases(case_key),
    workload TEXT NOT NULL,
    backend TEXT NOT NULL REFERENCES engines(backend),
    repetition INTEGER NOT NULL,
    order_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    success INTEGER NOT NULL,
    timeout INTEGER NOT NULL,
    target_wall_ns INTEGER,
    process_wall_ns INTEGER,
    observation_json TEXT NOT NULL,
    PRIMARY KEY (case_key, backend, repetition)
);
CREATE TABLE IF NOT EXISTS agreements (
    case_key TEXT NOT NULL REFERENCES cases(case_key),
    repetition INTEGER NOT NULL,
    lhs_backend TEXT NOT NULL,
    rhs_backend TEXT NOT NULL,
    status TEXT NOT NULL,
    success INTEGER NOT NULL,
    timing_eligible INTEGER NOT NULL,
    ratio REAL,
    agreement_json TEXT NOT NULL,
    PRIMARY KEY (case_key, repetition, lhs_backend, rhs_backend)
);
CREATE INDEX IF NOT EXISTS observations_workload_backend
    ON observations(workload, backend, case_key, repetition);
CREATE INDEX IF NOT EXISTS agreements_pair
    ON agreements(lhs_backend, rhs_backend, case_key, repetition);
"""

RUN_STATES = frozenset({"running", "budget_exhausted", "complete", "failed"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode_object(value: Any, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return payload


def _selected_backends(case: dict[str, Any], backends: list[str]) -> set[str]:
    eligible = set(case["eligible_backends"])
    if not eligible:
        return set(backends)
    eligible.add("silex")
    return set(backends) & eligible


class RunLedger:
    def __init__(self, run_dir: Path, *, create: bool = False) -> None:
        self.run_dir = run_dir.expanduser().absolute()
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "run.sqlite"
        existed = self.path.exists()
        if existed and not self.path.is_file():
            raise ValueError(f"run ledger is not a regular file: {self.path}")
        if not create and not existed:
            raise ValueError(f"run ledger does not exist: {self.path}")
        self.connection = sqlite3.connect(self.path, timeout=30)
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA busy_timeout = 30000")
            if not existed:
                with self.connection:
                    self.connection.executescript(SCHEMA_SQL)
                    self.connection.execute(
                        f"PRAGMA user_version = {CAMPAIGN_SCHEMA_VERSION}"
                    )
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version != CAMPAIGN_SCHEMA_VERSION:
                raise ValueError(
                    "run ledger schema must be "
                    f"{CAMPAIGN_SCHEMA_VERSION}, got {version}"
                )
            if existed:
                self._validate_existing_run()
            # These settings are intentionally applied only after an existing
            # ledger and its embedded manifest pass the read-only checks above.
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")
        except BaseException:
            self.connection.close()
            raise

    def _validate_existing_run(self) -> None:
        try:
            row = self.connection.execute(
                "SELECT schema_version, fingerprint, state, manifest_json "
                "FROM run WHERE singleton = 1"
            ).fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"invalid run ledger schema: {exc}") from exc
        if row is None:
            raise ValueError("run ledger is not initialized")
        if row["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(
                "run row schema_version must be "
                f"{CAMPAIGN_SCHEMA_VERSION}, got {row['schema_version']}"
            )
        if row["state"] not in RUN_STATES:
            raise ValueError(f"invalid run ledger state: {row['state']!r}")
        manifest = _decode_object(row["manifest_json"], "run manifest")
        validate_campaign_manifest(manifest)
        fingerprint = row["fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("run ledger fingerprint must be a lowercase SHA-256 digest")
        if row["fingerprint"] != manifest["run_fingerprint"]:
            raise ValueError(
                "run ledger fingerprint does not match the embedded manifest"
            )
        self._validate_existing_rows(manifest, state=row["state"])

    def _validate_existing_rows(self, manifest: dict[str, Any], *, state: str) -> None:
        plan = manifest["plan"]
        backends = list(plan["backends"])
        backend_set = set(backends)
        probes = {probe["backend"]: probe for probe in manifest["engine_probes"]}
        cases = {
            f"{case['workload']}:{case['id']}": case for case in plan["cases"]
        }
        repetitions = int(plan["execution"]["repetitions"])
        try:
            engine_rows = self.connection.execute(
                "SELECT backend, available, info_json FROM engines"
            ).fetchall()
            case_rows = self.connection.execute(
                "SELECT case_key, workload, case_json FROM cases"
            ).fetchall()
            observation_rows = self.connection.execute(
                "SELECT case_key, workload, backend, repetition, status, "
                "observation_json FROM observations"
            ).fetchall()
            agreement_rows = self.connection.execute(
                "SELECT case_key, repetition, lhs_backend, rhs_backend, status, "
                "agreement_json FROM agreements"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"invalid run ledger schema: {exc}") from exc

        stored_engines: set[str] = set()
        for row in engine_rows:
            backend = row["backend"]
            if backend not in backend_set:
                raise ValueError(
                    f"run ledger engine {backend!r} references an unplanned backend"
                )
            info = _decode_object(row["info_json"], f"engine {backend}")
            if info != probes[backend]:
                raise ValueError(
                    f"run ledger engine {backend!r} does not match its manifest probe"
                )
            if row["available"] not in {0, 1} or bool(row["available"]) is not info["available"]:
                raise ValueError(
                    f"run ledger engine {backend!r} availability is inconsistent"
                )
            stored_engines.add(backend)

        stored_cases: set[str] = set()
        for row in case_rows:
            case_key = row["case_key"]
            if case_key not in cases:
                raise ValueError(
                    f"run ledger case {case_key!r} references an unplanned case"
                )
            case = _decode_object(row["case_json"], f"case {case_key}")
            if case != cases[case_key] or row["workload"] != case["workload"]:
                raise ValueError(
                    f"run ledger case {case_key!r} does not match the manifest plan"
                )
            stored_cases.add(case_key)

        observation_statuses = {status.value for status in ObservationStatus}
        for row in observation_rows:
            case_key = row["case_key"]
            backend = row["backend"]
            if case_key not in stored_cases or case_key not in cases:
                raise ValueError("run ledger observation references an unknown case")
            if backend not in stored_engines or backend not in _selected_backends(
                cases[case_key], backends
            ):
                raise ValueError("run ledger observation references an invalid backend")
            repetition = row["repetition"]
            if type(repetition) is not int or not 0 <= repetition < repetitions:
                raise ValueError("run ledger observation repetition is outside the plan")
            if row["workload"] != cases[case_key]["workload"]:
                raise ValueError("run ledger observation workload does not match its case")
            if row["status"] not in observation_statuses:
                raise ValueError("run ledger observation has an invalid status")
            payload = _decode_object(
                row["observation_json"],
                f"observation {case_key}/{backend}/{repetition}",
            )
            references = {
                "case_key": case_key,
                "workload": row["workload"],
                "backend": backend,
                "repetition": repetition,
                "status": row["status"],
            }
            if any(payload.get(key) != value for key, value in references.items()):
                raise ValueError(
                    "run ledger observation columns do not match its JSON payload"
                )

        agreement_statuses = {status.value for status in AgreementStatus}
        for row in agreement_rows:
            case_key = row["case_key"]
            lhs = row["lhs_backend"]
            rhs = row["rhs_backend"]
            if case_key not in stored_cases or case_key not in cases:
                raise ValueError("run ledger agreement references an unknown case")
            selected = _selected_backends(cases[case_key], backends)
            if lhs == rhs or lhs not in selected or rhs not in selected:
                raise ValueError("run ledger agreement references an invalid backend pair")
            if lhs not in stored_engines or rhs not in stored_engines:
                raise ValueError("run ledger agreement references an unstored backend")
            repetition = row["repetition"]
            if type(repetition) is not int or not 0 <= repetition < repetitions:
                raise ValueError("run ledger agreement repetition is outside the plan")
            if row["status"] not in agreement_statuses:
                raise ValueError("run ledger agreement has an invalid status")
            payload = _decode_object(
                row["agreement_json"],
                f"agreement {case_key}/{lhs}/{rhs}/{repetition}",
            )
            references = {
                "case_key": case_key,
                "repetition": repetition,
                "lhs_backend": lhs,
                "rhs_backend": rhs,
                "status": row["status"],
            }
            if any(payload.get(key) != value for key, value in references.items()):
                raise ValueError(
                    "run ledger agreement columns do not match its JSON payload"
                )

        if state in {"complete", "failed"}:
            if stored_engines != backend_set:
                raise ValueError("terminal run ledger is missing planned engines")
            if stored_cases != set(cases):
                raise ValueError("terminal run ledger is missing planned cases")
            if len(observation_rows) != plan["sample_count"]:
                raise ValueError(
                    "terminal run ledger observation count does not match plan.sample_count"
                )

    def __enter__(self) -> "RunLedger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def initialize(self, fingerprint: str, manifest: dict[str, Any]) -> None:
        validate_campaign_manifest(manifest)
        if fingerprint != manifest["run_fingerprint"]:
            raise ValueError(
                "run ledger fingerprint does not match the embedded manifest"
            )
        now = _now()
        with self.connection:
            row = self.connection.execute("SELECT fingerprint FROM run WHERE singleton = 1").fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise ValueError("run ledger fingerprint does not match the resolved campaign")
                return
            self.connection.execute(
                "INSERT INTO run VALUES (1, ?, ?, ?, ?, ?, ?)",
                (
                    CAMPAIGN_SCHEMA_VERSION,
                    fingerprint,
                    "running",
                    now,
                    now,
                    _json(manifest),
                ),
            )

    def manifest(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT schema_version, manifest_json FROM run WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise ValueError("run ledger is not initialized")
        if row["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(
                "run row schema_version must be "
                f"{CAMPAIGN_SCHEMA_VERSION}, got {row['schema_version']}"
            )
        try:
            value = json.loads(row["manifest_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid run manifest JSON: {exc}") from exc
        return validate_campaign_manifest(value)

    def fingerprint(self) -> str:
        row = self.connection.execute("SELECT fingerprint FROM run WHERE singleton = 1").fetchone()
        if row is None:
            raise ValueError("run ledger is not initialized")
        return str(row["fingerprint"])

    def state(self) -> str:
        row = self.connection.execute("SELECT state FROM run WHERE singleton = 1").fetchone()
        return "uninitialized" if row is None else str(row["state"])

    def set_state(self, state: str) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"invalid run ledger state: {state!r}")
        with self.connection:
            self.connection.execute(
                "UPDATE run SET state = ?, updated_at = ? WHERE singleton = 1",
                (state, _now()),
            )

    def put_engines(self, engines: Iterable[EngineInfo]) -> None:
        with self.connection:
            for engine in engines:
                encoded = _json(engine.to_json())
                existing = self.connection.execute(
                    "SELECT info_json FROM engines WHERE backend = ?", (engine.backend,)
                ).fetchone()
                if existing is not None and existing["info_json"] != encoded:
                    raise ValueError(f"engine identity drift for {engine.backend}")
                self.connection.execute(
                    "INSERT OR IGNORE INTO engines VALUES (?, ?, ?)",
                    (engine.backend, int(engine.available), encoded),
                )

    def put_cases(self, cases: Iterable[Case]) -> None:
        with self.connection:
            for case in cases:
                encoded = _json(case.to_json())
                existing = self.connection.execute(
                    "SELECT case_json FROM cases WHERE case_key = ?", (case.key,)
                ).fetchone()
                if existing is not None and existing["case_json"] != encoded:
                    raise ValueError(f"case drift for {case.key}")
                self.connection.execute(
                    "INSERT OR IGNORE INTO cases VALUES (?, ?, ?)",
                    (case.key, case.workload, encoded),
                )

    def has_observation(self, case_key: str, backend: str, repetition: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM observations WHERE case_key = ? AND backend = ? AND repetition = ?",
            (case_key, backend, repetition),
        ).fetchone()
        return row is not None

    def put_observation(self, observation: Observation) -> None:
        encoded = _json(observation.to_json())
        with self.connection:
            existing = self.connection.execute(
                "SELECT observation_json FROM observations WHERE case_key = ? AND backend = ? AND repetition = ?",
                (observation.case_key, observation.backend, observation.repetition),
            ).fetchone()
            if existing is not None:
                if existing["observation_json"] != encoded:
                    raise ValueError(
                        f"observation collision for {observation.case_key}/{observation.backend}/{observation.repetition}"
                    )
                return
            self.connection.execute(
                """INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation.case_key,
                    observation.workload,
                    observation.backend,
                    observation.repetition,
                    observation.order_index,
                    observation.status.value,
                    int(observation.success),
                    int(observation.timeout),
                    observation.target_wall_ns,
                    observation.process_wall_ns,
                    encoded,
                ),
            )

    def observations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT observation_json FROM observations ORDER BY workload, case_key, repetition, order_index"
        ).fetchall()
        return [json.loads(row["observation_json"]) for row in rows]

    def observation(
        self, case_key: str, backend: str, repetition: int
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT observation_json FROM observations WHERE case_key = ? AND backend = ? AND repetition = ?",
            (case_key, backend, repetition),
        ).fetchone()
        return None if row is None else json.loads(row["observation_json"])

    def put_agreement(
        self,
        case_key: str,
        repetition: int,
        lhs_backend: str,
        rhs_backend: str,
        result: AgreementResult,
        *,
        timing_eligible: bool,
        ratio: float | None,
    ) -> None:
        payload = result.to_json()
        payload.update(
            {
                "case_key": case_key,
                "repetition": repetition,
                "lhs_backend": lhs_backend,
                "rhs_backend": rhs_backend,
                "timing_eligible": timing_eligible,
                "speedup_baseline_over_candidate": ratio,
            }
        )
        encoded = _json(payload)
        with self.connection:
            self.connection.execute(
                """INSERT INTO agreements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(case_key, repetition, lhs_backend, rhs_backend)
                   DO UPDATE SET status=excluded.status,
                                 success=excluded.success,
                                 timing_eligible=excluded.timing_eligible,
                                 ratio=excluded.ratio,
                                 agreement_json=excluded.agreement_json""",
                (
                    case_key,
                    repetition,
                    lhs_backend,
                    rhs_backend,
                    payload["status"],
                    int(result.success),
                    int(timing_eligible),
                    ratio,
                    encoded,
                ),
            )

    def agreements(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT agreement_json FROM agreements ORDER BY case_key, repetition, lhs_backend, rhs_backend"
        ).fetchall()
        return [json.loads(row["agreement_json"]) for row in rows]

    def engines(self) -> list[dict[str, Any]]:
        return [
            json.loads(row["info_json"])
            for row in self.connection.execute("SELECT info_json FROM engines ORDER BY backend")
        ]

    def cases(self) -> list[dict[str, Any]]:
        return [
            json.loads(row["case_json"])
            for row in self.connection.execute("SELECT case_json FROM cases ORDER BY workload, case_key")
        ]

    def snapshot(self) -> dict[str, Any]:
        self.connection.execute("BEGIN")
        try:
            return {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "fingerprint": self.fingerprint(),
                "state": self.state(),
                "manifest": self.manifest(),
                "engines": self.engines(),
                "cases": self.cases(),
                "observations": self.observations(),
                "agreements": self.agreements(),
            }
        finally:
            self.connection.execute("ROLLBACK")
