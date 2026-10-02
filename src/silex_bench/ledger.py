"""Transactional SQLite run ledger."""

from __future__ import annotations

import itertools
import json
import math
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
    TimingSample,
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
    agreement_json TEXT NOT NULL,
    PRIMARY KEY (case_key, repetition, lhs_backend, rhs_backend)
);
CREATE TABLE IF NOT EXISTS timing_samples (
    case_key TEXT NOT NULL,
    backend TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    variant TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    timeout INTEGER NOT NULL,
    target_wall_ns INTEGER,
    target_cpu_ns INTEGER,
    process_wall_ns INTEGER,
    timing_scope TEXT NOT NULL,
    effective_timeout_seconds REAL NOT NULL,
    sample_json TEXT NOT NULL,
    PRIMARY KEY (case_key, backend, repetition, variant, sample_index),
    FOREIGN KEY (case_key, backend, repetition)
        REFERENCES observations(case_key, backend, repetition)
);
CREATE INDEX IF NOT EXISTS observations_workload_backend
    ON observations(workload, backend, case_key, repetition);
CREATE INDEX IF NOT EXISTS agreements_pair
    ON agreements(lhs_backend, rhs_backend, case_key, repetition);
CREATE INDEX IF NOT EXISTS timing_samples_workload
    ON timing_samples(case_key, backend, repetition, variant, sample_index);
"""

RUN_STATES = frozenset(
    {"running", "interrupted", "budget_exhausted", "complete", "failed"}
)
LEGACY_RUN_STATES = frozenset({"running", "budget_exhausted", "complete", "failed"})


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


def _selected_backend_order(
    case: dict[str, Any], backends: list[str], execution: dict[str, Any]
) -> tuple[str, ...]:
    eligible = set(case["eligible_backends"])
    if not eligible:
        selected = tuple(backends)
    else:
        eligible.add("silex")
        selected = tuple(backend for backend in backends if backend in eligible)
    excluded = {
        item.get("backend")
        for item in execution.get("backend_exclusions", [])
        if isinstance(item, dict) and item.get("workload") == case["workload"]
    }
    return tuple(backend for backend in selected if backend not in excluded)


def _selected_backends(
    case: dict[str, Any], backends: list[str], execution: dict[str, Any]
) -> set[str]:
    return set(_selected_backend_order(case, backends, execution))


def _agreement_pairs(
    backends: tuple[str, ...], required_pairs: list[list[str]]
) -> tuple[tuple[str, str], ...]:
    pairs = [
        (str(pair[0]), str(pair[1]))
        for pair in required_pairs
        if pair[0] in backends and pair[1] in backends
    ]
    present = {frozenset(pair) for pair in pairs}
    for lhs, rhs in itertools.combinations(backends, 2):
        if frozenset((lhs, rhs)) in present:
            continue
        if rhs == "silex":
            lhs, rhs = rhs, lhs
        pairs.append((lhs, rhs))
    return tuple(pairs)


def _expected_timing_coordinates(
    case: dict[str, Any], backend: str, execution: dict[str, Any]
) -> tuple[tuple[str, int], ...]:
    if (
        backend == "hecke"
        and case["workload"] != "sunit_proven"
        and execution.get("jit_repetitions") == 1
    ):
        return (("first_call", 0), ("repeat_call", 1))
    return (("standard", 0),)


def _effective_deadline(
    case: dict[str, Any], execution: dict[str, Any]
) -> tuple[float, str]:
    ceiling = float(execution["timeout_seconds"])
    hint = case.get("input", {}).get("timeout_seconds")
    if (
        isinstance(hint, (int, float))
        and not isinstance(hint, bool)
        and float(hint) > 0
    ):
        return min(ceiling, float(hint)), "min(campaign_ceiling,case_timeout_hint)"
    return ceiling, "campaign_ceiling"


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
        self.schema_version = CAMPAIGN_SCHEMA_VERSION
        self.read_only = False
        if existed:
            self.connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro", uri=True, timeout=30
            )
        else:
            self.connection = sqlite3.connect(self.path, timeout=30)
        try:
            self.connection.row_factory = sqlite3.Row
            if not existed:
                with self.connection:
                    self.connection.executescript(SCHEMA_SQL)
                    self.connection.execute(
                        f"PRAGMA user_version = {CAMPAIGN_SCHEMA_VERSION}"
                    )
            version = int(
                self.connection.execute("PRAGMA user_version").fetchone()[0]
            )
            self.schema_version = version
            if version not in {1, CAMPAIGN_SCHEMA_VERSION}:
                raise ValueError(
                    "run ledger schema must be 1 (report-only) or "
                    f"{CAMPAIGN_SCHEMA_VERSION}, got {version}"
                )
            if existed:
                self._validate_existing_run()
            if version == 1:
                self.read_only = True
                return
            if existed:
                # Existing ledgers are validated through a read-only handle
                # before they are reopened for a v2 resume.
                self.connection.close()
                self.connection = sqlite3.connect(self.path, timeout=30)
                self.connection.row_factory = sqlite3.Row
                self.connection.execute("PRAGMA foreign_keys = ON")
                self._validate_existing_run()
            # These settings are intentionally applied only after an existing
            # ledger and its embedded manifest pass the read-only checks above.
            self.connection.execute("PRAGMA busy_timeout = 30000")
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
        if row["schema_version"] != self.schema_version:
            raise ValueError(
                "run row schema_version must be "
                f"{self.schema_version}, got {row['schema_version']}"
            )
        states = RUN_STATES if self.schema_version == CAMPAIGN_SCHEMA_VERSION else LEGACY_RUN_STATES
        if row["state"] not in states:
            raise ValueError(f"invalid run ledger state: {row['state']!r}")
        manifest = _decode_object(row["manifest_json"], "run manifest")
        validate_campaign_manifest(manifest)
        self._require_manifest_schema(manifest)
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
                "SELECT case_key, workload, backend, repetition, order_index, "
                "status, success, timeout, target_wall_ns, process_wall_ns, "
                "observation_json FROM observations"
            ).fetchall()
            agreement_columns = (
                "case_key, repetition, lhs_backend, rhs_backend, status, success, "
                + (
                    "timing_eligible, ratio, "
                    if self.schema_version == 1
                    else ""
                )
                + "agreement_json"
            )
            agreement_rows = self.connection.execute(
                f"SELECT {agreement_columns} FROM agreements"
            ).fetchall()
            timing_rows = (
                self.connection.execute(
                    "SELECT case_key, backend, repetition, variant, sample_index, "
                    "status, timeout, target_wall_ns, target_cpu_ns, "
                    "process_wall_ns, timing_scope, effective_timeout_seconds, "
                    "sample_json FROM timing_samples"
                ).fetchall()
                if self.schema_version == CAMPAIGN_SCHEMA_VERSION
                else []
            )
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
        observations_by_key: dict[tuple[str, str, int], sqlite3.Row] = {}
        for row in observation_rows:
            case_key = row["case_key"]
            backend = row["backend"]
            if case_key not in stored_cases or case_key not in cases:
                raise ValueError("run ledger observation references an unknown case")
            if backend not in stored_engines or backend not in _selected_backends(
                cases[case_key], backends, plan["execution"]
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
                "order_index": row["order_index"],
                "success": bool(row["success"]),
                "timeout": bool(row["timeout"]),
                "target_wall_ns": row["target_wall_ns"],
                "process_wall_ns": row["process_wall_ns"],
            }
            if any(payload.get(key) != value for key, value in references.items()):
                raise ValueError(
                    "run ledger observation columns do not match its JSON payload"
                )
            observations_by_key[(case_key, backend, repetition)] = row

        agreement_statuses = {status.value for status in AgreementStatus}
        for row in agreement_rows:
            case_key = row["case_key"]
            lhs = row["lhs_backend"]
            rhs = row["rhs_backend"]
            if case_key not in stored_cases or case_key not in cases:
                raise ValueError("run ledger agreement references an unknown case")
            selected = _selected_backends(
                cases[case_key], backends, plan["execution"]
            )
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
                "success": bool(row["success"]),
            }
            if any(payload.get(key) != value for key, value in references.items()):
                raise ValueError(
                    "run ledger agreement columns do not match its JSON payload"
                )
            if self.schema_version == 1:
                if payload.get("timing_eligible") is not bool(
                    row["timing_eligible"]
                ) or payload.get("speedup_baseline_over_candidate") != row["ratio"]:
                    raise ValueError(
                        "run ledger agreement timing columns do not match its JSON payload"
                    )

        observation_keys = {
            (row["case_key"], row["backend"], row["repetition"])
            for row in observation_rows
        }
        sample_keys: set[tuple[str, str, int, str, int]] = set()
        timing_rows_by_key: dict[
            tuple[str, str, int, str, int], sqlite3.Row
        ] = {}
        samples_by_parent: dict[
            tuple[str, str, int], set[tuple[str, int]]
        ] = {}
        for row in timing_rows:
            parent = (row["case_key"], row["backend"], row["repetition"])
            if parent not in observation_keys:
                raise ValueError("run ledger timing sample references an unknown observation")
            payload = _decode_object(
                row["sample_json"],
                "timing sample "
                f"{row['case_key']}/{row['backend']}/{row['repetition']}/"
                f"{row['variant']}/{row['sample_index']}",
            )
            references = {
                "case_key": row["case_key"],
                "workload": cases[row["case_key"]]["workload"],
                "backend": row["backend"],
                "repetition": row["repetition"],
                "variant": row["variant"],
                "sample_index": row["sample_index"],
                "status": row["status"],
                "timeout": bool(row["timeout"]),
                "target_wall_ns": row["target_wall_ns"],
                "target_cpu_ns": row["target_cpu_ns"],
                "process_wall_ns": row["process_wall_ns"],
                "timing_scope": row["timing_scope"],
                "effective_timeout_seconds": row["effective_timeout_seconds"],
            }
            if any(payload.get(key) != value for key, value in references.items()):
                raise ValueError(
                    "run ledger timing sample columns do not match its JSON payload"
                )
            if row["status"] not in observation_statuses:
                raise ValueError("run ledger timing sample has an invalid status")
            if bool(row["timeout"]) != (row["status"] == "timeout"):
                raise ValueError(
                    "run ledger timing sample timeout flag and status disagree"
                )
            if not isinstance(row["variant"], str) or not row["variant"]:
                raise ValueError("run ledger timing sample variant must be nonempty")
            if type(row["sample_index"]) is not int or row["sample_index"] < 0:
                raise ValueError("run ledger timing sample index must be nonnegative")
            if not isinstance(row["timing_scope"], str) or not row["timing_scope"]:
                raise ValueError("run ledger timing sample scope must be nonempty")
            if (
                not isinstance(row["effective_timeout_seconds"], (int, float))
                or not math.isfinite(float(row["effective_timeout_seconds"]))
                or row["effective_timeout_seconds"] <= 0
            ):
                raise ValueError("run ledger timing sample deadline must be positive")
            if not isinstance(payload.get("timeout_source"), str) or not payload[
                "timeout_source"
            ]:
                raise ValueError(
                    "run ledger timing sample timeout source must be nonempty"
                )
            expected_timeout, expected_source = _effective_deadline(
                cases[row["case_key"]], plan["execution"]
            )
            if (
                float(row["effective_timeout_seconds"]) != expected_timeout
                or payload["timeout_source"] != expected_source
            ):
                raise ValueError(
                    "run ledger timing sample deadline differs from the planned case deadline"
                )
            for clock in (
                "target_wall_ns",
                "target_cpu_ns",
                "process_wall_ns",
            ):
                value = row[clock]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(
                        f"run ledger timing sample {clock} must be nonnegative or null"
                    )
            if row["status"] == "ok" and type(row["target_wall_ns"]) is not int:
                raise ValueError(
                    "successful timing samples require a nonnegative target wall clock"
                )
            if row["status"] == "ok" and (
                not isinstance(payload.get("wall_clock"), str)
                or not payload["wall_clock"].strip()
            ):
                raise ValueError(
                    "successful timing samples require a declared wall clock"
                )
            key = (*parent, row["variant"], row["sample_index"])
            if key in sample_keys:
                raise ValueError("run ledger contains a duplicate timing sample")
            sample_keys.add(key)
            timing_rows_by_key[key] = row
            samples_by_parent.setdefault(parent, set()).add(
                (row["variant"], row["sample_index"])
            )

        if self.schema_version == CAMPAIGN_SCHEMA_VERSION:
            for parent, observation_row in observations_by_key.items():
                if not bool(observation_row["success"]):
                    continue
                case_key, backend, _ = parent
                expected = set(
                    _expected_timing_coordinates(
                        cases[case_key], backend, plan["execution"]
                    )
                )
                actual = samples_by_parent.get(parent, set())
                if actual != expected:
                    raise ValueError(
                        "successful observation timing variants do not match the plan: "
                        f"{case_key}/{backend} expected {sorted(expected)}, "
                        f"got {sorted(actual)}"
                    )
                for variant, sample_index in expected:
                    sample_row = timing_rows_by_key[
                        (*parent, variant, sample_index)
                    ]
                    if sample_row["status"] != ObservationStatus.OK.value:
                        raise ValueError(
                            "successful observation contains an unsuccessful timing sample"
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
            if self.schema_version == CAMPAIGN_SCHEMA_VERSION:
                expected_agreements = {
                    (case_key, repetition, lhs, rhs)
                    for case_key, case in cases.items()
                    for lhs, rhs in _agreement_pairs(
                        _selected_backend_order(case, backends, plan["execution"]),
                        plan["required_pairs"],
                    )
                    for repetition in range(repetitions)
                }
                actual_agreements = {
                    (
                        row["case_key"],
                        row["repetition"],
                        row["lhs_backend"],
                        row["rhs_backend"],
                    )
                    for row in agreement_rows
                }
                if actual_agreements != expected_agreements:
                    raise ValueError(
                        "terminal run ledger agreement coordinates do not match the plan"
                    )
            if self.schema_version == CAMPAIGN_SCHEMA_VERSION:
                successful = {
                    (row["case_key"], row["backend"], row["repetition"])
                    for row in observation_rows
                    if bool(row["success"])
                }
                sampled = {(row[0], row[1], row[2]) for row in sample_keys}
                if not successful.issubset(sampled):
                    raise ValueError(
                        "terminal run ledger is missing timing samples for successful observations"
                    )

    def __enter__(self) -> "RunLedger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _require_writable(self) -> None:
        if self.read_only:
            raise ValueError(
                "campaign schema v1 ledgers are report-only and cannot be resumed or mutated"
            )

    def _require_manifest_schema(self, manifest: dict[str, Any]) -> None:
        if manifest["schema_version"] != self.schema_version:
            raise ValueError(
                "manifest schema_version does not match the ledger: "
                f"{manifest['schema_version']} != {self.schema_version}"
            )

    def initialize(self, fingerprint: str, manifest: dict[str, Any]) -> None:
        self._require_writable()
        validate_campaign_manifest(manifest)
        self._require_manifest_schema(manifest)
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
        if row["schema_version"] != self.schema_version:
            raise ValueError(
                "run row schema_version must be "
                f"{self.schema_version}, got {row['schema_version']}"
            )
        try:
            value = json.loads(row["manifest_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid run manifest JSON: {exc}") from exc
        validate_campaign_manifest(value)
        self._require_manifest_schema(value)
        return value

    def fingerprint(self) -> str:
        row = self.connection.execute("SELECT fingerprint FROM run WHERE singleton = 1").fetchone()
        if row is None:
            raise ValueError("run ledger is not initialized")
        return str(row["fingerprint"])

    def state(self) -> str:
        row = self.connection.execute("SELECT state FROM run WHERE singleton = 1").fetchone()
        return "uninitialized" if row is None else str(row["state"])

    def set_state(self, state: str) -> None:
        self._require_writable()
        if state not in RUN_STATES:
            raise ValueError(f"invalid run ledger state: {state!r}")
        if state in {"complete", "failed"}:
            self._validate_existing_rows(self.manifest(), state=state)
        with self.connection:
            self.connection.execute(
                "UPDATE run SET state = ?, updated_at = ? WHERE singleton = 1",
                (state, _now()),
            )

    def put_engines(self, engines: Iterable[EngineInfo]) -> None:
        self._require_writable()
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
        self._require_writable()
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
        self._require_writable()
        case_row = self.connection.execute(
            "SELECT case_json FROM cases WHERE case_key = ?", (observation.case_key,)
        ).fetchone()
        if case_row is None:
            raise ValueError("observation references an unstored case")
        case_payload = _decode_object(
            case_row["case_json"], f"case {observation.case_key}"
        )
        execution = self.manifest()["plan"]["execution"]
        expected_timeout, expected_timeout_source = _effective_deadline(
            case_payload, execution
        )
        encoded = _json(observation.to_json())
        sample_rows: list[tuple[TimingSample, str]] = []
        sample_keys: set[tuple[str, int]] = set()
        for sample in observation.timing_samples:
            if (
                sample.case_key != observation.case_key
                or sample.workload != observation.workload
                or sample.backend != observation.backend
                or sample.repetition != observation.repetition
            ):
                raise ValueError("timing sample coordinates do not match observation")
            key = (sample.variant, sample.sample_index)
            if key in sample_keys:
                raise ValueError("timing sample variants and indexes must be unique")
            sample_keys.add(key)
            if (
                not isinstance(sample.variant, str)
                or not sample.variant
                or type(sample.sample_index) is not int
                or sample.sample_index < 0
            ):
                raise ValueError("timing sample identity is invalid")
            if type(sample.timeout) is not bool or sample.timeout != (sample.status is ObservationStatus.TIMEOUT):
                raise ValueError("timing sample timeout flag and status disagree")
            if (
                not math.isfinite(sample.effective_timeout_seconds)
                or sample.effective_timeout_seconds <= 0
                or not sample.timeout_source
                or not sample.timing_scope
            ):
                raise ValueError("timing sample scope or deadline is invalid")
            if (
                sample.effective_timeout_seconds != expected_timeout
                or sample.timeout_source != expected_timeout_source
            ):
                raise ValueError(
                    "timing sample deadline differs from the planned case deadline"
                )
            for clock in ("target_wall_ns", "target_cpu_ns", "process_wall_ns"):
                value = getattr(sample, clock)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"timing sample {clock} must be nonnegative or null")
            if sample.status is ObservationStatus.OK and type(sample.target_wall_ns) is not int:
                raise ValueError(
                    "successful timing samples require a nonnegative target wall clock"
                )
            if sample.status is ObservationStatus.OK and (
                not isinstance(sample.wall_clock, str)
                or not sample.wall_clock.strip()
            ):
                raise ValueError(
                    "successful timing samples require a declared wall clock"
                )
            sample_rows.append((sample, _json(sample.to_json())))
        if observation.success:
            expected_coordinates = set(
                _expected_timing_coordinates(
                    case_payload, observation.backend, execution
                )
            )
            if sample_keys != expected_coordinates:
                raise ValueError(
                    "successful observation timing variants do not match the plan: "
                    f"expected {sorted(expected_coordinates)}, got {sorted(sample_keys)}"
                )
            if any(
                sample.status is not ObservationStatus.OK for sample, _ in sample_rows
            ):
                raise ValueError(
                    "successful observation contains an unsuccessful timing sample"
                )
        with self.connection:
            existing = self.connection.execute(
                "SELECT observation_json FROM observations WHERE case_key = ? AND backend = ? AND repetition = ?",
                (observation.case_key, observation.backend, observation.repetition),
            ).fetchone()
            if existing is not None:
                existing_samples = self.connection.execute(
                    "SELECT variant, sample_index, sample_json FROM timing_samples "
                    "WHERE case_key = ? AND backend = ? AND repetition = ?",
                    (observation.case_key, observation.backend, observation.repetition),
                ).fetchall()
                children = {
                    (row["variant"], row["sample_index"]): row["sample_json"]
                    for row in existing_samples
                }
                requested_children = {
                    (sample.variant, sample.sample_index): sample_json
                    for sample, sample_json in sample_rows
                }
                if existing["observation_json"] != encoded or children != requested_children:
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
            for sample, sample_json in sample_rows:
                self.connection.execute(
                    """INSERT INTO timing_samples VALUES
                       (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sample.case_key,
                        sample.backend,
                        sample.repetition,
                        sample.variant,
                        sample.sample_index,
                        sample.status.value,
                        int(sample.timeout),
                        sample.target_wall_ns,
                        sample.target_cpu_ns,
                        sample.process_wall_ns,
                        sample.timing_scope,
                        sample.effective_timeout_seconds,
                        sample_json,
                    ),
                )

    def observation_count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
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
    ) -> None:
        self._require_writable()
        payload = result.to_json()
        payload.update(
            {
                "case_key": case_key,
                "repetition": repetition,
                "lhs_backend": lhs_backend,
                "rhs_backend": rhs_backend,
            }
        )
        encoded = _json(payload)
        with self.connection:
            self.connection.execute(
                """INSERT INTO agreements VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(case_key, repetition, lhs_backend, rhs_backend)
                   DO UPDATE SET status=excluded.status,
                                 success=excluded.success,
                                 agreement_json=excluded.agreement_json""",
                (
                    case_key,
                    repetition,
                    lhs_backend,
                    rhs_backend,
                    payload["status"],
                    int(result.success),
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

    def timing_samples(self) -> list[dict[str, Any]]:
        if self.schema_version == CAMPAIGN_SCHEMA_VERSION:
            rows = self.connection.execute(
                "SELECT sample_json FROM timing_samples "
                "ORDER BY case_key, backend, repetition, variant, sample_index"
            ).fetchall()
            return [json.loads(row["sample_json"]) for row in rows]

        # Schema-v1 observations carried one timing inline.  Normalize those
        # rows in memory without altering the legacy database or fingerprint.
        timeout_seconds = float(
            self.manifest()["plan"]["execution"]["timeout_seconds"]
        )
        samples: list[dict[str, Any]] = []
        for observation in self.observations():
            internal = observation.get("internal_timing")
            internal = internal if isinstance(internal, dict) else {}
            target_wall_ns = observation.get("target_wall_ns")
            target_cpu_ms = internal.get("target_cpu_ms")
            target_cpu_ns = (
                int(round(float(target_cpu_ms) * 1_000_000))
                if isinstance(target_cpu_ms, (int, float))
                and not isinstance(target_cpu_ms, bool)
                and target_cpu_ms >= 0
                else None
            )
            samples.append(
                {
                    "case_key": observation["case_key"],
                    "workload": observation["workload"],
                    "backend": observation["backend"],
                    "repetition": observation["repetition"],
                    "variant": "standard",
                    "sample_index": 0,
                    "status": observation["status"],
                    "timeout": observation.get("timeout") is True,
                    "target_wall_ns": target_wall_ns,
                    "target_cpu_ns": target_cpu_ns,
                    "process_wall_ns": observation.get("process_wall_ns"),
                    "timing_scope": str(
                        internal.get("scope") or "legacy_v1_observation_target"
                    ),
                    "effective_timeout_seconds": timeout_seconds,
                    "timeout_source": "legacy_campaign_ceiling",
                    "wall_clock": internal.get("wall_clock"),
                    "cpu_clock": internal.get("algorithm_clock"),
                    "internal_timing": internal,
                    "diagnostics": {"source_ledger_schema_version": 1},
                }
            )
        return samples

    def snapshot(self) -> dict[str, Any]:
        self.connection.execute("BEGIN")
        try:
            return {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "source_ledger_schema_version": self.schema_version,
                "fingerprint": self.fingerprint(),
                "state": self.state(),
                "manifest": self.manifest(),
                "engines": self.engines(),
                "cases": self.cases(),
                "observations": self.observations(),
                "timing_samples": self.timing_samples(),
                "agreements": self.agreements(),
            }
        finally:
            self.connection.execute("ROLLBACK")
