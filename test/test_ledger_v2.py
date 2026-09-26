from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import silex_bench.reporting as reporting
from silex_bench.campaign import (
    _manifest,
    build_plan,
    invocation_context,
    probe_engines,
    run_campaign,
)
from silex_bench.configuration import (
    ProfileConfig,
    RunOverrides,
    SuiteConfig,
    ToolConfig,
    resolved_fingerprint,
)
from silex_bench.contracts import (
    AgreementResult,
    BackendDescriptor,
    EngineInfo,
    Observation,
    ObservationStatus,
    Registry,
    TimingSample,
    campaign_manifest_fingerprint,
    canonical_json,
    validate_campaign_manifest,
)
from silex_bench.ledger import RunLedger
from silex_bench.reporting import generate_report
from silex_bench.workloads import MAXIMAL_ORDER, NumberFieldContract


class InvocationRecorder:
    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.interrupt_on_call = interrupt_on_call
        self.calls = 0
        self.contexts: list[tuple[str, float, str]] = []

    def record(self, backend: str, timeout: float, source: str) -> None:
        self.calls += 1
        self.contexts.append((backend, timeout, source))
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt


@dataclasses.dataclass(frozen=True)
class SyntheticImplementation:
    backend: str
    recorder: InvocationRecorder
    variants: tuple[str, ...] = ("standard",)
    workload: str = MAXIMAL_ORDER

    def run(self, case, *, repetition, order_index, warmup, context, contract):
        if warmup is not None:
            raise AssertionError("schema-v2 adapters must not receive warmup fields")
        self.recorder.record(
            self.backend, context.timeout_seconds, context.timeout_source
        )
        result = {"maximal_order_discriminant": "5"}
        validation = contract.validate_observation(case, self.backend, result, {})
        timing_samples = tuple(
            TimingSample(
                case_key=case.key,
                workload=case.workload,
                backend=self.backend,
                repetition=repetition,
                variant=variant,
                sample_index=index,
                status=ObservationStatus.OK,
                timeout=False,
                target_wall_ns=10_000_000 - index * 1_000_000,
                target_cpu_ns=9_000_000 - index * 1_000_000,
                process_wall_ns=22_000_000,
                timing_scope=contract.timing_scope,
                effective_timeout_seconds=context.timeout_seconds,
                timeout_source=context.timeout_source,
                wall_clock="CLOCK_MONOTONIC",
                cpu_clock="CLOCK_PROCESS_CPUTIME_ID",
                internal_timing={"fixture": True},
                diagnostics={},
            )
            for index, variant in enumerate(self.variants)
        )
        return Observation(
            case_key=case.key,
            workload=case.workload,
            backend=self.backend,
            repetition=repetition,
            order_index=order_index,
            status=ObservationStatus.OK,
            success=True,
            timeout=False,
            result=result,
            proof={},
            validation=validation,
            target_wall_ns=timing_samples[0].target_wall_ns,
            process_wall_ns=timing_samples[0].process_wall_ns,
            internal_timing={"fixture": True},
            engine_identity={"engine": self.backend, "version": "fixture"},
            command=(self.backend, "fixture-command"),
            stdout="",
            stderr="",
            timing_samples=timing_samples,
        )


def synthetic_registry(
    recorder: InvocationRecorder,
    *,
    variants: tuple[str, ...] = ("standard",),
    backends: tuple[str, ...] = ("silex", "pari"),
) -> Registry:
    contract = NumberFieldContract(MAXIMAL_ORDER, "Maximal order")
    descriptors: list[BackendDescriptor] = []
    for backend in backends:
        implementation = SyntheticImplementation(backend, recorder, variants)

        def probe(context, descriptor, *, selected=backend):
            del context, descriptor
            return EngineInfo(
                backend=selected,
                display_name=selected,
                available=True,
                capabilities=(MAXIMAL_ORDER,),
                identity={"engine": selected, "version": "fixture"},
            )

        descriptors.append(
            BackendDescriptor(
                id=backend,
                display_name=backend,
                implementations={MAXIMAL_ORDER: implementation},
                probe_callback=probe,
            )
        )
    return Registry((contract,), tuple(descriptors))


def build_fixture_plan(
    root: Path,
    registry: Registry,
    *,
    repetitions: int = 1,
    timeout_seconds: float = 10.0,
    case_timeout_seconds: float = 7.0,
    backends: tuple[str, ...] = ("silex", "pari"),
    jit_repetitions: int = 0,
):
    corpus = root / "fields.json"
    corpus.write_text(
        json.dumps(
            {
                "fields": [
                    {
                        "id": "real_quadratic_5_proven",
                        "coefficients_low_to_high": [-5, 0, 1],
                        "degree": 2,
                        "maximal_order_discriminant": 5,
                        "expected_success": True,
                        "optimization_external_engines": [
                            backend for backend in backends if backend != "silex"
                        ],
                        "timeout_seconds": case_timeout_seconds,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    marker = root / "config.toml"
    marker.write_text("schema_version = 1\n", encoding="utf-8")
    suite = SuiteConfig(
        path=marker,
        id="ledger-v2-fixture",
        title="ledger-v2-fixture",
        workloads=(MAXIMAL_ORDER,),
        backends=backends,
        required_pairs=(("silex", backends[1]),),
        corpora={"number_fields": corpus},
        reports={},
        sha256="suite-fixture",
    )
    profile = ProfileConfig(
        path=marker,
        id="ledger-v2-fixture",
        description="",
        include_tags=("quick",),
        exclude_tags=(),
        backend_exclusions=(),
        repetitions=repetitions,
        jit_repetitions=jit_repetitions,
        timeout_seconds=timeout_seconds,
        budget_seconds=None,
        cpu=None,
        threads=1,
        publication=False,
        minimum_repetitions=1,
        require_clean_sources=False,
        metrics={},
        sha256="profile-fixture",
    )
    return build_plan(
        root,
        suite,
        profile,
        ToolConfig(None, {}, None),
        RunOverrides(),
        registry,
        performance=True,
    )


def fixture_case_context(plan):
    case = plan.cases[0]
    ceiling = float(plan.execution["timeout_seconds"])
    hint = float(case.input["timeout_seconds"])
    return dataclasses.replace(
        invocation_context(plan),
        timeout_seconds=min(ceiling, hint),
        timeout_source="min(campaign_ceiling,case_timeout_hint)",
    )


def legacy_manifest(current: dict[str, object]) -> dict[str, object]:
    manifest = copy.deepcopy(current)
    manifest["schema_version"] = 1
    plan = manifest["plan"]
    assert isinstance(plan, dict)
    plan["schema_version"] = 1
    execution = plan["execution"]
    assert isinstance(execution, dict)
    execution["warmups"] = execution.pop("jit_repetitions")
    execution.pop("backend_exclusions")
    plan["nominal_timeout_product_seconds"] = (
        int(plan["sample_count"]) * float(execution["timeout_seconds"])
    )
    cases = plan["cases"]
    cases_sha256 = hashlib.sha256(canonical_json(cases).encode()).hexdigest()
    plan["plan_fingerprint"] = hashlib.sha256(
        canonical_json(
            {
                "schema_version": 1,
                "suite_sha256": plan["suite_sha256"],
                "profile_sha256": plan["profile_sha256"],
                "tools_sha256": plan["tools_sha256"],
                "workloads": plan["workloads"],
                "backends": plan["backends"],
                "adapter_capabilities": plan["adapter_capabilities"],
                "required_pairs": plan["required_pairs"],
                "execution": execution,
                "cases_sha256": cases_sha256,
                "overrides": plan["overrides"],
            }
        ).encode()
    ).hexdigest()
    manifest["timing"] = {
        "primary_clock": "target_wall_ns",
        "scope": "workload contract: supervisor_marked_target or whole_process",
        "ratio_definition": "baseline_over_candidate",
        "performance_execution": "serial_paired_blocks",
    }
    manifest["run_fingerprint"] = campaign_manifest_fingerprint(manifest)
    return validate_campaign_manifest(manifest)


def materialize_legacy_ledger(source: Path, destination: Path) -> Path:
    destination.mkdir()
    target = destination / "run.sqlite"
    with (
        contextlib.closing(sqlite3.connect(source)) as old,
        contextlib.closing(sqlite3.connect(target)) as legacy,
    ):
        old.backup(legacy)
        row = legacy.execute("SELECT manifest_json FROM run WHERE singleton = 1").fetchone()
        assert row is not None
        manifest = legacy_manifest(json.loads(row[0]))
        fingerprint = manifest["run_fingerprint"]
        agreements = legacy.execute(
            "SELECT case_key, repetition, lhs_backend, rhs_backend, agreement_json "
            "FROM agreements"
        ).fetchall()
        with legacy:
            legacy.execute(
                "ALTER TABLE agreements ADD COLUMN timing_eligible "
                "INTEGER NOT NULL DEFAULT 0"
            )
            legacy.execute("ALTER TABLE agreements ADD COLUMN ratio REAL")
            for case_key, repetition, lhs, rhs, encoded in agreements:
                payload = json.loads(encoded)
                payload["timing_eligible"] = False
                payload["speedup_baseline_over_candidate"] = None
                legacy.execute(
                    "UPDATE agreements SET timing_eligible = 0, ratio = NULL, "
                    "agreement_json = ? WHERE case_key = ? AND repetition = ? "
                    "AND lhs_backend = ? AND rhs_backend = ?",
                    (
                        canonical_json(payload),
                        case_key,
                        repetition,
                        lhs,
                        rhs,
                    ),
                )
            legacy.execute("DROP TABLE timing_samples")
            legacy.execute(
                "UPDATE run SET schema_version = 1, fingerprint = ?, manifest_json = ? "
                "WHERE singleton = 1",
                (fingerprint, canonical_json(manifest)),
            )
            legacy.execute("PRAGMA user_version = 1")
    return target


class TimingSampleLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _completed_run(self, name: str):
        recorder = InvocationRecorder()
        registry = synthetic_registry(recorder)
        plan = build_fixture_plan(self.root, registry)
        run_dir = self.root / name
        snapshot = run_campaign(plan, run_dir)
        self.assertEqual(snapshot["state"], "complete")
        return plan, run_dir, snapshot

    def test_parent_and_timing_child_persist_atomically(self) -> None:
        plan, source_dir, snapshot = self._completed_run("source")
        self.assertEqual(len(snapshot["observations"]), 2)
        self.assertEqual(len(snapshot["timing_samples"]), 2)
        for observation in snapshot["observations"]:
            self.assertNotIn("timing_samples", observation)
            children = [
                sample
                for sample in snapshot["timing_samples"]
                if (
                    sample["case_key"],
                    sample["backend"],
                    sample["repetition"],
                )
                == (
                    observation["case_key"],
                    observation["backend"],
                    observation["repetition"],
                )
            ]
            self.assertEqual(
                [(row["variant"], row["sample_index"]) for row in children],
                [("standard", 0)],
            )
            self.assertTrue(
                all("command" not in row and "engine_identity" not in row for row in children)
            )

        with RunLedger(source_dir) as source:
            manifest = source.manifest()
        empty_dir = self.root / "atomic"
        with RunLedger(empty_dir, create=True) as ledger:
            ledger.initialize(manifest["run_fingerprint"], manifest)
            ledger.put_engines(probe_engines(plan))
            ledger.put_cases(plan.cases)
            case = plan.cases[0]
            implementation = plan.registry.backends["silex"].implementations[
                case.workload
            ]
            observation = implementation.run(
                case,
                repetition=0,
                order_index=0,
                warmup=None,
                context=fixture_case_context(plan),
                contract=plan.registry.workloads[case.workload],
            )

            duplicate = dataclasses.replace(
                observation,
                timing_samples=(
                    observation.timing_samples[0],
                    observation.timing_samples[0],
                ),
            )
            with self.assertRaisesRegex(ValueError, "must be unique"):
                ledger.put_observation(duplicate)
            self.assertEqual(ledger.observation_count(), 0)
            self.assertEqual(ledger.timing_samples(), [])

            mismatched = dataclasses.replace(
                observation.timing_samples[0], backend="pari"
            )
            with self.assertRaisesRegex(ValueError, "coordinates do not match"):
                ledger.put_observation(
                    dataclasses.replace(observation, timing_samples=(mismatched,))
                )
            self.assertEqual(ledger.observation_count(), 0)
            self.assertEqual(ledger.timing_samples(), [])

            with ledger.connection:
                ledger.connection.execute(
                    "CREATE TRIGGER reject_fixture_timing_sample "
                    "BEFORE INSERT ON timing_samples BEGIN "
                    "SELECT RAISE(ABORT, 'fixture timing failure'); END"
                )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError, "fixture timing failure"
            ):
                ledger.put_observation(observation)
            self.assertEqual(ledger.observation_count(), 0)
            self.assertEqual(ledger.timing_samples(), [])
            with ledger.connection:
                ledger.connection.execute("DROP TRIGGER reject_fixture_timing_sample")

            ledger.put_observation(observation)
            self.assertEqual(ledger.observation_count(), 1)
            self.assertEqual(len(ledger.timing_samples()), 1)

    def test_duplicate_observation_checks_timing_children(self) -> None:
        plan, _, snapshot = self._completed_run("collision-source")
        case = plan.cases[0]
        observation = plan.registry.backends["silex"].implementations[case.workload].run(
            case, repetition=0, order_index=0, warmup=None,
            context=fixture_case_context(plan), contract=plan.registry.workloads[case.workload],
        )
        with RunLedger(self.root / "collision", create=True) as ledger:
            ledger.initialize(snapshot["fingerprint"], snapshot["manifest"])
            ledger.put_engines(probe_engines(plan))
            ledger.put_cases(plan.cases)
            ledger.put_observation(observation)
            original = ledger.timing_samples()
            ledger.put_observation(observation)
            for changes in ({"target_wall_ns": 77_777_777}, {"diagnostics": {"changed": True}}):
                with self.subTest(changes=changes):
                    altered = dataclasses.replace(observation.timing_samples[0], **changes)
                    with self.assertRaisesRegex(ValueError, "collision"):
                        ledger.put_observation(dataclasses.replace(observation, timing_samples=(altered,)))
                    self.assertEqual(ledger.timing_samples(), original)
                    self.assertEqual(ledger.observation_count(), 1)

    def test_invalid_timing_samples_are_rejected_before_checkpoint(self) -> None:
        plan, _, snapshot = self._completed_run("invalid-source")
        case = plan.cases[0]
        observation = plan.registry.backends["silex"].implementations[case.workload].run(
            case, repetition=0, order_index=0, warmup=None,
            context=fixture_case_context(plan), contract=plan.registry.workloads[case.workload],
        )
        invalid = [
            {clock: value}
            for clock in ("target_wall_ns", "target_cpu_ns", "process_wall_ns")
            for value in (-1, 1.5, True)
        ] + [{"sample_index": False}, {"sample_index": 0.0}, {"timeout": 0}]
        with RunLedger(self.root / "invalid", create=True) as ledger:
            ledger.initialize(snapshot["fingerprint"], snapshot["manifest"])
            ledger.put_engines(probe_engines(plan))
            ledger.put_cases(plan.cases)
            for changes in invalid:
                with self.subTest(changes=changes):
                    sample = dataclasses.replace(observation.timing_samples[0], **changes)
                    with self.assertRaises(ValueError):
                        ledger.put_observation(dataclasses.replace(observation, timing_samples=(sample,)))
                    self.assertEqual(ledger.observation_count(), 0)
                    self.assertEqual(ledger.timing_samples(), [])

    def test_initialize_rejects_a_legacy_manifest_in_a_v2_ledger(self) -> None:
        _, _, snapshot = self._completed_run("manifest-source")
        manifest = legacy_manifest(snapshot["manifest"])
        with RunLedger(self.root / "new", create=True) as ledger:
            with self.assertRaisesRegex(ValueError, "manifest schema_version.*ledger"):
                ledger.initialize(manifest["run_fingerprint"], manifest)
            self.assertEqual(ledger.state(), "uninitialized")

    def test_reopen_rejects_manifest_ledger_schema_mismatch_without_mutation(self) -> None:
        _, run_dir, snapshot = self._completed_run("schema-source")
        legacy_path = materialize_legacy_ledger(run_dir / "run.sqlite", self.root / "legacy")
        for path, manifest in (
            (run_dir / "run.sqlite", legacy_manifest(snapshot["manifest"])),
            (legacy_path, snapshot["manifest"]),
        ):
            with self.subTest(path=path):
                with contextlib.closing(sqlite3.connect(path)) as connection, connection:
                    connection.execute(
                        "UPDATE run SET manifest_json = ?, fingerprint = ?",
                        (canonical_json(manifest), manifest["run_fingerprint"]),
                    )
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "manifest schema_version.*ledger"):
                    RunLedger(path.parent)
                self.assertEqual(path.read_bytes(), original)

    def test_zero_resolution_clock_sample_is_retained_but_not_timing_eligible(self) -> None:
        plan, source_dir, _ = self._completed_run("zero-source")
        case = plan.cases[0]
        implementation = plan.registry.backends["pari"].implementations[
            case.workload
        ]
        observation = implementation.run(
            case,
            repetition=0,
            order_index=0,
            warmup=None,
            context=fixture_case_context(plan),
            contract=plan.registry.workloads[case.workload],
        )
        zero_sample = dataclasses.replace(
            observation.timing_samples[0], target_wall_ns=0
        )
        observation = dataclasses.replace(
            observation,
            target_wall_ns=0,
            timing_samples=(zero_sample,),
        )

        with RunLedger(source_dir) as source:
            manifest = source.manifest()
        run_dir = self.root / "zero-target"
        with RunLedger(run_dir, create=True) as ledger:
            ledger.initialize(manifest["run_fingerprint"], manifest)
            ledger.put_engines(probe_engines(plan))
            ledger.put_cases(plan.cases)
            ledger.put_observation(observation)

        with RunLedger(run_dir) as ledger:
            sample = ledger.timing_samples()[0]
        self.assertEqual(sample["target_wall_ns"], 0)
        self.assertFalse(zero_sample.timing_eligible)

    def test_v2_timing_sample_column_json_tamper_is_rejected(self) -> None:
        _, run_dir, _ = self._completed_run("tamper")
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                connection.execute(
                    "UPDATE timing_samples SET target_wall_ns = target_wall_ns + 1 "
                    "WHERE backend = 'pari'"
                )
        with self.assertRaisesRegex(ValueError, "timing sample columns do not match"):
            RunLedger(run_dir)

    def test_v2_agreement_schema_has_no_speedup_columns(self) -> None:
        _, run_dir, _ = self._completed_run("agreement-schema")
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(agreements)")
            }
        self.assertNotIn("timing_eligible", columns)
        self.assertNotIn("ratio", columns)

    def test_v2_orphan_timing_sample_is_rejected(self) -> None:
        _, run_dir, _ = self._completed_run("orphan")
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM observations WHERE backend = 'pari' AND repetition = 0"
                )
        with self.assertRaisesRegex(
            ValueError, "timing sample references an unknown observation"
        ):
            RunLedger(run_dir)

    def test_successful_observation_requires_the_exact_planned_variant(self) -> None:
        plan, run_dir, _ = self._completed_run("variant-source")
        case = plan.cases[0]
        implementation = plan.registry.backends["silex"].implementations[
            case.workload
        ]
        observation = implementation.run(
            case,
            repetition=0,
            order_index=0,
            warmup=None,
            context=fixture_case_context(plan),
            contract=plan.registry.workloads[case.workload],
        )
        extra = dataclasses.replace(
            observation.timing_samples[0], variant="first_call", sample_index=1
        )
        empty_dir = self.root / "variant-target"
        with RunLedger(empty_dir, create=True) as ledger:
            with RunLedger(run_dir) as source:
                manifest = source.manifest()
            ledger.initialize(manifest["run_fingerprint"], manifest)
            ledger.put_engines(probe_engines(plan))
            ledger.put_cases(plan.cases)
            with self.assertRaisesRegex(ValueError, "variants do not match"):
                ledger.put_observation(
                    dataclasses.replace(
                        observation,
                        timing_samples=(*observation.timing_samples, extra),
                    )
                )

    def test_ordinary_hecke_jit_observation_requires_first_and_repeat(self) -> None:
        recorder = InvocationRecorder()
        registry = synthetic_registry(
            recorder,
            variants=("first_call", "repeat_call"),
            backends=("silex", "hecke"),
        )
        plan = build_fixture_plan(
            self.root,
            registry,
            backends=("silex", "hecke"),
            jit_repetitions=1,
        )
        # The fixture emits the paired JIT shape for both adapters, so exercise
        # the Hecke row directly and verify the non-JIT adapter is rejected.
        run_dir = self.root / "hecke-shape"
        probes = probe_engines(plan)
        manifest = _manifest(plan, probes)
        manifest["run_fingerprint"] = resolved_fingerprint(manifest)
        case = plan.cases[0]
        context = fixture_case_context(plan)
        with RunLedger(run_dir, create=True) as ledger:
            ledger.initialize(manifest["run_fingerprint"], manifest)
            ledger.put_engines(probes)
            ledger.put_cases(plan.cases)
            hecke = plan.registry.backends["hecke"].implementations[case.workload].run(
                case,
                repetition=0,
                order_index=0,
                warmup=None,
                context=context,
                contract=plan.registry.workloads[case.workload],
            )
            ledger.put_observation(hecke)
            self.assertEqual(
                [(row["variant"], row["sample_index"]) for row in ledger.timing_samples()],
                [("first_call", 0), ("repeat_call", 1)],
            )

    def test_tampered_case_deadline_is_rejected_even_when_json_matches(self) -> None:
        _, run_dir, _ = self._completed_run("deadline")
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            row = connection.execute(
                "SELECT rowid, sample_json FROM timing_samples LIMIT 1"
            ).fetchone()
            payload = json.loads(row[1])
            payload["effective_timeout_seconds"] = 6.0
            payload["timeout_source"] = "campaign_ceiling"
            with connection:
                connection.execute(
                    "UPDATE timing_samples SET effective_timeout_seconds = 6.0, "
                    "sample_json = ? WHERE rowid = ?",
                    (canonical_json(payload), row[0]),
                )
        with self.assertRaisesRegex(ValueError, "planned case deadline"):
            RunLedger(run_dir)

    def test_terminal_run_requires_every_planned_agreement(self) -> None:
        _, run_dir, _ = self._completed_run("agreement")
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                connection.execute("DELETE FROM agreements")
        with self.assertRaisesRegex(ValueError, "agreement coordinates"):
            RunLedger(run_dir)

    def test_terminal_run_requires_the_timing_child_for_each_success(self) -> None:
        _, run_dir, _ = self._completed_run("missing-timing")
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM timing_samples WHERE backend = 'pari'"
                )
        with self.assertRaisesRegex(ValueError, "variants do not match"):
            RunLedger(run_dir)


class LegacyLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.recorder = InvocationRecorder()
        self.registry = synthetic_registry(self.recorder)
        self.plan = build_fixture_plan(self.root, self.registry)
        current = self.root / "current"
        run_campaign(self.plan, current)
        self.current_dir = current
        self.legacy_dir = self.root / "legacy"
        self.path = materialize_legacy_ledger(
            current / "run.sqlite", self.legacy_dir
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_v1_snapshot_synthesizes_standard_samples_without_mutation(self) -> None:
        original = self.path.read_bytes()
        original_mtime_ns = self.path.stat().st_mtime_ns
        with RunLedger(self.legacy_dir) as ledger:
            self.assertTrue(ledger.read_only)
            self.assertEqual(ledger.schema_version, 1)
            snapshot = ledger.snapshot()
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertEqual(snapshot["source_ledger_schema_version"], 1)
        self.assertEqual(
            len(snapshot["timing_samples"]), len(snapshot["observations"])
        )
        self.assertEqual(
            {sample["variant"] for sample in snapshot["timing_samples"]},
            {"standard"},
        )
        self.assertTrue(
            all(
                sample["diagnostics"] == {"source_ledger_schema_version": 1}
                for sample in snapshot["timing_samples"]
            )
        )
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.path.stat().st_mtime_ns, original_mtime_ns)

        with mock.patch.object(
            reporting,
            "_plots",
            return_value={"generated": False, "reason": "fixture"},
        ):
            report = generate_report(self.legacy_dir)
        self.assertTrue(Path(report["directory"]).is_dir())
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.path.stat().st_mtime_ns, original_mtime_ns)

    def test_v1_resume_and_every_mutation_entrypoint_are_rejected(self) -> None:
        original = self.path.read_bytes()
        original_mtime_ns = self.path.stat().st_mtime_ns
        with RunLedger(self.legacy_dir) as ledger:
            snapshot = ledger.snapshot()
            actions = {
                "initialize": lambda: ledger.initialize(
                    snapshot["fingerprint"], snapshot["manifest"]
                ),
                "set_state": lambda: ledger.set_state("running"),
                "put_engines": lambda: ledger.put_engines(()),
                "put_cases": lambda: ledger.put_cases(()),
                "put_observation": lambda: ledger.put_observation(None),
                "put_agreement": lambda: ledger.put_agreement(
                    "case",
                    0,
                    "silex",
                    "pari",
                    AgreementResult(True),
                ),
            }
            for label, action in actions.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "report-only"):
                        action()
        with self.assertRaisesRegex(ValueError, "report-only"):
            run_campaign(self.plan, self.legacy_dir, resume=True)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.path.stat().st_mtime_ns, original_mtime_ns)

    def test_early_v1_manifest_without_later_provenance_fields_is_reportable(self) -> None:
        with RunLedger(self.legacy_dir) as ledger:
            manifest = ledger.manifest()
        plan = manifest["plan"]
        plan.pop("invocation_root")
        plan.pop("adapter_capabilities")
        plan["execution"].pop("required_tags")
        source = manifest["sources"]["silex_bench"]
        source.pop("provenance")
        source.pop("package_version")
        source.pop("package_sha256")
        manifest["run_fingerprint"] = campaign_manifest_fingerprint(manifest)
        self.assertEqual(validate_campaign_manifest(manifest), manifest)

    def test_v1_publication_without_clean_enforcement_remains_reportable(self) -> None:
        with RunLedger(self.current_dir) as ledger:
            current = ledger.manifest()
        execution = current["plan"]["execution"]
        execution["publication"] = True
        execution["repetitions"] = 9
        execution["minimum_repetitions"] = 9
        execution["require_clean_sources"] = False
        current["plan"]["sample_count"] = 18
        manifest = legacy_manifest(current)
        with contextlib.closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute(
                    "UPDATE run SET state = 'running', fingerprint = ?, "
                    "manifest_json = ? WHERE singleton = 1",
                    (
                        manifest["run_fingerprint"],
                        canonical_json(manifest),
                    ),
                )

        with RunLedger(self.legacy_dir) as ledger:
            self.assertTrue(ledger.read_only)
            self.assertEqual(ledger.manifest(), manifest)
        with mock.patch.object(
            reporting,
            "_plots",
            return_value={"generated": False, "reason": "fixture"},
        ):
            report = generate_report(self.legacy_dir)
        self.assertTrue(Path(report["directory"]).is_dir())


class CampaignResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_case_timeout_is_the_same_effective_deadline_for_every_adapter(self) -> None:
        recorder = InvocationRecorder()
        plan = build_fixture_plan(
            self.root,
            synthetic_registry(recorder),
            timeout_seconds=10,
            case_timeout_seconds=4,
        )
        snapshot = run_campaign(plan, self.root / "timeout")
        self.assertEqual(snapshot["state"], "complete")
        self.assertEqual(
            set(recorder.contexts),
            {
                (
                    "silex",
                    4.0,
                    "min(campaign_ceiling,case_timeout_hint)",
                ),
                (
                    "pari",
                    4.0,
                    "min(campaign_ceiling,case_timeout_hint)",
                ),
            },
        )
        self.assertTrue(
            all(
                sample["effective_timeout_seconds"] == 4.0
                and sample["timeout_source"]
                == "min(campaign_ceiling,case_timeout_hint)"
                for sample in snapshot["timing_samples"]
            )
        )

    def test_keyboard_interrupt_preserves_committed_rows_and_resume_completes(self) -> None:
        recorder = InvocationRecorder(interrupt_on_call=2)
        plan = build_fixture_plan(self.root, synthetic_registry(recorder))
        run_dir = self.root / "interrupted"
        interrupted = run_campaign(plan, run_dir)
        self.assertEqual(interrupted["state"], "interrupted")
        self.assertEqual(len(interrupted["observations"]), 1)
        self.assertEqual(len(interrupted["timing_samples"]), 1)
        committed = copy.deepcopy(interrupted["observations"][0])

        with RunLedger(run_dir) as ledger:
            self.assertEqual(ledger.state(), "interrupted")
            self.assertEqual(ledger.observation_count(), 1)

        resumed = run_campaign(plan, run_dir, resume=True)
        self.assertEqual(resumed["state"], "complete")
        self.assertEqual(len(resumed["observations"]), 2)
        self.assertEqual(len(resumed["timing_samples"]), 2)
        self.assertIn(committed, resumed["observations"])
        self.assertEqual(recorder.calls, 3)


if __name__ == "__main__":
    unittest.main()
