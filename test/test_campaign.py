from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import silex_bench.reporting as reporting
from silex_bench.campaign import build_plan, invocation_context, run_campaign
from silex_bench.cli import _resume_plan, main
from silex_bench.configuration import (
    ProfileConfig,
    RunOverrides,
    SuiteConfig,
    ToolConfig,
    load_profile,
    load_suite,
    load_tools,
)
from silex_bench.contracts import (
    CAMPAIGN_SCHEMA_VERSION,
    BackendDescriptor,
    EngineInfo,
    Observation,
    ObservationStatus,
    Registry,
)
from silex_bench.ledger import RunLedger
from silex_bench.reporting import generate_report
from silex_bench.resources import builtin_path
from silex_bench.workloads import MAXIMAL_ORDER, NumberFieldContract


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_checked_in_quick_plan_uses_campaign_schema_v1(self) -> None:
        suite = load_suite(builtin_path("suites", "number-field"))
        profile = load_profile(builtin_path("profiles", "quick"))
        self.assertEqual(CAMPAIGN_SCHEMA_VERSION, 1)
        self.assertEqual(suite.id, "number-field")
        self.assertEqual(profile.id, "quick")
        self.assertEqual(suite.required_pairs, (("silex", "pari"),))

    def test_unknown_configuration_keys_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.toml"
            path.write_text(
                'schema_version = 1\nid = "bad"\ninclude_tags = []\n'
                'exclude_tags = []\nrepetitions = 1\nwarmups = 0\n'
                'timeout_seconds = 1\nthreads = 1\npublication = false\n'
                'minimum_repetitions = 1\nrequire_clean_sources = false\n'
                'surprise = true\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown profile keys"):
                load_profile(path)

    def test_publication_profile_requires_nine_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.toml"
            path.write_text(
                'schema_version = 1\nid = "bad"\ninclude_tags = []\n'
                'exclude_tags = []\nrepetitions = 8\nwarmups = 0\n'
                'timeout_seconds = 1\nthreads = 1\npublication = true\n'
                'minimum_repetitions = 9\nrequire_clean_sources = true\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "at least nine"):
                load_profile(path)

    def test_pre_release_campaign_config_is_rejected_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.toml"
            original = (
                'schema_version = 4\nid = "stale"\ninclude_tags = []\n'
                'exclude_tags = []\nrepetitions = 1\nwarmups = 0\n'
                'timeout_seconds = 1\nthreads = 1\npublication = false\n'
                'minimum_repetitions = 1\nrequire_clean_sources = false\n'
            ).encode()
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "schema_version must be 1"):
                load_profile(path)
            self.assertEqual(path.read_bytes(), original)


class CampaignSchemaTests(unittest.TestCase):
    _RUN_TABLE = """
        CREATE TABLE run (
            singleton INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            manifest_json TEXT NOT NULL
        );
    """

    @staticmethod
    def _ledger_state(path: Path) -> tuple[int, list[tuple[object, ...]]]:
        with contextlib.closing(sqlite3.connect(path)) as connection:
            with connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                rows = connection.execute(
                    "SELECT schema_version, fingerprint, state, manifest_json "
                    "FROM run ORDER BY singleton"
                ).fetchall()
        return version, rows

    def test_pre_release_ledger_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            path = run_dir / "run.sqlite"
            with contextlib.closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.executescript(self._RUN_TABLE)
                    connection.execute(
                        "INSERT INTO run VALUES (1, 4, 'old-fingerprint', "
                        "'complete', 'created', 'updated', ?)",
                        (json.dumps({"schema_version": 4}),),
                    )
                    connection.execute("PRAGMA user_version = 4")
            original_bytes = path.read_bytes()
            original_state = self._ledger_state(path)

            with self.assertRaisesRegex(ValueError, "run ledger schema must be 1, got 4"):
                RunLedger(run_dir, create=True)

            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(self._ledger_state(path), original_state)

    def test_resume_requires_ledger_even_when_a_sidecar_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            path = run_dir / "manifest.json"
            original = b'{"schema_version":4,"plan":{"must_not_be_read":true}}\n'
            path.write_bytes(original)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(["resume", str(run_dir)])
            self.assertEqual(status, 2)
            self.assertIn("run ledger does not exist", stderr.getvalue())
            self.assertEqual(path.read_bytes(), original)

    def test_pre_release_embedded_manifest_blocks_report_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            path = run_dir / "run.sqlite"
            with contextlib.closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.executescript(self._RUN_TABLE)
                    connection.execute(
                        "INSERT INTO run VALUES (1, 1, 'fingerprint', "
                        "'complete', 'created', 'updated', ?)",
                        (json.dumps({"schema_version": 4}),),
                    )
                    connection.execute("PRAGMA user_version = 1")
            original_bytes = path.read_bytes()
            original_state = self._ledger_state(path)

            with self.assertRaisesRegex(
                ValueError, "campaign manifest schema_version must be 1"
            ):
                generate_report(run_dir)

            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(self._ledger_state(path), original_state)


class RegistryTests(unittest.TestCase):
    def test_duplicate_workload_ids_fail(self) -> None:
        contract = NumberFieldContract(MAXIMAL_ORDER, "Maximal order")
        with self.assertRaisesRegex(ValueError, "duplicate workload"):
            Registry([contract, contract], [])

    def test_cli_lists_stable_extension_points(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["--json", "list"])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertIn("sunit_proven", {row["id"] for row in payload["workloads"]})
        magma = next(row for row in payload["backends"] if row["id"] == "magma")
        self.assertNotIn("sunit_proven", magma["capabilities"])

    def test_cli_can_select_one_workload_and_required_pair(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    "--json",
                    "plan",
                    "--suite",
                    "number-field",
                    "--profile",
                    "quick",
                    "--workload",
                    "maximal_order",
                    "--backend",
                    "silex",
                    "--backend",
                    "pari",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["workloads"], ["maximal_order"])
        self.assertEqual(payload["case_count"], 2)
        self.assertEqual(payload["sample_count"], 4)

    def test_quick_sunit_slice_avoids_the_long_full_proof_fixture(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    "--json",
                    "plan",
                    "--profile",
                    "quick",
                    "--workload",
                    "sunit_proven",
                    "--backend",
                    "silex",
                    "--backend",
                    "pari",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(
            {case["id"] for case in payload["cases"]},
            {"cubic_x3_minus_2_empty_s", "real_quadratic_5_ramified_5"},
        )

    def test_explicit_case_uses_quick_limits_without_the_quick_tag(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    "--json",
                    "plan",
                    "--profile",
                    "quick",
                    "--workload",
                    "sunit_proven",
                    "--case",
                    "real_quadratic_210_first_over_2",
                    "--backend",
                    "silex",
                    "--backend",
                    "pari",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(
            [case["id"] for case in payload["cases"]],
            ["real_quadratic_210_first_over_2"],
        )
        self.assertEqual(payload["execution"]["timeout_seconds"], 180.0)


@dataclass(frozen=True)
class FakeImplementation:
    backend: str
    workload: str = MAXIMAL_ORDER
    discriminant: str = "5"

    def run(self, case, *, repetition, order_index, warmup, context, contract):
        result = {"maximal_order_discriminant": self.discriminant}
        validation = contract.validate_observation(case, self.backend, result, {})
        elapsed = 10_000_000 if self.backend == "silex" else 20_000_000
        return Observation(
            case_key=case.key,
            workload=self.workload,
            backend=self.backend,
            repetition=repetition,
            order_index=order_index,
            status=ObservationStatus.OK,
            success=True,
            timeout=False,
            result=result,
            proof={},
            validation=validation,
            target_wall_ns=elapsed,
            process_wall_ns=elapsed + 1_000,
            internal_timing={},
            engine_identity={"engine": self.backend, "version": "test"},
            command=(self.backend,),
            stdout="",
            stderr="",
        )


def fake_registry(
    *, include_unavailable: bool = False, disagree: bool = False
) -> Registry:
    contract = NumberFieldContract(MAXIMAL_ORDER, "Maximal order")
    descriptors = []
    for name in ("silex", "pari"):
        implementation = FakeImplementation(
            name, discriminant="7" if disagree and name == "pari" else "5"
        )

        def probe(context, descriptor, *, backend=name):
            return EngineInfo(
                backend=backend,
                display_name=backend,
                available=True,
                capabilities=(MAXIMAL_ORDER,),
                identity={"engine": backend, "version": "test"},
            )

        descriptors.append(
            BackendDescriptor(
                id=name,
                display_name=name,
                implementations={MAXIMAL_ORDER: implementation},
                probe_callback=probe,
            )
        )
    if include_unavailable:
        implementation = FakeImplementation("hecke")

        def unavailable(context, descriptor):
            return EngineInfo(
                backend="hecke",
                display_name="hecke",
                available=False,
                capabilities=(MAXIMAL_ORDER,),
                identity={},
                error="fixture unavailable",
            )

        descriptors.append(
            BackendDescriptor(
                id="hecke",
                display_name="hecke",
                implementations={MAXIMAL_ORDER: implementation},
                probe_callback=unavailable,
            )
        )
    return Registry([contract], descriptors)


class ResumeLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.invocation_root = self.root / "original-invocation"
        self.invocation_root.mkdir()
        workspace = self.invocation_root / "original-workspace"
        (workspace / "silex").mkdir(parents=True)

        corpus = self.invocation_root / "fields.json"
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
                            "optimization_external_engines": ["pari"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        suite_path = self.invocation_root / "suite.toml"
        suite_path.write_text(
            "\n".join(
                (
                    "schema_version = 1",
                    'id = "resume-test"',
                    'title = "Resume test"',
                    'workloads = ["maximal_order"]',
                    'backends = ["silex", "pari"]',
                    'required_pairs = [["silex", "pari"]]',
                    "[corpora]",
                    f"number_fields = {json.dumps(str(corpus))}",
                    "[reports]",
                    'primary_clock = "target_wall_ns"',
                    'speedup = "baseline_over_candidate"',
                    "",
                )
            ),
            encoding="utf-8",
        )
        profile_path = self.invocation_root / "profile.toml"
        profile_path.write_text(
            "\n".join(
                (
                    "schema_version = 1",
                    'id = "resume-test"',
                    'description = "Resume test"',
                    'include_tags = ["quick"]',
                    "exclude_tags = []",
                    "repetitions = 1",
                    "warmups = 0",
                    "timeout_seconds = 1",
                    "threads = 1",
                    "publication = false",
                    "minimum_repetitions = 1",
                    "require_clean_sources = false",
                    "",
                )
            ),
            encoding="utf-8",
        )
        tools_path = self.invocation_root / "tools.toml"
        tools_path.write_text(
            'schema_version = 1\nworkspace = "original-workspace"\n',
            encoding="utf-8",
        )
        self.expected_workspace = workspace.absolute()
        self.plan = build_plan(
            self.invocation_root,
            load_suite(suite_path),
            load_profile(profile_path),
            load_tools(tools_path),
            RunOverrides(),
            fake_registry(),
            performance=True,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_resume_plan_preserves_embedded_invocation_context(self) -> None:
        run_dir = self.root / "preserved-context"
        run_campaign(self.plan, run_dir)

        with mock.patch(
            "silex_bench.cli.builtin_registry", return_value=fake_registry()
        ), mock.patch(
            "silex_bench.cli.bench_root",
            side_effect=AssertionError("resume consulted the current working directory"),
        ):
            resumed = _resume_plan(run_dir, None)

        self.assertEqual(resumed.bench_root, self.invocation_root.absolute())
        context = invocation_context(resumed)
        self.assertEqual(context.bench_root, self.invocation_root.absolute())
        self.assertEqual(context.workspace, self.expected_workspace)

    def test_cli_resume_repairs_sidecar_from_embedded_ledger_manifest(self) -> None:
        variants = {
            "missing": None,
            "corrupt": b"not JSON\n",
            "stale": b'{"schema_version":1,"plan":{"invocation_root":"/wrong"}}\n',
        }
        for label, replacement in variants.items():
            with self.subTest(sidecar=label):
                run_dir = self.root / f"sidecar-{label}"
                run_campaign(self.plan, run_dir)
                with RunLedger(run_dir) as ledger:
                    embedded = ledger.manifest()
                sidecar = run_dir / "manifest.json"
                if replacement is None:
                    sidecar.unlink()
                else:
                    sidecar.write_bytes(replacement)

                stdout = io.StringIO()
                with mock.patch(
                    "silex_bench.cli.builtin_registry", return_value=fake_registry()
                ), mock.patch(
                    "silex_bench.cli.bench_root",
                    side_effect=AssertionError(
                        "resume consulted the current working directory"
                    ),
                ), contextlib.redirect_stdout(stdout):
                    status = main(["--json", "resume", str(run_dir)])

                self.assertEqual(status, 0, stdout.getvalue())
                self.assertTrue(json.loads(stdout.getvalue())["success"])
                self.assertEqual(
                    json.loads(sidecar.read_text(encoding="utf-8")), embedded
                )


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        corpus = self.root / "fields.json"
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
                            "optimization_external_engines": ["pari", "hecke"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        marker = self.root / "config.toml"
        marker.write_text("schema_version = 1\n", encoding="utf-8")
        self.suite = SuiteConfig(
            path=marker,
            id="test",
            title="test",
            workloads=(MAXIMAL_ORDER,),
            backends=("silex", "pari"),
            required_pairs=(("silex", "pari"),),
            corpora={"number_fields": corpus},
            reports={},
            sha256="suite",
        )
        self.profile = ProfileConfig(
            path=marker,
            id="test",
            description="",
            include_tags=("quick",),
            exclude_tags=(),
            repetitions=2,
            warmups=0,
            timeout_seconds=1,
            budget_seconds=None,
            cpu=None,
            threads=1,
            publication=False,
            minimum_repetitions=1,
            require_clean_sources=False,
            metrics={},
            sha256="profile",
        )
        self.plan = build_plan(
            self.root,
            self.suite,
            self.profile,
            ToolConfig(None, {}, None),
            RunOverrides(),
            fake_registry(),
            performance=True,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_campaign_ledgers_agreement_and_paired_speedup(self) -> None:
        run_dir = self.root / "run"
        snapshot = run_campaign(self.plan, run_dir)
        self.assertEqual(snapshot["state"], "complete")
        self.assertEqual(len(snapshot["observations"]), 4)
        self.assertEqual(len(snapshot["agreements"]), 2)
        self.assertTrue(all(row["timing_eligible"] for row in snapshot["agreements"]))
        self.assertTrue(
            all(row["speedup_baseline_over_candidate"] == 2 for row in snapshot["agreements"])
        )
        with RunLedger(run_dir) as ledger:
            self.assertEqual(ledger.state(), "complete")

    def test_resume_is_idempotent_and_report_is_immutable(self) -> None:
        run_dir = self.root / "run"
        first = run_campaign(self.plan, run_dir)
        second = run_campaign(self.plan, run_dir, resume=True)
        self.assertEqual(first["observations"], second["observations"])
        report = generate_report(run_dir)
        repeated = generate_report(run_dir)
        self.assertEqual(report["directory"], repeated["directory"])
        destination = Path(report["directory"])
        self.assertTrue((destination / "summary.json").is_file())
        self.assertTrue((destination / "observations.jsonl").is_file())
        self.assertFalse(report["publication_eligible"])
        with self.assertRaisesRegex(ValueError, "publication export rejected"):
            generate_report(run_dir, publication=True)
        (destination / "summary.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            generate_report(run_dir)

    def test_resume_rejects_cpu_model_drift(self) -> None:
        run_dir = self.root / "cpu-drift-run"
        hardware = {
            "hostname": "fixture-host",
            "platform": "fixture-platform",
            "architecture": "x86_64",
            "python": "fixture-python",
            "cpu_model": "fixture-cpu-a",
            "available_affinity": [0, 1],
            "requested_cpu": None,
        }
        with mock.patch(
            "silex_bench.campaign.machine_identity", return_value=hardware
        ):
            run_campaign(self.plan, run_dir)
        with mock.patch(
            "silex_bench.campaign.machine_identity",
            return_value={**hardware, "cpu_model": "fixture-cpu-b"},
        ):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_campaign(self.plan, run_dir, resume=True)

    def test_report_destination_binds_plot_renderer_availability(self) -> None:
        run_dir = self.root / "renderer-run"
        run_campaign(self.plan, run_dir)
        unavailable_renderer = {
            "report_renderer": reporting.REPORT_RENDERER_VERSION,
            "plots": {
                "available": False,
                "reason": "matplotlib is not installed",
            },
        }
        available_renderer = {
            "report_renderer": reporting.REPORT_RENDERER_VERSION,
            "plots": {"available": True, "matplotlib_version": "test-version"},
        }
        unavailable_plots = {
            "generated": False,
            "reason": "matplotlib is not installed",
        }
        available_plots = {"generated": True, "files": []}
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.__version__ = "test-version"
        fake_matplotlib.__path__ = []
        fake_matplotlib.use = mock.Mock()
        fake_pyplot = types.ModuleType("matplotlib.pyplot")

        with (
            mock.patch.dict(sys.modules, {"matplotlib": None}),
            mock.patch.object(
                reporting, "_plots", return_value=unavailable_plots
            ),
        ):
            unavailable = generate_report(run_dir)
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "matplotlib": fake_matplotlib,
                    "matplotlib.pyplot": fake_pyplot,
                },
            ),
            mock.patch.object(reporting, "_plots", return_value=available_plots),
        ):
            available = generate_report(run_dir)

        self.assertNotEqual(unavailable["report_identity"], available["report_identity"])
        self.assertNotEqual(unavailable["directory"], available["directory"])
        self.assertTrue(Path(unavailable["directory"]).is_dir())
        self.assertTrue(Path(available["directory"]).is_dir())
        unavailable_bundle = json.loads(
            (Path(unavailable["directory"]) / "bundle.json").read_text(
                encoding="utf-8"
            )
        )
        available_bundle = json.loads(
            (Path(available["directory"]) / "bundle.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(unavailable_bundle["renderer"], unavailable_renderer)
        self.assertEqual(available_bundle["renderer"], available_renderer)

        with (
            mock.patch.dict(sys.modules, {"matplotlib": None}),
            mock.patch.object(
                reporting,
                "_plots",
                side_effect=AssertionError("immutable report should be reused"),
            ),
        ):
            repeated = generate_report(run_dir)
        self.assertEqual(repeated["directory"], unavailable["directory"])

    def test_tampered_required_observation_identity_blocks_publication(self) -> None:
        run_dir = self.root / "identity-tamper-run"
        run_campaign(self.plan, run_dir)
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                row = connection.execute(
                    "SELECT observation_json FROM observations "
                    "WHERE backend = 'pari' AND repetition = 0"
                ).fetchone()
                self.assertIsNotNone(row)
                observation = json.loads(row[0])
                observation["engine_identity"]["version"] = "tampered-version"
                connection.execute(
                    "UPDATE observations SET observation_json = ? "
                    "WHERE backend = 'pari' AND repetition = 0",
                    (json.dumps(observation, sort_keys=True, separators=(",", ":")),),
                )

        with self.assertRaisesRegex(
            ValueError,
            "required observation engine identity differs from its probe",
        ):
            generate_report(run_dir, publication=True)

    def test_tampered_required_observation_backend_blocks_publication(self) -> None:
        run_dir = self.root / "backend-tamper-run"
        run_campaign(self.plan, run_dir)
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                row = connection.execute(
                    "SELECT observation_json FROM observations "
                    "WHERE backend = 'pari' AND repetition = 0"
                ).fetchone()
                self.assertIsNotNone(row)
                observation = json.loads(row[0])
                observation["backend"] = "forged-backend"
                connection.execute(
                    "UPDATE observations SET observation_json = ? "
                    "WHERE backend = 'pari' AND repetition = 0",
                    (json.dumps(observation, sort_keys=True, separators=(",", ":")),),
                )

        with self.assertRaisesRegex(
            ValueError,
            "run ledger observation columns do not match",
        ):
            generate_report(run_dir, publication=True)

    def test_tampered_required_observation_workload_blocks_publication(self) -> None:
        run_dir = self.root / "workload-tamper-run"
        run_campaign(self.plan, run_dir)
        with contextlib.closing(sqlite3.connect(run_dir / "run.sqlite")) as connection:
            with connection:
                row = connection.execute(
                    "SELECT observation_json FROM observations "
                    "WHERE backend = 'pari' AND repetition = 0"
                ).fetchone()
                self.assertIsNotNone(row)
                observation = json.loads(row[0])
                observation["workload"] = "class_unit_proven"
                connection.execute(
                    "UPDATE observations SET observation_json = ? "
                    "WHERE backend = 'pari' AND repetition = 0",
                    (json.dumps(observation, sort_keys=True, separators=(",", ":")),),
                )

        with self.assertRaisesRegex(
            ValueError,
            "run ledger observation columns do not match",
        ):
            generate_report(run_dir, publication=True)

    def test_publication_digest_check_uses_required_silex_executable(self) -> None:
        run_dir = self.root / "slot-digest-run"
        snapshot = run_campaign(self.plan, run_dir)
        silex_identity = {
            "engine": "silex",
            "class_unit_executable": "/test/silex-class-unit",
            "operation_executable": "/test/silex-operation",
            "executable_sha256": {
                "class_unit_executable": None,
                "operation_executable": "a" * 64,
            },
        }
        for engine in snapshot["engines"]:
            if engine["backend"] == "silex":
                engine["identity"] = silex_identity
        for observation in snapshot["observations"]:
            if observation["backend"] == "silex":
                observation["engine_identity"] = silex_identity
        summary = reporting._summary(snapshot)

        errors = reporting._publication_errors(snapshot, summary)
        self.assertFalse(
            any(
                error.startswith("required engine lacks executable content identity")
                and "/silex/" in error
                for error in errors
            ),
            errors,
        )

        silex_identity["executable_sha256"]["operation_executable"] = None
        missing_errors = reporting._publication_errors(snapshot, summary)
        self.assertTrue(
            any(
                error.startswith("required engine lacks executable content identity")
                and "/silex/" in error
                for error in missing_errors
            ),
            missing_errors,
        )

    def test_plan_counts_eligible_backend_case_slots(self) -> None:
        self.assertEqual(self.plan.sample_count, 4)
        narrowed = build_plan(
            self.root,
            self.suite,
            self.profile,
            ToolConfig(None, {}, None),
            RunOverrides(repetitions=3, metric_maxima=(("degree", 2),)),
            fake_registry(),
            performance=True,
        )
        self.assertEqual(narrowed.sample_count, 6)
        self.assertEqual(narrowed.workloads, (MAXIMAL_ORDER,))

    def test_correctness_mode_forces_one_cold_observation(self) -> None:
        profile = ProfileConfig(
            **{
                **self.profile.__dict__,
                "repetitions": 7,
                "warmups": 1,
                "publication": True,
                "minimum_repetitions": 7,
            }
        )
        plan = build_plan(
            self.root,
            self.suite,
            profile,
            ToolConfig(None, {}, None),
            RunOverrides(),
            fake_registry(),
            performance=False,
        )
        self.assertFalse(plan.performance)
        self.assertEqual(plan.execution["repetitions"], 1)
        self.assertEqual(plan.execution["warmups"], 0)
        self.assertFalse(plan.execution["publication"])
        run_dir = self.root / "correctness-run"
        snapshot = run_campaign(plan, run_dir)
        self.assertTrue(all(not row["timing_eligible"] for row in snapshot["agreements"]))
        report = generate_report(run_dir)
        summary = json.loads(
            (Path(report["directory"]) / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["timings"], [])

    def test_unavailable_optional_backend_does_not_fail_required_pair(self) -> None:
        suite = SuiteConfig(
            **{
                **self.suite.__dict__,
                "backends": ("silex", "pari", "hecke"),
            }
        )
        plan = build_plan(
            self.root,
            suite,
            self.profile,
            ToolConfig(None, {}, None),
            RunOverrides(),
            fake_registry(include_unavailable=True),
            performance=True,
        )
        snapshot = run_campaign(plan, self.root / "optional-run")
        self.assertEqual(snapshot["state"], "complete")
        hecke = [row for row in snapshot["observations"] if row["backend"] == "hecke"]
        self.assertEqual({row["status"] for row in hecke}, {"unavailable"})

    def test_disagreement_is_never_admitted_to_timing_tables(self) -> None:
        corpus = self.suite.corpora["number_fields"]
        payload = json.loads(corpus.read_text(encoding="utf-8"))
        payload["fields"][0].pop("maximal_order_discriminant")
        corpus.write_text(json.dumps(payload), encoding="utf-8")
        plan = build_plan(
            self.root,
            self.suite,
            self.profile,
            ToolConfig(None, {}, None),
            RunOverrides(),
            fake_registry(disagree=True),
            performance=True,
        )
        run_dir = self.root / "disagree-run"
        snapshot = run_campaign(plan, run_dir)
        self.assertEqual(snapshot["state"], "failed")
        report = generate_report(run_dir)
        summary = json.loads(
            (Path(report["directory"]) / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["timings"], [])
        self.assertEqual(summary["ratios"], [])


if __name__ == "__main__":
    unittest.main()
