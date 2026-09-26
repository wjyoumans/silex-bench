from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

from silex_bench.campaign import build_plan, doctor, run_campaign
from silex_bench.cli import main
from silex_bench.configuration import (
    ProfileConfig,
    RunOverrides,
    SuiteConfig,
    ToolConfig,
)
from silex_bench.contracts import (
    AgreementStatus,
    BackendDescriptor,
    EngineInfo,
    Observation,
    ObservationStatus,
    Registry,
    ValidationResult,
)
from silex_bench.workloads import MAXIMAL_ORDER, NumberFieldContract


@dataclass(frozen=True)
class SyntheticImplementation:
    backend: str
    discriminant: str = "5"
    validation_success: bool = True
    identity_version: str = "test"
    trust_result: bool = False
    workload: str = MAXIMAL_ORDER

    def run(self, case, *, repetition, order_index, warmup, context, contract):
        result = {"maximal_order_discriminant": self.discriminant}
        if self.trust_result:
            validation = ValidationResult(True, (), {"synthetic_valid": True})
        elif self.validation_success:
            validation = contract.validate_observation(case, self.backend, result, {})
        else:
            validation = ValidationResult(
                False,
                ("synthetic invalid observation",),
                {"synthetic_validation": False},
            )
        elapsed = {
            "silex": 10_000_000,
            "pari": 20_000_000,
            "hecke": 30_000_000,
        }.get(self.backend, 40_000_000)
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
            internal_timing={
                "scope": "maximal_order_only",
                "wall_clock": {
                    "silex": "steady_clock",
                    "pari": "pari_getwalltime_ms",
                    "hecke": "julia_time_ns_monotonic",
                }.get(self.backend, "fixture_monotonic"),
            },
            engine_identity={"engine": self.backend, "version": self.identity_version},
            command=(self.backend,),
            stdout="",
            stderr="",
        )


def synthetic_backend(
    backend: str,
    *,
    available: bool = True,
    implements: bool = True,
    discriminant: str = "5",
    validation_success: bool = True,
    identity_version: str = "test",
    trust_result: bool = False,
) -> BackendDescriptor:
    implementations = (
        {
            MAXIMAL_ORDER: SyntheticImplementation(
                backend,
                discriminant=discriminant,
                validation_success=validation_success,
                identity_version=identity_version,
                trust_result=trust_result,
            )
        }
        if implements
        else {}
    )

    def probe(context, descriptor):
        return EngineInfo(
            backend=backend,
            display_name=backend,
            available=available,
            capabilities=descriptor.capabilities,
            identity={"engine": backend, "version": "test"} if available else {},
            error=None if available else "fixture unavailable",
        )

    return BackendDescriptor(
        id=backend,
        display_name=backend,
        implementations=implementations,
        probe_callback=probe,
    )


def synthetic_registry(*descriptors: BackendDescriptor) -> Registry:
    return Registry(
        [NumberFieldContract(MAXIMAL_ORDER, "Maximal order")],
        descriptors,
    )


class CampaignPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "fields.json"
        self._write_corpus(expected=True)
        marker = self.root / "config.toml"
        marker.write_text("schema_version = 1\n", encoding="utf-8")
        self.suite = SuiteConfig(
            path=marker,
            id="test",
            title="test",
            workloads=(MAXIMAL_ORDER,),
            backends=("silex", "pari"),
            required_pairs=(("silex", "pari"),),
            corpora={"number_fields": self.corpus},
            reports={},
            sha256="suite",
        )
        self.profile = ProfileConfig(
            path=marker,
            id="test",
            description="",
            include_tags=("quick",),
            exclude_tags=(),
            backend_exclusions=(),
            repetitions=1,
            jit_repetitions=0,
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_corpus(self, *, expected: bool) -> None:
        field = {
            "id": "real_quadratic_5_proven",
            "coefficients_low_to_high": [-5, 0, 1],
            "degree": 2,
            "expected_success": True,
        }
        field["maximal_order_discriminant"] = 5
        self.corpus.write_text(json.dumps({"fields": [field]}), encoding="utf-8")

    def _plan(
        self,
        registry: Registry,
        *,
        backends: tuple[str, ...] | None = None,
        overrides: RunOverrides = RunOverrides(),
    ):
        suite = replace(
            self.suite,
            backends=self.suite.backends if backends is None else backends,
        )
        return build_plan(
            self.root,
            suite,
            self.profile,
            ToolConfig(None, {}, None),
            overrides,
            registry,
            performance=True,
        )

    def test_required_pair_override_replaces_default_and_preserves_orientation(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
        )
        plan = self._plan(
            registry,
            overrides=RunOverrides(required_pairs=(("pari", "silex"),)),
        )

        self.assertEqual(plan.required_pairs, (("pari", "silex"),))
        self.assertNotIn(("silex", "pari"), plan.required_pairs)
        snapshot = run_campaign(plan, self.root / "oriented-run")
        self.assertEqual(snapshot["state"], "complete")
        self.assertEqual(len(snapshot["agreements"]), 1)
        agreement = snapshot["agreements"][0]
        self.assertEqual(
            (agreement["lhs_backend"], agreement["rhs_backend"]),
            ("pari", "silex"),
        )
        self.assertTrue(agreement["success"])
        self.assertNotIn("speedup_baseline_over_candidate", agreement)

    def test_require_all_adapters_promotes_declared_unavailability_to_failure(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
            synthetic_backend("hecke", available=False),
        )
        backends = ("silex", "pari", "hecke")

        optional_plan = self._plan(registry, backends=backends)
        optional_diagnostics = doctor(optional_plan)
        self.assertTrue(optional_diagnostics["success"])
        self.assertEqual(optional_diagnostics["required_errors"], [])
        self.assertEqual(len(optional_diagnostics["optional_errors"]), 1)
        optional_snapshot = run_campaign(optional_plan, self.root / "optional-run")
        self.assertEqual(optional_snapshot["state"], "complete")

        strict_plan = self._plan(
            registry,
            backends=backends,
            overrides=RunOverrides(require_all_adapters=True),
        )
        strict_diagnostics = doctor(strict_plan)
        self.assertFalse(strict_diagnostics["success"])
        self.assertEqual(len(strict_diagnostics["required_errors"]), 1)
        self.assertEqual(strict_diagnostics["optional_errors"], [])
        strict_snapshot = run_campaign(strict_plan, self.root / "strict-run")
        self.assertEqual(strict_snapshot["state"], "failed")
        self.assertEqual(
            {
                row["status"]
                for row in strict_snapshot["observations"]
                if row["backend"] == "hecke"
            },
            {ObservationStatus.UNAVAILABLE.value},
        )

    def test_require_all_adapters_promotes_optional_disagreement_to_failure(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
            synthetic_backend("magma", discriminant="7", trust_result=True),
        )
        backends = ("silex", "pari", "magma")
        optional_plan = self._plan(registry, backends=backends)
        strict_plan = self._plan(
            registry,
            backends=backends,
            overrides=RunOverrides(require_all_adapters=True),
        )

        for policy, plan, expected_state in (
            ("optional", optional_plan, "complete"),
            ("strict", strict_plan, "failed"),
        ):
            with self.subTest(policy=policy):
                snapshot = run_campaign(plan, self.root / f"{policy}-run")
                self.assertEqual(snapshot["state"], expected_state)
                self.assertEqual(
                    {
                        (row["lhs_backend"], row["rhs_backend"]): row["status"]
                        for row in snapshot["agreements"]
                    },
                    {
                        ("silex", "pari"): AgreementStatus.AGREE.value,
                        ("silex", "magma"): AgreementStatus.DISAGREE.value,
                        ("pari", "magma"): AgreementStatus.DISAGREE.value,
                    },
                )

    def test_unsupported_cells_are_explicit_and_do_not_fail_strict_mode(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
            synthetic_backend("magma", implements=False),
        )
        plan = self._plan(
            registry,
            backends=("silex", "pari", "magma"),
            overrides=RunOverrides(require_all_adapters=True),
        )

        self.assertTrue(doctor(plan)["success"])
        snapshot = run_campaign(plan, self.root / "unsupported-run")
        self.assertEqual(snapshot["state"], "complete")
        unsupported = [
            row for row in snapshot["observations"] if row["backend"] == "magma"
        ]
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0]["status"], ObservationStatus.UNSUPPORTED.value)
        self.assertFalse(unsupported[0]["success"])
        self.assertIn("does not support maximal_order", unsupported[0]["error"])
        magma_agreements = [
            row
            for row in snapshot["agreements"]
            if "magma" in {row["lhs_backend"], row["rhs_backend"]}
        ]
        self.assertTrue(magma_agreements)
        self.assertEqual(
            {row["status"] for row in magma_agreements},
            {AgreementStatus.UNSUPPORTED.value},
        )
        self.assertTrue(
            all("timing_eligible" not in row for row in magma_agreements)
        )

    def test_agreement_rows_distinguish_practical_statuses(self) -> None:
        self._write_corpus(expected=False)
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
            synthetic_backend("hecke", discriminant="7", trust_result=True),
            synthetic_backend("invalid", validation_success=False),
            synthetic_backend("offline", available=False),
            synthetic_backend("unsupported", implements=False),
        )
        plan = self._plan(
            registry,
            backends=(
                "silex",
                "pari",
                "hecke",
                "invalid",
                "offline",
                "unsupported",
            ),
        )

        snapshot = run_campaign(plan, self.root / "status-run")
        self.assertEqual(snapshot["state"], "complete")
        statuses = {
            (row["lhs_backend"], row["rhs_backend"]): row["status"]
            for row in snapshot["agreements"]
        }
        self.assertEqual(statuses[("silex", "pari")], AgreementStatus.AGREE.value)
        self.assertEqual(
            statuses[("silex", "hecke")], AgreementStatus.DISAGREE.value
        )
        self.assertEqual(
            statuses[("silex", "invalid")], AgreementStatus.INVALID.value
        )
        self.assertEqual(
            statuses[("silex", "offline")], AgreementStatus.UNAVAILABLE.value
        )
        self.assertEqual(
            statuses[("silex", "unsupported")], AgreementStatus.UNSUPPORTED.value
        )

    def test_budget_exhaustion_records_incomplete_pairs_for_resume(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
        )
        plan = self._plan(
            registry,
            overrides=RunOverrides(budget_seconds=1e-12),
        )

        snapshot = run_campaign(plan, self.root / "budget-run")
        self.assertEqual(snapshot["state"], "budget_exhausted")
        self.assertEqual(len(snapshot["agreements"]), 1)
        self.assertEqual(
            snapshot["agreements"][0]["status"], AgreementStatus.INCOMPLETE.value
        )

    def test_successful_observation_must_match_the_initial_engine_probe(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari", identity_version="changed-after-probe"),
        )
        plan = self._plan(registry)

        snapshot = run_campaign(plan, self.root / "engine-drift-run")

        self.assertEqual(snapshot["state"], "failed")
        pari = next(
            row for row in snapshot["observations"] if row["backend"] == "pari"
        )
        self.assertEqual(pari["status"], ObservationStatus.ERROR.value)
        self.assertIn("changed after the initial probe", pari["error"])

    def test_check_cli_prints_human_summary_and_generates_report(self) -> None:
        registry = synthetic_registry(
            synthetic_backend("silex"),
            synthetic_backend("pari"),
        )
        plan = self._plan(registry)
        run_dir = self.root / "cli-check"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("silex_bench.cli._load_plan", return_value=plan):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(["check", "--run-dir", str(run_dir)])

        self.assertEqual(status, 0, stderr.getvalue())
        self.assertIn("Campaign state: complete", stdout.getvalue())
        self.assertIn("Observations", stdout.getvalue())
        self.assertTrue(list((run_dir / "exports").glob("*/report.md")))


class CliPolicyTests(unittest.TestCase):
    @staticmethod
    def _invoke(argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_require_pair_cli_replaces_default_without_reorienting_it(self) -> None:
        status, stdout, stderr = self._invoke(
            [
                "plan",
                "--json",
                "--suite",
                "number-field",
                "--profile",
                "quick",
                "--workload",
                MAXIMAL_ORDER,
                "--backend",
                "silex",
                "--backend",
                "pari",
                "--require-pair",
                "pari:silex",
            ]
        )

        self.assertEqual(status, 0, stderr or stdout)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["required_pairs"], [["pari", "silex"]])
        self.assertEqual(
            payload["overrides"]["required_pairs"],
            [["pari", "silex"]],
        )

    def test_json_flag_is_position_independent_and_human_list_is_readable(self) -> None:
        before_status, before_stdout, before_stderr = self._invoke(["--json", "list"])
        after_status, after_stdout, after_stderr = self._invoke(["list", "--json"])

        self.assertEqual(before_status, 0, before_stderr or before_stdout)
        self.assertEqual(after_status, 0, after_stderr or after_stdout)
        self.assertEqual(before_stderr, "")
        self.assertEqual(after_stderr, "")
        self.assertEqual(json.loads(before_stdout), json.loads(after_stdout))

        human_status, human_stdout, human_stderr = self._invoke(["list"])
        self.assertEqual(human_status, 0, human_stderr or human_stdout)
        self.assertEqual(human_stderr, "")
        self.assertTrue(human_stdout.startswith("Suites\n"))
        for heading in ("Profiles", "Workloads", "Adapter capabilities"):
            self.assertIn(heading, human_stdout)
        self.assertIn("unsupported", human_stdout)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(human_stdout)

    def test_json_mode_covers_argument_and_normalization_errors(self) -> None:
        for argv in (
            ["--json", "unknown-command"],
            ["list", "--json", "--json"],
        ):
            with self.subTest(argv=argv):
                status, stdout, stderr = self._invoke(list(argv))
                self.assertEqual(status, 2)
                self.assertEqual(stderr, "")
                payload = json.loads(stdout)
                self.assertFalse(payload["success"])
                self.assertTrue(payload["error"]["message"])

    def test_tag_filter_narrows_instead_of_broadening_the_profile(self) -> None:
        base_status, base_stdout, base_stderr = self._invoke(
            ["plan", "--json", "--profile", "quick"]
        )
        filtered_status, filtered_stdout, filtered_stderr = self._invoke(
            ["plan", "--json", "--profile", "quick", "--tag", "publication"]
        )
        self.assertEqual(base_status, 0, base_stderr or base_stdout)
        self.assertEqual(filtered_status, 0, filtered_stderr or filtered_stdout)
        base = json.loads(base_stdout)
        filtered = json.loads(filtered_stdout)
        self.assertLessEqual(filtered["case_count"], base["case_count"])
        self.assertEqual(filtered["execution"]["include_tags"], ["quick"])
        self.assertEqual(filtered["execution"]["required_tags"], ["publication"])
        self.assertTrue(
            all(
                {"quick", "publication"}.issubset(case["tags"])
                for case in filtered["cases"]
            )
        )


if __name__ == "__main__":
    unittest.main()
