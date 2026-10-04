"""Conditional (grh) class/unit population: adapters, validation, corpus, reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import silex_bench.reporting as reporting
from silex_bench.backends.silex import SilexBackend
from silex_bench.configuration import load_suite
from silex_bench.contracts import Case
from silex_bench.model import SampleRequest, backend_result_contract_errors
from silex_bench.registry import builtin_registry
from silex_bench.resources import builtin_path
from silex_bench.workloads import (
    CLASS_UNIT,
    CLASS_UNIT_GRH,
    builtin_workloads,
    load_cases,
)

from test_backend_timing import (
    MARKED_PROCESS_AFFINITY,
    _hecke_class_unit_stdout,
    _magma_class_unit_stdout,
    _pari_class_unit_stdout,
    class_unit_payload,
    context,
    field,
    hecke_programs,
    magma_programs,
    marked_result,
    pari_programs,
)
from test_backend_timing import PariBackend, HeckeBackend, MagmaBackend


def _request(operation: str) -> SampleRequest:
    return SampleRequest(
        field=field("target", -5),
        operation=operation,
        sample_kind="cold_process",
        sample_index=0,
        warmup=None,
        seed=7,
    )


def _run(factory, module: str, stdout: str, operation: str = "class_unit_grh"):
    backend = factory()
    backend._probe = {
        "engine": backend.name,
        "available": True,
        "engine_identity": {"executable": "/usr/bin/engine", "project": None},
    }
    with tempfile.TemporaryDirectory() as temporary, mock.patch(
        f"silex_bench.backends.{module}.run_marked_process",
        return_value=marked_result(stdout),
    ):
        return backend.run(_request(operation), context(Path(temporary)))


def _grh_native() -> dict[str, object]:
    return {
        **class_unit_payload(),
        "mode": "grh",
        "certification_status": "grh",
        "class_group_proof_status": "grh",
        "unit_group_proof_status": "grh",
        "regulator_proof_status": "not_checked",
    }


def _run_silex(native: dict[str, object], operation: str = "class_unit_grh"):
    process = {
        **marked_result(json.dumps(native)),
        "process_wall_ms": 3.0,
        "cmd": ["silex"],
    }
    with tempfile.TemporaryDirectory() as temporary, mock.patch(
        "silex_bench.backends.silex.run_marked_process", return_value=process
    ) as runner:
        payload = SilexBackend()._run_class_unit(
            _request(operation), context(Path(temporary)), {}
        )
    return payload, runner.call_args.args[0]


class ProgramTextTests(unittest.TestCase):
    def test_pari_grh_runs_bnfinit_without_certification(self) -> None:
        _, target, final = pari_programs(_request("class_unit_grh"))
        self.assertIn("bnfinit(nf, 1)", target)
        self.assertNotIn("bnfcertify", target + final)
        _, proven_target, _ = pari_programs(_request("class_unit_proven"))
        self.assertIn("bnfcertify(b)", proven_target)

    def test_hecke_grh_sets_grh_true_in_both_calls(self) -> None:
        _, target, _ = hecke_programs(_request("class_unit_grh"))
        self.assertIn("target_O; GRH = true, redo = true, do_lll = false", target)
        self.assertIn("unit_group(target_O; GRH = true)", target)
        self.assertNotIn("GRH = false", target)
        _, proven, _ = hecke_programs(_request("class_unit_proven"))
        self.assertIn("unit_group(target_O; GRH = false)", proven)
        self.assertNotIn("GRH = true", proven)

    def test_hecke_grh_jit_pair_uses_grh_true(self) -> None:
        sample = SampleRequest(
            field=field("target", -5),
            operation="class_unit_grh",
            sample_kind="cold_process",
            sample_index=0,
            warmup=None,
            seed=7,
            jit_repetitions=1,
        )
        _, target, _ = hecke_programs(sample)
        self.assertEqual(target.count("GRH = true"), 4)
        self.assertNotIn("GRH = false", target)

    def test_magma_grh_uses_grh_proof_and_unit_flag(self) -> None:
        _, target, _ = magma_programs(_request("class_unit_grh"))
        self.assertIn('ClassGroup(O_target : Proof := "GRH")', target)
        self.assertIn("UnitGroup(O_target : GRH := true)", target)
        self.assertNotIn('"Full"', target)
        _, proven, _ = magma_programs(_request("class_unit_proven"))
        self.assertIn('Proof := "Full"', proven)
        self.assertIn("GRH := false", proven)


class SilexGrhTests(unittest.TestCase):
    def test_command_requests_grh_mode(self) -> None:
        payload, cmd = _run_silex(_grh_native())
        self.assertEqual(cmd[cmd.index("--mode") + 1], "grh")
        _, proven_cmd = _run_silex(class_unit_payload(), "class_unit_proven")
        self.assertEqual(proven_cmd[proven_cmd.index("--mode") + 1], "proven")
        self.assertTrue(payload["success"], payload["error"])
        self.assertFalse(payload["proof"]["proof_complete"])
        self.assertTrue(payload["proof"]["conditional_result_complete"])
        self.assertEqual(payload["proof"]["certification_status"], "grh")
        self.assertEqual(payload["timing"]["scope"], "class_and_unit_group_only")

    def test_regulator_fields_come_from_unit_group(self) -> None:
        native = _grh_native()
        native["unit_group"] = {
            "free_rank": 1,
            "regulator_decimal": "297.835315655189378793583",
            "regulator_midpoint": 297.83531565518937,
            "regulator_radius": 1.5e-36,
            "regulator_proof_status": "grh",
        }
        payload, _ = _run_silex(native)
        self.assertTrue(payload["success"], payload.get("error"))
        self.assertEqual(
            payload["result"]["regulator_decimal"], "297.835315655189378793583"
        )
        self.assertEqual(payload["result"]["regulator_midpoint"], 297.83531565518937)
        self.assertEqual(payload["result"]["regulator_radius"], 1.5e-36)
        self.assertEqual(payload["proof"]["regulator_proof_status"], "grh")

    def test_regulator_status_falls_back_to_top_level(self) -> None:
        payload, _ = _run_silex(_grh_native())
        self.assertEqual(payload["proof"]["regulator_proof_status"], "not_checked")
        self.assertIsNone(payload["result"]["regulator_decimal"])

    def test_stale_mode_echo_is_rejected(self) -> None:
        for echo in ("proven", None):
            with self.subTest(echo=echo):
                native = {**_grh_native(), "mode": echo}
                payload, _ = _run_silex(native)
                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "proof_contract")
                self.assertIn("grh-publication contract failed", payload["error"])

    def test_proven_labels_do_not_satisfy_the_grh_route(self) -> None:
        native = {**class_unit_payload(), "mode": "grh"}
        payload, _ = _run_silex(native)
        self.assertFalse(payload["success"])

    def test_proven_route_still_requires_proven_labels(self) -> None:
        payload, _ = _run_silex(_grh_native(), "class_unit_proven")
        self.assertFalse(payload["success"])


class ExternalGrhTests(unittest.TestCase):
    def test_pari_grh_labels_and_components(self) -> None:
        stdout = _pari_class_unit_stdout(1).replace("certified=1\n", "")
        payload = _run(PariBackend, "pari", stdout)
        self.assertTrue(payload["success"], payload["error"])
        proof = payload["proof"]
        self.assertEqual(proof["certification_status"], "grh")
        self.assertEqual(proof["class_group_proof_status"], "grh")
        self.assertEqual(proof["unit_group_proof_status"], "grh")
        self.assertFalse(proof["proof_complete"])
        self.assertTrue(proof["conditional_result_complete"])
        self.assertEqual(list(payload["timing"]["components_ms"]), ["bnfinit"])

    def test_pari_grh_unit_count_mismatch_is_unknown(self) -> None:
        stdout = _pari_class_unit_stdout(0).replace("certified=1\n", "")
        payload = _run(PariBackend, "pari", stdout)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["proof"]["certification_status"], "unknown")

    def test_hecke_grh_labels_follow_flags(self) -> None:
        # (class_grh_free, unit_grh_free) -> labels; a positive-rank grh run
        # has both flags false.
        for class_free, unit_free, class_label, unit_label in (
            ("false", "false", "grh", "grh"),
            ("false", "true", "grh", "proven"),
            ("true", "false", "proven", "grh"),
        ):
            with self.subTest(class_free=class_free, unit_free=unit_free):
                payload = _run(
                    HeckeBackend,
                    "hecke",
                    _hecke_class_unit_stdout(
                        1, class_grh_free=class_free, unit_grh_free=unit_free
                    ),
                )
                self.assertTrue(payload["success"], payload["error"])
                proof = payload["proof"]
                self.assertEqual(proof["certification_status"], "grh")
                self.assertEqual(proof["class_group_proof_status"], class_label)
                self.assertEqual(proof["unit_group_proof_status"], unit_label)
                self.assertFalse(proof["proof_complete"])

    def test_hecke_grh_missing_flags_are_unknown(self) -> None:
        payload = _run(
            HeckeBackend,
            "hecke",
            _hecke_class_unit_stdout(1, class_grh_free="", unit_grh_free=""),
        )
        self.assertEqual(payload["proof"]["class_group_proof_status"], "unknown")

    def test_magma_grh_labels(self) -> None:
        payload = _run(MagmaBackend, "magma", _magma_class_unit_stdout(1))
        self.assertTrue(payload["success"], payload["error"])
        proof = payload["proof"]
        self.assertEqual(proof["certification_status"], "grh")
        self.assertFalse(proof["proof_complete"])
        self.assertTrue(proof["conditional_result_complete"])
        self.assertEqual(payload["timing"]["scope"], "class_and_unit_group_only")

    def test_result_contract_is_shared(self) -> None:
        payload = _run(MagmaBackend, "magma", _magma_class_unit_stdout(1))
        self.assertEqual(
            backend_result_contract_errors(
                payload["result"],
                operation="class_unit_grh",
                expected_field_degree=2,
            ),
            [],
        )


def _contract(workload: str):
    return next(item for item in builtin_workloads() if item.id == workload)


def _case(workload: str, expected: dict[str, object] | None = None) -> Case:
    return Case(
        id="target",
        workload=workload,
        input={},
        tags=(),
        metrics={"degree": 2},
        expected=expected or {},
    )


_RESULT = {
    "class_order": "1",
    "class_invariants": [],
    "unit_rank": 1,
    "signature": [2, 0],
    "maximal_order_discriminant": "5",
}


def _proof(**overrides: object) -> dict[str, object]:
    return {
        "final_result_published": True,
        "certification_status": "grh",
        "class_group_proof_status": "grh",
        "unit_group_proof_status": "grh",
        "regulator_proof_status": "not_checked",
        **overrides,
    }


class GrhValidationTests(unittest.TestCase):
    def validate(self, proof, expected=None, result=None):
        return _contract(CLASS_UNIT_GRH).validate_observation(
            _case(CLASS_UNIT_GRH, expected), "silex", result or _RESULT, proof
        )

    def test_grh_labels_are_accepted_without_a_verified_regulator(self) -> None:
        self.assertTrue(self.validate(_proof()).success)
        self.assertTrue(
            self.validate(_proof(unit_group_proof_status="proven")).success
        )

    def test_unknown_heuristic_failed_or_proven_overall_is_rejected(self) -> None:
        for key in (
            "certification_status",
            "class_group_proof_status",
            "unit_group_proof_status",
        ):
            for value in ("unknown", "heuristic", "failed", None):
                with self.subTest(key=key, value=value):
                    self.assertFalse(self.validate(_proof(**{key: value})).success)
        # never relabelled proven at population level
        self.assertFalse(self.validate(_proof(certification_status="proven")).success)

    def test_unpublished_result_is_rejected(self) -> None:
        self.assertFalse(self.validate(_proof(final_result_published=False)).success)

    def test_result_differing_from_expectation_is_rejected(self) -> None:
        self.assertFalse(self.validate(_proof(), {"class_order": 2}).success)
        self.assertTrue(self.validate(_proof(), {"class_order": 1}).success)

    def test_proven_contract_rejects_grh_labels(self) -> None:
        validation = _contract(CLASS_UNIT).validate_observation(
            _case(CLASS_UNIT), "silex", _RESULT, _proof()
        )
        self.assertFalse(validation.success)

    def test_agreement_compares_results_not_labels(self) -> None:
        contract = _contract(CLASS_UNIT_GRH)
        other = {**_RESULT, "class_order": "2"}
        case = _case(CLASS_UNIT_GRH)
        self.assertTrue(contract.compare(case, "silex", _RESULT, "pari", dict(_RESULT)).success)
        self.assertFalse(contract.compare(case, "silex", _RESULT, "pari", other).success)


class GrhCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = load_suite(builtin_path("suites", "number-field"))

    def test_workload_is_registered_but_not_in_default_suite(self) -> None:
        self.assertIn(CLASS_UNIT_GRH, builtin_registry().workloads)
        self.assertNotIn(CLASS_UNIT_GRH, self.suite.workloads)
        for descriptor in builtin_registry().backends.values():
            self.assertIn(CLASS_UNIT_GRH, descriptor.implementations)

    def test_default_load_has_no_grh_cases(self) -> None:
        cases = load_cases(self.suite.corpora, self.suite.workloads)
        self.assertFalse([case for case in cases if case.workload == CLASS_UNIT_GRH])

    def test_cubic_1080004_has_grh_corpus_row(self) -> None:
        both = load_cases(self.suite.corpora, (CLASS_UNIT, CLASS_UNIT_GRH))
        grh = {case.id: case for case in both if case.workload == CLASS_UNIT_GRH}
        case = grh["cubic_disc1080004_proven"]
        self.assertEqual(case.expected["class_order"], 2)
        self.assertEqual(case.expected["unit_rank"], 1)
        self.assertEqual(case.expected["maximal_order_discriminant"], -1080004)
        self.assertEqual(case.expected_status, "success")

    def test_grh_cases_are_opt_in_per_row_and_keyed_apart(self) -> None:
        both = load_cases(self.suite.corpora, (CLASS_UNIT, CLASS_UNIT_GRH))
        proven = {case.id: case for case in both if case.workload == CLASS_UNIT}
        grh = {case.id: case for case in both if case.workload == CLASS_UNIT_GRH}
        self.assertTrue(grh)
        self.assertLess(len(grh), len(proven))
        for identifier, case in grh.items():
            with self.subTest(case=identifier):
                self.assertEqual(case.key, f"class_unit_grh:{identifier}")
                self.assertNotEqual(case.key, proven[identifier].key)
                self.assertNotIn("publication", case.tags)
                self.assertEqual(case.expected, proven[identifier].expected)
                self.assertEqual(case.expected_status, "success")

    def test_non_proven_mode_is_rejected(self) -> None:
        corpus = json.loads(self.suite.corpora["number_fields"].read_text("utf-8"))
        corpus["fields"][0]["mode"] = "grh"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.json"
            path.write_text(json.dumps(corpus), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mode must be 'proven'"):
                load_cases({"number_fields": path}, (CLASS_UNIT,))

    def test_malformed_grh_object_is_rejected(self) -> None:
        corpus = json.loads(self.suite.corpora["number_fields"].read_text("utf-8"))
        corpus["fields"][0]["grh"] = {"expected_success": True}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.json"
            path.write_text(json.dumps(corpus), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "grh must be an object"):
                load_cases({"number_fields": path}, (CLASS_UNIT_GRH,))

    def test_grh_expected_failure_is_diagnostic_only(self) -> None:
        corpus = json.loads(self.suite.corpora["number_fields"].read_text("utf-8"))
        corpus["fields"][0]["grh"] = {"expected_success": False, "source": "fixture"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.json"
            path.write_text(json.dumps(corpus), encoding="utf-8")
            cases = load_cases({"number_fields": path}, (CLASS_UNIT_GRH,))
        first = corpus["fields"][0]["id"]
        (case,) = [item for item in cases if item.id == first]
        self.assertEqual(case.expected_status, "failure")
        self.assertFalse(case.performance_eligible)


class GrhReportTests(unittest.TestCase):
    def test_markdown_separates_grh_timings(self) -> None:
        def timing(workload: str) -> dict[str, object]:
            return {
                "workload": workload,
                "case_id": "quartic",
                "backend": "pari",
                "variant": "standard",
                "timing_scope": "class_and_unit_group_only",
                "wall_clock": "pari_getwalltime_ms",
                "statistics": {
                    "count": 3,
                    "median": 5.0,
                    "mad": 1.0,
                    "bootstrap_95_low": 4.0,
                    "bootstrap_95_high": 6.0,
                },
            }

        def render(rows):
            return reporting._markdown(
                {
                    "state": "complete",
                    "fingerprint": "fp",
                    "manifest": {
                        "machine": {
                            "requested_cpu": 2,
                            "cpu_model": "cpu",
                            "platform": "Linux",
                            "architecture": "x86_64",
                        },
                        "plan": {"suite": "number-field", "execution": {"threads": 1}},
                    },
                },
                {
                    "primary_clock": "c",
                    "engines": [],
                    "backend_status": [],
                    "agreement_status": [],
                    "backend_exclusions": [],
                    "failures": [],
                    "timings": rows,
                },
                [],
                {"generated": False, "reason": "none"},
            )

        mixed = render([timing("class_unit_proven"), timing("class_unit_grh")])
        head, _, tail = mixed.partition("## GRH-conditional per-case timings")
        self.assertIn("| class_unit_proven |", head)
        self.assertNotIn("| class_unit_grh |", head)
        self.assertIn("requested assumption: GRH", tail)
        self.assertIn("| class_unit_grh |", tail)
        self.assertNotIn("| class_unit_proven |", tail)
        proven_only = render([timing("class_unit_proven")])
        self.assertNotIn("GRH-conditional", proven_only)
        self.assertEqual(
            reporting._EXPECTED_TIMING_SCOPES["class_unit_grh"],
            "class_and_unit_group_only",
        )

    def test_summary_status_rows_and_counts_are_split_per_workload(self) -> None:
        def observation(workload: str, status: str = "ok") -> dict[str, object]:
            return {
                "case_key": f"{workload}:quartic",
                "workload": workload,
                "backend": "pari",
                "repetition": 0,
                "status": status,
                "validation": {"success": status == "ok", "errors": []},
            }

        def agreement(workload: str, status: str) -> dict[str, object]:
            return {
                "case_key": f"{workload}:quartic",
                "lhs_backend": "silex",
                "rhs_backend": "pari",
                "repetition": 0,
                "success": status == "agree",
                "status": status,
            }

        snapshot = {
            "fingerprint": "fp",
            "state": "complete",
            "manifest": {"plan": {"mode": "performance"}},
            "cases": [],
            "engines": [],
            "observations": [
                observation("class_unit_proven"),
                observation("class_unit_grh"),
                observation("class_unit_grh", "error"),
            ],
            "timing_samples": [],
            "agreements": [
                agreement("class_unit_proven", "agree"),
                agreement("class_unit_grh", "disagree"),
            ],
        }
        summary = reporting._summary(snapshot)
        self.assertEqual(
            summary["status_counts"],
            {
                "class_unit_grh": {"error": 1, "ok": 1},
                "class_unit_proven": {"ok": 1},
            },
        )
        self.assertEqual(
            [(r["workload"], r["backend"], r.get("ok", 0), r.get("error", 0))
             for r in summary["backend_status"]],
            [("class_unit_grh", "pari", 1, 1), ("class_unit_proven", "pari", 1, 0)],
        )
        self.assertEqual(
            [(r["workload"], r.get("agree", 0), r.get("disagree", 0))
             for r in summary["agreement_status"]],
            [("class_unit_grh", 0, 1), ("class_unit_proven", 1, 0)],
        )
        text = reporting._markdown(
            {
                "state": "complete",
                "fingerprint": "fp",
                "manifest": {
                    "machine": {
                        "requested_cpu": 2,
                        "cpu_model": "cpu",
                        "platform": "Linux",
                        "architecture": "x86_64",
                    },
                    "plan": {"suite": "number-field", "execution": {"threads": 1}},
                },
            },
            summary,
            [],
            {"generated": False, "reason": "none"},
        )
        self.assertIn("| class_unit_grh | pari | 1 | 0 | 0 | 0 | 1 | 0 |", text)
        self.assertIn("| class_unit_proven | pari | 1 | 0 | 0 | 0 | 0 | 0 |", text)
        self.assertIn("| class_unit_grh | silex | pari | 0 | 1 |", text)
        self.assertEqual(reporting.REPORT_RENDERER_VERSION, 3)


if __name__ == "__main__":
    unittest.main()
