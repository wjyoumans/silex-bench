from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from silex_bench import family_columns
from silex_bench.backends.pari import (
    REPEAT_MIN_TOTAL_MS,
    REPEAT_THRESHOLD_MS,
    _programs,
    _repeat_timing,
)
from silex_bench.model import FieldSpec, SampleRequest
from silex_bench.process import parse_key_values
from test_backend_timing import PariBackend, _pari_class_unit_stdout, context, marked_result



def _request(operation: str) -> SampleRequest:
    return SampleRequest(
        field=_field(),
        operation=operation,
        sample_kind="cold_process",
        sample_index=0,
        warmup=None,
        seed=7,
    )


def _field() -> FieldSpec:
    from test_backend_timing import field

    return field("q", -5)


class RepeatTimingProgramTests(unittest.TestCase):
    def test_final_program_repeats_until_minimum_total_wall_time(self) -> None:
        for operation in ("class_unit_grh", "class_unit_proven"):
            with self.subTest(operation=operation):
                _, target, final = _programs(_request(operation))
                self.assertIn(f"target_internal_wall_ms < {REPEAT_THRESHOLD_MS}", final)
                self.assertIn(f"rep_min_ms = {REPEAT_MIN_TOTAL_MS}", final)
                self.assertIn("until (getwalltime() - rep_t0 >= rep_min_ms", final)
                self.assertIn('print("target_repeats="', final)
                self.assertIn('print("quadclassunit_repeats="', final)
                self.assertIn("poldegree(P) == 2", final)
                self.assertTrue(final.rstrip().endswith("quit"))
                # The headline target region is untouched.
                self.assertNotIn("rep_", target)
                self.assertNotIn("quadclassunit", target)
        proven = _programs(_request("class_unit_proven"))[2]
        grh = _programs(_request("class_unit_grh"))[2]
        self.assertIn("bnfcertify(bnf_rep)", proven)
        self.assertNotIn("bnfcertify(bnf_rep)", grh)

    def test_repeat_fields_are_recorded_in_the_raw_result(self) -> None:
        stdout = _pari_class_unit_stdout(1).replace("certified=1\n", "") + (
            "target_repeats=412\n"
            "target_repeat_ms=0.121359223300970873786407766990291262136\n"
            "quadclassunit_repeats=1503\n"
            "quadclassunit_repeat_ms=0.0332\n"
            "quadclassunit_class_order=1\n"
            "quadclassunit_class_invariants=[]\n"
        )
        backend = PariBackend()
        backend._probe = {
            "engine": "pari",
            "available": True,
            "engine_identity": {"executable": "/usr/bin/gp"},
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.pari.run_marked_process",
            return_value=marked_result(stdout),
        ):
            payload = backend.run(_request("class_unit_grh"), context(Path(temporary)))
        self.assertTrue(payload["success"], payload["error"])
        repeat = payload["timing"]["repeat_timing"]
        self.assertEqual(repeat["target_repeats"], 412)
        self.assertAlmostEqual(repeat["target_repeat_ms"], 0.12135922)
        self.assertEqual(repeat["family_reference"]["repeats"], 1503)
        self.assertEqual(repeat["family_reference"]["repeat_ms"], 0.0332)
        self.assertEqual(payload["timing"]["target_wall_ms"], 1.0)
        self.assertEqual(list(payload["timing"]["components_ms"]), ["bnfinit"])

    def test_no_repeat_fields_means_no_repeat_timing(self) -> None:
        self.assertIsNone(_repeat_timing(parse_key_values("class_order=1\n")))


def _row(workload, case, backend, median, degree=2, disc=5):
    return {
        "workload": workload,
        "case_key": f"{workload}:{case}",
        "case_id": case,
        "backend": backend,
        "variant": "standard",
        "statistics": {"median": median},
        "metrics": {"degree": degree, "maximal_order_discriminant": disc},
    }


class FamilyGroupingTests(unittest.TestCase):
    def test_grouping_rows_and_geomeans(self) -> None:
        w = "class_unit_grh"
        timings = []
        for case, disc, degree, silex, pari in (
            ("iq1", -47, 2, 2.0, 1.0),
            ("iq2", -23, 2, 8.0, 1.0),
            ("rq1", 5, 2, 3.0, 1.0),
            ("cubic", -23, 3, 5.0, 5.0),
        ):
            timings.append(_row(w, case, "silex", silex, degree, disc))
            timings.append(_row(w, case, "pari", pari, degree, disc))
        refs = {
            (w, f"{w}:iq1"): [0.5],
            (w, f"{w}:iq2"): [0.25, 0.75],
            (w, f"{w}:rq1"): [0.5],
        }
        columns = family_columns.family_columns(timings, refs)
        by = {c["column"]: c for c in columns}
        self.assertEqual(by["headline"]["rows"], 4)
        self.assertAlmostEqual(by["headline"]["geomean"], (2 * 8 * 3 * 1) ** 0.25)
        self.assertEqual(by["imaginary_quadratic"]["rows"], 2)
        self.assertAlmostEqual(by["imaginary_quadratic"]["geomean"], math.sqrt(4 * 16))
        self.assertEqual(by["real_quadratic"]["rows"], 1)
        self.assertAlmostEqual(by["real_quadratic"]["geomean"], 6.0)
        self.assertIn("exact", by["imaginary_quadratic"]["label"])
        self.assertIn("no unit", by["real_quadratic"]["label"])
        self.assertTrue(all(c["workload"] == w for c in columns))

    def test_rows_without_reference_are_excluded(self) -> None:
        w = "class_unit_proven"
        timings = [
            _row(w, "iq", "silex", 2.0, 2, -47),
            _row(w, "iq", "pari", 1.0, 2, -47),
        ]
        by = {c["column"]: c for c in family_columns.family_columns(timings, {})}
        self.assertEqual(by["imaginary_quadratic"]["rows"], 0)
        self.assertIsNone(by["imaginary_quadratic"]["geomean"])
        self.assertEqual(by["headline"]["rows"], 1)
        self.assertIn("-", "\n".join(family_columns.markdown_lines(list(by.values()), w)))

    def test_rows_differing_in_scope_are_not_overwritten(self) -> None:
        w = "class_unit_proven"
        a = _row(w, "iq", "silex", 2.0, 2, -47)
        b = dict(_row(w, "iq", "silex", 8.0, 2, -47), timing_scope="other")
        timings = [a, b, _row(w, "iq", "pari", 1.0, 2, -47)]
        by = {c["column"]: c for c in family_columns.family_columns(timings, {})}
        self.assertEqual(by["headline"]["rows"], 0)

    def test_summary_and_markdown_carry_sub_columns(self) -> None:
        from silex_bench import reporting

        w = "class_unit_proven"
        timings = [
            _row(w, "iq", "silex", 2.0, 2, -47),
            _row(w, "iq", "pari", 1.0, 2, -47),
        ]
        columns = family_columns.family_columns(timings, {(w, f"{w}:iq"): [0.5]})
        summary = {
            "primary_clock": "c",
            "engines": [],
            "backend_status": [],
            "agreement_status": [],
            "backend_exclusions": [],
            "failures": [],
            "timings": [],
            "family_columns": columns,
        }
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
        self.assertIn("Per-family PARI reference sub-columns (proven)", text)
        self.assertIn("warm, repeated", text)

    def test_reference_ms_validation(self) -> None:
        good = {"repeat_timing": {"family_reference": {"repeat_ms": 0.5, "repeats": 10}}}
        self.assertEqual(family_columns.reference_ms(good), 0.5)
        for bad in (
            None,
            {},
            {"repeat_timing": {"family_reference": {"repeat_ms": 0, "repeats": 10}}},
            {"repeat_timing": {"family_reference": {"repeat_ms": 0.5, "repeats": 0}}},
        ):
            self.assertIsNone(family_columns.reference_ms(bad))


if __name__ == "__main__":
    unittest.main()
