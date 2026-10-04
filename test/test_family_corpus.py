"""Consistency checks for the family spot-test rows of the class/unit corpus."""

from __future__ import annotations

import json
import unittest

from silex_bench.resources import builtin_path
from silex_bench.workloads import CLASS_UNIT, load_cases

FAMILIES = {
    "cyclotomic",
    "cyclotomic_real",
    "pure",
    "simplest_cubic",
    "simplest_quartic",
    "multiquadratic",
    "quadratic_large",
    "lmfdb",
}


def polynomial_discriminant(coefficients: list[int]) -> int:
    """Exact discriminant of a monic integer polynomial (low-to-high input)."""
    n = len(coefficients) - 1
    f = coefficients[::-1]
    g = [(n - i) * f[i] for i in range(n)]
    size = 2 * n - 1
    rows = [[0] * i + f + [0] * (size - len(f) - i) for i in range(n - 1)]
    rows += [[0] * i + g + [0] * (size - len(g) - i) for i in range(n)]
    previous, sign = 1, 1
    for k in range(size - 1):
        if rows[k][k] == 0:
            for r in range(k + 1, size):
                if rows[r][k] != 0:
                    rows[k], rows[r] = rows[r], rows[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                rows[i][j] = (rows[i][j] * rows[k][k] - rows[i][k] * rows[k][j]) // previous
        previous = rows[k][k]
    resultant = sign * rows[size - 1][size - 1]
    return (-1) ** (n * (n - 1) // 2) * resultant


def _family_rows() -> list[dict]:
    path = builtin_path("corpora", "class_unit_fields.json")
    rows = json.loads(path.read_text(encoding="utf-8"))["fields"]
    return [row for row in rows if "family" in row]


class FamilyCorpusTests(unittest.TestCase):
    def test_every_family_covered(self) -> None:
        self.assertEqual({row["family"] for row in _family_rows()}, FAMILIES)

    def test_rows_have_tag_provenance_and_unmeasured_state(self) -> None:
        for row in _family_rows():
            with self.subTest(row=row["id"]):
                self.assertIn(row["family"], FAMILIES)
                self.assertEqual(row["benchmark_role"], "family")
                self.assertEqual(row["status"], "family_unmeasured")
                # Not yet measured on Silex: pending the family measurement task.
                self.assertIs(row["expected_success"], False)
                source = row["source"]
                self.assertIn("T-099", source)
                self.assertTrue(
                    any(key in source for key in ("PARI/GP 2.17.4", "LMFDB", "Miller")),
                    source,
                )
                self.assertRegex(source, r"2026-10-0\d")
                self.assertTrue(
                    any(key in source for key in ("certified", "uncertified", "table")),
                    source,
                )

    def test_degree_signature_and_discriminants_are_consistent(self) -> None:
        for row in _family_rows():
            with self.subTest(row=row["id"]):
                coefficients = row["coefficients_low_to_high"]
                degree = len(coefficients) - 1
                self.assertEqual(row["degree"], degree)
                self.assertEqual(coefficients[-1], 1)
                r1, r2 = row["signature"]
                self.assertEqual(r1 + 2 * r2, degree)
                self.assertEqual(row["expected_unit_rank"], r1 + r2 - 1)
                poly_disc = row["polynomial_discriminant"]
                self.assertEqual(poly_disc, polynomial_discriminant(coefficients))
                maximal = row["maximal_order_discriminant"]
                self.assertEqual(poly_disc % maximal, 0)
                quotient = poly_disc // maximal
                self.assertGreater(quotient, 0)
                index = round(quotient**0.5) if quotient < 10**30 else None
                if index is not None:
                    self.assertEqual(index * index, quotient)
                self.assertEqual(1 if maximal > 0 else -1, (-1) ** r2)
                invariants = row.get("expected_class_invariants")
                if invariants is not None:
                    product = 1
                    for item in invariants:
                        product *= item
                    self.assertEqual(product, row["expected_class_order"])
                    for small, large in zip(invariants[1:], invariants):
                        self.assertEqual(large % small, 0)

    def test_high_degree_family_rows_have_nontrivial_class_groups(self) -> None:
        covered = {
            (row["degree"], tuple(row["signature"]))
            for row in _family_rows()
            if row["degree"] >= 6 and row["expected_class_order"] > 1
        }
        for key in ((6, (0, 3)), (10, (0, 5)), (12, (0, 6)), (10, (10, 0))):
            with self.subTest(degree_signature=key):
                self.assertIn(key, covered)

    def test_family_rows_are_isolated_from_measured_profiles(self) -> None:
        cases = [
            case
            for case in load_cases(
                {"number_fields": builtin_path("corpora", "class_unit_fields.json")},
                (CLASS_UNIT,),
            )
            if "family" in case.tags
        ]
        self.assertEqual(len(cases), len(_family_rows()))
        for case in cases:
            with self.subTest(case=case.id):
                self.assertTrue(any(tag.startswith("family:") for tag in case.tags))
                for tag in ("publication", "scale", "dev", "quick"):
                    self.assertNotIn(tag, case.tags)
                self.assertEqual(case.expected_status, "failure")
                self.assertFalse(case.performance_eligible)


if __name__ == "__main__":
    unittest.main()
