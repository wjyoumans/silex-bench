"""Expectation-tracking checks for the class/unit corpus."""

from __future__ import annotations

import json
import re
import unittest

from silex_bench.resources import builtin_path

TASK = re.compile(r"T-\d+")


def _rows() -> list[dict]:
    path = builtin_path("corpora", "class_unit_fields.json")
    return json.loads(path.read_text(encoding="utf-8"))["fields"]


def tracking_reference(row: dict) -> str | None:
    """Backlog task tracking a failing row, or None when absent.

    Measured rows carry an explicit ``tracking_task``; unmeasured family rows
    name their pending task in the ``source`` note.
    """
    explicit = row.get("tracking_task")
    if isinstance(explicit, str) and TASK.fullmatch(explicit):
        return explicit
    if "family" in row:
        match = TASK.search(row.get("source", ""))
        return match.group(0) if match else None
    return None


class ExpectedSuccessTrackingTests(unittest.TestCase):
    def test_every_false_row_names_a_tracking_task(self) -> None:
        failing = [row for row in _rows() if row["expected_success"] is False]
        self.assertTrue(failing)
        for row in failing:
            with self.subTest(row=row["id"]):
                self.assertIsNotNone(tracking_reference(row))

    def test_missing_reference_is_detected(self) -> None:
        self.assertIsNone(
            tracking_reference({"id": "x", "expected_success": False, "source": "none"})
        )
        self.assertEqual(
            tracking_reference({"expected_success": False, "tracking_task": "T-105"}),
            "T-105",
        )

    def test_true_rows_do_not_claim_a_failure_tracker(self) -> None:
        for row in _rows():
            if row["expected_success"] is True:
                with self.subTest(row=row["id"]):
                    self.assertNotIn("tracking_task", row)


if __name__ == "__main__":
    unittest.main()
