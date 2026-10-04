from __future__ import annotations

import unittest
from types import SimpleNamespace

from silex_bench.presentation import (
    render_campaign,
    render_doctor,
    render_list,
    render_plan,
    render_progress,
    table,
)


class PresentationTests(unittest.TestCase):
    def test_table_is_deterministic_and_rejects_ragged_rows(self) -> None:
        self.assertEqual(
            table(("Name", "Value"), (("short", 1), ("longer", None))),
            "Name    Value\n------  -----\nshort   1    \nlonger  -    ",
        )
        with self.assertRaisesRegex(ValueError, "header width"):
            table(("one",), (("a", "b"),))

    def test_list_shows_capabilities_and_unsupported_cells(self) -> None:
        text = render_list(
            {
                "suites": ["number-field"],
                "profiles": ["quick"],
                "workloads": [
                    {
                        "id": "ordinary",
                        "display_name": "Ordinary",
                        "timing_scope": "target",
                        "scale_axes": ["degree"],
                    },
                    {
                        "id": "sunit",
                        "display_name": "S-unit",
                        "timing_scope": "process",
                        "scale_axes": ["degree", "s_size"],
                    },
                ],
                "backends": [
                    {
                        "id": "magma",
                        "display_name": "Magma",
                        "capabilities": ["ordinary"],
                    }
                ],
            }
        )
        self.assertIn("Adapter capabilities", text)
        self.assertIn("Magma", text)
        self.assertIn("unsupported", text)

    def test_plan_and_doctor_distinguish_policy_from_availability(self) -> None:
        plan = render_plan(
            {
                "suite": "number-field",
                "profile": "quick",
                "mode": "correctness",
                "workloads": ["maximal_order"],
                "backends": ["silex", "pari"],
                "case_count": 2,
                "sample_count": 4,
                "required_pairs": [["silex", "pari"]],
                "cases": [
                    {
                        "id": "quadratic-5",
                        "workload": "maximal_order",
                        "metrics": {"degree": 2},
                        "tags": ["quick", "core"],
                    }
                ],
                "adapter_capabilities": [
                    {"backend": "silex", "capabilities": ["maximal_order"]},
                    {"backend": "pari", "capabilities": []},
                ],
                "execution": {
                    "repetitions": 1,
                    "timeout_seconds": 30,
                    "budget_seconds": 60,
                    "cpu": None,
                    "require_all_adapters": True,
                    "backend_exclusions": [
                        {
                            "backend": "hecke",
                            "workload": "sunit_proven",
                            "reason": "bounded campaign policy",
                        }
                    ],
                },
            }
        )
        self.assertIn("silex:pari", plan)
        self.assertIn("Require all adapters", plan)
        self.assertIn("Resolved cases", plan)
        self.assertIn("quadratic-5", plan)
        self.assertIn("degree=2", plan)
        self.assertIn("Adapter cells", plan)
        self.assertIn("unsupported", plan)
        self.assertIn("Observation timeout ceiling", plan)
        self.assertIn("Profile backend exclusions", plan)
        self.assertIn("bounded campaign policy", plan)
        doctor = render_doctor(
            {
                "engines": [
                    {
                        "backend": "magma",
                        "display_name": "Magma",
                        "available": False,
                        "capabilities": ["maximal_order"],
                        "identity": {},
                        "error": "license unavailable",
                    }
                ],
                "required_errors": [],
                "optional_errors": ["Magma: license unavailable"],
            }
        )
        self.assertIn("Required comparisons: PASS", doctor)
        self.assertIn("Optional adapter issues", doctor)

    def test_campaign_renders_correctness_and_artifacts_without_speedups(self) -> None:
        text = render_campaign(
            {
                "state": "complete",
                "run_dir": "/tmp/run",
                "backend_status": [{"workload": "class_unit_grh", "backend": "silex", "ok": 1}],
                "agreement_status": [
                    {"candidate": "silex", "baseline": "pari", "agree": 1}
                ],
                "aggregate_ratios": [
                    {
                        "workload": "maximal_order",
                        "candidate": "silex",
                        "baseline": "pari",
                        "case_count": 1,
                        "geometric_mean_speedup": 2.0,
                    }
                ],
                "report": {
                    "markdown": "/tmp/report.md",
                    "directory": "/tmp/export",
                    "plots": [],
                    "plot_reason": "matplotlib is not installed",
                },
            }
        )
        self.assertIn("Campaign state: complete", text)
        self.assertIn("class_unit_grh", text)
        self.assertIn("Workload", text)
        self.assertNotIn("speedup", text.lower())
        self.assertNotIn("2.000x", text)
        self.assertIn("/tmp/report.md", text)
        self.assertIn("Plots not generated", text)

    def test_progress_line_retains_each_recorded_timing_variant(self) -> None:
        observation = SimpleNamespace(
            case_key="class_unit_proven:quartic",
            backend="hecke",
            repetition=1,
            status=SimpleNamespace(value="ok"),
            timing_samples=(
                SimpleNamespace(
                    variant="first_call",
                    status=SimpleNamespace(value="ok"),
                    target_wall_ns=1_250_000_000,
                ),
                SimpleNamespace(
                    variant="repeat_call",
                    status=SimpleNamespace(value="ok"),
                    target_wall_ns=850_000_000,
                ),
            ),
        )

        self.assertEqual(
            render_progress(observation, 2, 7),
            "[2/7] class_unit_proven:quartic/hecke rep=2: ok; "
            "first_call=1.250 s, repeat_call=850.000 ms",
        )

    def test_progress_line_uses_legacy_observation_clock_as_standard_sample(self) -> None:
        observation = SimpleNamespace(
            case_key="maximal_order:quadratic",
            backend="pari",
            repetition=0,
            status=SimpleNamespace(value="ok"),
            target_wall_ns=875_000,
        )

        self.assertEqual(
            render_progress(observation, 1, 1),
            "[1/1] maximal_order:quadratic/pari rep=1: ok; standard=875.000 us",
        )


if __name__ == "__main__":
    unittest.main()
