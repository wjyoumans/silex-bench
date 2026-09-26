from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import silex_bench.reporting as reporting


def _observation(backend: str) -> dict[str, object]:
    return {
        "case_key": "class_unit_proven:quartic",
        "workload": "class_unit_proven",
        "backend": backend,
        "repetition": 0,
        "status": "ok",
        "success": True,
        "timeout": False,
        "validation": {"success": True, "errors": []},
        "engine_identity": {},
    }


def _sample(backend: str, variant: str, milliseconds: int) -> dict[str, object]:
    return {
        "case_key": "class_unit_proven:quartic",
        "backend": backend,
        "repetition": 0,
        "variant": variant,
        "sample_index": 0,
        "status": "ok",
        "timeout": False,
        "target_wall_ns": milliseconds * 1_000_000,
        "process_wall_ns": milliseconds * 2_000_000,
        "timing_scope": "class_and_unit_groups_only",
        "wall_clock": "fixture_monotonic",
        "timeout_source": "campaign_ceiling",
        "effective_timeout_seconds": 60,
        "internal_timing": {},
    }


class ReportingTests(unittest.TestCase):
    def test_summary_keeps_first_and_repeat_calls_as_distinct_series(self) -> None:
        snapshot = {
            "fingerprint": "fixture-fingerprint",
            "state": "complete",
            "source_ledger_schema_version": 2,
            "manifest": {"plan": {"mode": "performance"}},
            "cases": [
                {
                    "id": "quartic",
                    "workload": "class_unit_proven",
                    "metrics": {
                        "degree": 4,
                        "log10_abs_discriminant": 5.25,
                    },
                    "performance_eligible": True,
                }
            ],
            "engines": [],
            "observations": [_observation("silex"), _observation("hecke")],
            "timing_samples": [
                _sample("silex", "standard", 20),
                _sample("hecke", "first_call", 80),
                _sample("hecke", "repeat_call", 30),
            ],
            "agreements": [
                {
                    "case_key": "class_unit_proven:quartic",
                    "lhs_backend": "silex",
                    "rhs_backend": "hecke",
                    "repetition": 0,
                    "success": True,
                    "timing_eligible": False,
                    "status": "agree",
                }
            ],
        }

        summary = reporting._summary(snapshot)

        timings = {
            (row["backend"], row["variant"]): row["statistics"]["median"]
            for row in summary["timings"]
        }
        self.assertEqual(
            timings,
            {
                ("silex", "standard"): 20.0,
                ("hecke", "first_call"): 80.0,
                ("hecke", "repeat_call"): 30.0,
            },
        )
        self.assertNotIn("ratios", summary)
        self.assertNotIn("aggregate_ratios", summary)
        self.assertNotIn("speedup_definition", summary)
        self.assertEqual(
            {row["wall_clock"] for row in summary["timings"]},
            {"fixture_monotonic"},
        )

    def test_markdown_reports_absolute_variant_timings_without_speedups(self) -> None:
        timing = {
            "workload": "class_unit_proven",
            "case_id": "quartic",
            "backend": "hecke",
            "variant": "first_call",
            "timing_scope": "class_and_unit_groups_only",
            "wall_clock": "julia_time_ns_monotonic",
            "statistics": {
                "count": 3,
                "median": 80.0,
                "mad": 2.0,
                "bootstrap_95_low": 77.0,
                "bootstrap_95_high": 82.0,
            },
        }
        markdown = reporting._markdown(
            {
                "state": "complete",
                "fingerprint": "fixture-fingerprint",
                "manifest": {
                    "machine": {
                        "requested_cpu": 2,
                        "cpu_model": "fixture CPU",
                        "platform": "Linux",
                        "architecture": "x86_64",
                    },
                    "plan": {
                        "suite": "number-field",
                        "execution": {"threads": 1},
                    },
                },
            },
            {
                "primary_clock": "timing_samples.target_wall_ns",
                "engines": [],
                "backend_status": [],
                "agreement_status": [],
                "backend_exclusions": [
                    {
                        "backend": "hecke",
                        "workload": "sunit_proven",
                        "reason": "fixture exclusion",
                    }
                ],
                "failures": [],
                "timings": [timing],
            },
            [],
            {"generated": True, "files": ["plots/runtime.svg"]},
        )

        self.assertIn("| hecke | first_call |", markdown)
        self.assertIn("## Configured backend exclusions", markdown)
        self.assertIn("fixture exclusion", markdown)
        self.assertNotIn("speedup", markdown.lower())

    def test_plots_emit_only_deterministic_unconnected_svg_runtime_axes(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        summary = {
            "run_fingerprint": "fixture-fingerprint",
            "timings": [
                {
                    "workload": "class_unit_proven",
                    "case_key": "class_unit_proven:quartic",
                    "case_id": "quartic",
                    "backend": "hecke",
                    "variant": "first_call",
                    "timing_scope": "class_and_unit_groups_only",
                    "wall_clock": "julia_time_ns_monotonic",
                    "metrics": {
                        "degree": 4,
                        "log10_abs_discriminant": 5.25,
                    },
                    "units": "ms",
                    "statistics": {
                        "count": 3,
                        "median": 80.0,
                        "mad": 2.0,
                        "minimum": 77.0,
                        "maximum": 82.0,
                        "bootstrap_95_low": 77.0,
                        "bootstrap_95_high": 82.0,
                    },
                }
            ],
        }
        from matplotlib.axes import Axes

        calls: list[dict[str, object]] = []
        original_errorbar = Axes.errorbar

        def recording_errorbar(
            axis: Axes, *args: object, **kwargs: object
        ) -> object:
            calls.append(dict(kwargs))
            return original_errorbar(axis, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
            mock.patch.object(Axes, "errorbar", recording_errorbar),
        ):
            first = reporting._plots(Path(first_dir), summary)
            second = reporting._plots(Path(second_dir), summary)

            expected = [
                "plots/runtime-class_unit_proven-by-degree.svg",
                "plots/runtime-class_unit_proven-by-log10-abs-discriminant.svg",
            ]
            self.assertEqual(first, {"generated": True, "files": expected})
            self.assertEqual(second, first)
            self.assertTrue(all(Path(name).suffix == ".svg" for name in first["files"]))
            for relative in expected:
                first_bytes = (Path(first_dir) / relative).read_bytes()
                second_bytes = (Path(second_dir) / relative).read_bytes()
                self.assertEqual(first_bytes, second_bytes)
                self.assertNotIn(b"<dc:date>", first_bytes)
        self.assertTrue(calls)
        self.assertTrue(all(call["fmt"] == "o" for call in calls))
        self.assertTrue(all(call["linestyle"] == "none" for call in calls))

    def test_publication_gate_requires_three_not_nine_repetitions(self) -> None:
        def errors(repetitions: int, minimum: int) -> list[str]:
            snapshot = {
                "state": "incomplete",
                "manifest": {
                    "plan": {
                        "execution": {
                            "publication": True,
                            "repetitions": repetitions,
                            "minimum_repetitions": minimum,
                            "cpu": None,
                            "threads": 1,
                        },
                        "required_pairs": [],
                    },
                    "machine": {},
                    "sources": {},
                },
                "engines": [],
                "cases": [],
                "observations": [],
                "timing_samples": [],
            }
            return reporting._publication_errors(snapshot, {"agreements": []})

        self.assertTrue(any("three repetitions" in error for error in errors(2, 2)))
        self.assertFalse(any("repetitions" in error for error in errors(3, 3)))

    def test_publication_gate_rechecks_clean_source_enforcement(self) -> None:
        snapshot = {
            "state": "incomplete",
            "manifest": {
                "plan": {
                    "execution": {
                        "publication": True,
                        "repetitions": 3,
                        "minimum_repetitions": 3,
                        "cpu": None,
                        "threads": 1,
                        "require_clean_sources": False,
                    },
                    "required_pairs": [],
                },
                "machine": {},
                "sources": {
                    "silex": {"revision": "fixture", "dirty": True}
                },
            },
            "engines": [],
            "cases": [],
            "observations": [],
            "timing_samples": [],
        }
        errors = reporting._publication_errors(snapshot, {"agreements": []})
        self.assertIn("publication requires clean-source enforcement", errors)
        self.assertIn("publication source silex is not clean", errors)

    def test_legacy_discriminant_axis_is_recovered_without_mutation(self) -> None:
        case = {
            "id": "quartic",
            "workload": "class_unit_proven",
            "metrics": {"degree": 4, "discriminant_bits": 7},
            "expected": {"maximal_order_discriminant": -125},
            "input": {},
            "performance_eligible": True,
        }
        snapshot = {
            "fingerprint": "legacy-fixture",
            "state": "running",
            "source_ledger_schema_version": 1,
            "manifest": {
                "plan": {
                    "mode": "performance",
                    "execution": {"backend_exclusions": []},
                }
            },
            "cases": [case],
            "engines": [],
            "observations": [_observation("silex"), _observation("pari")],
            "timing_samples": [
                _sample("silex", "standard", 20),
                _sample("pari", "standard", 30),
            ],
            "agreements": [
                {
                    "case_key": "class_unit_proven:quartic",
                    "lhs_backend": "silex",
                    "rhs_backend": "pari",
                    "repetition": 0,
                    "success": True,
                    "timing_eligible": True,
                    "status": "agree",
                }
            ],
        }
        summary = reporting._summary(snapshot)
        self.assertEqual(case["metrics"], {"degree": 4, "discriminant_bits": 7})
        self.assertTrue(summary["timings"])
        for row in summary["timings"]:
            self.assertEqual(row["metrics"]["maximal_order_discriminant"], -125)
            self.assertAlmostEqual(
                row["metrics"]["log10_abs_discriminant"], 2.0969100130080562
            )

    def test_publication_gates_optional_backend_timings_in_the_summary(self) -> None:
        case = {
            "id": "quartic",
            "workload": "class_unit_proven",
            "input": {},
            "metrics": {"degree": 4, "log10_abs_discriminant": 5.25},
            "expected": {},
            "eligible_backends": [],
            "performance_eligible": True,
        }
        observations = [_observation(name) for name in ("silex", "pari", "hecke")]
        samples = [_sample(name, "standard", 20) for name in ("silex", "pari", "hecke")]
        expected_clocks = {
            "silex": "steady_clock",
            "pari": "pari_getwalltime_ms",
            "hecke": "incorrect_clock",
        }
        for sample in samples:
            sample["timing_scope"] = "class_and_unit_group_only"
            sample["wall_clock"] = expected_clocks[str(sample["backend"])]
        agreements = [
            {
                "case_key": "class_unit_proven:quartic",
                "lhs_backend": "silex",
                "rhs_backend": "hecke",
                "repetition": 0,
                "success": True,
                "status": "agree",
            }
        ]
        snapshot = {
            "fingerprint": "publication-fixture",
            "state": "complete",
            "source_ledger_schema_version": 2,
            "manifest": {
                "plan": {
                    "mode": "performance",
                    "backends": ["silex", "pari", "hecke"],
                    "cases": [case],
                    "required_pairs": [["silex", "pari"]],
                    "execution": {
                        "publication": True,
                        "repetitions": 3,
                        "minimum_repetitions": 3,
                        "jit_repetitions": 0,
                        "timeout_seconds": 60,
                        "backend_exclusions": [],
                        "cpu": 2,
                        "threads": 1,
                        "require_clean_sources": True,
                    },
                },
                "machine": {"system": "Linux", "cpu_model": "fixture CPU"},
                "sources": {},
            },
            "cases": [case],
            "engines": [
                {
                    "backend": name,
                    "available": True,
                    "identity": {"engine": name},
                }
                for name in ("silex", "pari", "hecke")
            ],
            "observations": observations,
            "timing_samples": samples,
            "agreements": agreements,
        }
        summary = reporting._summary(snapshot)
        errors = reporting._publication_errors(snapshot, summary)
        self.assertTrue(
            any(
                "unexpected scope or wall clock" in error
                and "/hecke/" in error
                for error in errors
            )
        )

    def test_publication_rejects_a_partial_optional_backend_series(self) -> None:
        case = {
            "id": "quartic",
            "workload": "class_unit_proven",
            "input": {},
            "metrics": {"degree": 4, "log10_abs_discriminant": 5.25},
            "expected": {},
            "eligible_backends": [],
            "performance_eligible": True,
        }
        observations = []
        samples = []
        agreements = []
        clocks = {
            "silex": "steady_clock",
            "pari": "pari_getwalltime_ms",
            "hecke": "julia_time_ns_monotonic",
        }
        for repetition in range(3):
            for backend in ("silex", "pari"):
                observation = _observation(backend)
                observation["repetition"] = repetition
                observations.append(observation)
                sample = _sample(backend, "standard", 20)
                sample["repetition"] = repetition
                sample["timing_scope"] = "class_and_unit_group_only"
                sample["wall_clock"] = clocks[backend]
                samples.append(sample)
            agreements.append(
                {
                    "case_key": "class_unit_proven:quartic",
                    "lhs_backend": "silex",
                    "rhs_backend": "pari",
                    "repetition": repetition,
                    "success": True,
                    "status": "agree",
                }
            )

        observations.append(_observation("hecke"))
        hecke_sample = _sample("hecke", "standard", 30)
        hecke_sample["timing_scope"] = "class_and_unit_group_only"
        hecke_sample["wall_clock"] = clocks["hecke"]
        samples.append(hecke_sample)
        agreements.append(
            {
                "case_key": "class_unit_proven:quartic",
                "lhs_backend": "silex",
                "rhs_backend": "hecke",
                "repetition": 0,
                "success": True,
                "status": "agree",
            }
        )
        snapshot = {
            "fingerprint": "publication-partial-optional-fixture",
            "state": "complete",
            "source_ledger_schema_version": 2,
            "manifest": {
                "plan": {
                    "mode": "performance",
                    "backends": ["silex", "pari", "hecke"],
                    "cases": [case],
                    "required_pairs": [["silex", "pari"]],
                    "execution": {
                        "publication": True,
                        "repetitions": 3,
                        "minimum_repetitions": 3,
                        "jit_repetitions": 0,
                        "timeout_seconds": 60,
                        "backend_exclusions": [],
                        "cpu": 2,
                        "threads": 1,
                        "require_clean_sources": True,
                    },
                },
                "machine": {"system": "Linux", "cpu_model": "fixture CPU"},
                "sources": {},
            },
            "cases": [case],
            "engines": [
                {
                    "backend": backend,
                    "available": True,
                    "identity": {"engine": backend},
                }
                for backend in ("silex", "pari", "hecke")
            ],
            "observations": observations,
            "timing_samples": samples,
            "agreements": agreements,
        }

        summary = reporting._summary(snapshot)
        self.assertEqual(
            next(
                row["statistics"]["count"]
                for row in summary["timings"]
                if row["backend"] == "hecke"
            ),
            1,
        )
        errors = reporting._publication_errors(snapshot, summary)
        self.assertTrue(
            any(
                "timing series lacks every planned repetition" in error
                and "/hecke" in error
                for error in errors
            )
        )


if __name__ == "__main__":
    unittest.main()
