"""Tests for installed built-in suites, profiles, and corpora."""

from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from silex_bench.campaign import build_plan
from silex_bench.configuration import (
    CONFIG_SCHEMA_VERSION,
    RunOverrides,
    effective_execution,
    load_profile,
    load_suite,
    load_tools,
)
from silex_bench.registry import builtin_registry
from silex_bench.resources import builtin_names, builtin_path, resolve_path
from silex_bench.workloads import (
    CLASS_UNIT,
    ELEMENT_SQUARE_ROOT,
    NUMBER_FIELD_WORKLOADS,
    SUNIT,
    load_cases,
    select_cases,
)


ROOT = Path(__file__).resolve().parents[1]


class BuiltinResourceTests(unittest.TestCase):
    def test_wheel_configuration_includes_user_and_license_documents(self) -> None:
        configuration = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(configuration["project"]["readme"], "README.md")
        self.assertEqual(
            configuration["project"]["license-files"],
            ["LICENSE", "NOTICE.md", "THIRD_PARTY_NOTICES.md"],
        )
        for name in ("README.md", "LICENSE", "NOTICE.md", "THIRD_PARTY_NOTICES.md"):
            self.assertTrue((ROOT / name).is_file(), name)
        self.assertIn(
            "PARI/GP source lineage",
            (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
        )

    def test_wheel_configuration_includes_every_resource_collection(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = set(project["tool"]["setuptools"]["package-data"]["silex_bench"])
        self.assertEqual(
            patterns,
            {
                "data/corpora/*.json",
                "data/profiles/*.toml",
                "data/suites/*.toml",
            },
        )

    def test_catalog_exposes_loadable_suite_profile_and_corpora(self) -> None:
        self.assertEqual(builtin_names("suites"), ("number-field",))
        self.assertEqual(
            builtin_names("profiles"),
            ("dev", "proven-focus", "publication", "quick", "scale"),
        )
        self.assertEqual(
            builtin_names("corpora"),
            ("class_unit_fields", "sunit_fields"),
        )

        suite = load_suite(builtin_path("suites", "number-field"))
        profile = load_profile(builtin_path("profiles", "quick.toml"))

        self.assertEqual(suite.id, "number-field")
        self.assertEqual(profile.id, "quick")
        self.assertEqual(
            suite.corpora["number_fields"],
            builtin_path("corpora", "class_unit_fields"),
        )
        self.assertEqual(
            suite.corpora["sunit"], builtin_path("corpora", "sunit_fields.json")
        )
        for corpus in suite.corpora.values():
            payload = json.loads(corpus.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)

    def test_profile_resources_keep_their_own_v1_schema_and_jit_policy(self) -> None:
        self.assertEqual(CONFIG_SCHEMA_VERSION, 1)
        expected = {
            "quick": (1, 0, 180.0),
            "dev": (3, 1, 300.0),
            "proven-focus": (9, 1, 60.0),
            "scale": (5, 1, 3600.0),
            "publication": (3, 1, 60.0),
        }
        for name, (repetitions, jit_repetitions, timeout) in expected.items():
            with self.subTest(profile=name):
                path = builtin_path("profiles", name)
                raw = tomllib.loads(path.read_text(encoding="utf-8"))
                profile = load_profile(path)
                self.assertEqual(raw["schema_version"], CONFIG_SCHEMA_VERSION)
                self.assertNotIn("warmups", raw)
                self.assertEqual(profile.repetitions, repetitions)
                self.assertEqual(profile.jit_repetitions, jit_repetitions)
                self.assertEqual(profile.timeout_seconds, timeout)

        publication = load_profile(builtin_path("profiles", "publication"))
        self.assertEqual(publication.minimum_repetitions, 3)
        self.assertEqual(len(publication.backend_exclusions), 1)
        exclusion = publication.backend_exclusions[0]
        self.assertEqual((exclusion.backend, exclusion.workload), ("hecke", SUNIT))
        self.assertIn("60-second", exclusion.reason)
        execution = effective_execution(publication, RunOverrides())
        self.assertEqual(
            execution["backend_exclusions"],
            [
                {
                    "backend": "hecke",
                    "workload": SUNIT,
                    "reason": exclusion.reason,
                }
            ],
        )

    def test_proven_focus_profile_keeps_bounded_exploratory_policy(self) -> None:
        profile = load_profile(builtin_path("profiles", "proven-focus"))
        execution = effective_execution(profile, RunOverrides())
        self.assertEqual(profile.id, "proven-focus")
        self.assertEqual(execution["include_tags"], ["dev"])
        self.assertEqual(execution["exclude_tags"], [])
        self.assertEqual(execution["backend_exclusions"], [])
        self.assertEqual(execution["metrics"], {"degree": [2.0, 6.0]})
        self.assertEqual(execution["repetitions"], 9)
        self.assertEqual(execution["jit_repetitions"], 1)
        self.assertEqual(execution["threads"], 1)
        self.assertEqual(execution["timeout_seconds"], 60.0)
        self.assertEqual(execution["budget_seconds"], 1200.0)
        self.assertTrue(execution["require_clean_sources"])
        self.assertFalse(execution["publication"])
        self.assertEqual(execution["minimum_repetitions"], 1)
        self.assertIsNone(execution["cpu"])

    def test_proven_focus_explicit_class_unit_plan_requires_silex_pari(self) -> None:
        expected_degrees = {
            "real_quadratic_5_proven": 2,
            "cubic_disc81_proven": 3,
            "quartic_disc1856_proven": 4,
            "quintic_disc4417_proven": 5,
        }
        plan = build_plan(
            ROOT,
            load_suite(builtin_path("suites", "number-field")),
            load_profile(builtin_path("profiles", "proven-focus")),
            load_tools(None),
            RunOverrides(
                workloads=(CLASS_UNIT,),
                backends=("silex", "pari"),
                required_pairs=(("silex", "pari"),),
                case_ids=tuple(expected_degrees),
            ),
            builtin_registry(),
            performance=True,
        )
        self.assertEqual(plan.workloads, (CLASS_UNIT,))
        self.assertEqual(plan.backends, ("silex", "pari"))
        self.assertEqual(plan.required_pairs, (("silex", "pari"),))
        self.assertEqual(len(plan.cases), 4)
        self.assertEqual(
            {case.id: case.metrics["degree"] for case in plan.cases},
            expected_degrees,
        )
        self.assertEqual(plan.sample_count, 72)
        # The profile ceiling preserves the corpus's shorter observation hints.
        self.assertEqual(plan.nominal_timeout_product_seconds, 720.0)
        self.assertLess(
            plan.nominal_timeout_product_seconds, plan.execution["budget_seconds"]
        )
        for case in plan.cases:
            with self.subTest(case=case.key):
                self.assertEqual(case.workload, CLASS_UNIT)
                self.assertIn("dev", case.tags)
                self.assertEqual(case.expected_status, "success")
                self.assertTrue(case.performance_eligible)
                self.assertEqual(case.input["timeout_seconds"], 10)

    def test_expected_failures_cannot_be_timing_successes(self) -> None:
        failure_ids = (
            "octodecic_x18_minus_x_plus_1_proven",
            "nonadecic_x19_minus_x_minus_1_proven",
        )
        suite = load_suite(builtin_path("suites", "number-field"))
        profile = load_profile(builtin_path("profiles", "scale"))
        cases = load_cases(suite.corpora, (CLASS_UNIT,))
        execution = effective_execution(profile, RunOverrides(case_ids=failure_ids))
        diagnostics = select_cases(cases, execution, performance=False)
        self.assertEqual({case.id for case in diagnostics}, set(failure_ids))
        self.assertEqual(select_cases(cases, execution, performance=True), [])
        contract = builtin_registry().workloads[CLASS_UNIT]
        for case in diagnostics:
            with self.subTest(case=case.key):
                self.assertEqual(case.expected_status, "failure")
                self.assertFalse(case.performance_eligible)
                self.assertNotIn("dev", case.tags)
                for backend in ("silex", "pari"):
                    with self.subTest(backend=backend):
                        validation = contract.validate_observation(case, backend, {}, {})
                        self.assertFalse(validation.success)
                        self.assertEqual(validation.checks, {"expected_success": False})
                        self.assertIn("expected-failure", validation.errors[0])

    def test_publication_corpus_uses_all_fields_and_bounds_square_roots(self) -> None:
        suite = load_suite(builtin_path("suites", "number-field"))
        profile = load_profile(builtin_path("profiles", "publication"))
        cases = load_cases(suite.corpora, (*NUMBER_FIELD_WORKLOADS, SUNIT))
        selected = select_cases(
            cases,
            effective_execution(profile, RunOverrides()),
            performance=True,
        )
        by_workload = {
            workload: [case for case in selected if case.workload == workload]
            for workload in (*NUMBER_FIELD_WORKLOADS, SUNIT)
        }
        self.assertEqual(len(selected), 233)
        self.assertEqual(len(by_workload[CLASS_UNIT]), 59)
        self.assertEqual(len(by_workload[ELEMENT_SQUARE_ROOT]), 46)
        self.assertEqual(
            max(case.metrics["degree"] for case in by_workload[ELEMENT_SQUARE_ROOT]),
            9,
        )
        self.assertEqual(len(by_workload[SUNIT]), 6)
        self.assertTrue(all(not case.eligible_backends for case in by_workload[SUNIT]))
        excluded_cells = {
            (row["backend"], row["workload"])
            for row in effective_execution(profile, RunOverrides())[
                "backend_exclusions"
            ]
        }
        self.assertNotIn(("magma", SUNIT), excluded_cells)
        backend_cells = sum(
            len(
                [
                    backend
                    for backend in suite.backends
                    if (
                        not case.eligible_backends
                        or backend == "silex"
                        or backend in case.eligible_backends
                    )
                    and (backend, case.workload) not in excluded_cells
                ]
            )
            for case in selected
        )
        self.assertEqual(backend_cells, 904)
        self.assertEqual(backend_cells * profile.repetitions, 2712)

        corpus = json.loads(
            suite.corpora["number_fields"].read_text(encoding="utf-8")
        )
        self.assertNotIn("benchmark_warmups", corpus)
        no_role_ids = {
            row["id"] for row in corpus["fields"] if "benchmark_role" not in row
        }
        selected_number_field_ids = {
            case.id for case in selected if case.workload in NUMBER_FIELD_WORKLOADS
        }
        self.assertLessEqual(no_role_ids, selected_number_field_ids)

    def test_every_case_has_exact_discriminant_metrics(self) -> None:
        suite = load_suite(builtin_path("suites", "number-field"))
        cases = load_cases(suite.corpora, (*NUMBER_FIELD_WORKLOADS, SUNIT))
        for case in cases:
            with self.subTest(case=case.key):
                self.assertGreater(case.metrics["discriminant_bits"], 0)
                self.assertGreaterEqual(case.metrics["log10_abs_discriminant"], 0.0)
                if case.workload in NUMBER_FIELD_WORKLOADS:
                    discriminant = int(case.expected["maximal_order_discriminant"])
                else:
                    discriminant = int(case.input["maximal_order_discriminant"])
                self.assertNotEqual(discriminant, 0)
                self.assertEqual(
                    case.metrics["maximal_order_discriminant"], discriminant
                )
                self.assertEqual(
                    case.metrics["discriminant_bits"], abs(discriminant).bit_length()
                )

        maximal_order_cases = {
            case.id: case
            for case in cases
            if case.workload == "maximal_order"
        }
        self.assertEqual(
            maximal_order_cases["random_cubic_proven"].expected[
                "maximal_order_discriminant"
            ],
            -23,
        )
        self.assertEqual(
            maximal_order_cases["quartic_disc35019_proven"].expected[
                "maximal_order_discriminant"
            ],
            -3891,
        )

    def test_number_field_loader_rejects_missing_or_zero_exact_discriminant(self) -> None:
        base_row = {
            "id": "missing_discriminant",
            "coefficients_low_to_high": [-5, 0, 1],
            "degree": 2,
            "expected_success": True,
        }
        variants = {
            "missing": base_row,
            "zero": {**base_row, "maximal_order_discriminant": 0},
            "inexact": {**base_row, "maximal_order_discriminant": 5.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, row in variants.items():
                with self.subTest(variant=name):
                    path = root / f"{name}.json"
                    path.write_text(json.dumps({"fields": [row]}), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        "exact signed nonzero maximal_order_discriminant",
                    ):
                        load_cases({"number_fields": path}, (CLASS_UNIT,))

    def test_resolve_path_prefers_builtins_for_plain_names(self) -> None:
        self.assertEqual(
            resolve_path("suites", "number-field"),
            builtin_path("suites", "number-field"),
        )

    def test_resolve_path_preserves_explicit_custom_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.toml"
            custom.write_text("schema_version = 1\n", encoding="utf-8")
            self.assertEqual(resolve_path("suites", custom), custom.absolute())

        missing = Path("custom") / "missing.toml"
        self.assertEqual(resolve_path("suites", missing), missing.absolute())

    def test_catalog_rejects_unknown_collection_and_unsafe_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "resource collection"):
            builtin_names("unknown")
        with self.assertRaisesRegex(ValueError, "plain resource name"):
            builtin_path("profiles", "../quick")


if __name__ == "__main__":
    unittest.main()
