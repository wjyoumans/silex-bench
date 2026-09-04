from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from silex_bench.resources import builtin_path
from silex_bench.workloads import sunit_module


def normalized_result(
    engine: str = "silex", ordinary_status: str = "proven"
) -> dict[str, object]:
    result: dict[str, object] = {
        "engine": engine,
        "success": True,
        "certification_status": ordinary_status,
        "class_group_proof_status": ordinary_status,
        "unit_group_proof_status": ordinary_status,
        "regulator_proof_status": (
            "verified" if engine == "silex" else ordinary_status
        ),
        "s_class_order": "1",
        "s_class_invariants": [],
        "torsion_order": "2",
        "ordinary_free_rank": 1,
        "nonunit_rank": 0,
        "free_rank": 1,
        "valuation_lattice_index": "1",
        "selected_primes": [],
        "canonical_prime_decompositions": [],
        "valuation_matrix": [],
        "regulator_midpoint": None,
        "membership_status": "verified",
        "mixed_round_trip_verified": True,
        "verified_round_trip_count": 1,
        "mixed_outcome": "verified",
        "outside_support_rejected": True,
        "outside_outcome": "not_sunit",
        "s_class_proof_status": "verified",
        "s_unit_proof_status": "verified",
        "s_regulator_proof_status": "verified",
        "final_result_published": True,
    }
    if engine == "silex":
        result["engine_identity"] = {
            "executable": "/tmp/silex-class-unit-instance",
            "executable_sha256": "d" * 64,
            "version": None,
            "source": "/tmp/silex",
        }
    elif engine == "pari":
        result["engine_identity"] = {
            "executable": "/tmp/gp",
            "executable_sha256": "d" * 64,
            "version": "2.17.3",
            "required_version": "2.17.3",
            "source": "/tmp/pari-2.17.3",
            "source_version": "2.17.3",
        }
    elif engine == "hecke":
        result["engine_identity"] = {
            "executable": "/tmp/julia",
            "executable_sha256": "d" * 64,
            "version": "1.12.6",
            "package_version": "0.39.19",
            "source": "/tmp/Hecke.jl",
        }
    return result


def native_process(**changes: object) -> dict[str, object]:
    process: dict[str, object] = {
        "available": True,
        "success": True,
        "timeout": False,
        "process_wall_ms": 1.0,
        "stderr": "",
    }
    process.update(changes)
    return process


def valuation_matrix_for(values: dict[str, object]) -> list[list[str]]:
    rank = values["nonunit_rank"]
    index = values["valuation_lattice_index"]
    if type(rank) is not int or type(index) is not str or rank == 0:
        return []
    return [
        [
            index if row == 0 and column == 0 else "1" if row == column else "0"
            for column in range(rank)
        ]
        for row in range(rank)
    ]


def native_payload(
    expected: dict[str, object] | None = None,
    *,
    selected_primes: object = None,
    canonical_decomposition: object = None,
    maximal_order_discriminant: object = None,
    maximal_order_basis_power: object = None,
) -> dict[str, object]:
    values = expected or {
        "s_class_order": "1",
        "s_class_invariants": [],
        "torsion_order": "2",
        "ordinary_free_rank": 1,
        "nonunit_rank": 0,
        "free_rank": 1,
        "valuation_lattice_index": "1",
    }
    selected = [] if selected_primes is None else selected_primes
    complete = selected if canonical_decomposition is None else canonical_decomposition
    return {
        "success": True,
        "timeout": False,
        "certification_status": "proven",
        "class_group_proof_status": "proven",
        "unit_group_proof_status": "proven",
        "regulator_proof_status": "verified",
        "final_result_published": True,
        "maximal_order_discriminant": maximal_order_discriminant,
        "maximal_order_basis_power": maximal_order_basis_power,
        "phase_timing_ms": {},
        "component_timing_ms": {},
        "sunit": {
            "success": True,
            "selected_primes": selected,
            "canonical_prime_decompositions": complete,
            "s_class_group": {
                "order": values["s_class_order"],
                "invariants": values["s_class_invariants"],
                "proof_status": "verified",
            },
            "s_unit_group": {
                "torsion_order": values["torsion_order"],
                "ordinary_free_rank": values["ordinary_free_rank"],
                "nonunit_rank": values["nonunit_rank"],
                "free_rank": values["free_rank"],
                "valuation_matrix": valuation_matrix_for(values),
                "valuation_lattice_index": values["valuation_lattice_index"],
                "regulator_midpoint": None,
                "proof_status": "verified",
                "regulator_proof_status": "verified",
            },
            "membership": {
                "status": "verified",
                "mixed_round_trip_verified": True,
                "verified_round_trip_count": 1,
                "mixed_outcome": "verified",
                "outside_support_rejected": True,
                "outside_outcome": "not_sunit",
            },
            "final_result_published": True,
            "timing_ms": {},
        },
    }


PRIME_INDEX_CONVENTION = (
    "zero-based authoritative manifest order of exact two-generator ideals "
    "(p, beta_power_basis) within each rational-prime decomposition"
)


def sunit_manifest(fields: list[object], *, schema_version: object = 2) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "prime_index_convention": PRIME_INDEX_CONVENTION,
        "fields": fields,
    }


class SUnitContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest_path = builtin_path("corpora", "sunit_fields")
        cls.manifest = json.loads(
            cls.manifest_path.read_text()
        )
        cls.module = sunit_module()

    def _run_silex_with_stdout(
        self, stdout: str, **process_changes: object
    ) -> tuple[dict[str, Any], Any]:
        process = {
            "available": True,
            "success": True,
            "timeout": False,
            "process_wall_ms": 1.0,
            "stdout": stdout,
            "stderr": "",
            "executable_sha256": "a" * 64,
        }
        process.update(process_changes)
        args = SimpleNamespace(
            build_dir=self.root / "build",
            silex_root=self.root,
            timeout=1.0,
        )
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        with mock.patch.object(
            self.module, "run_process", return_value=process
        ), mock.patch.object(
            self.module, "normalize_silex", return_value={"success": True}
        ) as normalize:
            result = self.module.run_silex(args, row)
        return result, normalize

    def test_third_party_notice_sregulator_anchor_covers_calculation(self) -> None:
        notice = (self.root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        anchor = re.search(
            r"`(src/silex_bench/sunit_backend\.py):(\d+)-(\d+)` translates",
            notice,
        )
        self.assertIsNotNone(anchor)
        if anchor is None:
            self.fail("missing S-regulator source anchor")
        source_path = self.root / anchor.group(1)
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        first_line = int(anchor.group(2))
        last_line = int(anchor.group(3))
        self.assertGreaterEqual(first_line, 1)
        self.assertGreaterEqual(last_line, first_line)
        self.assertLessEqual(last_line, len(source_lines))
        anchored_source = "\n".join(source_lines[first_line - 1 : last_line])
        for required_source in (
            "ordinary_regulator = regulator",
            "sregulator = Float64(ordinary_regulator)",
            "sregulator *= Float64(order(Q))",
            "sregulator *= log(Float64(norm(P)))",
        ):
            with self.subTest(required_source=required_source):
                self.assertIn(required_source, anchored_source)

    def test_corpus_has_exact_schema_and_six_unique_rows(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 2)
        rows = self.manifest["fields"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["id"] for row in rows}), 6)
        self.assertTrue(all("expected" in row for row in rows))

    def test_external_value_parser_rejects_duplicate_fields(self) -> None:
        self.assertEqual(
            self.module.parse_values("certified=0\ncertified=1\n"),
            {},
        )

    def test_load_field_rejects_invalid_hecke_certification(self) -> None:
        source_row = self.manifest["fields"][0]
        for invalid_status in ("verified", "unknown", "", None, 1):
            with self.subTest(invalid_status=invalid_status):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "sunit-fields.json"
                    path.write_text(
                        json.dumps(
                            sunit_manifest(
                                [
                                    {
                                        **source_row,
                                        "external_hecke_certification": invalid_status,
                                    }
                                ]
                            )
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError, "external_hecke_certification"
                    ):
                        self.module.load_field(path, source_row["id"])

    def test_load_field_rejects_incomplete_expectations(self) -> None:
        source_row = self.manifest["fields"][0]
        expectation_keys = tuple(source_row["expected"])
        invalid_expectations = [
            {},
            *(
                {
                    key: value
                    for key, value in source_row["expected"].items()
                    if key != missing
                }
                for missing in expectation_keys
            ),
        ]
        for expected in invalid_expectations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "sunit-fields.json"
                    path.write_text(
                        json.dumps(
                            sunit_manifest(
                                [{**source_row, "expected": expected}]
                            )
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "expected"):
                        self.module.load_field(path, source_row["id"])

    def test_load_field_rejects_mistyped_expectations(self) -> None:
        source_row = self.manifest["fields"][0]
        invalid_values = {
            "s_class_order": 1,
            "s_class_invariants": [1],
            "torsion_order": 2,
            "ordinary_free_rank": "1",
            "nonunit_rank": "0",
            "free_rank": "1",
            "valuation_lattice_index": 1,
        }
        for key, invalid_value in invalid_values.items():
            with self.subTest(key=key):
                row = {
                    **source_row,
                    "expected": {
                        **source_row["expected"],
                        key: invalid_value,
                    },
                }
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "sunit-fields.json"
                    path.write_text(
                        json.dumps(sunit_manifest([row])),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, f"expected\\.{key}"):
                        self.module.load_field(path, source_row["id"])

    def test_manifest_rejects_noncanonical_exact_integers_and_invalid_ranks(
        self,
    ) -> None:
        source_row = self.manifest["fields"][0]
        mutations = {
            **{
                f"{key}={value!r}": (key, value)
                for key in (
                    "s_class_order",
                    "torsion_order",
                    "valuation_lattice_index",
                )
                for value in ("01", "+1", "1.0", " 1")
            },
            "noncanonical invariant": ("s_class_invariants", ["02"]),
            "unit invariant": ("s_class_invariants", ["1"]),
            "unsorted invariants": ("s_class_invariants", ["3", "2"]),
            "negative ordinary rank": ("ordinary_free_rank", -1),
            "negative nonunit rank": ("nonunit_rank", -1),
            "negative free rank": ("free_rank", -1),
            "inconsistent free rank": ("free_rank", 2),
        }
        for case, (key, value) in mutations.items():
            with self.subTest(case=case):
                row = json.loads(json.dumps(source_row))
                row["expected"][key] = value
                with self.assertRaisesRegex(ValueError, "expected"):
                    self.module.validate_manifest(sunit_manifest([row]))

    def test_manifest_rejects_impossible_group_invariants_and_torsion(self) -> None:
        source_row = self.manifest["fields"][0]
        cases = (
            {"s_class_order": "6", "s_class_invariants": ["2", "2"]},
            {"s_class_order": "6", "s_class_invariants": ["2", "3"]},
            {"torsion_order": "1"},
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                row = json.loads(json.dumps(source_row))
                row["expected"].update(mutation)
                with self.assertRaisesRegex(ValueError, "expected"):
                    self.module.validate_manifest(
                        sunit_manifest([row])
                    )

    def test_manifest_requires_the_declared_prime_index_convention(self) -> None:
        source_row = self.manifest["fields"][0]
        for convention in (None, "one-based reverse HNF order", 1):
            with self.subTest(convention=convention), self.assertRaisesRegex(
                ValueError, "prime_index_convention"
            ):
                manifest = {"schema_version": 2, "fields": [source_row]}
                if convention is not None:
                    manifest["prime_index_convention"] = convention
                self.module.validate_manifest(manifest)

    def test_manifest_requires_authoritative_prime_ideal_witnesses(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        row = next(
            item
            for item in manifest["fields"]
            if item["id"] == "real_quadratic_5_split_11"
        )

        row["prime_ideal_witnesses"] = []
        with self.assertRaisesRegex(ValueError, "prime_ideal_witnesses"):
            self.module.validate_manifest(manifest)

        row["prime_ideal_witnesses"] = [
            {
                "p": 11,
                "canonical_index": 0,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["-4", "1"],
            },
            {
                "p": 11,
                "canonical_index": 1,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["not-a-rational", "1"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "prime_ideal_witnesses"):
            self.module.validate_manifest(manifest)

    def test_manifest_requires_monic_polynomial_and_prime_selectors(self) -> None:
        source_row = self.manifest["fields"][2]
        mutations = {
            "nonmonic polynomial": lambda row: row.__setitem__(
                "coefficients_low_to_high", [23, 0, 2]
            ),
            "composite rational prime": lambda row: row["selected_primes"][
                0
            ].__setitem__("p", 4),
            "strong pseudoprime rational prime": lambda row: row[
                "selected_primes"
            ][0].__setitem__("p", 341550071728321),
            "rational prime outside bounded domain": lambda row: row[
                "selected_primes"
            ][0].__setitem__("p", 1 << 64),
            "rational prime outside native signed-word domain": lambda row: row[
                "selected_primes"
            ][0].__setitem__("p", 18446744073709551557),
        }
        for case, mutate in mutations.items():
            with self.subTest(case=case):
                row = json.loads(json.dumps(source_row))
                mutate(row)
                with self.assertRaises(ValueError):
                    self.module.validate_manifest(sunit_manifest([row]))

    def test_load_field_rejects_invalid_manifest_structure(self) -> None:
        source_row = self.manifest["fields"][0]
        invalid_manifests = (
            (
                "schema_version",
                sunit_manifest([source_row], schema_version=1),
            ),
            (
                "schema_version",
                sunit_manifest([source_row], schema_version=True),
            ),
            (
                "fields",
                sunit_manifest([42]),
            ),
            (
                "coefficients_low_to_high",
                sunit_manifest(
                    [
                        {
                            key: value
                            for key, value in source_row.items()
                            if key != "coefficients_low_to_high"
                        }
                    ]
                ),
            ),
            (
                "selected_primes",
                sunit_manifest(
                    [
                        {
                            key: value
                            for key, value in source_row.items()
                            if key != "selected_primes"
                        }
                    ]
                ),
            ),
            (
                "selected_primes",
                sunit_manifest(
                    [
                        {
                            **source_row,
                            "selected_primes": [{"p": "2", "index": 0}],
                        }
                    ]
                ),
            ),
            (
                "duplicate",
                sunit_manifest([source_row, source_row]),
            ),
        )
        for error_pattern, manifest in invalid_manifests:
            with self.subTest(error_pattern=error_pattern):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "sunit-fields.json"
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        self.module.load_field(path, source_row["id"])

    def test_runner_owns_fixture_loading_and_normalized_agreement(self) -> None:
        path = self.manifest_path
        row = self.module.load_field(path, "cubic_x3_minus_2_empty_s")
        self.assertEqual(row["selected_primes"], [])
        normalized = normalized_result()
        agreement = self.module.build_agreement({"silex": normalized}, row)
        self.assertTrue(agreement["success"])

    def test_silex_normalization_rejects_mistyped_success(self) -> None:
        normalized = self.module.normalize_silex(
            {
                "available": True,
                "success": True,
                "timeout": False,
                "process_wall_ms": 1.0,
                "stderr": "",
            },
            {"success": "false", "sunit": {"selected_primes": []}},
            self.root / "build",
            self.root,
        )

        self.assertFalse(normalized["success"])

    def test_silex_normalization_rejects_other_mistyped_booleans(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )

        def mutate(
            payload: dict[str, object], path: tuple[str, ...], value: object
        ) -> None:
            target = payload
            for key in path[:-1]:
                child = target[key]
                self.assertIsInstance(child, dict)
                target = cast(dict[str, object], child)
            target[path[-1]] = value

        mutations = {
            "payload timeout true": (("timeout",), True),
            "payload timeout string": (("timeout",), "false"),
            "process timeout string": (("process", "timeout"), "false"),
            "process availability string": (("process", "available"), "false"),
            "mixed membership string": (
                ("sunit", "membership", "mixed_round_trip_verified"),
                "false",
            ),
            "outside membership integer": (
                ("sunit", "membership", "outside_support_rejected"),
                1,
            ),
            "publication integer": (("sunit", "final_result_published"), 1),
        }
        for label, (path, value) in mutations.items():
            with self.subTest(label=label):
                process = native_process()
                payload = native_payload(row["expected"])
                if path[0] == "process":
                    mutate(process, path[1:], value)
                else:
                    mutate(payload, path, value)
                normalized = self.module.normalize_silex(
                    process, payload, self.root / "build", self.root, row
                )
                self.assertFalse(normalized["success"])
                self.assertEqual(
                    normalized["failure_reason"], "invalid_silex_payload_schema"
                )
                self.assertFalse(
                    self.module.build_agreement({"silex": normalized}, row)["success"]
                )

    def test_silex_normalization_requires_exact_timing_map_shapes(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        cases = (
            ("phase_timing_ms",),
            ("component_timing_ms",),
            ("sunit", "timing_ms"),
        )
        for path in cases:
            with self.subTest(path=path):
                payload = native_payload(row["expected"])
                target = payload
                for key in path[:-1]:
                    target = cast(dict[str, object], target[key])
                target[path[-1]] = []
                normalized = self.module.normalize_silex(
                    native_process(), payload, self.root / "build", self.root, row
                )

                self.assertFalse(normalized["success"])
                self.assertEqual(
                    normalized["failure_reason"], "invalid_silex_payload_schema"
                )

    def test_silex_normalization_replaces_malformed_failure_reason(self) -> None:
        for location in ("payload", "sunit"):
            with self.subTest(location=location):
                payload = native_payload()
                payload["success"] = False
                target = (
                    payload
                    if location == "payload"
                    else cast(dict[str, object], payload["sunit"])
                )
                target["failure_reason"] = []

                normalized = self.module.normalize_silex(
                    native_process(), payload, self.root / "build", self.root
                )

                self.assertFalse(normalized["success"])
                self.assertIsInstance(normalized["failure_reason"], str)
                self.assertTrue(normalized["failure_reason"])

    def test_silex_normalization_binds_ordinary_publication_state(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        for value in (False, 0, "false"):
            with self.subTest(value=value):
                payload = native_payload(row["expected"])
                payload["final_result_published"] = value
                normalized = self.module.normalize_silex(
                    native_process(), payload, self.root / "build", self.root, row
                )
                self.assertFalse(normalized["success"])
                self.assertEqual(
                    normalized["failure_reason"], "invalid_silex_payload_schema"
                )

    def test_silex_normalization_validates_membership_evidence(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        mutations = {
            "round-trip count": ("verified_round_trip_count", 0),
            "mixed outcome": ("mixed_outcome", "not_sunit"),
            "outside outcome": ("outside_outcome", "verified"),
        }
        for label, (key, value) in mutations.items():
            with self.subTest(label=label):
                payload = native_payload(row["expected"])
                membership = cast(
                    dict[str, object],
                    cast(dict[str, object], payload["sunit"])["membership"],
                )
                membership[key] = value
                normalized = self.module.normalize_silex(
                    native_process(), payload, self.root / "build", self.root, row
                )
                self.assertFalse(normalized["success"])

    def test_silex_normalization_rejects_oversized_hnf_integer_without_crash(
        self,
    ) -> None:
        row = json.loads(
            json.dumps(
                self.module.load_field(
                    self.manifest_path,
                    "real_quadratic_5_split_11",
                )
            )
        )
        row["selected_primes"] = [{"p": 11, "index": 0}]
        row["expected"]["nonunit_rank"] = 1
        row["expected"]["free_rank"] = 2
        oversized = {
            "p": "11",
            "canonical_index": 0,
            "e": 1,
            "f": 1,
            "hnf": [["1", "9" * 5000], ["0", "11"]],
        }
        normalized = self.module.normalize_silex(
            native_process(),
            native_payload(
                row["expected"],
                selected_primes=[oversized],
                canonical_decomposition=[oversized],
            ),
            self.root / "build",
            self.root,
            row,
        )
        self.assertFalse(normalized["success"])
        self.assertEqual(
            normalized["failure_reason"], "invalid_silex_payload_schema"
        )

    def test_silex_normalization_requires_exact_nested_sunit_success(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        for invalid_success in ("false", 1):
            with self.subTest(invalid_success=invalid_success):
                payload = native_payload()
                sunit = payload["sunit"]
                self.assertIsInstance(sunit, dict)
                sunit["success"] = invalid_success  # type: ignore[index]

                normalized = self.module.normalize_silex(
                    native_process(), payload, self.root / "build", self.root, row
                )

                self.assertFalse(normalized["success"])
                self.assertEqual(
                    normalized["failure_reason"], "invalid_silex_payload_schema"
                )
                self.assertFalse(
                    self.module.build_agreement({"silex": normalized}, row)[
                        "success"
                    ]
                )

    def test_silex_normalization_rejects_malformed_container_shapes(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )

        def replace_sunit(key: str, value: object) -> object:
            payload = native_payload()
            sunit = payload["sunit"]
            self.assertIsInstance(sunit, dict)
            sunit[key] = value  # type: ignore[index]
            return payload

        malformed_payloads = {
            "array payload": [],
            "scalar payload": 1,
            "null payload": None,
            "string payload": "invalid",
            "array sunit": {"success": True, "sunit": []},
            "object selected primes": replace_sunit("selected_primes", {}),
            "non-object descriptor": replace_sunit("selected_primes", [[]]),
            "array class group": replace_sunit("s_class_group", []),
            "array unit group": replace_sunit("s_unit_group", []),
            "array membership": replace_sunit("membership", []),
            "oversized descriptor prime": replace_sunit(
                "selected_primes",
                [
                    {
                        "p": "9" * 5000,
                        "e": 1,
                        "f": 1,
                        "hnf": [["1"]],
                    }
                ],
            ),
        }
        for case, payload in malformed_payloads.items():
            with self.subTest(case=case):
                normalized = self.module.normalize_silex(
                    native_process(), payload, self.root / "build", self.root
                )
                self.assertFalse(normalized["success"])
                self.assertEqual(
                    normalized["failure_reason"], "invalid_silex_payload_schema"
                )
                self.assertFalse(
                    self.module.build_agreement({"silex": normalized}, row)[
                        "success"
                    ]
                )

    def test_silex_normalization_rejects_boolean_descriptor_integers(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        descriptors = [
            {
                "p": "11",
                "e": 1,
                "f": 1,
                "hnf": [["1", "0"], ["0", "11"]],
            },
            {
                "p": "11",
                "e": 1,
                "f": 1,
                "hnf": [["11", "0"], ["0", "1"]],
            },
        ]
        for key in ("e", "f"):
            with self.subTest(key=key):
                malformed = json.loads(json.dumps(descriptors))
                malformed[0][key] = True
                normalized = self.module.normalize_silex(
                    native_process(),
                    native_payload(row["expected"], selected_primes=malformed),
                    self.root / "build",
                    self.root,
                )
                self.assertFalse(normalized["success"])
                self.assertFalse(
                    self.module.build_agreement({"silex": normalized}, row)[
                        "success"
                    ]
                )

    def test_silex_normalization_requires_native_canonical_prime_index(self) -> None:
        row = json.loads(
            json.dumps(
                self.module.load_field(
                    self.manifest_path,
                    "real_quadratic_5_split_11",
                )
            )
        )
        row["selected_primes"] = [{"p": 11, "index": 0}]
        row["expected"]["nonunit_rank"] = 1
        row["expected"]["free_rank"] = 2
        descriptor_without_index = {
            "p": "11",
            "e": 1,
            "f": 1,
            "hnf": [["1", "8"], ["0", "11"]],
        }

        normalized = self.module.normalize_silex(
            native_process(),
            native_payload(
                row["expected"], selected_primes=[descriptor_without_index]
            ),
            self.root / "build",
            self.root,
            row,
        )

        self.assertFalse(normalized["success"])
        self.assertEqual(
            normalized["failure_reason"], "invalid_silex_payload_schema"
        )

    def test_silex_normalization_preserves_canonical_prime_index_order(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        descriptors = [
            {
                "p": "11",
                "canonical_index": 0,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["-4", "1"],
                "hnf": [["1", "8"], ["0", "11"]],
            },
            {
                "p": "11",
                "canonical_index": 1,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["3", "1"],
                "hnf": [["1", "4"], ["0", "11"]],
            },
        ]
        normalized = self.module.normalize_silex(
            native_process(),
            native_payload(
                row["expected"],
                selected_primes=descriptors,
                maximal_order_discriminant="5",
                maximal_order_basis_power=[["1", "0"], ["0", "1"]],
            ),
            self.root / "build",
            self.root,
            row,
        )
        self.assertTrue(normalized["success"])
        self.assertEqual(
            [item["canonical_index"] for item in normalized["selected_primes"]],
            [0, 1],
        )
        for label, selected, complete in (
            ("selected order", list(reversed(descriptors)), descriptors),
            ("complete order", descriptors, list(reversed(descriptors))),
            (
                "swapped HNFs",
                [
                    {**descriptors[0], "hnf": descriptors[1]["hnf"]},
                    {**descriptors[1], "hnf": descriptors[0]["hnf"]},
                ],
                descriptors,
            ),
        ):
            with self.subTest(label=label):
                rejected = self.module.normalize_silex(
                    native_process(),
                    native_payload(
                        row["expected"],
                        selected_primes=selected,
                        canonical_decomposition=complete,
                        maximal_order_discriminant="5",
                        maximal_order_basis_power=[["1", "0"], ["0", "1"]],
                    ),
                    self.root / "build",
                    self.root,
                    row,
                )
                self.assertFalse(rejected["success"])
                self.assertEqual(
                    rejected["failure_reason"], "invalid_silex_payload_schema"
                )

    def test_external_descriptor_parser_rejects_malformed_integer(self) -> None:
        values = {
            "selected_prime_count": "1",
            "prime_1_p": "11",
            "prime_1_canonical_index": "0",
            "prime_1_e": "not-an-integer",
            "prime_1_f": "1",
            "prime_1_hnf_rows": "1",
            "prime_1_hnf_cols": "1",
            "prime_1_hnf_1_1": "11",
        }
        self.assertEqual(
            self.module.parse_external_descriptors(values, "pari_bnf_integral_basis_rows"),
            [],
        )
        for key in ("prime_1_e", "prime_1_f"):
            for invalid_value in ("+1", "01", " 1"):
                with self.subTest(key=key, invalid_value=invalid_value):
                    malformed = dict(values)
                    malformed["prime_1_e"] = "1"
                    malformed[key] = invalid_value
                    self.assertEqual(
                        self.module.parse_external_descriptors(
                            malformed, "pari_bnf_integral_basis_rows"
                        ),
                        [],
                    )

    def test_external_descriptor_parser_requires_canonical_prime_index(self) -> None:
        values = {
            "selected_prime_count": "1",
            "prime_1_p": "11",
            "prime_1_e": "1",
            "prime_1_f": "1",
            "prime_1_beta_count": "2",
            "prime_1_beta_1": "-4",
            "prime_1_beta_2": "1",
            "prime_1_hnf_rows": "2",
            "prime_1_hnf_cols": "2",
            "prime_1_hnf_1_1": "1",
            "prime_1_hnf_1_2": "4",
            "prime_1_hnf_2_1": "0",
            "prime_1_hnf_2_2": "11",
        }
        self.assertEqual(
            self.module.parse_external_descriptors(
                values, "pari_bnf_integral_basis_rows"
            ),
            [],
        )
        for invalid_index in ("not-an-integer", "+0", "00", " 0"):
            with self.subTest(invalid_index=invalid_index):
                values["prime_1_canonical_index"] = invalid_index
                self.assertEqual(
                    self.module.parse_external_descriptors(
                        values, "pari_bnf_integral_basis_rows"
                    ),
                    [],
                )
        values["prime_1_canonical_index"] = "0"
        self.assertEqual(
            self.module.parse_external_descriptors(
                values, "pari_bnf_integral_basis_rows"
            ),
            [
                {
                    "p": "11",
                    "canonical_index": 0,
                    "e": 1,
                    "f": 1,
                    "beta_power_basis": ["-4", "1"],
                    "hnf": [["1", "4"], ["0", "11"]],
                    "hnf_basis": "pari_bnf_integral_basis_rows",
                }
            ],
        )

    def test_external_descriptor_parser_supports_complete_decomposition_prefix(
        self,
    ) -> None:
        values = {
            "canonical_prime_count": "1",
            "canonical_prime_1_p": "11",
            "canonical_prime_1_canonical_index": "0",
            "canonical_prime_1_e": "1",
            "canonical_prime_1_f": "2",
            "canonical_prime_1_beta_count": "2",
            "canonical_prime_1_beta_1": "0",
            "canonical_prime_1_beta_2": "1",
            "canonical_prime_1_hnf_rows": "2",
            "canonical_prime_1_hnf_cols": "2",
            "canonical_prime_1_hnf_1_1": "1",
            "canonical_prime_1_hnf_1_2": "0",
            "canonical_prime_1_hnf_2_1": "0",
            "canonical_prime_1_hnf_2_2": "121",
        }
        self.assertEqual(
            self.module.parse_external_descriptors(
                values,
                "pari_bnf_integral_basis_rows",
                count_key="canonical_prime_count",
                prefix="canonical_prime",
            ),
            [
                {
                    "p": "11",
                    "canonical_index": 0,
                    "e": 1,
                    "f": 2,
                    "beta_power_basis": ["0", "1"],
                    "hnf": [["1", "0"], ["0", "121"]],
                    "hnf_basis": "pari_bnf_integral_basis_rows",
                }
            ],
        )

    def test_agreement_rejects_nonobject_backend_result(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        agreement = self.module.build_agreement({"silex": []}, row)
        self.assertFalse(agreement["backend_identities_valid"])
        self.assertFalse(agreement["success"])

    def test_agreement_requires_complete_ordinary_proof_states(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        proof_fields = (
            "certification_status",
            "class_group_proof_status",
            "unit_group_proof_status",
            "regulator_proof_status",
        )
        for key in proof_fields:
            with self.subTest(missing=key):
                normalized = normalized_result()
                del normalized[key]
                agreement = self.module.build_agreement({"silex": normalized}, row)
                self.assertFalse(agreement["proof_complete"])
                self.assertFalse(agreement["success"])

    def test_agreement_rejects_missing_or_mistyped_exact_results(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        invalid_values = {
            "s_class_order": 1,
            "s_class_invariants": [1],
            "torsion_order": 2,
            "ordinary_free_rank": "1",
            "nonunit_rank": "0",
            "free_rank": "1",
            "valuation_lattice_index": 1,
        }
        for key, invalid_value in invalid_values.items():
            for case in ("missing", "mistyped"):
                with self.subTest(key=key, case=case):
                    normalized = normalized_result()
                    if case == "missing":
                        del normalized[key]
                    else:
                        normalized[key] = invalid_value
                    agreement = self.module.build_agreement(
                        {"silex": normalized}, row
                    )
                    self.assertFalse(agreement["fields"][key])
                    self.assertFalse(agreement["backend_results_agree"])
                    self.assertFalse(agreement["success"])

    def test_agreement_rejects_noncanonical_exact_results_and_invalid_ranks(
        self,
    ) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        mutations = {
            "noncanonical class order": ("s_class_order", "01"),
            "noncanonical torsion order": ("torsion_order", "+2"),
            "noncanonical lattice index": ("valuation_lattice_index", " 1"),
            "noncanonical invariant": ("s_class_invariants", ["02"]),
            "negative ordinary rank": ("ordinary_free_rank", -1),
            "negative nonunit rank": ("nonunit_rank", -1),
            "negative free rank": ("free_rank", -1),
            "inconsistent free rank": ("free_rank", 2),
        }
        for case, (key, value) in mutations.items():
            with self.subTest(case=case):
                malformed_row = json.loads(json.dumps(row))
                malformed_row["expected"][key] = value
                normalized = normalized_result()
                normalized[key] = value
                agreement = self.module.build_agreement(
                    {"silex": normalized}, malformed_row
                )
                self.assertFalse(agreement["success"])

    def test_agreement_accepts_large_canonical_exact_integer_text(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        large_order = "9" * 5000
        row["expected"]["s_class_order"] = large_order
        row["expected"]["s_class_invariants"] = [large_order]
        normalized = normalized_result()
        normalized["s_class_order"] = large_order
        normalized["s_class_invariants"] = [large_order]

        agreement = self.module.build_agreement({"silex": normalized}, row)

        self.assertTrue(agreement["fields"]["s_class_order"])
        self.assertTrue(agreement["manifest_contract_valid"])
        self.assertTrue(agreement["success"])

    def test_external_regulator_requires_the_ordinary_proof_state(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        normalized = normalized_result("pari")
        normalized["regulator_proof_status"] = "verified"
        agreement = self.module.build_agreement({"pari": normalized}, row)
        self.assertFalse(agreement["proof_complete"])
        self.assertFalse(agreement["success"])

    def test_agreement_rejects_nonfinite_regulator_midpoints(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                normalized = normalized_result()
                normalized["regulator_midpoint"] = value
                agreement = self.module.build_agreement({"silex": normalized}, row)
                self.assertFalse(agreement["regulator_agrees"])
                self.assertFalse(agreement["backend_results_agree"])
                self.assertFalse(agreement["success"])

    def test_agreement_binds_backend_slot_to_engine_identity(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        mislabeled = normalized_result("pari")
        agreement = self.module.build_agreement(
            {"hecke:external": mislabeled}, row
        )
        self.assertFalse(agreement["success"])
        self.assertFalse(agreement["backend_identities_valid"])

    def test_agreement_requires_silex_native_executable_identity(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        attacks = {
            "missing": None,
            "malformed digest": {
                "executable": "/tmp/silex-class-unit-instance",
                "executable_sha256": "not-a-digest",
                "version": None,
                "source": "/tmp/silex",
            },
            "extra field": {
                "executable": "/tmp/silex-class-unit-instance",
                "executable_sha256": "d" * 64,
                "version": None,
                "source": "/tmp/silex",
                "unbound": True,
            },
        }
        for attack, identity in attacks.items():
            with self.subTest(attack=attack):
                result = normalized_result()
                if identity is None:
                    result.pop("engine_identity")
                else:
                    result["engine_identity"] = identity
                agreement = self.module.build_agreement({"silex": result}, row)
                self.assertFalse(agreement["backend_identities_valid"])
                self.assertFalse(agreement["success"])

    def test_agreement_requires_manifest_selected_prime_descriptors(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )

        def descriptor(index: int, hnf: list[list[str]]) -> dict[str, object]:
            beta = (["-4", "1"], ["3", "1"])[index]
            return {
                "p": "11",
                "canonical_index": index,
                "e": 1,
                "f": 1,
                "beta_power_basis": beta,
                "hnf": hnf,
                "hnf_basis": "silex_maximal_order_basis_rows",
            }

        complete = [
            descriptor(0, [["1", "8"], ["0", "11"]]),
            descriptor(1, [["1", "4"], ["0", "11"]]),
        ]

        def result(
            selected: list[dict[str, object]],
            decomposition: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
            normalized = normalized_result()
            normalized.update(row["expected"])
            normalized["valuation_matrix"] = valuation_matrix_for(row["expected"])
            normalized["maximal_order_discriminant"] = "5"
            normalized["maximal_order_basis_power"] = [["1", "0"], ["0", "1"]]
            normalized["selected_primes"] = selected
            if decomposition is not None:
                normalized["canonical_prime_decompositions"] = decomposition
            return normalized

        invalid_cases = {
            "missing selected": result([], complete),
            "missing decomposition": result(complete),
            "incomplete selected": result(complete[:1], complete),
            "reversed decomposition": result(complete, list(reversed(complete))),
            "swapped selected HNFs": result(
                [
                    {**complete[0], "hnf": complete[1]["hnf"]},
                    {**complete[1], "hnf": complete[0]["hnf"]},
                ],
                complete,
            ),
            "duplicate decomposition HNF": result(
                complete, [complete[0], {**complete[1], "hnf": complete[0]["hnf"]}]
            ),
            "wrong prime": result(
                [{**item, "p": "13"} for item in complete], complete
            ),
            "malformed HNF": result(
                [{**complete[0], "hnf": [["bad", "4"], ["0", "11"]]}, complete[1]],
                complete,
            ),
        }
        for label, normalized in invalid_cases.items():
            with self.subTest(label=label):
                observed = self.module.build_agreement({"silex": normalized}, row)
                self.assertFalse(observed["fields"]["selected_prime_specification"])
                self.assertFalse(observed["success"])

        valid = self.module.build_agreement(
            {"silex": result(complete, complete)}, row
        )
        self.assertTrue(valid["fields"]["selected_prime_specification"])
        self.assertTrue(valid["success"])

    def test_hnf_orientation_uses_the_correct_reduction_pivot(self) -> None:
        self.assertTrue(
            self.module.matrix_is_row_or_transposed_row_hnf([[1, 0], [4, 5]])
        )
        self.assertFalse(
            self.module.matrix_is_row_or_transposed_row_hnf([[5, 0], [4, 1]])
        )

    def test_agreement_validates_valuation_lattice_evidence(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        complete = [
            {
                "p": "11",
                "canonical_index": 0,
                "e": 1,
                "f": 1,
                "hnf": [["1", "4"], ["0", "11"]],
                "hnf_basis": "silex_maximal_order_basis_rows",
            },
            {
                "p": "11",
                "canonical_index": 1,
                "e": 1,
                "f": 1,
                "hnf": [["1", "8"], ["0", "11"]],
                "hnf_basis": "silex_maximal_order_basis_rows",
            },
        ]
        normalized = normalized_result()
        normalized.update(row["expected"])
        normalized["selected_primes"] = complete
        normalized["canonical_prime_decompositions"] = complete
        normalized["valuation_matrix"] = [["0", "0"], ["0", "0"]]

        agreement = self.module.build_agreement({"silex": normalized}, row)

        self.assertFalse(agreement["fields"]["valuation_lattice_evidence"])
        self.assertFalse(agreement["success"])

        normalized["valuation_matrix"] = [["1", "-0"], ["0", "1"]]
        agreement = self.module.build_agreement({"silex": normalized}, row)

        self.assertFalse(agreement["fields"]["valuation_lattice_evidence"])
        self.assertFalse(agreement["success"])

    def test_external_generators_decompose_each_rational_prime_once(self) -> None:
        row = json.loads(
            json.dumps(
                self.module.load_field(
                    self.manifest_path,
                    "real_quadratic_5_split_11",
                )
            )
        )
        row["selected_primes"] = [
            {"p": 11, "index": 0},
            {"p": 11, "index": 1},
        ]

        gp = self.module.gp_selection_code(row)
        julia = self.module.julia_selection_code(row)

        self.assertEqual(gp.count("idealprimedec"), 1)
        self.assertEqual(julia.count("prime_decomposition(O, 11)"), 1)

    def test_manifest_selector_order_is_explicit_for_all_adapters(self) -> None:
        row = json.loads(
            json.dumps(
                self.module.load_field(
                    self.manifest_path,
                    "real_quadratic_5_split_11",
                )
            )
        )
        row["selected_primes"] = [
            {"p": 11, "index": 1},
            {"p": 11, "index": 0},
        ]

        selection = self.module.manifest_witness_selection(row)

        self.assertEqual(
            [
                (witness["canonical_index"], selection_index)
                for witness, selection_index in selection
            ],
            [(0, 1), (1, 0)],
        )
        gp_selected = [
            line
            for line in self.module.gp_selection_code(row).splitlines()
            if line.startswith("S = concat")
        ]
        julia_selected = [
            line
            for line in self.module.julia_selection_code(row).splitlines()
            if line.startswith("push!(S,")
        ]
        self.assertIn("D_1_match_2", gp_selected[0])
        self.assertIn("D_1_match_1", gp_selected[1])
        self.assertIn("D_1_match_2", julia_selected[0])
        self.assertIn("D_1_match_1", julia_selected[1])

    def test_pari_matches_manifest_witness_before_serialization(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        programs: list[str] = []
        original_run_process = self.module.run_process
        original_which = self.module.shutil.which
        original_source_version = self.module.pari_source_version

        def capture_process(
            command: list[str], *, timeout: float, stdin: str | None = None
        ) -> dict[str, object]:
            del timeout
            if "--version-short" in command:
                return {
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                    "stderr": "",
                    "process_wall_ms": 0.0,
                }
            programs.append(stdin or "")
            return {
                "available": True,
                "success": False,
                "timeout": False,
                "stdout": "",
                "stderr": "",
                "process_wall_ms": 0.0,
            }

        try:
            setattr(self.module, "run_process", capture_process)
            setattr(self.module.shutil, "which", lambda _value: "/tmp/gp")
            setattr(self.module, "pari_source_version", lambda _path: "2.17.3")
            self.module.run_pari(
                SimpleNamespace(
                    gp="/tmp/gp",
                    pari_source=Path("/tmp/pari-2.17.3"),
                    pari_version="2.17.3",
                    timeout=1.0,
                ),
                row,
            )
        finally:
            setattr(self.module, "run_process", original_run_process)
            setattr(self.module.shutil, "which", original_which)
            setattr(self.module, "pari_source_version", original_source_version)

        self.assertEqual(len(programs), 1)
        program = programs[0]
        self.assertIn(
            "idealhnf(b, 11, D_1_beta_1)",
            program,
        )
        self.assertIn(
            "manifest prime witness did not match exactly one PARI prime ideal",
            program,
        )
        self.assertIn("H = mattranspose(idealhnf(b, S[i]))", program)
        self.assertIn("H = mattranspose(idealhnf(b, S_all[i]))", program)
        self.assertNotIn("vecsort", program)

    def test_hecke_outside_support_uses_actual_sunit_membership_query(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        commands: list[list[str]] = []
        original_run_process = self.module.run_process

        def capture_process(
            command: list[str],
            *,
            timeout: float,
            stdin: str | None = None,
            env: dict[str, str] | None = None,
        ) -> dict[str, object]:
            del timeout, stdin, env
            commands.append(command)
            return {
                "available": True,
                "success": False,
                "timeout": False,
                "stdout": "",
                "stderr": "",
                "process_wall_ms": 0.0,
            }

        try:
            setattr(self.module, "run_process", capture_process)
            self.module.run_hecke(
                SimpleNamespace(
                    julia="/usr/bin/julia",
                    hecke_project=None,
                    timeout=1.0,
                ),
                row,
            )
        finally:
            setattr(self.module, "run_process", original_run_process)

        self.assertEqual(len(commands), 1)
        program = commands[0][-1]
        self.assertIn("outside_support_present = any(", program)
        self.assertIn("preimage(mG, outside_element)", program)
        self.assertIn("image(mG, outside_coordinates) == outside_element", program)
        self.assertIn(
            "P == ideal(O, ZZ(11), O(",
            program,
        )
        self.assertIn("H = ideal_basis_rows(S[i])", program)
        self.assertIn("H = ideal_basis_rows(S_all[i])", program)
        self.assertIn("class_group(O; GRH = false)", program)
        self.assertIn("unit_group_fac_elem(O; GRH = false)", program)
        self.assertIn("sunit_group_fac_elem(S; GRH = false)", program)
        self.assertNotIn("sort!", program)

    def test_agreement_binds_explicit_selector_to_canonical_prime_index(self) -> None:
        row = json.loads(
            json.dumps(
                self.module.load_field(
                    self.manifest_path,
                    "real_quadratic_5_split_11",
                )
            )
        )
        row["selected_primes"] = [{"p": 11, "index": 0}]
        row["expected"]["nonunit_rank"] = 1
        row["expected"]["free_rank"] = 2
        complete = [
            {
                "p": "11",
                "canonical_index": 0,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["-4", "1"],
                "hnf": [["1", "8"], ["0", "11"]],
                "hnf_basis": "silex_maximal_order_basis_rows",
            },
            {
                "p": "11",
                "canonical_index": 1,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["3", "1"],
                "hnf": [["1", "4"], ["0", "11"]],
                "hnf_basis": "silex_maximal_order_basis_rows",
            },
        ]

        def result_for(index: object, hnf_index: int) -> dict[str, object]:
            result = normalized_result()
            result.update(row["expected"])
            result["valuation_matrix"] = valuation_matrix_for(row["expected"])
            result["maximal_order_discriminant"] = "5"
            result["maximal_order_basis_power"] = [["1", "0"], ["0", "1"]]
            result["selected_primes"] = [
                {**complete[hnf_index], "canonical_index": index}
            ]
            result["canonical_prime_decompositions"] = complete
            return result

        for label, result in (
            ("wrong index", result_for(1, 1)),
            ("relabeled wrong HNF", result_for(0, 1)),
        ):
            with self.subTest(label=label):
                observed = self.module.build_agreement({"silex": result}, row)
                self.assertFalse(observed["fields"]["selected_prime_specification"])
                self.assertFalse(observed["success"])

        for invalid_index in (None, True, -1, 2, "0"):
            with self.subTest(invalid_index=invalid_index):
                observed = self.module.build_agreement(
                    {"silex": result_for(invalid_index, 0)}, row
                )
                self.assertFalse(observed["fields"]["selected_prime_specification"])
                self.assertFalse(observed["success"])

        matched = self.module.build_agreement(
            {"silex": result_for(0, 0)}, row
        )
        self.assertTrue(matched["fields"]["selected_prime_specification"])
        self.assertTrue(matched["success"])

    def test_agreement_rejects_correct_shape_nonideal_hnf_forgery(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        descriptors = [
            {
                "p": "11",
                "canonical_index": 0,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["-4", "1"],
                "hnf": [["1", "0"], ["0", "11"]],
                "hnf_basis": "silex_maximal_order_basis_rows",
            },
            {
                "p": "11",
                "canonical_index": 1,
                "e": 1,
                "f": 1,
                "beta_power_basis": ["3", "1"],
                "hnf": [["1", "8"], ["0", "11"]],
                "hnf_basis": "silex_maximal_order_basis_rows",
            },
        ]
        result = normalized_result()
        result.update(row["expected"])
        result["valuation_matrix"] = valuation_matrix_for(row["expected"])
        result["maximal_order_basis_power"] = [["1", "0"], ["0", "1"]]
        result["selected_primes"] = descriptors
        result["canonical_prime_decompositions"] = descriptors

        agreement = self.module.build_agreement({"silex": result}, row)

        self.assertFalse(agreement["fields"]["prime_ideal_witnesses"])
        self.assertFalse(agreement["success"])

    def test_agreement_rejects_cross_engine_witness_index_relabeling(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )

        def descriptors(engine: str, swapped: bool) -> list[dict[str, object]]:
            betas = (
                (["3", "1"], ["-4", "1"])
                if swapped
                else (["-4", "1"], ["3", "1"])
            )
            basis = {
                "silex": "silex_maximal_order_basis_rows",
                "pari": "pari_bnf_integral_basis_rows",
            }[engine]
            return [
                {
                    "p": "11",
                    "canonical_index": index,
                    "e": 1,
                    "f": 1,
                    "beta_power_basis": betas[index],
                    "hnf": [["1", str(4 + 4 * index)], ["0", "11"]],
                    "hnf_basis": basis,
                }
                for index in range(2)
            ]

        results: dict[str, dict[str, object]] = {}
        for engine, swapped in (("silex", False), ("pari", True)):
            result = normalized_result(engine)
            result.update(row["expected"])
            result["valuation_matrix"] = valuation_matrix_for(row["expected"])
            result["maximal_order_basis_power"] = [["1", "0"], ["0", "1"]]
            result["selected_primes"] = descriptors(engine, swapped)
            result["canonical_prime_decompositions"] = descriptors(engine, swapped)
            results[engine] = result

        agreement = self.module.build_agreement(results, row)

        self.assertFalse(agreement["fields"]["prime_ideal_witnesses"])
        self.assertFalse(agreement["success"])

    def test_agreement_compares_complete_prime_decomposition_rational_data(self) -> None:
        row = json.loads(
            json.dumps(
                self.module.load_field(
                    self.manifest_path,
                    "cubic_x3_minus_2_empty_s",
                )
            )
        )
        row["selected_primes"] = [{"p": 2, "index": 0}]
        row["expected"]["nonunit_rank"] = 1
        row["expected"]["free_rank"] = (
            row["expected"]["ordinary_free_rank"] + 1
        )
        hnfs = (
            [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "2"]],
            [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "1"]],
            [["2", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
        )

        def descriptor(
            engine: str,
            index: int,
            e: int,
        ) -> dict[str, object]:
            basis = {
                "silex": "silex_maximal_order_basis_rows",
                "pari": "pari_bnf_integral_basis_rows",
            }[engine]
            return {
                "p": "2",
                "canonical_index": index,
                "e": e,
                "f": 1,
                "hnf": hnfs[index],
                "hnf_basis": basis,
            }

        silex = normalized_result("silex")
        silex.update(row["expected"])
        silex["selected_primes"] = [descriptor("silex", 0, 1)]
        silex["canonical_prime_decompositions"] = [
            descriptor("silex", index, 1) for index in range(3)
        ]
        pari = normalized_result("pari")
        pari.update(row["expected"])
        pari["selected_primes"] = [descriptor("pari", 0, 1)]
        pari["canonical_prime_decompositions"] = [
            descriptor("pari", 0, 1),
            descriptor("pari", 1, 2),
        ]

        agreement = self.module.build_agreement(
            {"silex": silex, "pari": pari}, row
        )

        self.assertFalse(
            agreement["fields"]["canonical_prime_rational_data"]
        )
        self.assertFalse(agreement["success"])

    def test_agreement_rejects_duplicate_hnf_under_distinct_indices(self) -> None:
        row = self.module.load_field(
            self.manifest_path,
            "real_quadratic_5_split_11",
        )
        repeated_hnf = [["1", "4"], ["0", "11"]]
        normalized = normalized_result()
        normalized.update(row["expected"])
        normalized["selected_primes"] = [
            {
                "p": "11",
                "canonical_index": canonical_index,
                "e": 1,
                "f": 1,
                "hnf": repeated_hnf,
                "hnf_basis": "silex_maximal_order_basis_rows",
            }
            for canonical_index in (0, 1)
        ]
        normalized["canonical_prime_decompositions"] = normalized["selected_primes"]

        agreement = self.module.build_agreement({"silex": normalized}, row)

        self.assertFalse(
            agreement["fields"]["selected_prime_specification"]
        )
        self.assertFalse(agreement["success"])

    def test_explicit_hecke_grh_state_applies_to_every_ordinary_proof(self) -> None:
        row = dict(
            self.module.load_field(
                self.manifest_path,
                "cubic_x3_minus_2_empty_s",
            )
        )
        row["external_hecke_certification"] = "grh"
        normalized = normalized_result("hecke", "grh")
        self.assertTrue(
            self.module.build_agreement({"hecke": normalized}, row)["success"]
        )
        for key in (
            "certification_status",
            "class_group_proof_status",
            "unit_group_proof_status",
            "regulator_proof_status",
        ):
            with self.subTest(mismatched=key):
                mismatched = dict(normalized)
                mismatched[key] = "proven"
                agreement = self.module.build_agreement({"hecke": mismatched}, row)
                self.assertFalse(agreement["proof_complete"])
                self.assertFalse(agreement["success"])

    def test_invalid_hecke_certification_cannot_complete_agreement(self) -> None:
        row = dict(
            self.module.load_field(
                self.manifest_path,
                "cubic_x3_minus_2_empty_s",
            )
        )
        for invalid_status in ("verified", "unknown"):
            with self.subTest(invalid_status=invalid_status):
                row["external_hecke_certification"] = invalid_status
                normalized = normalized_result("hecke", invalid_status)
                agreement = self.module.build_agreement({"hecke": normalized}, row)
                self.assertFalse(agreement["proof_complete"])
                self.assertFalse(agreement["success"])

    def test_runner_missing_executable_is_a_structured_failure(self) -> None:
        result = self.module.run_process(
            [str(self.root / "definitely-missing-sunit-executable")],
            timeout=0.1,
        )
        self.assertFalse(result["available"])
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("error", result)

    def test_silex_native_executes_from_snapshot_without_driver_delegation(self) -> None:
        from silex_bench import process as process_module
        import subprocess
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silex_root = root / "silex"
            driver = silex_root / "tools/bench/run-sunit-instance.py"
            driver.parent.mkdir(parents=True)
            driver_marker = root / "driver-ran"
            driver.write_text(
                f"import pathlib\npathlib.Path({str(driver_marker)!r}).write_text('ran')\n"
            )
            build_dir = root / "build"
            build_dir.mkdir()
            executable = build_dir / "silex-class-unit-instance"
            original_executable = (
                f"#!{sys.executable}\n"
                "import json\nprint(json.dumps({'native': 'original'}))\n"
            ).encode()
            replacement_executable = (
                f"#!{sys.executable}\n"
                "import json\nprint(json.dumps({'native': 'replacement'}))\n"
            ).encode()
            executable.write_bytes(original_executable)
            executable.chmod(0o755)
            real_popen = subprocess.Popen

            def replace_during_spawn(command: list[str], **kwargs: Any):
                executable.write_bytes(replacement_executable)
                try:
                    return real_popen(command, **kwargs)
                finally:
                    executable.write_bytes(original_executable)

            def expose_process(
                process: dict[str, object], payload: dict[str, object], *_args: object
            ) -> dict[str, object]:
                return {"process": process, "payload": payload}

            args = SimpleNamespace(
                silex_root=silex_root,
                manifest=self.manifest_path,
                build_dir=build_dir,
                timeout=1.0,
            )
            row = self.module.load_field(
                args.manifest,
                "cubic_x3_minus_2_empty_s",
            )
            with mock.patch.object(
                process_module.subprocess,
                "Popen",
                side_effect=replace_during_spawn,
            ), mock.patch.object(
                self.module,
                "normalize_silex",
                side_effect=expose_process,
            ):
                result = self.module.run_silex(args, row)

            payload = cast(dict[str, object], result["payload"])
            process = cast(dict[str, object], result["process"])
            self.assertEqual(payload["native"], "original")
            self.assertFalse(driver_marker.exists())
            self.assertEqual(
                process["executable_sha256"],
                hashlib.sha256(original_executable).hexdigest(),
            )

    def test_silex_native_output_rejects_duplicate_json_fields(self) -> None:
        result, normalize = self._run_silex_with_stdout(
            '{"success":false,"success":true}'
        )

        self.assertFalse(result["success"])
        self.assertIn("duplicate key", result["error"])
        normalize.assert_not_called()

    def test_silex_native_output_rejects_nonfinite_json_numbers(self) -> None:
        for value in ("NaN", "Infinity", "-Infinity", "1e1000000"):
            with self.subTest(value=value):
                result, normalize = self._run_silex_with_stdout(
                    f'{{"success":true,"value":{value}}}'
                )

                self.assertFalse(result["success"])
                self.assertIn("non-finite", result["error"])
                normalize.assert_not_called()

    def test_silex_native_output_enforces_json_byte_limit(self) -> None:
        self.assertEqual(self.module.MAX_SILEX_JSON_OUTPUT_BYTES, 1 << 20)
        stdout = '{"success":true,"padding":"xxxxxxxxxxxxxxxx"}'
        with mock.patch.object(
            self.module,
            "MAX_SILEX_JSON_OUTPUT_BYTES",
            len(stdout.encode("utf-8")) - 1,
            create=True,
        ):
            result, normalize = self._run_silex_with_stdout(stdout)

        self.assertFalse(result["success"])
        self.assertIn("size limit", result["error"])
        normalize.assert_not_called()

    def test_silex_native_output_enforces_json_nesting_limit(self) -> None:
        with mock.patch.object(
            self.module, "MAX_JSON_NESTING_DEPTH", 2, create=True
        ):
            result, normalize = self._run_silex_with_stdout(
                '{"success":true,"nested":{"deeper":{}}}'
            )

        self.assertFalse(result["success"])
        self.assertIn("nesting limit", result["error"])
        normalize.assert_not_called()

    def test_silex_native_output_rejects_utf8_decode_replacement(self) -> None:
        result, normalize = self._run_silex_with_stdout(
            '{"success":true,"value":"\ufffd"}'
        )

        self.assertFalse(result["success"])
        self.assertIn("UTF-8", result["error"])
        normalize.assert_not_called()

    def test_silex_native_output_rejects_non_object_roots(self) -> None:
        for stdout in ("[]", '"success"', "null", "1"):
            with self.subTest(stdout=stdout):
                result, normalize = self._run_silex_with_stdout(stdout)

                self.assertFalse(result["success"])
                self.assertIn("object", result["error"])
                normalize.assert_not_called()

    def test_silex_native_output_rejects_invalid_process_state_before_parse(
        self,
    ) -> None:
        invalid_states = (
            {"available": "false"},
            {"success": "true"},
            {"timeout": "false"},
        )
        for state in invalid_states:
            with self.subTest(state=state):
                result, normalize = self._run_silex_with_stdout(
                    '{"success":true}', **state
                )

                self.assertFalse(result["success"])
                self.assertIsInstance(result["error"], str)
                self.assertTrue(result["error"])
                normalize.assert_not_called()

    def test_pari_version_probe_rejects_mistyped_process_success(self) -> None:
        args = SimpleNamespace(
            gp="gp",
            timeout=1.0,
            pari_source=self.root,
            pari_version="2.17.3",
        )
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        process = {
            "available": True,
            "success": "false",
            "timeout": False,
            "stdout": "2.17.3\n",
            "stderr": "",
            "executable_sha256": "a" * 64,
        }
        with mock.patch.object(
            self.module.shutil, "which", return_value="/usr/bin/gp"
        ), mock.patch.object(
            self.module, "pari_source_version", return_value="2.17.3"
        ), mock.patch.object(
            self.module, "run_process", return_value=process
        ) as run:
            result = self.module.run_pari(args, row)

        self.assertEqual(run.call_count, 1)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "engine_provenance")
        self.assertIsInstance(result["failure_reason"], str)
        self.assertTrue(result["failure_reason"])

    def test_pari_provenance_failure_has_nonempty_text_diagnostic(self) -> None:
        args = SimpleNamespace(
            gp="gp",
            timeout=1.0,
            pari_source=self.root,
            pari_version="2.17.3",
        )
        row = self.module.load_field(
            self.manifest_path,
            "cubic_x3_minus_2_empty_s",
        )
        version_process = {
            "available": True,
            "success": True,
            "timeout": False,
            "stdout": "2.17.3\n",
            "stderr": "",
            "executable_sha256": "a" * 64,
        }
        with mock.patch.object(
            self.module.shutil, "which", return_value="/usr/bin/gp"
        ), mock.patch.object(
            self.module, "run_process", return_value=version_process
        ), mock.patch.object(
            self.module, "pari_source_version", side_effect=ValueError("")
        ):
            result = self.module.run_pari(args, row)

        self.assertFalse(result["success"])
        self.assertIsInstance(result["failure_reason"], str)
        self.assertTrue(result["failure_reason"])

    def test_missing_external_executables_have_complete_failure_envelopes(
        self,
    ) -> None:
        args = SimpleNamespace(
            gp="missing-gp",
            julia="missing-julia",
            hecke_project=None,
        )
        with mock.patch.object(self.module.shutil, "which", return_value=None):
            results = (
                self.module.run_pari(args, {}),
                self.module.run_hecke(args, {}),
            )

        for result in results:
            with self.subTest(engine=result.get("engine")):
                self.assertIs(result.get("available"), False)
                self.assertIs(result.get("success"), False)
                self.assertIs(result.get("timeout"), False)
                diagnostic = result.get("failure_reason", result.get("error"))
                self.assertIsInstance(diagnostic, str)
                self.assertTrue(diagnostic)

    def test_standalone_scalar_parsers_fail_closed(self) -> None:
        self.assertIsNone(self.module.parse_int({"value": "9" * 5000}, "value"))
        for value in ("nan", "inf", "-inf", "1e1000000"):
            with self.subTest(value=value):
                self.assertIsNone(
                    self.module.parse_float({"value": value}, "value")
                )

    def test_sunit_manifest_input_is_bounded_and_nofollow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "manifest.json"
            serialized = json.dumps(self.manifest)
            target.write_text(serialized)
            linked = root / "linked.json"
            linked.symlink_to(target)
            with self.assertRaises((OSError, ValueError)):
                self.module.load_field(linked, "cubic_x3_minus_2_empty_s")

            oversized = root / "oversized.json"
            oversized.write_text(serialized + " ")
            with mock.patch.object(
                self.module,
                "MAX_SUNIT_MANIFEST_BYTES",
                len(serialized.encode()),
                create=True,
            ), self.assertRaisesRegex(ValueError, "size limit"):
                self.module.load_field(oversized, "cubic_x3_minus_2_empty_s")

    def test_sunit_result_output_is_atomic_and_nofollow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("sentinel\n")
            output = root / "result.json"
            output.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "regular file|symlink"):
                self.module.write_result({"success": True}, output)

            self.assertEqual(target.read_text(), "sentinel\n")

    def test_pari_source_version_input_is_bounded_and_nofollow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari"
            config = source / "config"
            config.mkdir(parents=True)
            target = root / "version-target"
            target.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            version = config / "version"
            version.symlink_to(target)
            with self.assertRaises((OSError, ValueError)):
                self.module.pari_source_version(source)

            version.unlink()
            version.write_text(target.read_text() + "# padding\n")
            with mock.patch.object(
                self.module,
                "MAX_PARI_VERSION_BYTES",
                len(target.read_bytes()),
                create=True,
            ), self.assertRaisesRegex(ValueError, "size limit"):
                self.module.pari_source_version(source)

    def test_pari_source_version_rejects_duplicate_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari"
            config = source / "config"
            config.mkdir(parents=True)
            (config / "version").write_text(
                "VersionMajor='9'\n"
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
            )

            with self.assertRaisesRegex(ValueError, "duplicate.*VersionMajor"):
                self.module.pari_source_version(source)

    def test_pari_source_version_rejects_compound_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari"
            config = source / "config"
            config.mkdir(parents=True)
            (config / "version").write_text(
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
                "VersionMajor='9'; VersionMajor='8'\n"
            )

            with self.assertRaisesRegex(ValueError, "ambiguous.*VersionMajor"):
                self.module.pari_source_version(source)

    def test_pari_source_version_rejects_alternate_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari"
            config = source / "config"
            config.mkdir(parents=True)
            (config / "version").write_text(
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
                "VersionMinor+='8'\n"
            )

            with self.assertRaisesRegex(ValueError, "ambiguous.*VersionMinor"):
                self.module.pari_source_version(source)

    def test_pari_source_version_rejects_continued_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari"
            config = source / "config"
            config.mkdir(parents=True)
            (config / "version").write_text(
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
                "pat\\\nch='9'\n"
            )

            with self.assertRaisesRegex(ValueError, "ambiguous.*patch"):
                self.module.pari_source_version(source)

    def test_pari_source_version_rejects_escaped_indirect_assignment(self) -> None:
        escaped_key = f"Version{chr(92)}115inor"
        escaped_eval = f"e{chr(92)}val"
        escaped_printf = f"p{chr(92)}rintf"
        fragmented_eval = "e''val"
        fragmented_printf = 'p""rintf'
        ansi_eval = r"$'\145\166\141\154'"
        ansi_printf = r"$'\160\162\151\156\164\146'"
        ansi_key = r"\126\145\162\163\151\157\156\115\151\156\157\162"
        expanded_commands = (
            "a=e\n"
            "b=val\n"
            "p=pri\n"
            "q=ntf\n"
            f'"$a$b" "$("$p$q" \'{escaped_key}\')=\'8\'"\n'
        )
        dynamic_export = (
            "name=Version\n"
            'name="${name}Minor"\n'
            'export "$name=8"\n'
        )
        evasions = (
            f"eval \"$(printf '{escaped_key}')='8'\"\n",
            (
                f"{escaped_eval} \"$("
                f"{escaped_printf} '{escaped_key}')='8'\"\n"
            ),
            (
                f"{fragmented_eval} \"$("
                f"{fragmented_printf} '{escaped_key}')='8'\"\n"
            ),
            f'{ansi_eval} "$({ansi_printf} \'{ansi_key}\')=\'8\'"\n',
            expanded_commands,
            dynamic_export,
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "pari"
                    config = source / "config"
                    config.mkdir(parents=True)
                    (config / "version").write_text(
                        "VersionMajor='2'\n"
                        "VersionMinor='17'\n"
                        "patch='3'\n"
                        + evasion
                    )

                    with self.assertRaisesRegex(ValueError, "ambiguous"):
                        self.module.pari_source_version(source)

    def test_pari_source_version_rejects_direct_escaped_required_assignment(
        self,
    ) -> None:
        evasions = (
            f"VersionM{chr(92)}inor='8'\n",
            "VersionM''inor='8'\n",
            'VersionM""inor=\'8\'\n',
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "pari"
                config = source / "config"
                config.mkdir(parents=True)
                (config / "version").write_text(
                    "VersionMajor='2'\n"
                    "VersionMinor='17'\n"
                    "patch='3'\n"
                    + evasion
                )

                with self.assertRaisesRegex(ValueError, "ambiguous.*VersionMinor"):
                    self.module.pari_source_version(source)

    def test_pari_source_version_rejects_computed_arithmetic_assignment(
        self,
    ) -> None:
        evasions = (
            ': "$(( $name = 8 ))"\n',
            'calculated="$(( $name = 8 ))"\n',
            'echo "$(( $name = 8 ))"\n',
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "pari"
                config = source / "config"
                config.mkdir(parents=True)
                (config / "version").write_text(
                    "VersionMajor='2'\n"
                    "VersionMinor='17'\n"
                    "patch='3'\n"
                    "left=VersionM\n"
                    "right=inor\n"
                    'name="$left$right"\n'
                    + evasion
                )

                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    self.module.pari_source_version(source)

    def test_pari_source_version_rejects_conditional_required_assignments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari"
            config = source / "config"
            config.mkdir(parents=True)
            (config / "version").write_text(
                "if true; then\n"
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
                "fi\n"
            )

            with self.assertRaisesRegex(ValueError, "noncanonical content"):
                self.module.pari_source_version(source)

    def test_runner_timeout_kills_descendant_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "descendant-survived"
            child = (
                "import pathlib,time; "
                "time.sleep(0.8); "
                f"pathlib.Path({str(marker)!r}).write_text('survived')"
            )
            parent = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', "
                f"{child!r}], stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL); "
                "print('spawned', flush=True); time.sleep(5)"
            )
            result = self.module.run_process(
                [sys.executable, "-u", "-c", parent],
                timeout=0.1,
            )
            self.assertTrue(result["available"])
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            self.assertIn("spawned", result["stdout"])
            time.sleep(0.9)
            self.assertFalse(marker.exists())

    def test_runner_output_is_bounded(self) -> None:
        limit = getattr(self.module, "MAX_CAPTURE_BYTES", 1 << 20)
        result = self.module.run_process(
            [
                sys.executable,
                "-c",
                f"import os; os.write(1, b'x' * ({limit} + 1))",
            ],
            timeout=2.0,
        )
        self.assertFalse(result["success"])
        self.assertIn("output limit", result["error"])
        self.assertLessEqual(len(result["stdout"].encode()), limit)

    def test_native_adapter_location_is_explicit(self) -> None:
        source = (
            self.root / "src/silex_bench/sunit_backend.py"
        ).read_text()
        self.assertIn("--silex-root", source)
        self.assertIn('args.build_dir / "silex-class-unit-instance"', source)
        self.assertNotIn('tools/bench/run-sunit-instance.py"', source)

    def test_local_pari_source_release_is_exact(self) -> None:
        configured = os.environ.get("SILEX_BENCH_PARI_SOURCE")
        source = (
            Path(configured).expanduser()
            if configured
            else Path.home() / "pari-2.17.3"
        )
        if not source.is_dir():
            self.skipTest("workspace PARI source is unavailable")
        self.assertEqual(self.module.pari_source_version(source), "2.17.3")


if __name__ == "__main__":
    unittest.main()
