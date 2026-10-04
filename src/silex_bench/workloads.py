"""Built-in workload contracts and deterministic corpus materialization."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .contracts import AgreementResult, Case, ValidationResult, WorkloadContract, canonical_json
from .util import read_json_nofollow


CLASS_UNIT = "class_unit_proven"
MAXIMAL_ORDER = "maximal_order"
IDEAL_MULTIPLY = "ideal_multiply"
ELEMENT_SQUARE_ROOT = "element_square_root"
SUNIT = "sunit_proven"
NUMBER_FIELD_WORKLOADS = (
    CLASS_UNIT,
    MAXIMAL_ORDER,
    IDEAL_MULTIPLY,
    ELEMENT_SQUARE_ROOT,
)
ALL_WORKLOADS = (*NUMBER_FIELD_WORKLOADS, SUNIT)
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_QUICK_FIELDS = {"real_quadratic_5_proven", "imaginary_quadratic_47_proven"}
_QUICK_SUNIT_FIELDS = {"cubic_x3_minus_2_empty_s", "real_quadratic_5_ramified_5"}


def _integer_text(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and _INTEGER.fullmatch(value):
        return value
    return None


def _invariants(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    normalized: list[int] = []
    for item in value:
        text = _integer_text(item)
        if text is None:
            return None
        integer = int(text)
        if integer < 1:
            return None
        if integer > 1:
            normalized.append(integer)
    normalized.sort()
    return [str(item) for item in normalized]


def _product(values: list[str]) -> str:
    result = 1
    for value in values:
        result *= int(value)
    return str(result)


def _comparison(
    lhs: Mapping[str, Any], rhs: Mapping[str, Any], fields: tuple[str, ...]
) -> AgreementResult:
    checks: dict[str, bool] = {}
    differences: list[str] = []
    for field in fields:
        checks[field] = lhs.get(field) == rhs.get(field)
        if not checks[field]:
            differences.append(
                f"{field}: {lhs.get(field)!r} != {rhs.get(field)!r}"
            )
    return AgreementResult(
        success=all(checks.values()),
        differences=tuple(differences),
        checks=checks,
    )


class NumberFieldContract(WorkloadContract):
    timing_scope = "backend_internal_target"
    scale_axes = (
        "degree",
        "log10_abs_discriminant",
        "discriminant_bits",
        "coefficient_bits",
        "height",
    )

    def __init__(self, identifier: str, display_name: str) -> None:
        self.id = identifier
        self.display_name = display_name

    def validate_case(self, case: Case) -> ValidationResult:
        errors: list[str] = []
        coefficients = case.input.get("coefficients_low_to_high")
        if (
            not isinstance(coefficients, list)
            or len(coefficients) < 3
            or any(type(value) is not int for value in coefficients)
            or coefficients[-1] != 1
        ):
            errors.append("coefficients must be an exact low-to-high monic integer array")
        degree = case.metrics.get("degree")
        if type(degree) is not int or not isinstance(coefficients, list) or degree != len(coefficients) - 1:
            errors.append("degree metric must match the coefficient array")
        if case.expected_status not in {"success", "failure"}:
            errors.append("expected_status must be success or failure")
        return ValidationResult(not errors, tuple(errors), {"case_schema": not errors})

    def validate_observation(
        self,
        case: Case,
        backend: str,
        result: Mapping[str, Any],
        proof: Mapping[str, Any],
    ) -> ValidationResult:
        if case.expected_status == "failure":
            return ValidationResult(
                False,
                ("expected-failure cases are correctness diagnostics, not successful observations",),
                {"expected_success": False},
            )
        errors: list[str] = []
        checks: dict[str, bool] = {}
        degree = int(case.metrics["degree"])
        if self.id == CLASS_UNIT:
            order = _integer_text(result.get("class_order"))
            invariants = _invariants(result.get("class_invariants"))
            rank = result.get("unit_rank")
            signature = result.get("signature")
            discriminant = _integer_text(result.get("maximal_order_discriminant"))
            checks.update(
                {
                    "class_order": order is not None,
                    "class_invariants": invariants is not None,
                    "unit_rank": type(rank) is int and rank >= 0,
                    "signature": (
                        isinstance(signature, list)
                        and len(signature) == 2
                        and all(type(value) is int and value >= 0 for value in signature)
                    ),
                    "maximal_order_discriminant": discriminant is not None,
                }
            )
            if invariants is not None and order is not None:
                checks["group_relation"] = _product(invariants) == order
            else:
                checks["group_relation"] = False
            if checks["signature"]:
                r1, r2 = signature
                checks["signature_degree"] = r1 + 2 * r2 == degree
                checks["rank_relation"] = rank == r1 + r2 - 1
            else:
                checks["signature_degree"] = False
                checks["rank_relation"] = False
            expected = case.expected
            expected_map = {
                "class_order": order,
                "class_invariants": invariants,
                "unit_rank": rank,
                "maximal_order_discriminant": discriminant,
            }
            for key, actual in expected_map.items():
                if key not in expected:
                    continue
                wanted: Any = expected[key]
                if key in {"class_order", "maximal_order_discriminant"}:
                    wanted = _integer_text(wanted)
                elif key == "class_invariants":
                    wanted = _invariants(wanted)
                checks[f"expected_{key}"] = actual == wanted
            final = proof.get("final_result_published") is True
            certification = proof.get("certification_status") == "proven"
            class_proof = proof.get("class_group_proof_status") == "proven"
            unit_proof = proof.get("unit_group_proof_status") == "proven"
            regulator = proof.get("regulator_proof_status") in {"proven", "verified"}
            checks.update(
                {
                    "final_result_published": final,
                    "certification": certification,
                    "class_group_proof": class_proof,
                    "unit_group_proof": unit_proof,
                    "regulator_proof": regulator,
                }
            )
        elif self.id == MAXIMAL_ORDER:
            value = _integer_text(result.get("maximal_order_discriminant"))
            checks["maximal_order_discriminant"] = value is not None
            if "maximal_order_discriminant" in case.expected:
                checks["expected_maximal_order_discriminant"] = value == _integer_text(
                    case.expected["maximal_order_discriminant"]
                )
        elif self.id == IDEAL_MULTIPLY:
            value = _integer_text(result.get("ideal_norm"))
            checks["ideal_norm"] = value == str(6**degree)
        elif self.id == ELEMENT_SQUARE_ROOT:
            checks["root_found"] = result.get("root_found") is True
            checks["root_verified"] = result.get("root_verified") is True
        else:
            errors.append(f"unknown workload contract: {self.id}")
        for name, success in checks.items():
            if not success:
                errors.append(f"failed check: {name}")
        return ValidationResult(not errors, tuple(errors), checks)

    def compare(
        self,
        case: Case,
        lhs_backend: str,
        lhs: Mapping[str, Any],
        rhs_backend: str,
        rhs: Mapping[str, Any],
    ) -> AgreementResult:
        if self.id == CLASS_UNIT:
            return _comparison(
                lhs,
                rhs,
                (
                    "class_order",
                    "class_invariants",
                    "unit_rank",
                    "signature",
                    "maximal_order_discriminant",
                ),
            )
        if self.id == MAXIMAL_ORDER:
            return _comparison(lhs, rhs, ("maximal_order_discriminant",))
        if self.id == IDEAL_MULTIPLY:
            return _comparison(lhs, rhs, ("ideal_norm",))
        return _comparison(lhs, rhs, ("root_found", "root_verified"))


class SUnitContract(WorkloadContract):
    id = SUNIT
    display_name = "Proven S-class and S-unit groups"
    timing_scope = "whole_process"
    scale_axes = (
        "degree",
        "log10_abs_discriminant",
        "s_size",
        "discriminant_bits",
        "coefficient_bits",
    )

    @staticmethod
    def _slot(backend: str) -> str:
        return backend if backend == "silex" else f"{backend}:external"

    def validate_case(self, case: Case) -> ValidationResult:
        row = case.input
        errors: list[str] = []
        coefficients = row.get("coefficients_low_to_high")
        if not isinstance(coefficients, list) or len(coefficients) < 3 or any(type(value) is not int for value in coefficients):
            errors.append("S-unit coefficients must be an integer array")
        if not isinstance(row.get("selected_primes"), list) or not isinstance(row.get("expected"), dict):
            errors.append("S-unit case requires selected_primes and expected objects")
        if not errors:
            module = sunit_module()
            try:
                module.validate_manifest(
                    {
                        "schema_version": module.SUNIT_MANIFEST_SCHEMA_VERSION,
                        "prime_index_convention": module.PRIME_INDEX_CONVENTION,
                        "fields": [dict(row)],
                    }
                )
            except ValueError as exc:
                errors.append(f"invalid S-unit manifest row: {exc}")
        return ValidationResult(not errors, tuple(errors), {"case_schema": not errors})

    def validate_observation(
        self,
        case: Case,
        backend: str,
        result: Mapping[str, Any],
        proof: Mapping[str, Any],
    ) -> ValidationResult:
        if backend == "magma":
            return ValidationResult(False, ("Magma S-unit comparison is not implemented",), {"supported": False})
        module = sunit_module()
        agreement = module.build_agreement({self._slot(backend): dict(result)}, case.input)
        errors = () if agreement.get("success") is True else (
            "backend failed the S-unit result, proof, membership, or manifest contract",
        )
        checks = {
            "backend_result": agreement.get("backend_results_agree") is True,
            "proof_complete": agreement.get("proof_complete") is True,
            "membership_verified": agreement.get("membership_verified") is True,
            "manifest_expectations": agreement.get("manifest_expectations_match") is True,
        }
        return ValidationResult(not errors, errors, checks)

    def compare(
        self,
        case: Case,
        lhs_backend: str,
        lhs: Mapping[str, Any],
        rhs_backend: str,
        rhs: Mapping[str, Any],
    ) -> AgreementResult:
        module = sunit_module()
        agreement = module.build_agreement(
            {
                self._slot(lhs_backend): dict(lhs),
                self._slot(rhs_backend): dict(rhs),
            },
            case.input,
        )
        checks = {
            "backend_results": agreement.get("backend_results_agree") is True,
            "proof_complete": agreement.get("proof_complete") is True,
            "membership_verified": agreement.get("membership_verified") is True,
            "manifest_expectations": agreement.get("manifest_expectations_match") is True,
        }
        differences = () if agreement.get("success") is True else (
            "S-unit canonical results or proof/membership evidence disagree",
        )
        return AgreementResult(not differences, differences, checks)


@lru_cache(maxsize=1)
def sunit_module() -> Any:
    from . import sunit_backend

    return sunit_backend


def builtin_workloads() -> list[WorkloadContract]:
    return [
        NumberFieldContract(CLASS_UNIT, "Proven class and unit groups"),
        NumberFieldContract(MAXIMAL_ORDER, "Maximal order"),
        NumberFieldContract(IDEAL_MULTIPLY, "Integral ideal multiplication"),
        NumberFieldContract(ELEMENT_SQUARE_ROOT, "Number-field element square root"),
        SUnitContract(),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"corpus root must be an object: {path}")
    return payload


def _field_tags(row: Mapping[str, Any]) -> set[str]:
    identifier = str(row.get("id", ""))
    role = row.get("benchmark_role")
    status = row.get("status")
    family = row.get("family")
    if isinstance(family, str) and family:
        # Family spot-test rows are unmeasured reference rows: tagged by
        # family only, never selected by the publication/scale/dev/quick
        # profiles until a measurement promotes them.
        tags = {"all", "number-field", "family", f"family:{family}"}
        if isinstance(role, str) and role:
            tags.add(role)
        if isinstance(status, str) and status:
            tags.add(status)
        return tags
    tags = {"all", "number-field", "publication"}
    if isinstance(role, str) and role:
        tags.add(role)
    if isinstance(status, str) and status:
        tags.add(status)
    if identifier in _QUICK_FIELDS:
        tags.add("quick")
    if row.get("expected_success") is True and isinstance(row.get("degree"), int) and row["degree"] <= 6:
        tags.add("dev")
    if role in {"core", "scale", "diversity", "holdout"}:
        tags.add("scale")
    return tags


def _exact_discriminant(row: Mapping[str, Any], label: str) -> int:
    text = _integer_text(row.get("maximal_order_discriminant"))
    if text is None or int(text) == 0:
        raise ValueError(
            f"{label} requires an exact signed nonzero maximal_order_discriminant"
        )
    return int(text)


def _discriminant_metrics(discriminant: int) -> dict[str, int | float]:
    magnitude = abs(discriminant)
    return {
        "maximal_order_discriminant": discriminant,
        "discriminant_bits": magnitude.bit_length(),
        "log10_abs_discriminant": math.log10(magnitude),
    }


def _field_metrics(row: Mapping[str, Any]) -> dict[str, int | float]:
    coefficients = row["coefficients_low_to_high"]
    height = max(abs(value) for value in coefficients)
    discriminant = _exact_discriminant(
        row, f"number-field corpus row {row.get('id', '<unknown>')}"
    )
    return {
        "degree": len(coefficients) - 1,
        "height": height,
        "coefficient_bits": max(1, height.bit_length()),
        **_discriminant_metrics(discriminant),
    }


def load_cases(corpora: Mapping[str, Path], workload_ids: tuple[str, ...]) -> list[Case]:
    cases: list[Case] = []
    number_path = corpora.get("number_fields")
    if any(item in NUMBER_FIELD_WORKLOADS for item in workload_ids):
        if number_path is None:
            raise ValueError("suite needs corpora.number_fields")
        payload = _read_json(number_path)
        rows = payload.get("fields")
        if not isinstance(rows, list):
            raise ValueError("number-field corpus fields must be an array")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("number-field corpus rows must be objects")
            identifier = row.get("id")
            coefficients = row.get("coefficients_low_to_high")
            if not isinstance(identifier, str) or not isinstance(coefficients, list):
                raise ValueError("number-field corpus row needs id and coefficients")
            metrics = _field_metrics(row)
            common_expected = {
                key: row[value]
                for key, value in (
                    ("class_order", "expected_class_order"),
                    ("class_invariants", "expected_class_invariants"),
                    ("unit_rank", "expected_unit_rank"),
                    ("maximal_order_discriminant", "maximal_order_discriminant"),
                )
                if value in row
            }
            for workload in workload_ids:
                if workload not in NUMBER_FIELD_WORKLOADS:
                    continue
                expected_success = row.get("expected_success", True) if workload == CLASS_UNIT else True
                backends = row.get("optimization_external_engines")
                eligible = ("silex", *backends) if isinstance(backends, list) else ()
                tags = _field_tags(row)
                if workload == ELEMENT_SQUARE_ROOT and metrics["degree"] > 9:
                    tags.discard("publication")
                cases.append(
                    Case(
                        id=identifier,
                        workload=workload,
                        input={
                            "coefficients_low_to_high": list(coefficients),
                            "timeout_seconds": row.get("timeout_seconds"),
                        },
                        tags=tuple(sorted(tags)),
                        metrics=metrics,
                        expected=common_expected,
                        expected_status="success" if expected_success else "failure",
                        performance_eligible=bool(expected_success),
                        eligible_backends=tuple(str(item) for item in eligible),
                        source=str(row.get("source", "")),
                    )
                )
    if SUNIT in workload_ids:
        path = corpora.get("sunit")
        if path is None:
            raise ValueError("suite needs corpora.sunit")
        module = sunit_module()
        payload = read_json_nofollow(
            path,
            root=path.parent,
            max_bytes=module.MAX_SUNIT_MANIFEST_BYTES,
        )
        rows = module.validate_manifest(payload)
        for row in rows:
            coefficients = row["coefficients_low_to_high"]
            discriminant = _exact_discriminant(
                row, f"S-unit corpus row {row.get('id', '<unknown>')}"
            )
            height = max(abs(value) for value in coefficients)
            tags = {"all", "sunit", "dev", "scale", "publication"}
            # `quick` membership is decided solely by this explicit allow-list,
            # never by manifest row position, so reordering
            # corpora/sunit_fields.json can never silently swap a slow/full-proof
            # row into the bounded `quick` gate. See
            # test_quick_sunit_tag_is_independent_of_manifest_row_order in
            # test/test_sunit_corpus_loading.py.
            if row["id"] in _QUICK_SUNIT_FIELDS:
                tags.add("quick")
            cases.append(
                Case(
                    id=row["id"],
                    workload=SUNIT,
                    input=dict(row),
                    tags=tuple(sorted(tags)),
                    metrics={
                        "degree": len(coefficients) - 1,
                        "s_size": len(row.get("selected_primes", [])),
                        "coefficient_bits": max(1, height.bit_length()),
                        **_discriminant_metrics(discriminant),
                    },
                    expected=dict(row.get("expected", {})),
                    performance_eligible=True,
                    eligible_backends=(),
                    source="S-class/S-unit comparison corpus",
                )
            )
    keys = [case.key for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("materialized case keys must be unique")
    return cases


def select_cases(
    cases: list[Case],
    execution: Mapping[str, Any],
    *,
    performance: bool,
) -> list[Case]:
    include = set(execution.get("include_tags", []))
    required = set(execution.get("required_tags", []))
    exclude = set(execution.get("exclude_tags", []))
    case_ids = set(execution.get("case_ids", []))
    metrics = execution.get("metrics", {})
    selected: list[Case] = []
    for case in cases:
        tags = set(case.tags)
        if case_ids:
            if case.id not in case_ids and case.key not in case_ids:
                continue
        elif include and not include.intersection(tags):
            continue
        if not required.issubset(tags):
            continue
        if exclude.intersection(tags):
            continue
        if performance and not case.performance_eligible:
            continue
        keep = True
        for name, bounds in metrics.items():
            if name not in case.metrics:
                keep = False
                break
            value = float(case.metrics[name])
            minimum, maximum = bounds
            if minimum is not None and value < minimum:
                keep = False
            if maximum is not None and value > maximum:
                keep = False
        if keep:
            selected.append(case)
    return sorted(selected, key=lambda item: (item.workload, item.id))


def cases_digest(cases: list[Case]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json([case.to_json() for case in cases]).encode()).hexdigest()
