#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare one proven S-class/S-unit fixture across Silex, PARI, and Hecke."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, cast


from silex_bench.backends.hecke import (
    hecke_load_error,
    julia_command,
    julia_environment,
    resolve_hecke_project,
    resolve_julia_runtime,
)
from silex_bench.backends.pari import parse_pari_source_version_bytes
from silex_bench.process import run_process as run_bounded_process
from silex_bench.resources import builtin_path
from silex_bench.util import (
    MAX_JSON_NESTING_DEPTH,
    atomic_write_json,
    ensure_absolute_directory_nofollow,
    parse_bounded_json_bytes,
    read_bytes_nofollow,
    read_json_nofollow,
)


HECKE_CERTIFICATION_STATUSES = frozenset({"proven", "grh"})
EXACT_RESULT_FIELDS = (
    "s_class_order",
    "s_class_invariants",
    "torsion_order",
    "ordinary_free_rank",
    "nonunit_rank",
    "free_rank",
    "valuation_lattice_index",
)
STRING_RESULT_FIELDS = frozenset(
    {"s_class_order", "torsion_order", "valuation_lattice_index"}
)
INTEGER_RESULT_FIELDS = frozenset(
    {"ordinary_free_rank", "nonunit_rank", "free_rank"}
)
BACKEND_ENGINES = frozenset({"silex", "pari", "hecke"})
HNF_BASES = {
    "silex": "silex_maximal_order_basis_rows",
    "pari": "pari_bnf_integral_basis_rows",
    "hecke": "hecke_maximal_order_basis_rows",
}
CANONICAL_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
MAX_RATIONAL_PRIME = (1 << 63) - 1
MAX_MATRIX_INTEGER_DIGITS = 4096
MAX_WITNESS_DEGREE = 16
MAX_RATIONAL_TEXT_CHARS = 8192
PRIME_INDEX_CONVENTION = (
    "zero-based authoritative manifest order of exact two-generator ideals "
    "(p, beta_power_basis) within each rational-prime decomposition"
)
SUNIT_MANIFEST_SCHEMA_VERSION = 2
MAX_SUNIT_MANIFEST_BYTES = 1 << 20
MAX_PARI_VERSION_BYTES = 64 << 10
MAX_SILEX_JSON_OUTPUT_BYTES = 1 << 20


def repo_root() -> Path:
    """Return the caller's working directory for child-process execution."""

    return Path.cwd().absolute()


def default_workspace() -> Path:
    """Find a conventional workspace without depending on package location."""

    current = Path.cwd().absolute()
    if (current / "silex").is_dir():
        return current
    if current.name == "silex-bench" and (current.parent / "silex").is_dir():
        return current.parent
    return current


def hecke_certification_status(row: dict[str, Any]) -> str | None:
    status = row.get("external_hecke_certification", "proven")
    if isinstance(status, str) and status in HECKE_CERTIFICATION_STATUSES:
        return status
    return None


def rational_prime_is_valid(value: Any) -> bool:
    """Return whether value is prime in the bounded manifest integer domain."""
    if type(value) is not int or value < 2 or value > MAX_RATIONAL_PRIME:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1
    # Jim Sinclair's seven-base set is recorded as deterministic through 2^64:
    # https://miller-rabin.appspot.com/
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def parse_canonical_rational(value: Any) -> Fraction | None:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_RATIONAL_TEXT_CHARS
    ):
        return None
    if "/" not in value:
        if (
            re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value) is None
            or value == "-0"
        ):
            return None
        return Fraction(int(value), 1)
    if value.count("/") != 1:
        return None
    numerator_text, denominator_text = value.split("/", 1)
    if (
        re.fullmatch(r"-?[1-9][0-9]*", numerator_text) is None
        or re.fullmatch(r"[1-9][0-9]*", denominator_text) is None
    ):
        return None
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator < 2 or math.gcd(abs(numerator), denominator) != 1:
        return None
    return Fraction(numerator, denominator)


def parse_fraction_matrix(value: Any, size: int) -> list[list[Fraction]] | None:
    if (
        type(size) is not int
        or size < 1
        or size > MAX_WITNESS_DEGREE
        or type(value) is not list
        or len(value) != size
        or any(type(row) is not list or len(row) != size for row in value)
    ):
        return None
    matrix: list[list[Fraction]] = []
    for row in value:
        parsed_row = [parse_canonical_rational(entry) for entry in row]
        if any(entry is None for entry in parsed_row):
            return None
        matrix.append(cast(list[Fraction], parsed_row))
    return matrix


def fraction_matrix_inverse(
    matrix: list[list[Fraction]],
) -> list[list[Fraction]] | None:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        return None
    work = [
        list(row)
        + [
            Fraction(1 if row_index == column else 0)
            for column in range(size)
        ]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot_row = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return None
        work[column], work[pivot_row] = work[pivot_row], work[column]
        pivot = work[column][column]
        work[column] = [entry / pivot for entry in work[column]]
        for row in range(size):
            if row == column or work[row][column] == 0:
                continue
            multiplier = work[row][column]
            work[row] = [
                entry - multiplier * pivot_entry
                for entry, pivot_entry in zip(work[row], work[column])
            ]
    return [row[size:] for row in work]


def row_times_matrix(
    row: list[Fraction], matrix: list[list[Fraction]]
) -> list[Fraction]:
    return [
        sum(
            (row[index] * matrix[index][column] for index in range(len(row))),
            Fraction(0),
        )
        for column in range(len(matrix[0]))
    ]


def polynomial_product_mod(
    left: list[Fraction],
    right: list[Fraction],
    polynomial: list[int],
) -> list[Fraction]:
    degree = len(polynomial) - 1
    product = [Fraction(0) for _ in range(2 * degree - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            product[left_index + right_index] += left_value * right_value
    for power in range(len(product) - 1, degree - 1, -1):
        coefficient = product[power]
        if coefficient == 0:
            continue
        for lower_power in range(degree):
            product[power - degree + lower_power] -= (
                coefficient * polynomial[lower_power]
            )
    return product[:degree]


def integer_coordinates(
    power_row: list[Fraction], basis_inverse: list[list[Fraction]]
) -> list[int] | None:
    coordinates = row_times_matrix(power_row, basis_inverse)
    if any(value.denominator != 1 for value in coordinates):
        return None
    return [value.numerator for value in coordinates]


def integer_matrix_product(
    left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    return [
        [
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(len(right))
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def modular_rank(matrix: list[list[int]], modulus: int) -> int:
    if not matrix:
        return 0
    work = [[entry % modulus for entry in row] for row in matrix]
    row_count = len(work)
    column_count = len(work[0])
    pivot_row = 0
    for column in range(column_count):
        source = next(
            (row for row in range(pivot_row, row_count) if work[row][column]),
            None,
        )
        if source is None:
            continue
        work[pivot_row], work[source] = work[source], work[pivot_row]
        inverse = pow(work[pivot_row][column], -1, modulus)
        work[pivot_row] = [
            (entry * inverse) % modulus for entry in work[pivot_row]
        ]
        for row in range(row_count):
            if row == pivot_row or work[row][column] == 0:
                continue
            multiplier = work[row][column]
            work[row] = [
                (entry - multiplier * pivot_entry) % modulus
                for entry, pivot_entry in zip(work[row], work[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def maximal_order_evidence(
    result: dict[str, Any], row: dict[str, Any]
) -> tuple[list[list[Fraction]], list[list[Fraction]]] | None:
    polynomial = row.get("coefficients_low_to_high")
    expected_discriminant = row.get("maximal_order_discriminant")
    if (
        type(polynomial) is not list
        or not 2 <= len(polynomial) <= MAX_WITNESS_DEGREE + 1
        or any(type(coefficient) is not int for coefficient in polynomial)
        or polynomial[-1] != 1
        or type(expected_discriminant) is not str
        or re.fullmatch(r"-?[1-9][0-9]*", expected_discriminant) is None
        or len(expected_discriminant.lstrip("-")) > MAX_MATRIX_INTEGER_DIGITS
        or result.get("maximal_order_discriminant") != expected_discriminant
    ):
        return None
    degree = len(polynomial) - 1
    basis = parse_fraction_matrix(result.get("maximal_order_basis_power"), degree)
    if basis is None:
        return None
    basis_inverse = fraction_matrix_inverse(basis)
    if basis_inverse is None:
        return None
    one_coordinates = integer_coordinates(
        [Fraction(1)] + [Fraction(0) for _ in range(degree - 1)],
        basis_inverse,
    )
    if one_coordinates is None:
        return None

    multiplication_matrices: list[list[list[int]]] = []
    for multiplier in basis:
        multiplication_matrix: list[list[int]] = []
        for multiplicand in basis:
            coordinates = integer_coordinates(
                polynomial_product_mod(multiplicand, multiplier, polynomial),
                basis_inverse,
            )
            if coordinates is None:
                return None
            multiplication_matrix.append(coordinates)
        multiplication_matrices.append(multiplication_matrix)
    trace_matrix: list[list[int]] = []
    for left in range(degree):
        trace_row: list[int] = []
        for right in range(degree):
            product = integer_matrix_product(
                multiplication_matrices[left], multiplication_matrices[right]
            )
            trace_row.append(sum(product[index][index] for index in range(degree)))
        trace_matrix.append(trace_row)
    if integer_matrix_determinant(trace_matrix) != int(expected_discriminant):
        return None
    return basis, basis_inverse


def prime_ideal_lattice_matches_witness(
    descriptor: dict[str, Any],
    polynomial: list[int],
    basis: list[list[Fraction]],
    basis_inverse: list[list[Fraction]],
) -> bool:
    degree = len(polynomial) - 1
    beta_raw = descriptor.get("beta_power_basis")
    if type(beta_raw) is not list or len(beta_raw) != degree:
        return False
    beta_entries = [parse_canonical_rational(entry) for entry in beta_raw]
    if any(entry is None for entry in beta_entries):
        return False
    beta = cast(list[Fraction], beta_entries)
    hnf_raw = descriptor.get("hnf")
    if (
        type(hnf_raw) is not list
        or len(hnf_raw) != degree
        or any(type(row) is not list or len(row) != degree for row in hnf_raw)
    ):
        return False
    hnf: list[list[int]] = []
    for row_values in hnf_raw:
        integer_row: list[int] = []
        for entry in row_values:
            if (
                type(entry) is not str
                or re.fullmatch(r"(?:0|-?[1-9][0-9]*)", entry) is None
                or entry == "-0"
                or len(entry.lstrip("-")) > MAX_MATRIX_INTEGER_DIGITS
            ):
                return False
            integer_row.append(int(entry))
        hnf.append(integer_row)
    p = int(descriptor["p"])
    if abs(integer_matrix_determinant(hnf)) != p ** descriptor["f"]:
        return False
    hnf_inverse = fraction_matrix_inverse(
        [[Fraction(entry) for entry in row] for row in hnf]
    )
    if hnf_inverse is None:
        return False

    beta_multiplication: list[list[int]] = []
    for multiplicand in basis:
        coordinates = integer_coordinates(
            polynomial_product_mod(multiplicand, beta, polynomial), basis_inverse
        )
        if coordinates is None:
            return False
        beta_multiplication.append(coordinates)
    ideal_generators = [
        [p if row == column else 0 for column in range(degree)]
        for row in range(degree)
    ] + beta_multiplication
    for generator in ideal_generators:
        coordinates = row_times_matrix(
            [Fraction(entry) for entry in generator], hnf_inverse
        )
        if any(value.denominator != 1 for value in coordinates):
            return False
    beta_rank = modular_rank(beta_multiplication, p)
    if any(
        modular_rank(beta_multiplication + [row], p) != beta_rank for row in hnf
    ):
        return False
    return True


def canonical_positive_integer_text(value: Any, *, minimum: int = 1) -> bool:
    if type(value) is not str or CANONICAL_POSITIVE_INTEGER.fullmatch(value) is None:
        return False
    minimum_text = str(minimum)
    return len(value) > len(minimum_text) or (
        len(value) == len(minimum_text) and value >= minimum_text
    )


def exact_result_value_is_valid(key: str, value: Any) -> bool:
    if key == "torsion_order":
        return canonical_positive_integer_text(value, minimum=2)
    if key in STRING_RESULT_FIELDS:
        return canonical_positive_integer_text(value)
    if key in INTEGER_RESULT_FIELDS:
        return type(value) is int and value >= 0
    if key == "s_class_invariants":
        return (
            type(value) is list
            and all(
                canonical_positive_integer_text(item, minimum=2) for item in value
            )
            and value == sorted(value, key=lambda item: (len(item), item))
        )
    return False


def decimal_product(left: str, right: str) -> str:
    digits = [0] * (len(left) + len(right))
    for left_offset, left_digit in enumerate(reversed(left)):
        for right_offset, right_digit in enumerate(reversed(right)):
            digits[left_offset + right_offset] += int(left_digit) * int(right_digit)
    for index in range(len(digits) - 1):
        carry, digits[index] = divmod(digits[index], 10)
        digits[index + 1] += carry
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
    return "".join(str(digit) for digit in reversed(digits))


def decimal_compare(left: str, right: str) -> int:
    if len(left) != len(right):
        return -1 if len(left) < len(right) else 1
    return (left > right) - (left < right)


def decimal_subtract(left: str, right: str) -> str:
    result: list[int] = []
    borrow = 0
    right_digits = list(reversed(right))
    for index, left_digit in enumerate(reversed(left)):
        value = int(left_digit) - borrow
        if index < len(right_digits):
            value -= int(right_digits[index])
        if value < 0:
            value += 10
            borrow = 1
        else:
            borrow = 0
        result.append(value)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return "".join(str(digit) for digit in reversed(result))


def decimal_divides(divisor: str, dividend: str) -> bool:
    remainder = "0"
    for digit in dividend:
        remainder = ("" if remainder == "0" else remainder) + digit
        remainder = remainder.lstrip("0") or "0"
        while decimal_compare(remainder, divisor) >= 0:
            remainder = decimal_subtract(remainder, divisor)
    return remainder == "0"


def exact_group_relations_are_valid(result: Any) -> bool:
    if type(result) is not dict:
        return False
    order = result.get("s_class_order")
    invariants = result.get("s_class_invariants")
    torsion = result.get("torsion_order")
    if (
        type(order) is not str
        or type(invariants) is not list
        or any(type(invariant) is not str for invariant in invariants)
        or type(torsion) is not str
        or not exact_result_value_is_valid("s_class_order", order)
        or not exact_result_value_is_valid("s_class_invariants", invariants)
        or not exact_result_value_is_valid("torsion_order", torsion)
    ):
        return False
    string_invariants = cast(list[str], invariants)
    product = "1"
    for invariant in string_invariants:
        product = decimal_product(product, invariant)
    if product != order:
        return False
    return all(
        decimal_divides(left, right)
        for left, right in zip(string_invariants, string_invariants[1:])
    )


def exact_rank_relation_is_valid(result: Any) -> bool:
    if type(result) is not dict:
        return False
    ordinary_rank = result.get("ordinary_free_rank")
    nonunit_rank = result.get("nonunit_rank")
    free_rank = result.get("free_rank")
    return (
        type(ordinary_rank) is int
        and ordinary_rank >= 0
        and type(nonunit_rank) is int
        and nonunit_rank >= 0
        and type(free_rank) is int
        and free_rank >= 0
        and free_rank == ordinary_rank + nonunit_rank
    )


def expected_engine_for_slot(slot: str) -> str | None:
    engine = slot.partition(":")[0]
    return engine if engine in BACKEND_ENGINES else None


def standalone_engine_identity_is_valid(engine: str | None, result: Any) -> bool:
    if engine is None or type(result) is not dict or result.get("engine") != engine:
        return False
    identity = result.get("engine_identity")
    if type(identity) is not dict:
        return False
    expected_keys = {
        "silex": {"executable", "executable_sha256", "version", "source"},
        "pari": {
            "executable",
            "executable_sha256",
            "version",
            "required_version",
            "source",
            "source_version",
        },
        "hecke": {
            "executable",
            "executable_sha256",
            "version",
            "package_version",
            "source",
        },
    }.get(engine)
    if expected_keys is None or set(identity) != expected_keys:
        return False
    if (
        type(identity.get("executable")) is not str
        or not identity["executable"]
        or not Path(identity["executable"]).is_absolute()
        or type(identity.get("executable_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", identity["executable_sha256"]) is None
    ):
        return False
    if engine == "silex":
        return (
            identity.get("version") is None
            and type(identity.get("source")) is str
            and bool(identity["source"])
        )
    if engine == "pari":
        version = identity.get("version")
        required_version = identity.get("required_version")
        source = identity.get("source")
        source_version = identity.get("source_version")
        if type(version) is not str or not version:
            return False
        if any(
            value is not None and (type(value) is not str or not value)
            for value in (required_version, source, source_version)
        ):
            return False
        if source is None and source_version is not None:
            return False
        if source is not None and source_version is None:
            return False
        if required_version is not None and required_version != version:
            return False
        return source_version is None or source_version == version
    return all(
        type(identity.get(key)) is str and bool(identity[key])
        for key in ("version", "package_version", "source")
    )


def integer_matrix_determinant(matrix: list[list[int]]) -> int:
    """Compute an exact determinant with fraction-free elimination."""
    size = len(matrix)
    if size == 0:
        return 1
    work = [list(row) for row in matrix]
    sign = 1
    previous_pivot = 1
    for column in range(size - 1):
        pivot_row = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            sign = -sign
        pivot = work[column][column]
        for row in range(column + 1, size):
            for inner_column in range(column + 1, size):
                numerator = (
                    work[row][inner_column] * pivot
                    - work[row][column] * work[column][inner_column]
                )
                work[row][inner_column] = numerator // previous_pivot
            work[row][column] = 0
        previous_pivot = pivot
    return sign * work[-1][-1]


def matrix_is_row_or_transposed_row_hnf(matrix: list[list[int]]) -> bool:
    size = len(matrix)
    diagonal = [matrix[index][index] for index in range(size)]
    if any(value <= 0 for value in diagonal):
        return False
    upper = all(
        matrix[row][column] == 0
        for row in range(size)
        for column in range(row)
    ) and all(
        0 <= matrix[row][column] < diagonal[column]
        for row in range(size)
        for column in range(row + 1, size)
    )
    lower = all(
        matrix[row][column] == 0
        for row in range(size)
        for column in range(row + 1, size)
    ) and all(
        0 <= matrix[row][column] < diagonal[row]
        for row in range(size)
        for column in range(row)
    )
    return upper or lower


def valuation_lattice_evidence_is_valid(result: Any) -> bool:
    if type(result) is not dict:
        return False
    rank = result.get("nonunit_rank")
    matrix = result.get("valuation_matrix")
    index = result.get("valuation_lattice_index")
    if (
        type(rank) is not int
        or rank < 0
        or type(matrix) is not list
        or not canonical_positive_integer_text(index)
    ):
        return False
    if rank == 0:
        return matrix == [] and index == "1"
    if len(matrix) != rank or any(
        type(row) is not list or len(row) != rank for row in matrix
    ):
        return False
    integer_matrix: list[list[int]] = []
    try:
        for row in matrix:
            integer_row: list[int] = []
            for entry in row:
                if (
                    type(entry) is not str
                    or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", entry) is None
                    or len(entry.lstrip("-")) > MAX_MATRIX_INTEGER_DIGITS
                    or str(int(entry)) != entry
                ):
                    return False
                integer_row.append(int(entry))
            integer_matrix.append(integer_row)
    except ValueError:
        return False
    return str(abs(integer_matrix_determinant(integer_matrix))) == index


def validated_prime_descriptors(
    result: dict[str, Any],
    degree: int,
    key: str = "selected_primes",
) -> list[dict[str, Any]] | None:
    engine = result.get("engine")
    if type(engine) is not str:
        return None
    expected_basis = HNF_BASES.get(engine)
    if expected_basis is None or type(degree) is not int or degree < 1:
        return None
    raw_descriptors = result.get(key)
    if type(raw_descriptors) is not list:
        return None
    descriptors: list[dict[str, Any]] = []
    for item in raw_descriptors:
        if type(item) is not dict:
            return None
        raw_p = item.get("p")
        try:
            if type(raw_p) is int:
                p = raw_p
            elif (
                type(raw_p) is str
                and raw_p.isdigit()
                and str(int(raw_p)) == raw_p
            ):
                p = int(raw_p)
            else:
                return None
        except ValueError:
            return None
        e = item.get("e")
        f = item.get("f")
        canonical_index = item.get("canonical_index")
        beta_power_basis = item.get("beta_power_basis")
        hnf = item.get("hnf")
        basis = item.get("hnf_basis")
        if (
            not rational_prime_is_valid(p)
            or type(canonical_index) is not int
            or canonical_index < 0
            or canonical_index >= degree
            or type(e) is not int
            or e < 1
            or e > degree
            or type(f) is not int
            or f < 1
            or f > degree
            or e * f > degree
            or type(beta_power_basis) is not list
            or len(beta_power_basis) != degree
            or any(
                parse_canonical_rational(coefficient) is None
                for coefficient in beta_power_basis
            )
            or type(hnf) is not list
            or len(hnf) != degree
            or any(type(row) is not list or len(row) != degree for row in hnf)
            or basis != expected_basis
        ):
            return None
        integer_hnf: list[list[int]] = []
        try:
            for hnf_row in hnf:
                integer_row: list[int] = []
                for entry in hnf_row:
                    if (
                        type(entry) is not str
                        or len(entry.lstrip("-")) > MAX_MATRIX_INTEGER_DIGITS
                        or str(int(entry)) != entry
                    ):
                        return None
                    integer_row.append(int(entry))
                integer_hnf.append(integer_row)
        except ValueError:
            return None
        if abs(integer_matrix_determinant(integer_hnf)) != p**f:
            return None
        descriptors.append(
            {
                "p": str(p),
                "canonical_index": canonical_index,
                "e": e,
                "f": f,
                "beta_power_basis": beta_power_basis,
                "hnf": hnf,
                "hnf_basis": basis,
            }
        )
    return descriptors


def selected_prime_selector_groups(
    row: dict[str, Any],
) -> dict[int, list[str | int]] | None:
    selectors = row.get("selected_primes")
    if type(selectors) is not list:
        return None
    selectors_by_prime: dict[int, list[str | int]] = {}
    seen_selectors: set[tuple[int, str | int]] = set()
    for selector in selectors:
        if type(selector) is not dict:
            return None
        p = selector.get("p")
        index = selector.get("index")
        if (
            type(p) is not int
            or not rational_prime_is_valid(p)
            or not (
                index == "all"
                or (type(index) is int and index >= 0)
            )
            or (p, index) in seen_selectors
        ):
            return None
        seen_selectors.add((p, index))
        selectors_by_prime.setdefault(p, []).append(index)
    if any(
        "all" in indices and indices != ["all"]
        for indices in selectors_by_prime.values()
    ):
        return None
    return selectors_by_prime


def validated_manifest_prime_witnesses(
    row: dict[str, Any],
) -> list[dict[str, Any]] | None:
    coefficients = row.get("coefficients_low_to_high")
    selectors_by_prime = selected_prime_selector_groups(row)
    discriminant = row.get("maximal_order_discriminant")
    witnesses = row.get("prime_ideal_witnesses")
    if (
        type(coefficients) is not list
        or not 2 <= len(coefficients) <= MAX_WITNESS_DEGREE + 1
        or selectors_by_prime is None
        or type(discriminant) is not str
        or re.fullmatch(r"-?[1-9][0-9]*", discriminant) is None
        or len(discriminant.lstrip("-")) > MAX_MATRIX_INTEGER_DIGITS
        or type(witnesses) is not list
    ):
        return None
    degree = len(coefficients) - 1
    normalized: list[dict[str, Any]] = []
    for witness in witnesses:
        if type(witness) is not dict or set(witness) != {
            "p",
            "canonical_index",
            "e",
            "f",
            "beta_power_basis",
        }:
            return None
        p = witness.get("p")
        canonical_index = witness.get("canonical_index")
        e = witness.get("e")
        f = witness.get("f")
        beta = witness.get("beta_power_basis")
        if (
            type(p) is not int
            or not rational_prime_is_valid(p)
            or type(canonical_index) is not int
            or canonical_index < 0
            or canonical_index >= degree
            or type(e) is not int
            or e < 1
            or e > degree
            or type(f) is not int
            or f < 1
            or f > degree
            or e * f > degree
            or type(beta) is not list
            or len(beta) != degree
            or any(parse_canonical_rational(entry) is None for entry in beta)
        ):
            return None
        normalized.append(
            {
                "p": p,
                "canonical_index": canonical_index,
                "e": e,
                "f": f,
                "beta_power_basis": beta,
            }
        )

    witnesses_by_prime: dict[int, list[dict[str, Any]]] = {}
    for witness in normalized:
        witnesses_by_prime.setdefault(witness["p"], []).append(witness)
    if list(witnesses_by_prime) != list(selectors_by_prime):
        return None
    selected_count = 0
    for p, selectors in selectors_by_prime.items():
        complete = witnesses_by_prime[p]
        if (
            [witness["canonical_index"] for witness in complete]
            != list(range(len(complete)))
            or sum(witness["e"] * witness["f"] for witness in complete) != degree
            or len(
                {
                    tuple(witness["beta_power_basis"])
                    for witness in complete
                }
            )
            != len(complete)
        ):
            return None
        if selectors == ["all"]:
            selected_count += len(complete)
        else:
            if any(
                type(index) is not int or index >= len(complete)
                for index in selectors
            ):
                return None
            selected_count += len(selectors)
    expected = row.get("expected")
    if type(expected) is not dict or expected.get("nonunit_rank") != selected_count:
        return None
    return normalized


def witness_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item["p"]),
        item["canonical_index"],
        item["e"],
        item["f"],
        tuple(item["beta_power_basis"]),
    )


def selected_prime_specification_matches(
    descriptors: list[dict[str, Any]],
    canonical_decomposition: list[dict[str, Any]],
    row: dict[str, Any],
) -> bool:
    expected = row.get("expected")
    selectors_by_prime = selected_prime_selector_groups(row)
    manifest_witnesses = validated_manifest_prime_witnesses(row)
    if (
        selectors_by_prime is None
        or manifest_witnesses is None
        or type(expected) is not dict
    ):
        return False
    expected_rank = expected.get("nonunit_rank")
    if type(expected_rank) is not int or expected_rank < 0:
        return False

    def local_identity(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            witness_identity(item),
            tuple(tuple(row) for row in item["hnf"]),
            item["hnf_basis"],
        )

    descriptor_keys = [local_identity(item) for item in descriptors]
    decomposition_keys = [local_identity(item) for item in canonical_decomposition]
    if (
        len(set(descriptor_keys)) != len(descriptor_keys)
        or len(set(decomposition_keys)) != len(decomposition_keys)
    ):
        return False
    if len(descriptors) != expected_rank:
        return False

    decomposition_by_prime: dict[int, list[dict[str, Any]]] = {}
    for descriptor in canonical_decomposition:
        p = int(descriptor["p"])
        decomposition_by_prime.setdefault(p, []).append(descriptor)
    witness_by_prime: dict[int, list[dict[str, Any]]] = {}
    for witness in manifest_witnesses:
        witness_by_prime.setdefault(witness["p"], []).append(witness)
    if (
        list(decomposition_by_prime) != list(selectors_by_prime)
        or list(witness_by_prime) != list(selectors_by_prime)
    ):
        return False

    expected_selected: list[dict[str, Any]] = []
    for p, selectors in selectors_by_prime.items():
        complete = decomposition_by_prime[p]
        expected_complete = witness_by_prime[p]
        if (
            [item["canonical_index"] for item in complete]
            != list(range(len(complete)))
            or sum(item["e"] * item["f"] for item in complete) != len(
                complete[0]["hnf"]
            )
            or [witness_identity(item) for item in complete]
            != [witness_identity(item) for item in expected_complete]
        ):
            return False
        hnf_keys = [
            tuple(int(entry) for row_values in item["hnf"] for entry in row_values)
            for item in complete
        ]
        if len(set(hnf_keys)) != len(hnf_keys):
            return False
        if "all" in selectors:
            expected_selected.extend(complete)
            continue
        for index in selectors:
            if type(index) is not int or index >= len(complete):
                return False
            expected_selected.append(complete[index])
    return descriptor_keys == [local_identity(item) for item in expected_selected]


def validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if (
        type(manifest) is not dict
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SUNIT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            f"manifest schema_version must be {SUNIT_MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("prime_index_convention") != PRIME_INDEX_CONVENTION:
        raise ValueError(
            "manifest prime_index_convention must declare the exact canonical "
            "two-generator witness ordering"
        )
    rows = manifest.get("fields")
    if type(rows) is not list or not rows:
        raise ValueError("manifest fields must be a non-empty list")

    seen_ids: set[str] = set()
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("manifest fields entries must be objects")
        field_id = row.get("id")
        if type(field_id) is not str or not field_id:
            raise ValueError("manifest field id must be a non-empty string")
        if field_id in seen_ids:
            raise ValueError(f"duplicate manifest field id: {field_id}")
        seen_ids.add(field_id)

        coefficients = row.get("coefficients_low_to_high")
        if (
            type(coefficients) is not list
            or len(coefficients) < 2
            or len(coefficients) > MAX_WITNESS_DEGREE + 1
            or any(type(coefficient) is not int for coefficient in coefficients)
            or coefficients[-1] != 1
        ):
            raise ValueError(
                f"field {field_id} coefficients_low_to_high must be a "
                "monic nonconstant integer polynomial"
            )
        if selected_prime_selector_groups(row) is None:
            raise ValueError(f"field {field_id} selected_primes is invalid")

        expected = row.get("expected")
        if type(expected) is not dict or set(expected) != set(EXACT_RESULT_FIELDS):
            raise ValueError(
                f"field {field_id} expected must contain exactly: "
                + ", ".join(EXACT_RESULT_FIELDS)
            )
        for key in EXACT_RESULT_FIELDS:
            if not exact_result_value_is_valid(key, expected[key]):
                raise ValueError(
                    f"field {field_id} expected.{key} has an invalid value"
                )
        if not exact_rank_relation_is_valid(expected):
            raise ValueError(
                f"field {field_id} expected.free_rank must equal "
                "ordinary_free_rank + nonunit_rank"
            )
        if not exact_group_relations_are_valid(expected):
            raise ValueError(
                f"field {field_id} expected group invariants or torsion are invalid"
            )
        if validated_manifest_prime_witnesses(row) is None:
            raise ValueError(
                f"field {field_id} prime_ideal_witnesses or "
                "maximal_order_discriminant is invalid"
            )
        if hecke_certification_status(row) is None:
            raise ValueError(
                f"field {field_id} external_hecke_certification must be "
                "'proven' or 'grh'"
            )
        validated_rows.append(row)
    return validated_rows


def default_julia() -> str:
    resolved, _requested = resolve_julia_runtime()
    return resolved or "julia"


def load_field(path: Path, field_id: str) -> dict[str, Any]:
    manifest = read_json_nofollow(
        path,
        root=path.parent,
        max_bytes=MAX_SUNIT_MANIFEST_BYTES,
    )
    for row in validate_manifest(manifest):
        if row.get("id") == field_id:
            return row
    raise ValueError(f"unknown S-unit field id: {field_id}")


def polynomial_expr(coeffs: list[int], variable: str = "x") -> str:
    terms: list[tuple[int, str]] = []
    for power, coefficient in enumerate(coeffs):
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        if power == 0:
            term = str(magnitude)
        else:
            monomial = variable if power == 1 else f"{variable}^{power}"
            term = monomial if magnitude == 1 else f"{magnitude}*{monomial}"
        terms.append((1 if coefficient > 0 else -1, term))
    sign, term = terms[-1]
    result = term if sign > 0 else f"-{term}"
    for sign, term in reversed(terms[:-1]):
        result += (" + " if sign > 0 else " - ") + term
    return result


def rational_polynomial_expr(
    coefficients: list[str], variable: str, *, julia: bool = False
) -> str:
    terms: list[str] = []
    for power, coefficient in enumerate(coefficients):
        value = coefficient.replace("/", "//") if julia else coefficient
        terms.append(f"({value})*{variable}^{power}")
    return " + ".join(terms) if terms else "0"


def manifest_witness_selection(
    row: dict[str, Any],
) -> list[tuple[dict[str, Any], int | None]]:
    selectors_by_prime = selected_prime_selector_groups(row) or {}
    witnesses = validated_manifest_prime_witnesses(row) or []
    witnesses_by_prime: dict[int, list[dict[str, Any]]] = {}
    for witness in witnesses:
        witnesses_by_prime.setdefault(witness["p"], []).append(witness)
    selection_indices: dict[tuple[int, int], int] = {}
    next_index = 0
    for p, selectors in selectors_by_prime.items():
        complete = witnesses_by_prime[p]
        expanded = range(len(complete)) if selectors == ["all"] else selectors
        for canonical_index in expanded:
            selection_indices[(p, canonical_index)] = next_index
            next_index += 1
    return [
        (
            witness,
            selection_indices.get(
                (witness["p"], witness["canonical_index"])
            ),
        )
        for witness in witnesses
    ]


def parse_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            key, value = line.split("=", 1)
            if key and all(ch.isalnum() or ch == "_" for ch in key):
                if key in values:
                    return {}
                values[key] = value.strip()
    return values


def process_state_is_valid(process: Any) -> bool:
    return (
        type(process) is dict
        and type(process.get("available")) is bool
        and type(process.get("success")) is bool
        and type(process.get("timeout")) is bool
        and not (process["success"] and not process["available"])
        and not (process["success"] and process["timeout"])
    )


def nonempty_diagnostic(*values: Any, fallback: str) -> str:
    for value in values:
        if type(value) is str and value.strip():
            return value
    return fallback


def parse_int(values: dict[str, str], key: str) -> int | None:
    value = values.get(key)
    if value is None or re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value) is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(values: dict[str, str], key: str) -> float | None:
    try:
        parsed = float(values[key])
    except (KeyError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def pari_source_version(source: Path) -> str:
    version_file = source / "config/version"
    encoded = read_bytes_nofollow(
        version_file,
        root=source,
        max_bytes=MAX_PARI_VERSION_BYTES,
    )
    return parse_pari_source_version_bytes(encoded, version_file)


def parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def run_process(
    command: list[str],
    *,
    timeout: float,
    stdin: str | None = None,
    cpu: int | None = None,
    env: dict[str, str] | None = None,
    immutable_path_arguments: tuple[int, ...] = (),
) -> dict[str, Any]:
    return run_bounded_process(
        command,
        timeout=timeout,
        cwd=repo_root(),
        stdin=stdin,
        cpu=cpu,
        env=env,
        immutable_path_arguments=immutable_path_arguments,
    )


def first_outside_prime(row: dict[str, Any]) -> int:
    selected = {int(selector["p"]) for selector in row.get("selected_primes", [])}
    rational_prime = 2
    while rational_prime <= MAX_RATIONAL_PRIME:
        if rational_prime not in selected:
            if rational_prime_is_valid(rational_prime):
                return rational_prime
        rational_prime += 1
    raise ValueError("fixture leaves no rational prime for the outside-support audit")


def normalize_hnf_entries(matrix: list[list[Any]]) -> list[list[str]]:
    return [[str(entry) for entry in row] for row in matrix]


def normalize_silex_descriptors(value: Any) -> list[dict[str, Any]] | None:
    if type(value) is not list:
        return None
    selected: list[dict[str, Any]] = []
    for descriptor in value:
        if type(descriptor) is not dict:
            return None
        p = descriptor.get("p")
        canonical_index = descriptor.get("canonical_index")
        e = descriptor.get("e")
        f = descriptor.get("f")
        beta_power_basis = descriptor.get("beta_power_basis")
        hnf = descriptor.get("hnf")
        if (
            type(p) is not str
            or re.fullmatch(r"[1-9][0-9]*", p) is None
            or len(p) > 20
            or not rational_prime_is_valid(int(p))
            or type(canonical_index) is not int
            or canonical_index < 0
            or type(e) is not int
            or e < 1
            or type(f) is not int
            or f < 1
            or type(beta_power_basis) is not list
            or any(
                parse_canonical_rational(coefficient) is None
                for coefficient in beta_power_basis
            )
            or type(hnf) is not list
            or any(
                type(row) is not list
                or any(
                    type(entry) is not str
                    or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", entry) is None
                    or len(entry.lstrip("-")) > MAX_MATRIX_INTEGER_DIGITS
                    for entry in row
                )
                for row in hnf
            )
        ):
            return None
        selected.append(
            {
                "p": p,
                "canonical_index": canonical_index,
                "e": e,
                "f": f,
                "beta_power_basis": beta_power_basis,
                "hnf": hnf,
                "hnf_basis": "silex_maximal_order_basis_rows",
            }
        )
    return selected


def bind_descriptor_canonical_indices(
    descriptors: list[dict[str, Any]],
    canonical_decomposition: list[dict[str, Any]],
    row: Any,
) -> list[dict[str, Any]] | None:
    if type(row) is not dict or not selected_prime_specification_matches(
        descriptors, canonical_decomposition, row
    ):
        return None
    return descriptors


def normalize_silex(
    process: Any,
    payload: Any,
    build_dir: Path,
    silex_root: Path,
    row: Any = None,
) -> dict[str, Any]:
    process_payload = process if type(process) is dict else {}
    payload_object = payload if type(payload) is dict else {}
    raw_sunit = payload_object.get("sunit")
    sunit = raw_sunit if type(raw_sunit) is dict else {}
    raw_class_group = sunit.get("s_class_group")
    class_group = raw_class_group if type(raw_class_group) is dict else {}
    raw_unit_group = sunit.get("s_unit_group")
    unit_group = raw_unit_group if type(raw_unit_group) is dict else {}
    raw_membership = sunit.get("membership")
    membership = raw_membership if type(raw_membership) is dict else {}
    selected = normalize_silex_descriptors(sunit.get("selected_primes"))
    canonical_decomposition = normalize_silex_descriptors(
        sunit.get("canonical_prime_decompositions")
    )
    if selected is not None and canonical_decomposition is not None:
        if row is not None:
            selected = bind_descriptor_canonical_indices(
                selected, canonical_decomposition, row
            )
        elif selected or canonical_decomposition:
            selected = None
    prime_evidence_valid = selected == [] and canonical_decomposition == []
    if (
        type(row) is dict
        and selected is not None
        and canonical_decomposition is not None
        and (selected or canonical_decomposition)
    ):
        evidence_result = {
            "maximal_order_discriminant": payload_object.get(
                "maximal_order_discriminant"
            ),
            "maximal_order_basis_power": payload_object.get(
                "maximal_order_basis_power"
            ),
        }
        order_evidence = maximal_order_evidence(evidence_result, row)
        polynomial = row.get("coefficients_low_to_high")
        prime_evidence_valid = (
            order_evidence is not None
            and type(polynomial) is list
            and all(
                prime_ideal_lattice_matches_witness(
                    descriptor, polynomial, order_evidence[0], order_evidence[1]
                )
                for descriptor in canonical_decomposition
            )
            and all(
                prime_ideal_lattice_matches_witness(
                    descriptor, polynomial, order_evidence[0], order_evidence[1]
                )
                for descriptor in selected
            )
        )
    normalized_math = {
        "s_class_order": class_group.get("order"),
        "s_class_invariants": class_group.get("invariants"),
        "torsion_order": unit_group.get("torsion_order"),
        "ordinary_free_rank": unit_group.get("ordinary_free_rank"),
        "nonunit_rank": unit_group.get("nonunit_rank"),
        "free_rank": unit_group.get("free_rank"),
        "valuation_matrix": unit_group.get("valuation_matrix"),
        "valuation_lattice_index": unit_group.get("valuation_lattice_index"),
    }
    schema_valid = (
        process_state_is_valid(process)
        and type(payload) is dict
        and type(payload_object.get("success")) is bool
        and type(payload_object.get("timeout")) is bool
        and type(payload_object.get("final_result_published")) is bool
        and type(payload_object.get("phase_timing_ms")) is dict
        and type(payload_object.get("component_timing_ms")) is dict
        and (
            payload_object.get("failure_stage") is None
            or type(payload_object.get("failure_stage")) is str
        )
        and (
            payload_object.get("failure_reason") is None
            or type(payload_object.get("failure_reason")) is str
        )
        and type(raw_sunit) is dict
        and type(sunit.get("success")) is bool
        and type(sunit.get("timing_ms")) is dict
        and (
            sunit.get("failure_stage") is None
            or type(sunit.get("failure_stage")) is str
        )
        and (
            sunit.get("failure_reason") is None
            or type(sunit.get("failure_reason")) is str
        )
        and type(raw_class_group) is dict
        and type(raw_unit_group) is dict
        and type(raw_membership) is dict
        and type(membership.get("mixed_round_trip_verified")) is bool
        and type(membership.get("verified_round_trip_count")) is int
        and type(membership.get("mixed_outcome")) is str
        and type(membership.get("outside_support_rejected")) is bool
        and type(membership.get("outside_outcome")) is str
        and type(sunit.get("final_result_published")) is bool
        and all(
            exact_result_value_is_valid(key, normalized_math.get(key))
            for key in EXACT_RESULT_FIELDS
        )
        and exact_rank_relation_is_valid(normalized_math)
        and exact_group_relations_are_valid(normalized_math)
        and valuation_lattice_evidence_is_valid(normalized_math)
        and selected is not None
        and canonical_decomposition is not None
        and prime_evidence_valid
        and not (
            process_payload.get("success") is True
            and process_payload.get("timeout") is True
        )
        and not (
            payload_object.get("success") is True
            and payload_object.get("timeout") is True
        )
        and (
            payload_object.get("success") is not True
            or (
                payload_object.get("final_result_published") is True
                and sunit.get("success") is True
                and sunit.get("final_result_published") is True
                and membership.get("status") == "verified"
                and membership.get("mixed_round_trip_verified") is True
                and membership.get("verified_round_trip_count") == 1
                and membership.get("mixed_outcome") == "verified"
                and membership.get("outside_support_rejected") is True
                and membership.get("outside_outcome") == "not_sunit"
            )
        )
    )
    success = (
        schema_valid
        and process_payload.get("available") is True
        and process_payload.get("success") is True
        and process_payload.get("timeout") is False
        and payload_object.get("success") is True
        and payload_object.get("timeout") is False
        and sunit.get("success") is True
    )
    failure_stage = sunit.get("failure_stage") or payload_object.get("failure_stage")
    failure_reason = sunit.get("failure_reason") or payload_object.get(
        "failure_reason"
    )
    if not schema_valid:
        failure_stage = "invalid_silex_payload_schema"
        failure_reason = "invalid_silex_payload_schema"
    elif not success:
        failure_stage = nonempty_diagnostic(
            failure_stage, fallback="silex_native_failure"
        )
        failure_reason = nonempty_diagnostic(
            failure_reason, fallback="silex_native_failure"
        )
    return {
        "engine": "silex",
        "engine_identity": {
            "executable": str(
                (build_dir / "silex-class-unit-instance").resolve()
            ),
            "version": None,
            "source": str(silex_root.resolve()),
        },
        "available": process_payload.get("available") is True,
        "success": success,
        "timeout": process_payload.get("timeout") is True
        or payload_object.get("timeout") is True,
        "process_wall_ms": process_payload.get("process_wall_ms"),
        "effective_affinity": process_payload.get("effective_affinity"),
        "launcher_executable": process_payload.get("launcher_executable"),
        "launcher_executable_sha256": process_payload.get(
            "launcher_executable_sha256"
        ),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "certification_status": payload_object.get("certification_status"),
        "class_group_proof_status": payload_object.get("class_group_proof_status"),
        "unit_group_proof_status": payload_object.get("unit_group_proof_status"),
        "regulator_proof_status": payload_object.get("regulator_proof_status"),
        "selected_primes": selected or [],
        "canonical_prime_decompositions": canonical_decomposition or [],
        "maximal_order_discriminant": payload_object.get(
            "maximal_order_discriminant"
        ),
        "maximal_order_basis_power": payload_object.get(
            "maximal_order_basis_power"
        ),
        "s_class_order": class_group.get("order"),
        "s_class_invariants": class_group.get("invariants"),
        "s_class_proof_status": class_group.get("proof_status"),
        "torsion_order": unit_group.get("torsion_order"),
        "ordinary_free_rank": unit_group.get("ordinary_free_rank"),
        "nonunit_rank": unit_group.get("nonunit_rank"),
        "free_rank": unit_group.get("free_rank"),
        "valuation_matrix": unit_group.get("valuation_matrix"),
        "valuation_lattice_index": unit_group.get("valuation_lattice_index"),
        "regulator_midpoint": unit_group.get("regulator_midpoint"),
        "s_unit_proof_status": unit_group.get("proof_status"),
        "s_regulator_proof_status": unit_group.get("regulator_proof_status"),
        "membership_status": membership.get("status"),
        "mixed_round_trip_verified": membership.get("mixed_round_trip_verified")
        is True,
        "verified_round_trip_count": membership.get("verified_round_trip_count"),
        "mixed_outcome": membership.get("mixed_outcome"),
        "outside_support_rejected": membership.get("outside_support_rejected") is True,
        "outside_outcome": membership.get("outside_outcome"),
        "final_result_published": payload_object.get("final_result_published") is True
        and sunit.get("final_result_published") is True,
        "phase_timing_ms": payload_object.get("phase_timing_ms", {}),
        "sunit_timing_ms": sunit.get("timing_ms", {}),
        "component_timing_ms": payload_object.get("component_timing_ms", {}),
        "stderr": process_payload.get("stderr", ""),
    }


def parse_silex_json_output(output: Any) -> dict[str, Any]:
    if type(output) is not str:
        raise ValueError("Silex output must be text")
    if "\ufffd" in output:
        raise ValueError("Silex output contains invalid UTF-8")
    payload = parse_bounded_json_bytes(
        output.encode("utf-8"),
        source="Silex S-unit output",
        max_bytes=MAX_SILEX_JSON_OUTPUT_BYTES,
        max_depth=MAX_JSON_NESTING_DEPTH,
    )
    if type(payload) is not dict:
        raise ValueError("Silex output JSON root must be an object")
    return payload


def run_silex(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    executable = args.build_dir / "silex-class-unit-instance"
    command = [
        str(executable),
        "--coeffs",
        ",".join(str(value) for value in row["coefficients_low_to_high"]),
        "--mode",
        "proven",
        "--precision",
        "128",
        "--compute-sunit",
    ]
    for witness, selection_index in manifest_witness_selection(row):
        command.extend(
            [
                "--s-prime-witness",
                f"{witness['p']}:{witness['canonical_index']}:"
                f"{selection_index if selection_index is not None else -1}:"
                + ",".join(witness["beta_power_basis"]),
            ]
        )
    process = run_process(
        command,
        timeout=args.timeout,
        **(
            {"cpu": args.cpu}
            if getattr(args, "cpu", None) is not None
            else {}
        ),
    )
    if not process_state_is_valid(process):
        process_payload = process if type(process) is dict else {}
        result = dict(process_payload)
        result.update(
            {
                "available": process_payload.get("available") is True,
                "success": False,
                "timeout": process_payload.get("timeout") is True,
                "error": "invalid Silex process state",
            }
        )
        return result
    try:
        payload = parse_silex_json_output(process.get("stdout", ""))
    except (UnicodeError, ValueError) as exc:
        process["success"] = False
        process["error"] = f"invalid Silex JSON output: {exc}"
        return process
    if type(payload) is dict and "timeout" not in payload:
        payload["timeout"] = process.get("timeout")
    result = normalize_silex(process, payload, args.build_dir, args.silex_root, row)
    engine_identity = result.get("engine_identity")
    if type(engine_identity) is dict:
        engine_identity["executable_sha256"] = process.get("executable_sha256")
    if (
        type(process.get("executable_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", process["executable_sha256"]) is None
    ):
        result["success"] = False
        result["failure_stage"] = "engine_provenance"
        result["failure_reason"] = "Silex native execution identity is incomplete"
        result["final_result_published"] = False
    return result


def gp_selection_code(row: dict[str, Any]) -> str:
    lines = [
        "S = [];",
        "S_indices = [];",
        "S_betas = [];",
        "S_all = [];",
        "S_all_indices = [];",
        "S_all_betas = [];",
    ]
    witnesses_by_prime: dict[int, list[tuple[dict[str, Any], int | None]]] = {}
    for witness, selection_index in manifest_witness_selection(row):
        witnesses_by_prime.setdefault(witness["p"], []).append(
            (witness, selection_index)
        )
    selected_lines: list[tuple[int, list[str]]] = []
    for position, (rational_prime, witnesses) in enumerate(
        witnesses_by_prime.items(), start=1
    ):
        decomposition = f"D_{position}"
        lines.extend(
            [
                f"{decomposition} = idealprimedec(b, {rational_prime});",
                f"D_{position}_matched = [];",
            ]
        )
        for witness_position, (witness, selection_index) in enumerate(
            witnesses, start=1
        ):
            beta = rational_polynomial_expr(witness["beta_power_basis"], "x")
            match = f"D_{position}_match_{witness_position}"
            beta_name = f"D_{position}_beta_{witness_position}"
            lines.extend(
                [
                    f"{beta_name} = {beta};",
                    f"{match} = select(j -> idealhnf(b, {decomposition}[j]) == "
                    f"idealhnf(b, {rational_prime}, {beta_name}), "
                    f"vector(#{decomposition}, j, j));",
                    f"if(#{match} != 1, error(\"manifest prime witness did not "
                    "match exactly one PARI prime ideal\"));",
                    f"if(sum(j = 1, #D_{position}_matched, "
                    f"D_{position}_matched[j] == {match}[1]), "
                    "error(\"manifest prime witnesses are not distinct\"));",
                    f"D_{position}_matched = concat(D_{position}_matched, {match});",
                    f"S_all = concat(S_all, [{decomposition}[{match}[1]]]);",
                    f"S_all_indices = concat(S_all_indices, "
                    f"[{witness['canonical_index']}]);",
                    f"S_all_betas = concat(S_all_betas, [{beta_name}]);",
                ]
            )
            if selection_index is not None:
                selected_lines.append(
                    (
                        selection_index,
                        [
                            f"S = concat(S, [{decomposition}[{match}[1]]]);",
                            f"S_indices = concat(S_indices, "
                            f"[{witness['canonical_index']}]);",
                            f"S_betas = concat(S_betas, [{beta_name}]);",
                        ],
                    )
                )
        lines.append(
            f"if(#D_{position}_matched != #{decomposition}, "
            "error(\"manifest witnesses do not cover the PARI decomposition\"));"
        )
    for _, selected in sorted(selected_lines):
        lines.extend(selected)
    return "\n".join(lines)


def parse_external_descriptors(
    values: dict[str, str],
    basis_name: str,
    *,
    count_key: str = "selected_prime_count",
    prefix: str = "prime",
) -> list[dict[str, Any]]:
    count = parse_int(values, count_key) or 0
    descriptors: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        required = (
            f"{prefix}_{i}_p",
            f"{prefix}_{i}_canonical_index",
            f"{prefix}_{i}_e",
            f"{prefix}_{i}_f",
        )
        if any(key not in values for key in required):
            return []
        parsed_integers: list[int] = []
        try:
            for key in required[1:]:
                raw_value = values[key]
                if re.fullmatch(r"0|[1-9][0-9]*", raw_value) is None:
                    return []
                parsed_integers.append(int(raw_value))
        except ValueError:
            return []
        canonical_index, e, f = parsed_integers
        if e < 1 or f < 1:
            return []
        rows = parse_int(values, f"{prefix}_{i}_hnf_rows") or 0
        cols = parse_int(values, f"{prefix}_{i}_hnf_cols") or 0
        beta_count = parse_int(values, f"{prefix}_{i}_beta_count") or 0
        if beta_count != rows:
            return []
        try:
            beta = [
                values[f"{prefix}_{i}_beta_{coefficient}"]
                for coefficient in range(1, beta_count + 1)
            ]
            matrix = [
                [
                    values[f"{prefix}_{i}_hnf_{r}_{c}"]
                    for c in range(1, cols + 1)
                ]
                for r in range(1, rows + 1)
            ]
        except KeyError:
            return []
        descriptors.append(
            {
                "p": values[f"{prefix}_{i}_p"],
                "canonical_index": canonical_index,
                "e": e,
                "f": f,
                "beta_power_basis": beta,
                "hnf": matrix,
                "hnf_basis": basis_name,
            }
        )
    return descriptors


def parse_external_matrix(values: dict[str, str], prefix: str) -> list[list[str]]:
    rows = parse_int(values, f"{prefix}_rows") or 0
    cols = parse_int(values, f"{prefix}_cols") or 0
    try:
        return [
            [values[f"{prefix}_{r}_{c}"] for c in range(1, cols + 1)]
            for r in range(1, rows + 1)
        ]
    except KeyError:
        return []


def _argument_environment(args: argparse.Namespace) -> dict[str, str]:
    configured = getattr(args, "environment", None)
    if type(configured) is not dict:
        return {}
    return {str(key): str(value) for key, value in configured.items()}


def _configured_argument(
    args: argparse.Namespace,
    name: str,
    environment_name: str,
    *,
    default: str | None = None,
) -> str | None:
    value = getattr(args, name, None)
    if value is not None and str(value).strip():
        return str(value).strip()
    environment = _argument_environment(args)
    value = environment.get(environment_name) or os.environ.get(environment_name)
    if value is not None and str(value).strip():
        return str(value).strip()
    return default


def _resolve_command(requested: str) -> str | None:
    candidate = Path(requested).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(requested)


def run_pari(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    requested_gp = _configured_argument(
        args, "gp", "SILEX_BENCH_GP", default="gp"
    )
    assert requested_gp is not None
    gp = _resolve_command(requested_gp)
    if gp is None:
        return {
            "engine": "pari",
            "available": False,
            "success": False,
            "timeout": False,
            "failure_stage": "engine_provenance",
            "failure_reason": f"PARI executable not found: {requested_gp}",
        }
    deadline = time.monotonic() + float(args.timeout)
    version_process = run_process(
        [gp, "--version-short"],
        timeout=args.timeout,
        **(
            {"cpu": args.cpu}
            if getattr(args, "cpu", None) is not None
            else {}
        ),
    )
    if (
        not process_state_is_valid(version_process)
        or version_process["available"] is not True
        or version_process["success"] is not True
    ):
        return {
            "engine": "pari",
            "available": False,
            "success": False,
            "timeout": (
                type(version_process) is dict
                and version_process.get("timeout") is True
            ),
            "failure_stage": "engine_provenance",
            "failure_reason": "invalid PARI version process state",
        }
    version_lines = version_process.get("stdout", "").strip().splitlines()
    observed_version = version_lines[-1].strip() if version_lines else None
    if not observed_version:
        return {
            "engine": "pari",
            "available": False,
            "success": False,
            "timeout": False,
            "failure_stage": "engine_provenance",
            "failure_reason": "PARI/GP version probe returned no version",
        }
    configured_source = _configured_argument(
        args, "pari_source", "SILEX_BENCH_PARI_SOURCE"
    )
    source_path = (
        Path(os.path.abspath(os.fspath(Path(configured_source).expanduser())))
        if configured_source is not None
        else None
    )
    source_version = None
    if source_path is not None:
        try:
            source_version = pari_source_version(source_path)
        except (OSError, ValueError) as exc:
            return {
                "engine": "pari",
                "available": True,
                "success": False,
                "timeout": False,
                "failure_stage": "engine_provenance",
                "failure_reason": nonempty_diagnostic(
                    str(exc), fallback="PARI source provenance validation failed"
                ),
            }
    required_version = _configured_argument(
        args, "pari_version", "SILEX_BENCH_PARI_VERSION"
    )
    if required_version is None:
        required_version = source_version
    identity = {
        "executable": str(Path(gp).resolve()),
        "executable_sha256": version_process.get("executable_sha256"),
        "version": observed_version,
        "required_version": required_version,
        "source": str(source_path) if source_path is not None else None,
        "source_version": source_version,
    }
    provenance_errors = []
    if required_version is not None and observed_version != required_version:
        provenance_errors.append(
            f"PARI/GP executable version {observed_version!r} does not match "
            f"required version {required_version!r}"
        )
    if (
        source_version is not None
        and required_version is not None
        and source_version != required_version
    ):
        provenance_errors.append(
            f"PARI source version {source_version!r} does not match required "
            f"version {required_version!r}"
        )
    if provenance_errors:
        return {
            "engine": "pari",
            "engine_identity": identity,
            "available": True,
            "success": False,
            "timeout": False,
            "failure_stage": "engine_provenance",
            "failure_reason": "; ".join(provenance_errors),
        }
    polynomial = polynomial_expr(row["coefficients_low_to_high"])
    outside_prime = first_outside_prime(row)
    program = f"""
P = {polynomial};
gettime();
b = bnfinit(P, 1);
bnfinit_ms = gettime();
gettime();
certified = bnfcertify(b);
certification_ms = gettime();
gettime();
{gp_selection_code(row)}
prime_selection_ms = gettime();
gettime();
B = bnfsunit(b, S);
sunit_ms = gettime();
s = #S;
V = matrix(s, s, i, j, idealval(b, B[1][i], S[j]));
valuation_index = if(s, abs(matdet(V)), 1);
gettime();
mixed = b.tu[2];
for(i = 1, #b.fu, mixed *= b.fu[i]^i);
for(i = 1, #B[1], mixed *= B[1][i]^i);
mixed_coordinates = bnfissunit(b, B, mixed);
mixed_verified = (#mixed_coordinates == #B[1] + #b.fu + 1);
outside_rejected = (#bnfissunit(b, B, {outside_prime}) == 0);
membership_ms = gettime();
print("certified=", certified);
print("bnfinit_ms=", bnfinit_ms);
print("certification_ms=", certification_ms);
print("prime_selection_ms=", prime_selection_ms);
print("sunit_ms=", sunit_ms);
print("membership_ms=", membership_ms);
n = poldegree(P);
print("maximal_order_discriminant=", b.nf.disc);
print("maximal_order_basis_power_rows=", n);
print("maximal_order_basis_power_cols=", n);
for(r = 1, n, for(c = 1, n, print("maximal_order_basis_power_", r, "_", c, "=", polcoef(lift(b.nf.zk[r]), c - 1))));
print("selected_prime_count=", s);
for(i = 1, s, H = mattranspose(idealhnf(b, S[i])); print("prime_", i, "_p=", S[i][1]); print("prime_", i, "_canonical_index=", S_indices[i]); print("prime_", i, "_e=", S[i][3]); print("prime_", i, "_f=", S[i][4]); print("prime_", i, "_beta_count=", n); for(c = 1, n, print("prime_", i, "_beta_", c, "=", polcoef(S_betas[i], c - 1))); print("prime_", i, "_hnf_rows=", matsize(H)[1]); print("prime_", i, "_hnf_cols=", matsize(H)[2]); for(r = 1, matsize(H)[1], for(c = 1, matsize(H)[2], print("prime_", i, "_hnf_", r, "_", c, "=", H[r,c]))));
canonical_s = #S_all;
print("canonical_prime_count=", canonical_s);
for(i = 1, canonical_s, H = mattranspose(idealhnf(b, S_all[i])); print("canonical_prime_", i, "_p=", S_all[i][1]); print("canonical_prime_", i, "_canonical_index=", S_all_indices[i]); print("canonical_prime_", i, "_e=", S_all[i][3]); print("canonical_prime_", i, "_f=", S_all[i][4]); print("canonical_prime_", i, "_beta_count=", n); for(c = 1, n, print("canonical_prime_", i, "_beta_", c, "=", polcoef(S_all_betas[i], c - 1))); print("canonical_prime_", i, "_hnf_rows=", matsize(H)[1]); print("canonical_prime_", i, "_hnf_cols=", matsize(H)[2]); for(r = 1, matsize(H)[1], for(c = 1, matsize(H)[2], print("canonical_prime_", i, "_hnf_", r, "_", c, "=", H[r,c]))));
print("s_class_order=", B[5][1]);
print("s_class_invariants=", B[5][2]);
print("torsion_order=", b.tu[1]);
print("ordinary_free_rank=", #b.fu);
print("nonunit_rank=", #B[1]);
print("free_rank=", #b.fu + #B[1]);
print("valuation_rows=", s);
print("valuation_cols=", s);
for(i = 1, s, for(j = 1, s, print("valuation_", i, "_", j, "=", V[i,j])));
print("valuation_lattice_index=", valuation_index);
print("regulator_midpoint=", B[4]);
print("mixed_verified=", mixed_verified);
print("outside_rejected=", outside_rejected);
quit
"""
    remaining_timeout = deadline - time.monotonic()
    if remaining_timeout <= 0.0:
        return {
            "engine": "pari",
            "engine_identity": identity,
            "available": True,
            "success": False,
            "timeout": True,
            "failure_stage": "observation_deadline",
            "failure_reason": (
                "PARI observation deadline exhausted before target execution"
            ),
        }
    process = run_process(
        [gp, "-q"],
        timeout=remaining_timeout,
        stdin=program,
        **(
            {"cpu": args.cpu}
            if getattr(args, "cpu", None) is not None
            else {}
        ),
    )
    values = parse_values(process.get("stdout", ""))
    descriptors = parse_external_descriptors(
        values, "pari_bnf_integral_basis_rows"
    )
    maximal_order_basis = parse_external_matrix(
        values, "maximal_order_basis_power"
    )
    valuation_matrix = parse_external_matrix(values, "valuation")
    selected_count = parse_int(values, "selected_prime_count")
    canonical_count = parse_int(values, "canonical_prime_count")
    canonical_decomposition = parse_external_descriptors(
        values,
        "pari_bnf_integral_basis_rows",
        count_key="canonical_prime_count",
        prefix="canonical_prime",
    )
    success = (
        process_state_is_valid(process)
        and process.get("available") is True
        and process.get("success") is True
        and process.get("executable_sha256") == identity["executable_sha256"]
        and "***" not in process.get("stderr", "")
        and values.get("certified") == "1"
        and selected_count is not None
        and len(descriptors) == selected_count
        and canonical_count is not None
        and len(canonical_decomposition) == canonical_count
        and len(valuation_matrix) == selected_count
        and len(maximal_order_basis) == len(row["coefficients_low_to_high"]) - 1
        and values.get("maximal_order_discriminant")
        == row.get("maximal_order_discriminant")
    )
    return {
        "engine": "pari",
        "engine_identity": identity,
        "available": True,
        "success": success,
        "timeout": process.get("timeout") is True,
        "process_wall_ms": process.get("process_wall_ms"),
        "effective_affinity": process.get("effective_affinity"),
        "launcher_executable": process.get("launcher_executable"),
        "launcher_executable_sha256": process.get(
            "launcher_executable_sha256"
        ),
        "failure_stage": None if success else "external_execution",
        "failure_reason": None if success else "pari_bnfsunit_failed",
        "certification_status": "proven" if success else "unknown",
        "class_group_proof_status": "proven" if success else "unknown",
        "unit_group_proof_status": "proven" if success else "unknown",
        "regulator_proof_status": "proven" if success else "unknown",
        "selected_primes": descriptors,
        "canonical_prime_decompositions": canonical_decomposition,
        "maximal_order_discriminant": values.get(
            "maximal_order_discriminant"
        ),
        "maximal_order_basis_power": maximal_order_basis,
        "s_class_order": values.get("s_class_order"),
        "s_class_invariants": parse_list(values.get("s_class_invariants")),
        "s_class_proof_status": "verified" if success else "unknown",
        "torsion_order": values.get("torsion_order"),
        "ordinary_free_rank": parse_int(values, "ordinary_free_rank"),
        "nonunit_rank": parse_int(values, "nonunit_rank"),
        "free_rank": parse_int(values, "free_rank"),
        "valuation_matrix": valuation_matrix,
        "valuation_lattice_index": values.get("valuation_lattice_index"),
        "regulator_midpoint": parse_float(values, "regulator_midpoint"),
        "s_unit_proof_status": "verified" if success else "unknown",
        "s_regulator_proof_status": "verified" if success else "unknown",
        "membership_status": "verified"
        if values.get("mixed_verified") == "1"
        and values.get("outside_rejected") == "1"
        else "unknown",
        "mixed_round_trip_verified": values.get("mixed_verified") == "1",
        "outside_support_rejected": values.get("outside_rejected") == "1",
        "final_result_published": success,
        "phase_timing_ms": {
            "bnfinit": parse_float(values, "bnfinit_ms"),
            "certification": parse_float(values, "certification_ms"),
        },
        "sunit_timing_ms": {
            "prime_selection": parse_float(values, "prime_selection_ms"),
            "construction": parse_float(values, "sunit_ms"),
            "membership": parse_float(values, "membership_ms"),
        },
        "stderr": process.get("stderr", ""),
    }


def julia_selection_code(row: dict[str, Any]) -> str:
    lines = [
        "S = typeof(ideal(O, 1))[]",
        "S_indices = Int[]",
        "S_beta_text = Vector{String}[]",
        "S_all = typeof(ideal(O, 1))[]",
        "S_all_indices = Int[]",
        "S_all_beta_text = Vector{String}[]",
    ]
    witnesses_by_prime: dict[int, list[tuple[dict[str, Any], int | None]]] = {}
    for witness, selection_index in manifest_witness_selection(row):
        witnesses_by_prime.setdefault(witness["p"], []).append(
            (witness, selection_index)
        )
    selected_lines: list[tuple[int, list[str]]] = []
    for position, (rational_prime, witnesses) in enumerate(
        witnesses_by_prime.items(), start=1
    ):
        decomposition = f"D_{position}"
        lines.extend(
            [
                f"{decomposition} = [entry[1] for entry in "
                f"prime_decomposition(O, {rational_prime})]",
                f"D_{position}_matched = Int[]",
            ]
        )
        for witness_position, (witness, selection_index) in enumerate(
            witnesses, start=1
        ):
            beta = rational_polynomial_expr(
                witness["beta_power_basis"], "a", julia=True
            )
            match = f"D_{position}_matches_{witness_position}"
            beta_text = json.dumps(witness["beta_power_basis"])
            lines.extend(
                [
                    f"{match} = findall(P -> P == ideal(O, "
                    f"ZZ({rational_prime}), O({beta})), {decomposition})",
                    f"length({match}) == 1 || error(\"manifest prime witness "
                    "did not match exactly one Hecke prime ideal\")",
                    f"D_{position}_match_{witness_position} = only({match})",
                    f"D_{position}_match_{witness_position} in D_{position}_matched "
                    "&& error(\"manifest prime witnesses are not distinct\")",
                    f"push!(D_{position}_matched, "
                    f"D_{position}_match_{witness_position})",
                    f"push!(S_all, {decomposition}[D_{position}_match_"
                    f"{witness_position}])",
                    f"push!(S_all_indices, {witness['canonical_index']})",
                    f"push!(S_all_beta_text, {beta_text})",
                ]
            )
            if selection_index is not None:
                selected_lines.append(
                    (
                        selection_index,
                        [
                            f"push!(S, {decomposition}[D_{position}_match_"
                            f"{witness_position}])",
                            f"push!(S_indices, {witness['canonical_index']})",
                            f"push!(S_beta_text, {beta_text})",
                        ],
                    )
                )
        lines.append(
            f"length(D_{position}_matched) == length({decomposition}) || "
            "error(\"manifest witnesses do not cover the Hecke decomposition\")"
        )
    for _, selected in sorted(selected_lines):
        lines.extend(selected)
    return "\n".join(lines)


def run_hecke(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    environment_overrides = _argument_environment(args)
    julia, requested_julia = resolve_julia_runtime(
        getattr(args, "julia", None), environment=environment_overrides
    )
    if julia is None:
        return {
            "engine": "hecke",
            "available": False,
            "success": False,
            "timeout": False,
            "failure_stage": "engine_provenance",
            "failure_reason": f"Julia executable not found: {requested_julia}",
        }
    project, project_error = resolve_hecke_project(
        getattr(args, "hecke_project", None),
        environment=environment_overrides,
    )
    if project_error is not None:
        return {
            "engine": "hecke",
            "available": False,
            "success": False,
            "timeout": False,
            "failure_stage": "engine_provenance",
            "failure_reason": project_error,
        }
    polynomial = polynomial_expr(row["coefficients_low_to_high"])
    outside_prime = first_outside_prime(row)
    hecke_certification = hecke_certification_status(row)
    if hecke_certification is None:
        return {
            "engine": "hecke",
            "available": False,
            "success": False,
            "timeout": False,
            "error": "invalid external_hecke_certification",
        }
    grh = "true" if hecke_certification == "grh" else "false"
    environment = julia_environment(environment_overrides, julia=julia)
    code = f"""
using Hecke
Qx, x = polynomial_ring(QQ, "x")
function canonical_rational_text(value)
  denominator(value) == 1 && return string(numerator(value))
  return string(numerator(value), "/", denominator(value))
end
function ideal_basis_rows(P)
  elements = basis(P)
  n = length(elements)
  result = zero_matrix(ZZ, n, n)
  for row in 1:n
    coordinates_row = coordinates(elements[row])
    for column in 1:n
      result[row, column] = coordinates_row[column]
    end
  end
  return result
end
f = {polynomial}
function silex_compare_sunit(f)
t0 = time_ns()
K, a = number_field(f, "a")
O = maximal_order(K)
class_t0 = time_ns()
C, mC = class_group(O; GRH = {grh})
class_group_ms = (time_ns() - class_t0) / 1.0e6
unit_t0 = time_ns()
U, mU = unit_group_fac_elem(O; GRH = {grh})
unit_group_ms = (time_ns() - unit_t0) / 1.0e6
prime_t0 = time_ns()
{julia_selection_code(row)}
prime_selection_ms = (time_ns() - prime_t0) / 1.0e6
sunit_t0 = time_ns()
if isempty(S)
  Q = C
  G, mG = U, mU
else
  Q, mQ = quo(C, elem_type(C)[preimage(mC, P) for P in S], false)
  G, mG = sunit_group_fac_elem(S; GRH = {grh})
end
sunit_ms = (time_ns() - sunit_t0) / 1.0e6
s = length(S)
V = zero_matrix(ZZ, s, s)
gg = gens(G)
for i in 1:s, j in 1:s
  V[i, j] = valuation(image(mG, gg[ngens(U) + i]), S[j])
end
valuation_index = iszero(s) ? ZZ(1) : abs(det(V))
membership_t0 = time_ns()
mixed = zero(G)
for i in 1:ngens(G)
  mixed += i * gg[i]
end
mixed_verified = preimage(mG, image(mG, mixed)) == mixed
outside_element = FacElem(K({outside_prime}))
outside_factorization = factor(ideal(O, {outside_prime}))
outside_support_present = any(
  all(P != selected for selected in S)
  for (P, exponent) in outside_factorization if !iszero(exponent)
)
outside_query_rejected = false
try
  outside_coordinates = preimage(mG, outside_element)
  outside_query_rejected = !(image(mG, outside_coordinates) == outside_element)
catch error
  expected_nonmembership_error = error isa AssertionError || (
    error isa ErrorException &&
    error.msg == "Something wrong in conjugates_arb_log"
  )
  expected_nonmembership_error || rethrow(error)
  outside_query_rejected = true
end
outside_rejected = outside_support_present && outside_query_rejected
membership_ms = (time_ns() - membership_t0) / 1.0e6
invariants = [d for d in elementary_divisors(Q) if d > 1]
unit_divisors = [d for d in elementary_divisors(U) if !iszero(d)]
torsion_order = first(unit_divisors)
ordinary_free_rank = signature(K)[1] + signature(K)[2] - 1
ordinary_regulator = regulator(O; GRH = {grh})
sregulator = Float64(ordinary_regulator)
if !isempty(S)
  sregulator *= Float64(order(Q))
  for P in S
    sregulator *= log(Float64(norm(P)))
  end
end
field_degree = degree(K)
order_basis = basis(O)
println("maximal_order_discriminant=", discriminant(O))
println("maximal_order_basis_power_rows=", field_degree)
println("maximal_order_basis_power_cols=", field_degree)
for row in 1:field_degree, column in 1:field_degree
  value = coeff(elem_in_nf(order_basis[row]), column - 1)
  println("maximal_order_basis_power_", row, "_", column, "=", canonical_rational_text(value))
end
println("selected_prime_count=", s)
for i in 1:s
  H = ideal_basis_rows(S[i])
  println("prime_", i, "_p=", minimum(S[i]))
  println("prime_", i, "_canonical_index=", S_indices[i])
  println("prime_", i, "_e=", ramification_index(S[i]))
  println("prime_", i, "_f=", degree(S[i]))
  println("prime_", i, "_beta_count=", field_degree)
  for coefficient in 1:field_degree
    println("prime_", i, "_beta_", coefficient, "=", S_beta_text[i][coefficient])
  end
  println("prime_", i, "_hnf_rows=", nrows(H))
  println("prime_", i, "_hnf_cols=", ncols(H))
  for r in 1:nrows(H), c in 1:ncols(H)
    println("prime_", i, "_hnf_", r, "_", c, "=", H[r,c])
  end
end
canonical_s = length(S_all)
println("canonical_prime_count=", canonical_s)
for i in 1:canonical_s
  H = ideal_basis_rows(S_all[i])
  println("canonical_prime_", i, "_p=", minimum(S_all[i]))
  println("canonical_prime_", i, "_canonical_index=", S_all_indices[i])
  println("canonical_prime_", i, "_e=", ramification_index(S_all[i]))
  println("canonical_prime_", i, "_f=", degree(S_all[i]))
  println("canonical_prime_", i, "_beta_count=", field_degree)
  for coefficient in 1:field_degree
    println("canonical_prime_", i, "_beta_", coefficient, "=", S_all_beta_text[i][coefficient])
  end
  println("canonical_prime_", i, "_hnf_rows=", nrows(H))
  println("canonical_prime_", i, "_hnf_cols=", ncols(H))
  for r in 1:nrows(H), c in 1:ncols(H)
    println("canonical_prime_", i, "_hnf_", r, "_", c, "=", H[r,c])
  end
end
println("s_class_order=", order(Q))
println("s_class_invariants=", join(string.(invariants), ","))
println("torsion_order=", torsion_order)
println("ordinary_free_rank=", ordinary_free_rank)
println("nonunit_rank=", s)
println("free_rank=", ordinary_free_rank + s)
println("valuation_rows=", s)
println("valuation_cols=", s)
for i in 1:s, j in 1:s
  println("valuation_", i, "_", j, "=", V[i,j])
end
println("valuation_lattice_index=", valuation_index)
println("regulator_midpoint=", sregulator)
println("mixed_verified=", mixed_verified ? 1 : 0)
println("outside_rejected=", outside_rejected ? 1 : 0)
println("class_group_ms=", class_group_ms)
println("unit_group_ms=", unit_group_ms)
println("prime_selection_ms=", prime_selection_ms)
println("sunit_ms=", sunit_ms)
println("membership_ms=", membership_ms)
println("total_ms=", (time_ns() - t0) / 1.0e6)
println("julia_version=", VERSION)
println("hecke_version=", Base.pkgversion(Hecke))
println("hecke_source=", pathof(Hecke))
end
silex_compare_sunit(f)
"""
    process = run_process(
        [*julia_command(julia, project), "-e", code],
        timeout=args.timeout,
        env=environment,
        **(
            {"cpu": args.cpu}
            if getattr(args, "cpu", None) is not None
            else {}
        ),
    )
    values = parse_values(process.get("stdout", ""))
    selected = parse_external_descriptors(
        values, "hecke_maximal_order_basis_rows"
    )
    maximal_order_basis = parse_external_matrix(
        values, "maximal_order_basis_power"
    )
    canonical_decomposition = parse_external_descriptors(
        values,
        "hecke_maximal_order_basis_rows",
        count_key="canonical_prime_count",
        prefix="canonical_prime",
    )
    selected_count = parse_int(values, "selected_prime_count")
    canonical_count = parse_int(values, "canonical_prime_count")
    success = (
        process_state_is_valid(process)
        and process.get("available") is True
        and process.get("success") is True
        and type(process.get("executable_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", process["executable_sha256"]) is not None
        and "s_class_order" in values
        and selected_count is not None
        and len(selected) == selected_count
        and canonical_count is not None
        and len(canonical_decomposition) == canonical_count
        and len(maximal_order_basis) == len(row["coefficients_low_to_high"]) - 1
        and values.get("maximal_order_discriminant")
        == row.get("maximal_order_discriminant")
    )
    return {
        "engine": "hecke",
        "engine_identity": {
            "executable": str(Path(julia).resolve()),
            "executable_sha256": process.get("executable_sha256"),
            "version": values.get("julia_version"),
            "package_version": values.get("hecke_version"),
            "source": values.get("hecke_source"),
        },
        "available": True,
        "success": success,
        "timeout": process.get("timeout") is True,
        "process_wall_ms": process.get("process_wall_ms"),
        "effective_affinity": process.get("effective_affinity"),
        "launcher_executable": process.get("launcher_executable"),
        "launcher_executable_sha256": process.get(
            "launcher_executable_sha256"
        ),
        "failure_stage": None if success else "external_execution",
        "failure_reason": None if success else hecke_load_error(process, project),
        "certification_status": hecke_certification if success else "unknown",
        "class_group_proof_status": hecke_certification if success else "unknown",
        "unit_group_proof_status": hecke_certification if success else "unknown",
        "regulator_proof_status": hecke_certification if success else "unknown",
        "selected_primes": selected,
        "canonical_prime_decompositions": canonical_decomposition,
        "maximal_order_discriminant": values.get(
            "maximal_order_discriminant"
        ),
        "maximal_order_basis_power": maximal_order_basis,
        "s_class_order": values.get("s_class_order"),
        "s_class_invariants": parse_list(values.get("s_class_invariants")),
        "s_class_proof_status": "verified" if success else "unknown",
        "torsion_order": values.get("torsion_order"),
        "ordinary_free_rank": parse_int(values, "ordinary_free_rank"),
        "nonunit_rank": parse_int(values, "nonunit_rank"),
        "free_rank": parse_int(values, "free_rank"),
        "valuation_matrix": parse_external_matrix(values, "valuation"),
        "valuation_lattice_index": values.get("valuation_lattice_index"),
        "regulator_midpoint": parse_float(values, "regulator_midpoint"),
        "s_unit_proof_status": "verified" if success else "unknown",
        "s_regulator_proof_status": "verified" if success else "unknown",
        "membership_status": "verified"
        if values.get("mixed_verified") == "1"
        and values.get("outside_rejected") == "1"
        else "unknown",
        "mixed_round_trip_verified": values.get("mixed_verified") == "1",
        "outside_support_rejected": values.get("outside_rejected") == "1",
        "final_result_published": success,
        "phase_timing_ms": {
            "class_group": parse_float(values, "class_group_ms"),
            "unit_group": parse_float(values, "unit_group_ms"),
        },
        "sunit_timing_ms": {
            "prime_selection": parse_float(values, "prime_selection_ms"),
            "construction": parse_float(values, "sunit_ms"),
            "membership": parse_float(values, "membership_ms"),
        },
        "component_timing_ms": {"total": parse_float(values, "total_ms")},
        "stderr": process.get("stderr", ""),
    }


def manifest_matches(result: dict[str, Any], row: dict[str, Any]) -> bool:
    actual = {
        "s_class_order": result.get("s_class_order"),
        "s_class_invariants": result.get("s_class_invariants"),
        "torsion_order": result.get("torsion_order"),
        "ordinary_free_rank": result.get("ordinary_free_rank"),
        "nonunit_rank": result.get("nonunit_rank"),
        "free_rank": result.get("free_rank"),
        "valuation_lattice_index": result.get("valuation_lattice_index"),
    }
    return all(actual.get(key) == value for key, value in row["expected"].items())


def build_agreement(backends: Any, row: Any) -> dict[str, Any]:
    backend_items = list(backends.items()) if type(backends) is dict else []
    backend_shapes_valid = bool(backend_items) and all(
        type(slot) is str and bool(slot) and type(result) is dict
        for slot, result in backend_items
    )
    successful_items = [
        (slot, result)
        for slot, result in backend_items
        if type(result) is dict and result.get("success") is True
    ]
    successful = [result for _, result in successful_items]
    backend_identities_valid = backend_shapes_valid and all(
        standalone_engine_identity_is_valid(expected_engine_for_slot(slot), result)
        for slot, result in backend_items
    )
    row_payload = row if type(row) is dict else {}
    try:
        validate_manifest(
            {
                "schema_version": SUNIT_MANIFEST_SCHEMA_VERSION,
                "prime_index_convention": PRIME_INDEX_CONVENTION,
                "fields": [row_payload],
            }
        )
    except ValueError:
        row_contract_valid = False
    else:
        row_contract_valid = True
    fields: dict[str, bool] = {}
    for key in EXACT_RESULT_FIELDS:
        valid_values = [
            result[key]
            for result in successful
            if key in result and exact_result_value_is_valid(key, result[key])
        ]
        fields[key] = (
            bool(successful)
            and len(valid_values) == len(successful)
            and len(
                {json.dumps(value, sort_keys=True) for value in valid_values}
            )
            <= 1
        )
    fields["free_rank"] = fields["free_rank"] and all(
        exact_rank_relation_is_valid(result) for result in successful
    )
    fields["group_relations"] = bool(successful) and all(
        exact_group_relations_are_valid(result) for result in successful
    )
    fields["valuation_lattice_evidence"] = bool(successful) and all(
        valuation_lattice_evidence_is_valid(result) for result in successful
    )
    coefficients = row_payload.get("coefficients_low_to_high")
    degree = len(coefficients) - 1 if type(coefficients) is list else 0
    descriptor_sets = {
        slot: validated_prime_descriptors(result, degree)
        for slot, result in successful_items
    }
    decomposition_sets = {
        slot: validated_prime_descriptors(
            result, degree, "canonical_prime_decompositions"
        )
        for slot, result in successful_items
    }
    descriptors_complete = bool(successful_items) and all(
        descriptors is not None for descriptors in descriptor_sets.values()
    ) and all(
        decomposition is not None for decomposition in decomposition_sets.values()
    )
    rational_descriptors = [
        sorted(
            witness_identity(item)
            for item in descriptors
        )
        for descriptors in descriptor_sets.values()
        if descriptors is not None
    ]
    fields["selected_prime_rational_data"] = (
        descriptors_complete
        and len({json.dumps(value) for value in rational_descriptors}) <= 1
    )
    canonical_rational_descriptors = [
        sorted(
            witness_identity(item)
            for item in decomposition
        )
        for decomposition in decomposition_sets.values()
        if decomposition is not None
    ]
    fields["canonical_prime_rational_data"] = (
        descriptors_complete
        and len(
            {json.dumps(value) for value in canonical_rational_descriptors}
        )
        <= 1
    )
    fields["selected_prime_specification"] = descriptors_complete and all(
        selected_prime_specification_matches(
            descriptors,
            decomposition_sets[slot] or [],
            row_payload,
        )
        for slot, descriptors in descriptor_sets.items()
        if descriptors is not None and decomposition_sets[slot] is not None
    )
    prime_ideal_witnesses_valid = descriptors_complete
    if prime_ideal_witnesses_valid:
        for slot, result in successful_items:
            descriptors = descriptor_sets[slot]
            decomposition = decomposition_sets[slot]
            if descriptors is None or decomposition is None:
                prime_ideal_witnesses_valid = False
                break
            if not descriptors and not decomposition:
                continue
            order_evidence = maximal_order_evidence(result, row_payload)
            if order_evidence is None or type(coefficients) is not list:
                prime_ideal_witnesses_valid = False
                break
            if not all(
                prime_ideal_lattice_matches_witness(
                    descriptor,
                    coefficients,
                    order_evidence[0],
                    order_evidence[1],
                )
                for descriptor in decomposition + descriptors
            ):
                prime_ideal_witnesses_valid = False
                break
    fields["prime_ideal_witnesses"] = prime_ideal_witnesses_valid
    silex_hnf_descriptors = [
        [
            (
                item["p"],
                item["canonical_index"],
                item["e"],
                item["f"],
                item["hnf"],
            )
            for item in descriptors
        ]
        for slot, descriptors in descriptor_sets.items()
        if expected_engine_for_slot(slot) == "silex"
        and descriptors is not None
    ]
    fields["silex_prime_hnf"] = descriptors_complete and len(
        {json.dumps(value, sort_keys=True) for value in silex_hnf_descriptors}
    ) <= 1
    regulator_values = [
        result["regulator_midpoint"]
        for result in successful
        if result.get("regulator_midpoint") is not None
    ]
    regulators: list[float] = []
    regulators_valid = True
    for value in regulator_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            regulators_valid = False
            continue
        try:
            regulator = float(value)
        except OverflowError:
            regulators_valid = False
            continue
        if not math.isfinite(regulator):
            regulators_valid = False
            continue
        regulators.append(regulator)
    regulator_agrees = regulators_valid and (
        not regulators
        or all(
            math.isclose(regulators[0], value, rel_tol=1e-10, abs_tol=1e-12)
            for value in regulators[1:]
        )
    )
    membership_verified = all(
        result.get("membership_status") == "verified"
        and result.get("mixed_round_trip_verified") is True
        and result.get("outside_support_rejected") is True
        and (
            expected_engine_for_slot(slot) != "silex"
            or (
                result.get("verified_round_trip_count") == 1
                and result.get("mixed_outcome") == "verified"
                and result.get("outside_outcome") == "not_sunit"
            )
        )
        for slot, result in successful_items
    )
    expected_hecke_certification = hecke_certification_status(row_payload)
    hecke_certification_valid = expected_hecke_certification is not None
    ordinary_proof_states = [
        (
            expected_engine_for_slot(slot),
            result,
            expected_hecke_certification
            if expected_engine_for_slot(slot) == "hecke"
            else "proven",
        )
        for slot, result in successful_items
    ]
    proof_complete = (
        backend_identities_valid
        and hecke_certification_valid
        and bool(ordinary_proof_states)
        and all(
            result.get("certification_status") == ordinary_status
            and result.get("class_group_proof_status") == ordinary_status
            and result.get("unit_group_proof_status") == ordinary_status
            and result.get("regulator_proof_status")
            == ("verified" if engine == "silex" else ordinary_status)
            and result.get("s_class_proof_status") == "verified"
            and result.get("s_unit_proof_status") == "verified"
            and result.get("s_regulator_proof_status") == "verified"
            and result.get("final_result_published") is True
            for engine, result, ordinary_status in ordinary_proof_states
        )
    )
    manifest_ok = row_contract_valid and all(
        manifest_matches(result, row_payload) for result in successful
    )
    return {
        "fields": fields,
        "backend_results_agree": row_contract_valid
        and all(fields.values())
        and regulator_agrees,
        "regulator_agrees": regulator_agrees,
        "backend_identities_valid": backend_identities_valid,
        "membership_verified": membership_verified,
        "proof_complete": proof_complete,
        "manifest_contract_valid": row_contract_valid,
        "manifest_expectations_match": manifest_ok,
        "success": bool(successful)
        and len(successful) == len(backend_items)
        and backend_identities_valid
        and all(fields.values())
        and regulator_agrees
        and membership_verified
        and proof_complete
        and manifest_ok
    }


def write_result(result: dict[str, Any], out: Path | None) -> None:
    if out is None:
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        sys.stdout.write(text)
    else:
        ensure_absolute_directory_nofollow(out.parent)
        atomic_write_json(out, result, root=out.parent)


def main() -> int:
    workspace = default_workspace()
    silex_root = workspace / "silex"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=builtin_path("corpora", "sunit_fields")
    )
    parser.add_argument("--field-id", required=True)
    parser.add_argument(
        "--backend", action="append", choices=["silex", "pari", "hecke"]
    )
    parser.add_argument(
        "--silex-root", type=Path, default=silex_root
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=silex_root / "build/benchmark-adapters",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--gp")
    parser.add_argument("--pari-version")
    parser.add_argument("--pari-source", type=Path)
    parser.add_argument("--julia")
    parser.add_argument(
        "--hecke-project",
        help="Julia project override; default uses the active depot environment",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        row = load_field(args.manifest, args.field_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        write_result({"success": False, "error": str(exc)}, args.out)
        return 2
    selected_backends = args.backend or ["silex", "pari", "hecke"]
    backends: dict[str, dict[str, Any]] = {}
    if "silex" in selected_backends:
        backends["silex"] = run_silex(args, row)
    if "pari" in selected_backends:
        backends["pari:external"] = run_pari(args, row)
    if "hecke" in selected_backends:
        backends["hecke:external"] = run_hecke(args, row)

    agreement = build_agreement(backends, row)
    result = {
        "schema_version": SUNIT_MANIFEST_SCHEMA_VERSION,
        "success": agreement["success"],
        "field": {
            "id": row["id"],
            "coefficients_low_to_high": row["coefficients_low_to_high"],
            "selected_prime_specification": row.get("selected_primes", []),
        },
        "engine_schema": "pari|hecke|silex",
        "agreement": agreement,
        "backends": backends,
    }
    write_result(result, args.out)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
