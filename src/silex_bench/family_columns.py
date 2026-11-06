"""Per-family PARI reference sub-columns for the class/unit comparison.

The headline compares Silex with PARI's bnfinit (plus bnfcertify on the
proven route) on every row and is unchanged.  Beside it, quadratic fields are
also compared with PARI's quadclassunit, which is much faster but returns
less (PARI usersch3.tex, quadclassunit): exactly for imaginary quadratic
fields (class group only, no units to omit), and labelled "no unit" for real
quadratic fields.  Cyclotomic fields of degree above 2 are compared with the
K+ route (galoissubcyclo, bnfinit(K+), subcyclohminus, plus bnfcertify(K+) on
the proven route), labelled partial: it gives the class number h only, with
no class group structure or generators, and one unit is missing when the unit
index Q is 2.  The PARI driver times that call by repetition
(backends/pari.py), so sub-millisecond readings are not clipped at the 1 ms
getwalltime resolution.  Further families add an entry to
FAMILY_COLUMNS.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

CLASS_UNIT_WORKLOADS = ("class_unit_proven", "class_unit_grh")
REFERENCE_KEY = "family_reference"


def _quadratic_sign(metrics: Mapping[str, Any]) -> int | None:
    degree = metrics.get("degree")
    disc = metrics.get("maximal_order_discriminant")
    if degree != 2 or type(disc) is not int or disc == 0:
        return None
    return 1 if disc > 0 else -1


KPLUS_REFERENCE = "cyclotomic_kplus"
_DEFAULT_REFERENCE = "quadclassunit"
_REFERENCE_DISPLAY = {KPLUS_REFERENCE: "galoissubcyclo + bnfinit(K+) + subcyclohminus (+ bnfcertify(K+) proven)"}

# key, label, reference name, row predicate.  The reference name also selects
# the reading: a row's PARI sample records which reference it timed.
FAMILY_COLUMNS: tuple[tuple[str, str, str, Callable[[Mapping[str, Any]], bool]], ...] = (
    (
        "imaginary_quadratic",
        "imaginary quadratic vs quadclassunit (exact)",
        "quadclassunit",
        lambda metrics: _quadratic_sign(metrics) == -1,
    ),
    (
        "real_quadratic",
        "real quadratic vs quadclassunit (no unit)",
        "quadclassunit",
        lambda metrics: _quadratic_sign(metrics) == 1,
    ),
    (
        "cyclotomic",
        "cyclotomic vs K+ route (partial: h only, no structure or generators; "
        "one unit missing when Q=2)",
        KPLUS_REFERENCE,
        lambda metrics: True,
    ),
)


def reference_name(internal_timing: Any) -> str:
    """Name of the family reference a PARI sample timed (quadclassunit if absent)."""

    repeat = (
        internal_timing.get("repeat_timing")
        if isinstance(internal_timing, Mapping)
        else None
    )
    reference = repeat.get(REFERENCE_KEY) if isinstance(repeat, Mapping) else None
    name = (
        reference.get("family_reference_name")
        if isinstance(reference, Mapping)
        else None
    )
    return name if isinstance(name, str) else _DEFAULT_REFERENCE


def reference_ms(internal_timing: Any) -> float | None:
    """Repeat-mode family reference time (ms per call) from a PARI sample."""

    if not isinstance(internal_timing, Mapping):
        return None
    repeat = internal_timing.get("repeat_timing")
    reference = repeat.get(REFERENCE_KEY) if isinstance(repeat, Mapping) else None
    if not isinstance(reference, Mapping):
        return None
    value = reference.get("repeat_ms")
    repeats = reference.get("repeats")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or type(repeats) is not int
        or repeats < 1
    ):
        return None
    return float(value)


def geomean(values: Iterable[float]) -> float | None:
    logs = [math.log(value) for value in values if value > 0]
    return math.exp(sum(logs) / len(logs)) if logs else None


def family_columns(
    timings: list[Mapping[str, Any]],
    references: Mapping[tuple[str, str], list[float]],
    reference_names: Mapping[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """Return one headline plus per-family rows for each class/unit workload.

    `timings` are the report's per-case timing rows (median Silex and PARI
    wall times); `references` maps (workload, case_key) to PARI family
    reference readings.  Ratios are Silex median over PARI time, so below 1
    means Silex is faster.  The headline pairs Silex with PARI's single-call
    bnfinit-route target time exactly as the per-case table reports it.
    `reference_names` maps the same keys to the reference each row timed
    (default quadclassunit); a column counts only rows whose reference matches.
    """

    medians: dict[tuple[str, str, str], tuple[float, Mapping[str, Any]]] = {}
    # Rows that differ only in timing scope or wall clock stay separate in the
    # per-case table; a key seen twice is ambiguous here and is left out.
    ambiguous: set[tuple[str, str, str]] = set()
    for row in timings:
        if row.get("workload") not in CLASS_UNIT_WORKLOADS:
            continue
        if row.get("variant", "standard") != "standard":
            continue
        median = row["statistics"].get("median")
        if isinstance(median, (int, float)) and median > 0:
            key = (row["workload"], row["case_key"], row["backend"])
            if key in medians:
                ambiguous.add(key)
            medians[key] = (float(median), row.get("metrics") or {})
    for key in ambiguous:
        del medians[key]
    output: list[dict[str, Any]] = []
    for workload in CLASS_UNIT_WORKLOADS:
        cases = sorted(
            {key[1] for key in medians if key[0] == workload}
        )
        if not cases:
            continue
        headline = [
            medians[(workload, case, "silex")][0] / medians[(workload, case, "pari")][0]
            for case in cases
            if (workload, case, "silex") in medians
            and (workload, case, "pari") in medians
        ]
        output.append(
            {
                "workload": workload,
                "column": "headline",
                "label": "all rows vs bnfinit"
                + (" + bnfcertify" if workload == "class_unit_proven" else ""),
                "reference": "bnfinit",
                "rows": len(headline),
                "geomean": geomean(headline),
            }
        )
        for key, label, reference, predicate in FAMILY_COLUMNS:
            ratios: list[float] = []
            for case in cases:
                silex = medians.get((workload, case, "silex"))
                readings = references.get((workload, case))
                if silex is None or not readings or not predicate(silex[1]):
                    continue
                if (reference_names or {}).get(
                    (workload, case), _DEFAULT_REFERENCE
                ) != reference:
                    continue
                ordered = sorted(readings)
                middle = len(ordered) // 2
                ref = (
                    ordered[middle]
                    if len(ordered) % 2
                    else (ordered[middle - 1] + ordered[middle]) / 2
                )
                ratios.append(silex[0] / ref)
            output.append(
                {
                    "workload": workload,
                    "column": key,
                    "label": label,
                    "reference": reference,
                    "rows": len(ratios),
                    "geomean": geomean(ratios),
                }
            )
    return output


def markdown_lines(columns: list[Mapping[str, Any]], workload: str) -> list[str]:
    """Table lines for one workload, empty when it has no columns."""

    rows = [column for column in columns if column["workload"] == workload]
    if not rows:
        return []
    lines = [
        "| Column | Reference | Rows | Geomean Silex/PARI |",
        "|---|---|---:|---:|",
    ]
    for column in rows:
        value = column["geomean"]
        lines.append(
            f"| {column['label']} | "
            f"{_REFERENCE_DISPLAY.get(column['reference'], column['reference'])} | {column['rows']} | "
            + ("-" if value is None else f"{value:.3f}")
            + " |"
        )
    return lines
