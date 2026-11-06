"""Per-family PARI reference sub-columns for the class/unit comparison.

The headline compares Silex with PARI's bnfinit (plus bnfcertify on the
proven route) on every row and is unchanged.  Beside it, quadratic fields are
also compared with PARI's quadclassunit, which is much faster but returns
less (PARI usersch3.tex, quadclassunit): exactly for imaginary quadratic
fields (class group only, no units to omit), and labelled "no unit" for real
quadratic fields.  The PARI driver times that call by repetition
(backends/pari.py), so sub-millisecond readings are not clipped at the 1 ms
getwalltime resolution.  Further families (the cyclotomic K+ route) add an
entry to FAMILY_COLUMNS.
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


# key, label, reference name, row predicate
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
)


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
) -> list[dict[str, Any]]:
    """Return one headline plus per-family rows for each class/unit workload.

    `timings` are the report's per-case timing rows (median Silex and PARI
    wall times); `references` maps (workload, case_key) to PARI family
    reference readings.  Ratios are Silex median over PARI time, so below 1
    means Silex is faster.  The headline pairs Silex with PARI's single-call
    bnfinit-route target time exactly as the per-case table reports it.
    """

    medians: dict[tuple[str, str, str], tuple[float, Mapping[str, Any]]] = {}
    for row in timings:
        if row.get("workload") not in CLASS_UNIT_WORKLOADS:
            continue
        if row.get("variant", "standard") != "standard":
            continue
        median = row["statistics"].get("median")
        if isinstance(median, (int, float)) and median > 0:
            medians[(row["workload"], row["case_key"], row["backend"])] = (
                float(median),
                row.get("metrics") or {},
            )
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
            f"| {column['label']} | {column['reference']} | {column['rows']} | "
            + ("-" if value is None else f"{value:.3f}")
            + " |"
        )
    return lines
