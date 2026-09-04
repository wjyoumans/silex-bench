"""Dependency-free, deterministic terminal presentation for Silex Bench."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render one compact ASCII table without terminal-dependent styling."""

    materialized = [[_text(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in materialized:
        if len(row) != len(headers):
            raise ValueError("table rows must match the header width")
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "  ".join(value.ljust(width) for value, width in zip(headers, widths))
    rule = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in materialized
    ]
    return "\n".join((header, rule, *body))


def render_list(payload: Mapping[str, Any]) -> str:
    workloads = list(payload.get("workloads", []))
    backends = list(payload.get("backends", []))
    workload_ids = [str(row.get("id")) for row in workloads]
    lines = ["Suites", "  " + ", ".join(payload.get("suites", [])), "", "Profiles", "  " + ", ".join(payload.get("profiles", [])), ""]
    lines.extend(
        (
            "Workloads",
            table(
                ("ID", "Name", "Timing scope", "Scale axes"),
                (
                    (
                        row.get("id"),
                        row.get("display_name"),
                        row.get("timing_scope"),
                        ", ".join(row.get("scale_axes", [])),
                    )
                    for row in workloads
                ),
            ),
            "",
            "Adapter capabilities",
            table(
                ("Adapter", *workload_ids),
                (
                    (
                        row.get("display_name", row.get("id")),
                        *(
                            "yes" if workload in set(row.get("capabilities", [])) else "unsupported"
                            for workload in workload_ids
                        ),
                    )
                    for row in backends
                ),
            ),
        )
    )
    return "\n".join(lines)


def _seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:g}s"


def render_plan(payload: Mapping[str, Any]) -> str:
    execution = payload.get("execution", {})
    pairs = payload.get("required_pairs", [])
    pair_text = ", ".join(f"{lhs}:{rhs}" for lhs, rhs in pairs) or "none"
    rows = (
        ("Suite", payload.get("suite")),
        ("Profile", payload.get("profile")),
        ("Mode", payload.get("mode")),
        ("Workloads", ", ".join(payload.get("workloads", []))),
        ("Adapters", ", ".join(payload.get("backends", []))),
        ("Cases", payload.get("case_count")),
        ("Observations", payload.get("sample_count")),
        ("Repetitions", execution.get("repetitions")),
        ("Per-observation timeout", _seconds(execution.get("timeout_seconds"))),
        ("Campaign budget", _seconds(execution.get("budget_seconds"))),
        ("CPU", execution.get("cpu")),
        ("Required pairs", pair_text),
        ("Require all adapters", execution.get("require_all_adapters", False)),
    )
    lines = ["Campaign plan", table(("Setting", "Value"), rows)]
    cases = payload.get("cases", [])
    if cases:
        lines.extend(
            (
                "",
                "Resolved cases",
                table(
                    ("Workload", "Case", "Metrics", "Tags"),
                    (
                        (
                            case.get("workload"),
                            case.get("id"),
                            ", ".join(
                                f"{name}={value}"
                                for name, value in sorted(
                                    case.get("metrics", {}).items()
                                )
                            )
                            or "-",
                            ", ".join(case.get("tags", [])) or "-",
                        )
                        for case in cases
                    ),
                ),
            )
        )
    capabilities = payload.get("adapter_capabilities", [])
    workloads = list(payload.get("workloads", []))
    if capabilities and workloads:
        lines.extend(
            (
                "",
                "Adapter cells",
                table(
                    ("Adapter", *workloads),
                    (
                        (
                            row.get("backend"),
                            *(
                                "yes"
                                if workload in set(row.get("capabilities", []))
                                else "unsupported"
                                for workload in workloads
                            ),
                        )
                        for row in capabilities
                    ),
                ),
            )
        )
    return "\n".join(lines)


def render_doctor(payload: Mapping[str, Any]) -> str:
    rows = []
    for engine in payload.get("engines", []):
        identity = engine.get("identity", {})
        version = identity.get("package_version") or identity.get("version")
        executable = identity.get("executable")
        if executable is None:
            executable = identity.get("operation_executable") or identity.get("class_unit_executable")
        rows.append(
            (
                engine.get("display_name", engine.get("backend")),
                "available" if engine.get("available") is True else "unavailable",
                version,
                executable,
                ", ".join(engine.get("capabilities", [])),
                engine.get("error"),
            )
        )
    required = payload.get("required_errors", [])
    optional = payload.get("optional_errors", [])
    lines = [
        "Adapter diagnostics",
        table(("Adapter", "Status", "Version", "Executable", "Capabilities", "Detail"), rows),
        "",
        "Required comparisons: " + ("PASS" if not required else "FAIL"),
    ]
    lines.extend(f"  - {error}" for error in required)
    if optional:
        lines.extend(("", "Optional adapter issues:"))
        lines.extend(f"  - {error}" for error in optional)
    return "\n".join(lines)


def render_campaign(payload: Mapping[str, Any]) -> str:
    lines = [
        f"Campaign state: {payload.get('state', 'unknown')}",
        f"Run directory: {payload.get('run_dir', '-')}",
    ]
    backend_rows = payload.get("backend_status", [])
    if backend_rows:
        lines.extend(
            (
                "",
                "Observations",
                table(
                    ("Adapter", "OK", "Unsupported", "Unavailable", "Timeout", "Error", "Invalid"),
                    (
                        (
                            row.get("backend"),
                            row.get("ok", 0),
                            row.get("unsupported", 0),
                            row.get("unavailable", 0),
                            row.get("timeout", 0),
                            row.get("error", 0),
                            row.get("invalid", 0),
                        )
                        for row in backend_rows
                    ),
                ),
            )
        )
    agreement_rows = payload.get("agreement_status", [])
    if agreement_rows:
        lines.extend(
            (
                "",
                "Agreements",
                table(
                    ("Candidate", "Baseline", "Agree", "Disagree", "Unavailable", "Unsupported", "Invalid", "Incomplete"),
                    (
                        (
                            row.get("candidate"),
                            row.get("baseline"),
                            row.get("agree", 0),
                            row.get("disagree", 0),
                            row.get("unavailable", 0),
                            row.get("unsupported", 0),
                            row.get("invalid", 0),
                            row.get("incomplete", 0),
                        )
                        for row in agreement_rows
                    ),
                ),
            )
        )
    failures = payload.get("failures", [])
    if failures:
        lines.extend(("", "Failures"))
        lines.extend(
            f"  - {row.get('case_key')}/{row.get('backend')}: "
            f"{row.get('status')} ({row.get('error') or 'no detail'})"
            for row in failures
        )
    ratios = payload.get("aggregate_ratios", [])
    if ratios:
        lines.extend(
            (
                "",
                "Aggregate speedups (baseline / candidate)",
                table(
                    ("Workload", "Candidate", "Baseline", "Cases", "Speedup"),
                    (
                        (
                            row.get("workload"),
                            row.get("candidate"),
                            row.get("baseline"),
                            row.get("case_count"),
                            f"{row.get('geometric_mean_speedup'):.3f}x",
                        )
                        for row in ratios
                    ),
                ),
            )
        )
    report = payload.get("report")
    if isinstance(report, Mapping):
        lines.extend(
            (
                "",
                f"Report: {report.get('markdown', '-')}",
                f"Artifacts: {report.get('directory', '-')}",
            )
        )
        plots = report.get("plots", [])
        if plots:
            plot_directory = str(plots[0])
            if "/" in plot_directory:
                plot_directory = plot_directory.rsplit("/", 1)[0]
            lines.append(f"Plots: {len(plots)} files in {plot_directory}")
        elif report.get("plot_reason"):
            lines.append("Plots not generated: " + str(report["plot_reason"]))
    return "\n".join(lines)
