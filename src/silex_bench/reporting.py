"""Deterministic reports and immutable export bundles for campaign ledgers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import stat
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from .contracts import CAMPAIGN_SCHEMA_VERSION, engine_identity_binding
from .ledger import RunLedger


BOOTSTRAP_SAMPLES = 2_000
REPORT_RENDERER_VERSION = 1


def _timing_scope(workload: str) -> str:
    return "whole_process" if workload == "sunit_proven" else "supervisor_marked_target"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _mad(values: list[float], center: float | None) -> float | None:
    return None if center is None else float(statistics.median(abs(value - center) for value in values))


def _bootstrap_interval(values: list[float], seed: int) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    generator = random.Random(seed)
    samples = sorted(
        float(statistics.median(generator.choice(values) for _ in values))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def statistics_row(values: Iterable[float], *, seed_text: str) -> dict[str, Any]:
    data = [float(value) for value in values if math.isfinite(float(value)) and value > 0]
    center = _median(data)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    low, high = _bootstrap_interval(data, seed)
    return {
        "count": len(data),
        "median": center,
        "mad": _mad(data, center),
        "minimum": min(data) if data else None,
        "maximum": max(data) if data else None,
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def _observation_status(observation: dict[str, Any]) -> str:
    status = str(observation.get("status", "error"))
    if status == "ok" and observation.get("validation", {}).get("success") is not True:
        return "invalid"
    return status


def _agreement_status(agreement: dict[str, Any]) -> str:
    status = agreement.get("status")
    if isinstance(status, str) and status:
        return status
    return "agree" if agreement.get("success") is True else "invalid"


def _summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    cases = {row["workload"] + ":" + row["id"]: row for row in snapshot["cases"]}
    timing_slots: set[tuple[str, str, int]] = set()
    for agreement in snapshot["agreements"]:
        if agreement.get("timing_eligible") is True:
            timing_slots.add(
                (
                    agreement["case_key"],
                    agreement["lhs_backend"],
                    int(agreement["repetition"]),
                )
            )
            timing_slots.add(
                (
                    agreement["case_key"],
                    agreement["rhs_backend"],
                    int(agreement["repetition"]),
                )
            )
    timing_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    status_counts: dict[str, int] = defaultdict(int)
    backend_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures: list[dict[str, Any]] = []
    for observation in snapshot["observations"]:
        effective_status = _observation_status(observation)
        status_counts[effective_status] += 1
        backend_status[str(observation["backend"])][effective_status] += 1
        if effective_status not in {"ok", "unsupported"}:
            failures.append(
                {
                    "case_key": observation.get("case_key"),
                    "backend": observation.get("backend"),
                    "status": effective_status,
                    "error": observation.get("error")
                    or "; ".join(observation.get("validation", {}).get("errors", [])),
                }
            )
        if (
            observation.get("success") is True
            and observation.get("validation", {}).get("success") is True
            and isinstance(observation.get("target_wall_ns"), int)
            and observation["target_wall_ns"] > 0
            and (
                observation["case_key"],
                observation["backend"],
                int(observation["repetition"]),
            )
            in timing_slots
        ):
            timing_groups[
                (
                    observation["workload"],
                    observation["case_key"],
                    observation["backend"],
                )
            ].append(observation["target_wall_ns"] / 1_000_000.0)
    timings: list[dict[str, Any]] = []
    for (workload, case_key, backend), values in sorted(timing_groups.items()):
        case = cases[case_key]
        timings.append(
            {
                "workload": workload,
                "case_key": case_key,
                "case_id": case["id"],
                "backend": backend,
                "timing_scope": _timing_scope(workload),
                "metrics": case["metrics"],
                "units": "ms",
                "statistics": statistics_row(
                    values,
                    seed_text=f"{snapshot['fingerprint']}:{workload}:{case_key}:{backend}",
                ),
            }
        )
    ratio_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for agreement in snapshot["agreements"]:
        ratio = agreement.get("speedup_baseline_over_candidate")
        if agreement.get("timing_eligible") is True and isinstance(ratio, (int, float)) and ratio > 0:
            workload = cases[agreement["case_key"]]["workload"]
            ratio_groups[
                (
                    workload,
                    agreement["case_key"],
                    agreement["lhs_backend"],
                    agreement["rhs_backend"],
                )
            ].append(float(ratio))
    ratios: list[dict[str, Any]] = []
    for (workload, case_key, candidate, baseline), values in sorted(ratio_groups.items()):
        ratios.append(
            {
                "workload": workload,
                "case_key": case_key,
                "case_id": cases[case_key]["id"],
                "candidate": candidate,
                "baseline": baseline,
                "definition": "baseline_time / candidate_time",
                "timing_scope": _timing_scope(workload),
                "metrics": cases[case_key]["metrics"],
                "statistics": statistics_row(
                    values,
                    seed_text=f"{snapshot['fingerprint']}:{case_key}:{candidate}:{baseline}",
                ),
            }
        )
    aggregate_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ratios:
        value = row["statistics"]["median"]
        if isinstance(value, (int, float)) and value > 0:
            aggregate_groups[(row["workload"], row["candidate"], row["baseline"])].append(math.log(value))
    aggregates = [
        {
            "workload": workload,
            "candidate": candidate,
            "baseline": baseline,
            "case_count": len(log_values),
            "geometric_mean_speedup": math.exp(statistics.mean(log_values)),
        }
        for (workload, candidate, baseline), log_values in sorted(aggregate_groups.items())
    ]
    agreement_status: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for agreement in snapshot["agreements"]:
        agreement_status[
            (str(agreement["lhs_backend"]), str(agreement["rhs_backend"]))
        ][_agreement_status(agreement)] += 1
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "run_fingerprint": snapshot["fingerprint"],
        "run_state": snapshot["state"],
        "primary_clock": "target_wall_ns with timing_scope declared per row",
        "speedup_definition": "baseline_time / candidate_time; values above 1 mean the candidate is faster",
        "status_counts": dict(sorted(status_counts.items())),
        "backend_status": [
            {"backend": backend, **dict(sorted(counts.items()))}
            for backend, counts in sorted(backend_status.items())
        ],
        "agreement_status": [
            {"candidate": candidate, "baseline": baseline, **dict(sorted(counts.items()))}
            for (candidate, baseline), counts in sorted(agreement_status.items())
        ],
        "engines": snapshot["engines"],
        "failures": failures,
        "timings": timings,
        "ratios": ratios,
        "aggregate_ratios": aggregates,
        "agreements": snapshot["agreements"],
    }


def _publication_errors(snapshot: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    plan = snapshot["manifest"].get("plan", {})
    execution = plan.get("execution", {})
    if snapshot["state"] != "complete":
        errors.append("run state is not complete")
    if execution.get("publication") is not True:
        errors.append("run did not use a publication profile")
    repetitions = execution.get("repetitions")
    minimum = execution.get("minimum_repetitions")
    if type(repetitions) is not int or type(minimum) is not int or repetitions < minimum or minimum < 9:
        errors.append("publication requires at least nine repetitions")
    if execution.get("cpu") is None:
        errors.append("publication requires an explicit CPU affinity")
    if snapshot["manifest"].get("machine", {}).get("system") != "Linux":
        errors.append("publication timing currently requires Linux")
    cpu_model = snapshot["manifest"].get("machine", {}).get("cpu_model")
    if not isinstance(cpu_model, str) or not cpu_model.strip():
        errors.append("publication requires a recorded CPU model")
    if execution.get("threads") != 1:
        errors.append("publication currently requires one engine thread")
    for name, source in snapshot["manifest"].get("sources", {}).items():
        if source.get("revision") is None:
            errors.append(f"publication source {name} has no commit identity")
        if execution.get("require_clean_sources") is True and source.get("dirty") is not False:
            errors.append(f"publication source {name} is not clean")
    benchmark_source = snapshot["manifest"].get("sources", {}).get("silex_bench", {})
    if benchmark_source.get("provenance") != "source_checkout":
        errors.append("publication requires silex-bench to run from its source checkout")
    package_sha256 = benchmark_source.get("package_sha256")
    if not isinstance(package_sha256, str) or len(package_sha256) != 64:
        errors.append("publication silex-bench package content identity is incomplete")
    required_pairs = {tuple(pair) for pair in plan.get("required_pairs", [])}
    required_backends = {backend for pair in required_pairs for backend in pair}
    agreements = [
        row
        for row in summary["agreements"]
        if (row.get("lhs_backend"), row.get("rhs_backend")) in required_pairs
    ]
    if not agreements:
        errors.append("publication has no required-pair agreement rows")
    for row in agreements:
        if row.get("success") is not True or row.get("timing_eligible") is not True:
            errors.append(
                f"required pair is not eligible: {row.get('case_key')}/"
                f"{row.get('lhs_backend')}/{row.get('rhs_backend')}/"
                f"{row.get('repetition')}"
            )

    probes = {
        engine.get("backend"): engine
        for engine in snapshot["engines"]
        if isinstance(engine.get("backend"), str)
    }
    probe_identities: dict[str, Mapping[str, Any]] = {}
    for backend in sorted(required_backends):
        engine = probes.get(backend, {})
        identity = engine.get("identity", {})
        if (
            engine.get("available") is not True
            or not isinstance(identity, Mapping)
            or not identity
        ):
            errors.append(f"required engine lacks a complete identity: {backend}")
            identity = {}
        probe_identities[backend] = identity
        if backend == "pari":
            version = identity.get("version")
            required_version = identity.get("required_version")
            source_version = identity.get("source_version")
            if (
                not identity.get("source")
                or not isinstance(identity.get("source_version_file_sha256"), str)
                or len(identity["source_version_file_sha256"]) != 64
                or not isinstance(required_version, str)
                or version != required_version
                or source_version != required_version
            ):
                errors.append(
                    "publication PARI identity requires a source tree and matching "
                    "executable, required, and source versions"
                )

    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in snapshot["cases"]:
        case_id = case.get("id")
        workload = case.get("workload")
        if isinstance(case_id, str) and isinstance(workload, str):
            cases[f"{workload}:{case_id}"].append(case)
    observations: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for observation in snapshot["observations"]:
        case_key = observation.get("case_key")
        backend = observation.get("backend")
        repetition = observation.get("repetition")
        if (
            isinstance(case_key, str)
            and isinstance(backend, str)
            and type(repetition) is int
        ):
            observations[(case_key, backend, repetition)].append(observation)
    required_slots: set[tuple[str, str, int]] = set()
    for agreement in agreements:
        case_key = agreement.get("case_key")
        repetition = agreement.get("repetition")
        if not isinstance(case_key, str) or type(repetition) is not int:
            errors.append("required agreement has invalid observation coordinates")
            continue
        required_slots.add((case_key, str(agreement["lhs_backend"]), repetition))
        required_slots.add((case_key, str(agreement["rhs_backend"]), repetition))

    requested_cpu = execution.get("cpu")
    for case_key, backend, repetition in sorted(required_slots):
        case_rows = cases.get(case_key, [])
        observation_rows = observations.get((case_key, backend, repetition), [])
        if len(case_rows) != 1 or len(observation_rows) != 1:
            errors.append(
                "required observation is missing or duplicated: "
                f"{case_key}/{backend}/{repetition}"
            )
            continue
        workload = case_rows[0]["workload"]
        observation = observation_rows[0]
        validation = observation.get("validation", {})
        if (
            observation.get("success") is not True
            or not isinstance(validation, Mapping)
            or validation.get("success") is not True
        ):
            errors.append(
                "required observation is not successful and validated: "
                f"{case_key}/{backend}/{repetition}"
            )
        if observation.get("workload") != workload:
            errors.append(
                "required observation workload differs from its case: "
                f"{case_key}/{backend}/{repetition}"
            )
        if requested_cpu is not None:
            timing = observation.get("internal_timing", {})
            affinity = None
            if isinstance(timing, Mapping):
                affinity = (
                    timing.get("effective_affinity")
                    if workload == "sunit_proven"
                    else timing.get("marked_process_affinity")
                )
            if affinity != [requested_cpu]:
                errors.append(
                    "required observation lacks the requested singleton affinity: "
                    f"{case_key}/{backend}/{repetition}"
                )
        probe_identity = probe_identities.get(backend, {})
        observed_identity = observation.get("engine_identity", {})
        if not isinstance(probe_identity, Mapping) or not isinstance(
            observed_identity, Mapping
        ):
            identity_matches = False
        else:
            expected = engine_identity_binding(
                backend, workload, probe_identity
            )
            actual = engine_identity_binding(
                backend, workload, observed_identity
            )
            identity_matches = expected == actual
            executable = expected.get("executable")
            executable_sha256 = expected.get("executable_sha256")
            if (
                not isinstance(executable, str)
                or not executable
                or not isinstance(executable_sha256, str)
                or len(executable_sha256) != 64
            ):
                errors.append(
                    "required engine lacks executable content identity: "
                    f"{case_key}/{backend}/{repetition}"
                )
        if not identity_matches:
            errors.append(
                "required observation engine identity differs from its probe: "
                f"{case_key}/{backend}/{repetition}"
            )
    return errors


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened: list[dict[str, Any]] = []
    keys: set[str] = set()
    for row in rows:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    result[f"{key}.{nested_key}"] = nested_value
            elif isinstance(value, list):
                result[key] = json.dumps(value, sort_keys=True)
            else:
                result[key] = value
        flattened.append(result)
        keys.update(result)
    fields = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _markdown(
    snapshot: dict[str, Any],
    summary: dict[str, Any],
    publication_errors: list[str],
    plots: dict[str, Any],
) -> str:
    machine = snapshot["manifest"].get("machine", {})
    execution = snapshot["manifest"]["plan"].get("execution", {})
    requested_cpu = machine.get("requested_cpu")
    lines = [
        f"# {snapshot['manifest']['plan']['suite']} benchmark report",
        "",
        f"- Run state: `{snapshot['state']}`",
        f"- Run fingerprint: `{snapshot['fingerprint']}`",
        f"- CPU: {machine.get('cpu_model') or 'unknown'}",
        f"- Platform: {machine.get('platform') or 'unknown'} ({machine.get('architecture') or 'unknown'})",
        f"- CPU affinity: {'not pinned' if requested_cpu is None else requested_cpu}",
        f"- Engine threads: {execution.get('threads', 'unknown')}",
        f"- Primary clock: `{summary['primary_clock']}`",
        f"- Speedup: {summary['speedup_definition']}",
        "",
    ]
    lines.extend(
        [
            "## Adapter availability",
            "",
            "| Adapter | Available | Version | Capabilities | Detail |",
            "|---|---:|---|---|---|",
        ]
    )
    for engine in summary["engines"]:
        identity = engine.get("identity", {})
        version = identity.get("package_version") or identity.get("version") or "-"
        capabilities = ", ".join(engine.get("capabilities", [])) or "-"
        detail = str(engine.get("error") or "-").replace("|", "\\|")
        lines.append(
            f"| {engine.get('display_name', engine.get('backend'))} | "
            f"{'yes' if engine.get('available') is True else 'no'} | {version} | "
            f"{capabilities} | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Observation status",
            "",
            "| Adapter | OK | Unsupported | Unavailable | Timeout | Error | Invalid |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["backend_status"]:
        lines.append(
            f"| {row['backend']} | {row.get('ok', 0)} | {row.get('unsupported', 0)} | "
            f"{row.get('unavailable', 0)} | {row.get('timeout', 0)} | "
            f"{row.get('error', 0)} | {row.get('invalid', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise agreement",
            "",
            "| Candidate | Baseline | Agree | Disagree | Unavailable | Unsupported | Invalid | Incomplete |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["agreement_status"]:
        lines.append(
            f"| {row['candidate']} | {row['baseline']} | {row.get('agree', 0)} | "
            f"{row.get('disagree', 0)} | {row.get('unavailable', 0)} | "
            f"{row.get('unsupported', 0)} | {row.get('invalid', 0)} | "
            f"{row.get('incomplete', 0)} |"
        )
    if summary["failures"]:
        lines.extend(
            [
                "",
                "## Observation failures",
                "",
                "| Workload/case | Adapter | Status | Detail |",
                "|---|---|---|---|",
            ]
        )
        for row in summary["failures"]:
            detail = str(row.get("error") or "-").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['case_key']} | {row['backend']} | {row['status']} | {detail} |"
            )
    if publication_errors:
        lines.extend(["", "## Publication blockers", ""])
        lines.extend(f"- {error}" for error in publication_errors)
        lines.append("")
    lines.extend(
        [
            "## Aggregate speedups",
            "",
            "| Workload | Candidate | Baseline | Cases | Geometric mean speedup |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregate_ratios"]:
        lines.append(
            f"| {row['workload']} | {row['candidate']} | {row['baseline']} | "
            f"{row['case_count']} | {row['geometric_mean_speedup']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Per-case timings",
            "",
            "| Workload | Case | Backend | Scope | n | Median ms | MAD ms | 95% bootstrap interval |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["timings"]:
        stats = row["statistics"]
        interval = (
            "exploratory"
            if stats["bootstrap_95_low"] is None
            else f"[{stats['bootstrap_95_low']:.3f}, {stats['bootstrap_95_high']:.3f}]"
        )
        lines.append(
            f"| {row['workload']} | {row['case_id']} | {row['backend']} | "
            f"{row['timing_scope']} | "
            f"{stats['count']} | {stats['median']:.3f} | {stats['mad']:.3f} | {interval} |"
        )
    lines.extend(["", "## Plots", ""])
    if plots.get("generated") is True:
        lines.extend(f"- `{path}`" for path in plots.get("files", []))
    else:
        lines.append(
            "Plots were not generated: "
            + str(plots.get("reason") or "no eligible timing data")
            + ". Install the `plot` extra to enable matplotlib output."
        )
    lines.extend(["", "All failures, timeouts, exclusions, and pairwise checks are retained in the JSON and CSV artifacts.", ""])
    return "\n".join(lines)


def _plots(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if not summary["timings"] and not summary["ratios"]:
        return {"generated": False, "reason": "no eligible timing data"}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"generated": False, "reason": "matplotlib is not installed"}
    plot_dir = directory / "plots"
    plot_dir.mkdir()
    files: list[str] = []
    workloads = sorted({row["workload"] for row in summary["timings"]})
    for workload in workloads:
        rows = [row for row in summary["timings"] if row["workload"] == workload]
        axes = [name for name in ("degree", "discriminant_bits", "coefficient_bits", "s_size", "height") if any(name in row["metrics"] for row in rows)]
        if not axes:
            continue
        axis_name = axes[0]
        figure, axis = plt.subplots(figsize=(8, 5))
        for backend in sorted({row["backend"] for row in rows}):
            points = sorted(
                (
                    row["metrics"][axis_name],
                    row["statistics"]["median"],
                    row["statistics"]["bootstrap_95_low"],
                    row["statistics"]["bootstrap_95_high"],
                )
                for row in rows
                if row["backend"] == backend and axis_name in row["metrics"]
            )
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            low = [0 if point[2] is None else point[1] - point[2] for point in points]
            high = [0 if point[3] is None else point[3] - point[1] for point in points]
            axis.errorbar(x, y, yerr=[low, high], marker="o", label=backend)
        axis.set_yscale("log")
        axis.set_xlabel(axis_name.replace("_", " "))
        axis.set_ylabel("target wall time (ms; see row scope)")
        axis.set_title(workload)
        axis.legend()
        figure.tight_layout()
        for extension in ("png", "svg", "pdf"):
            name = f"runtime-{workload}.{extension}"
            figure.savefig(plot_dir / name, dpi=160)
            files.append(f"plots/{name}")
        plt.close(figure)
    for workload in sorted({row["workload"] for row in summary["ratios"]}):
        rows = [row for row in summary["ratios"] if row["workload"] == workload]
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.5), 5))
        labels = [f"{row['case_id']}\n{row['candidate']}:{row['baseline']}" for row in rows]
        values = [row["statistics"]["median"] for row in rows]
        axis.bar(range(len(rows)), values)
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_xticks(range(len(rows)), labels, rotation=90, fontsize=7)
        axis.set_ylabel("speedup (baseline / candidate)")
        axis.set_title(workload)
        figure.tight_layout()
        for extension in ("png", "svg", "pdf"):
            name = f"speedup-{workload}.{extension}"
            figure.savefig(plot_dir / name, dpi=160)
            files.append(f"plots/{name}")
        plt.close(figure)
    return {"generated": True, "files": sorted(files)}


def _renderer_identity(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary["timings"] and not summary["ratios"]:
        plots = {"available": False, "reason": "no eligible timing data"}
    else:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot  # noqa: F401
        except ImportError:
            plots = {"available": False, "reason": "matplotlib is not installed"}
        else:
            plots = {"available": True, "matplotlib_version": matplotlib.__version__}
    return {"report_renderer": REPORT_RENDERER_VERSION, "plots": plots}


def _verify_existing_bundle(
    destination: Path, *, report_identity: str, publication: bool
) -> None:
    bundle_path = destination / "bundle.json"
    try:
        bundle_info = bundle_path.lstat()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing report bundle is unreadable: {destination}") from exc
    if not stat.S_ISREG(bundle_info.st_mode) or not isinstance(bundle, dict):
        raise ValueError(f"existing report bundle is not a regular object: {destination}")
    if (
        bundle.get("report_identity") != report_identity
        or bundle.get("publication") is not publication
        or not isinstance(bundle.get("artifacts"), dict)
    ):
        raise ValueError(f"existing report bundle identity is invalid: {destination}")
    expected_paths = set(bundle["artifacts"])
    actual_paths: set[str] = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix()
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"existing report contains a special file: {relative}")
        if relative != "bundle.json":
            actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise ValueError(f"existing report artifact set is invalid: {destination}")
    for relative, identity in bundle["artifacts"].items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(identity, dict)
        ):
            raise ValueError(f"existing report artifact identity is invalid: {relative!r}")
        path = destination / relative
        data = path.read_bytes()
        if (
            identity.get("size") != len(data)
            or identity.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(f"existing report artifact changed: {relative}")


def generate_report(run_dir: Path, *, publication: bool = False) -> dict[str, Any]:
    with RunLedger(run_dir) as ledger:
        snapshot = ledger.snapshot()
    summary = _summary(snapshot)
    publication_errors = _publication_errors(snapshot, summary)
    if publication and publication_errors:
        raise ValueError("publication export rejected: " + "; ".join(publication_errors))
    renderer = _renderer_identity(summary)
    report_identity = hashlib.sha256(
        _canonical(
            {
                "fingerprint": snapshot["fingerprint"],
                "state": snapshot["state"],
                "summary": summary,
                "publication": publication,
                "renderer": renderer,
            }
        )
    ).hexdigest()
    exports = run_dir / "exports"
    exports.mkdir(exist_ok=True)
    destination = exports / report_identity[:16]
    if destination.exists():
        _verify_existing_bundle(
            destination,
            report_identity=report_identity,
            publication=publication,
        )
        return {
            "success": True,
            "publication_eligible": not publication_errors,
            "publication_errors": publication_errors,
            "directory": str(destination),
            "report_identity": report_identity,
        }
    temporary = Path(tempfile.mkdtemp(prefix=".report-", dir=exports))
    try:
        _write_json(temporary / "summary.json", summary)
        _write_json(temporary / "manifest.json", snapshot["manifest"])
        _write_jsonl(temporary / "observations.jsonl", snapshot["observations"])
        _write_csv(temporary / "timings.csv", summary["timings"])
        _write_csv(temporary / "ratios.csv", summary["ratios"])
        _write_csv(temporary / "agreements.csv", summary["agreements"])
        plots = _plots(temporary, summary)
        (temporary / "report.md").write_text(
            _markdown(snapshot, summary, publication_errors, plots), encoding="utf-8"
        )
        artifact_entries: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary).as_posix()
            data = path.read_bytes()
            artifact_entries[relative] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        bundle = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "report_identity": report_identity,
            "publication": publication,
            "publication_eligible": not publication_errors,
            "publication_errors": publication_errors,
            "renderer": renderer,
            "plots": plots,
            "artifacts": artifact_entries,
        }
        _write_json(temporary / "bundle.json", bundle)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "success": True,
        "publication_eligible": not publication_errors,
        "publication_errors": publication_errors,
        "directory": str(destination),
        "report_identity": report_identity,
    }
