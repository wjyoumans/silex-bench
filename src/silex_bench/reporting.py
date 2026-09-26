"""Deterministic reports and immutable export bundles for campaign ledgers."""

from __future__ import annotations

import csv
import copy
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
from .ledger import (
    RunLedger,
    _effective_deadline,
    _expected_timing_coordinates,
    _selected_backend_order,
)


BOOTSTRAP_SAMPLES = 2_000
REPORT_RENDERER_VERSION = 2
_EXPECTED_TIMING_SCOPES = {
    "class_unit_proven": "class_and_unit_group_only",
    "maximal_order": "maximal_order_only",
    "ideal_multiply": "ideal_multiplication_only",
    "element_square_root": "number_field_element_is_square_only",
    "sunit_proven": "whole_process",
}
_EXPECTED_WALL_CLOCKS = {
    "silex": "steady_clock",
    "pari": "pari_getwalltime_ms",
    "hecke": "julia_time_ns_monotonic",
    "magma": "magma_realtime",
}
_SUNIT_WALL_CLOCK = "python_perf_counter_monotonic"


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


def _timing_samples(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return persisted timing samples, synthesizing legacy standard rows."""

    samples = snapshot.get("timing_samples")
    if isinstance(samples, list):
        return samples
    synthesized: list[dict[str, Any]] = []
    for observation in snapshot.get("observations", []):
        synthesized.append(
            {
                "case_key": observation.get("case_key"),
                "backend": observation.get("backend"),
                "repetition": observation.get("repetition"),
                "variant": "standard",
                "sample_index": 0,
                "status": observation.get("status", "error"),
                "timeout": observation.get("timeout") is True,
                "target_wall_ns": observation.get("target_wall_ns"),
                "process_wall_ns": observation.get("process_wall_ns"),
                "timing_scope": _timing_scope(str(observation.get("workload", ""))),
                "effective_timeout_seconds": None,
                "internal_timing": observation.get("internal_timing", {}),
                "diagnostics": {
                    "legacy_observation": True,
                    "error": observation.get("error"),
                },
            }
        )
    return synthesized


def _timing_sample_status(sample: Mapping[str, Any]) -> str:
    status = sample.get("status", "error")
    if sample.get("timeout") is True:
        return "timeout"
    return str(status)


def _series_label(backend: str, variant: str) -> str:
    return backend if variant == "standard" else f"{backend} ({variant.replace('_', ' ')})"


def _exact_discriminant(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value != 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if str(parsed) == value.strip() and parsed != 0 else None
    return None


def _report_cases(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recover exact discriminant axes for legacy ledgers in memory only."""

    cases = [copy.deepcopy(case) for case in snapshot.get("cases", [])]
    by_key = {
        f"{case.get('workload')}:{case.get('id')}": case
        for case in cases
        if isinstance(case, dict)
    }
    discriminants_by_id: dict[str, set[int]] = defaultdict(set)
    for case in cases:
        identifier = case.get("id")
        if not isinstance(identifier, str):
            continue
        for container in (case.get("expected"), case.get("input")):
            if not isinstance(container, Mapping):
                continue
            value = _exact_discriminant(
                container.get("maximal_order_discriminant")
            )
            if value is not None:
                discriminants_by_id[identifier].add(value)
    for observation in snapshot.get("observations", []):
        if (
            observation.get("success") is not True
            or observation.get("validation", {}).get("success") is not True
        ):
            continue
        case = by_key.get(str(observation.get("case_key")))
        result = observation.get("result")
        if case is None or not isinstance(result, Mapping):
            continue
        value = _exact_discriminant(result.get("maximal_order_discriminant"))
        if value is not None and isinstance(case.get("id"), str):
            discriminants_by_id[case["id"]].add(value)
    for case in cases:
        metrics = case.get("metrics")
        metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
        value = _exact_discriminant(metrics.get("maximal_order_discriminant"))
        if value is None:
            identifier = case.get("id")
            candidates = (
                discriminants_by_id.get(identifier, set())
                if isinstance(identifier, str)
                else set()
            )
            if len(candidates) == 1:
                value = next(iter(candidates))
        if value is not None:
            magnitude = abs(value)
            metrics.update(
                {
                    "maximal_order_discriminant": value,
                    "discriminant_bits": magnitude.bit_length(),
                    "log10_abs_discriminant": math.log10(magnitude),
                }
            )
        case["metrics"] = metrics
    return cases


def _agreement_admits_timings(
    snapshot: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    agreement: Mapping[str, Any],
) -> bool:
    if snapshot.get("source_ledger_schema_version") == 1:
        return agreement.get("timing_eligible") is True
    plan = snapshot.get("manifest", {}).get("plan", {})
    case = cases.get(str(agreement.get("case_key")))
    return bool(
        plan.get("mode") == "performance"
        and case is not None
        and case.get("performance_eligible") is True
        and agreement.get("success") is True
        and _agreement_status(dict(agreement)) == "agree"
    )


def _admitted_timing_slots(
    snapshot: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]]
) -> set[tuple[str, str, int]]:
    slots: set[tuple[str, str, int]] = set()
    for agreement in snapshot.get("agreements", []):
        if not _agreement_admits_timings(snapshot, cases, agreement):
            continue
        case_key = agreement.get("case_key")
        repetition = agreement.get("repetition")
        if not isinstance(case_key, str) or type(repetition) is not int:
            continue
        for key in ("lhs_backend", "rhs_backend"):
            backend = agreement.get(key)
            if isinstance(backend, str):
                slots.add((case_key, backend, repetition))
    return slots


def _summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    cases = {
        row["workload"] + ":" + row["id"]: row
        for row in _report_cases(snapshot)
    }
    timing_slots = _admitted_timing_slots(snapshot, cases)
    observations_by_slot = {
        (
            str(observation.get("case_key")),
            str(observation.get("backend")),
            int(observation.get("repetition")),
        ): observation
        for observation in snapshot["observations"]
        if isinstance(observation.get("case_key"), str)
        and isinstance(observation.get("backend"), str)
        and type(observation.get("repetition")) is int
    }
    timing_groups: dict[
        tuple[str, str, str, str, str, str], list[float]
    ] = defaultdict(list)
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
    sample_status: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for sample in _timing_samples(snapshot):
        backend = str(sample.get("backend", ""))
        variant = str(sample.get("variant", "standard"))
        sample_status[(backend, variant)][_timing_sample_status(sample)] += 1
        case_key = sample.get("case_key")
        repetition = sample.get("repetition")
        if not isinstance(case_key, str) or type(repetition) is not int:
            continue
        slot = (case_key, backend, repetition)
        observation = observations_by_slot.get(slot)
        case = cases.get(case_key)
        target_wall_ns = sample.get("target_wall_ns")
        timing_scope = sample.get("timing_scope")
        wall_clock = sample.get("wall_clock")
        if (
            (not isinstance(wall_clock, str) or not wall_clock)
            and snapshot.get("source_ledger_schema_version") == 1
        ):
            wall_clock = "legacy_supervisor_or_backend_clock"
        if (
            observation is None
            or case is None
            or observation.get("success") is not True
            or observation.get("validation", {}).get("success") is not True
            or slot not in timing_slots
            or _timing_sample_status(sample) != "ok"
            or type(target_wall_ns) is not int
            or target_wall_ns <= 0
            or not isinstance(timing_scope, str)
            or not timing_scope
            or not isinstance(wall_clock, str)
            or not wall_clock
        ):
            continue
        timing_groups[
            (
                str(case["workload"]),
                case_key,
                backend,
                variant,
                timing_scope,
                wall_clock,
            )
        ].append(target_wall_ns / 1_000_000.0)
    timings: list[dict[str, Any]] = []
    for (
        workload,
        case_key,
        backend,
        variant,
        timing_scope,
        wall_clock,
    ), values in sorted(
        timing_groups.items()
    ):
        case = cases[case_key]
        timings.append(
            {
                "workload": workload,
                "case_key": case_key,
                "case_id": case["id"],
                "backend": backend,
                "variant": variant,
                "series": _series_label(backend, variant),
                "timing_scope": timing_scope,
                "wall_clock": wall_clock,
                "metrics": case["metrics"],
                "units": "ms",
                "statistics": statistics_row(
                    values,
                    seed_text=(
                        f"{snapshot['fingerprint']}:{workload}:{case_key}:"
                        f"{backend}:{variant}:{timing_scope}:{wall_clock}"
                    ),
                ),
            }
        )
    agreement_status: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for agreement in snapshot["agreements"]:
        agreement_status[
            (str(agreement["lhs_backend"]), str(agreement["rhs_backend"]))
        ][_agreement_status(agreement)] += 1
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "source_ledger_schema_version": snapshot.get(
            "source_ledger_schema_version", snapshot.get("schema_version")
        ),
        "run_fingerprint": snapshot["fingerprint"],
        "run_state": snapshot["state"],
        "primary_clock": (
            "timing_samples.target_wall_ns (backend-internal target clocks for "
            "ordinary workloads; harness whole-process monotonic clock for integrated "
            "S-unit workloads; timing_scope and wall_clock declared per row)"
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "backend_status": [
            {"backend": backend, **dict(sorted(counts.items()))}
            for backend, counts in sorted(backend_status.items())
        ],
        "agreement_status": [
            {"candidate": candidate, "baseline": baseline, **dict(sorted(counts.items()))}
            for (candidate, baseline), counts in sorted(agreement_status.items())
        ],
        "timing_sample_status": [
            {"backend": backend, "variant": variant, **dict(sorted(counts.items()))}
            for (backend, variant), counts in sorted(sample_status.items())
        ],
        "engines": snapshot["engines"],
        "backend_exclusions": copy.deepcopy(
            snapshot.get("manifest", {})
            .get("plan", {})
            .get("execution", {})
            .get("backend_exclusions", [])
        ),
        "failures": failures,
        "timings": timings,
        "agreements": snapshot["agreements"],
    }


def _publication_errors(snapshot: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    plan = snapshot["manifest"].get("plan", {})
    execution = plan.get("execution", {})
    if snapshot.get("source_ledger_schema_version") not in {
        None,
        CAMPAIGN_SCHEMA_VERSION,
    }:
        errors.append("publication requires the current campaign ledger schema")
    if snapshot["state"] != "complete":
        errors.append("run state is not complete")
    if execution.get("publication") is not True:
        errors.append("run did not use a publication profile")
    repetitions = execution.get("repetitions")
    minimum = execution.get("minimum_repetitions")
    if type(repetitions) is not int or type(minimum) is not int or repetitions < minimum or minimum < 3:
        errors.append("publication requires at least three repetitions")
    if execution.get("cpu") is None:
        errors.append("publication requires an explicit CPU affinity")
    if snapshot["manifest"].get("machine", {}).get("system") != "Linux":
        errors.append("publication timing currently requires Linux")
    cpu_model = snapshot["manifest"].get("machine", {}).get("cpu_model")
    if not isinstance(cpu_model, str) or not cpu_model.strip():
        errors.append("publication requires a recorded CPU model")
    if execution.get("threads") != 1:
        errors.append("publication currently requires one engine thread")
    if execution.get("require_clean_sources") is not True:
        errors.append("publication requires clean-source enforcement")
    for name, source in snapshot["manifest"].get("sources", {}).items():
        if source.get("revision") is None:
            errors.append(f"publication source {name} has no commit identity")
        if source.get("dirty") is not False:
            errors.append(f"publication source {name} is not clean")
    benchmark_source = snapshot["manifest"].get("sources", {}).get("silex_bench", {})
    if benchmark_source.get("provenance") != "source_checkout":
        errors.append("publication requires silex-bench to run from its source checkout")
    package_sha256 = benchmark_source.get("package_sha256")
    if not isinstance(package_sha256, str) or len(package_sha256) != 64:
        errors.append("publication silex-bench package content identity is incomplete")
    required_pairs = {
        tuple(pair)
        for pair in plan.get("required_pairs", [])
        if isinstance(pair, list) and len(pair) == 2
    }
    required_backends = {backend for pair in required_pairs for backend in pair}
    planned_cases = {
        f"{case.get('workload')}:{case.get('id')}": case
        for case in plan.get("cases", [])
        if isinstance(case, dict)
    }
    backends = [
        backend for backend in plan.get("backends", []) if isinstance(backend, str)
    ]
    expected_required_agreements = {
        (case_key, repetition, lhs, rhs)
        for case_key, case in planned_cases.items()
        for lhs, rhs in required_pairs
        if lhs in _selected_backend_order(case, backends, execution)
        and rhs in _selected_backend_order(case, backends, execution)
        for repetition in range(repetitions if type(repetitions) is int else 0)
    }
    agreement_index = {
        (
            row.get("case_key"),
            row.get("repetition"),
            row.get("lhs_backend"),
            row.get("rhs_backend"),
        ): row
        for row in summary["agreements"]
    }
    if not expected_required_agreements:
        errors.append("publication has no required-pair agreement rows")
    case_index = {
        f"{case.get('workload')}:{case.get('id')}": case
        for case in _report_cases(snapshot)
    }
    for key in sorted(expected_required_agreements):
        row = agreement_index.get(key)
        if row is None:
            errors.append(
                "required agreement is missing: "
                f"{key[0]}/{key[2]}/{key[3]}/{key[1]}"
            )
        elif not _agreement_admits_timings(snapshot, case_index, row):
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
    published_slots = _admitted_timing_slots(snapshot, case_index)
    publication_repetitions = (
        range(repetitions) if type(repetitions) is int else range(0)
    )
    published_series = {
        (case_key, backend) for case_key, backend, _ in published_slots
    }
    complete_published_slots = {
        (case_key, backend, repetition)
        for case_key, backend in published_series
        for repetition in publication_repetitions
    }
    for case_key, backend in sorted(published_series):
        actual_repetitions = {
            repetition
            for slot_case, slot_backend, repetition in published_slots
            if slot_case == case_key and slot_backend == backend
        }
        expected_repetitions = set(publication_repetitions)
        if actual_repetitions != expected_repetitions:
            errors.append(
                "publication timing series lacks every planned repetition: "
                f"{case_key}/{backend} has {sorted(actual_repetitions)}, "
                f"expected {sorted(expected_repetitions)}"
            )
    gated_backends = required_backends | {
        backend for _, backend, _ in published_slots
    }
    for backend in sorted(gated_backends):
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
    timing_samples: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in _timing_samples(snapshot):
        case_key = sample.get("case_key")
        backend = sample.get("backend")
        repetition = sample.get("repetition")
        if (
            isinstance(case_key, str)
            and isinstance(backend, str)
            and type(repetition) is int
        ):
            timing_samples[(case_key, backend, repetition)].append(sample)
    required_slots = {
        (case_key, backend, repetition)
        for case_key, repetition, lhs, rhs in expected_required_agreements
        for backend in (lhs, rhs)
    }
    gated_slots = required_slots | complete_published_slots

    requested_cpu = execution.get("cpu")
    for case_key, backend, repetition in sorted(gated_slots):
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
        sample_rows = timing_samples.get((case_key, backend, repetition), [])
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
        if not sample_rows:
            errors.append(
                "required observation has no timing samples: "
                f"{case_key}/{backend}/{repetition}"
            )
        expected_coordinates = set(
            _expected_timing_coordinates(
                case_rows[0], backend, execution
            )
        )
        actual_coordinates = {
            (sample.get("variant"), sample.get("sample_index"))
            for sample in sample_rows
        }
        if actual_coordinates != expected_coordinates:
            errors.append(
                "required timing variants differ from the plan: "
                f"{case_key}/{backend}/{repetition}"
            )
        expected_timeout, expected_timeout_source = _effective_deadline(
            case_rows[0], execution
        )
        expected_scope = _EXPECTED_TIMING_SCOPES.get(str(workload))
        expected_clock = (
            _SUNIT_WALL_CLOCK
            if workload == "sunit_proven"
            else _EXPECTED_WALL_CLOCKS.get(backend)
        )
        for sample in sample_rows:
            if (
                _timing_sample_status(sample) != "ok"
                or type(sample.get("target_wall_ns")) is not int
                or sample["target_wall_ns"] <= 0
            ):
                errors.append(
                    "required timing sample is not successful: "
                    f"{case_key}/{backend}/{repetition}/"
                    f"{sample.get('variant', 'standard')}"
                )
            if (
                sample.get("effective_timeout_seconds") != expected_timeout
                or sample.get("timeout_source") != expected_timeout_source
            ):
                errors.append(
                    "required timing sample deadline differs from the plan: "
                    f"{case_key}/{backend}/{repetition}/"
                    f"{sample.get('variant', 'standard')}"
                )
            if (
                sample.get("timing_scope") != expected_scope
                or sample.get("wall_clock") != expected_clock
            ):
                errors.append(
                    "required timing sample has an unexpected scope or wall clock: "
                    f"{case_key}/{backend}/{repetition}/"
                    f"{sample.get('variant', 'standard')}"
                )
        if observation.get("workload") != workload:
            errors.append(
                "required observation workload differs from its case: "
                f"{case_key}/{backend}/{repetition}"
            )
        if requested_cpu is not None:
            for sample in sample_rows:
                timing = sample.get("internal_timing", {})
                if not isinstance(timing, Mapping):
                    timing = {}
                affinity = (
                    timing.get("effective_affinity")
                    if workload == "sunit_proven"
                    else timing.get("marked_process_affinity")
                )
                if affinity != [requested_cpu]:
                    errors.append(
                        "required timing sample lacks the requested singleton affinity: "
                        f"{case_key}/{backend}/{repetition}/"
                        f"{sample.get('variant', 'standard')}"
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
    lines.extend(["", "## Configured backend exclusions", ""])
    exclusions = summary.get("backend_exclusions", [])
    if exclusions:
        lines.extend(
            [
                "| Adapter | Workload | Reason |",
                "|---|---|---|",
            ]
        )
        for exclusion in exclusions:
            reason = str(exclusion.get("reason", "-")).replace("|", "\\|")
            lines.append(
                f"| {exclusion.get('backend', '-')} | "
                f"{exclusion.get('workload', '-')} | {reason} |"
            )
    else:
        lines.append("None.")
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
            "",
            "## Per-case timings",
            "",
            "| Workload | Case | Backend | Sample | Scope | Wall clock | n | Median ms | MAD ms | 95% bootstrap interval |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|",
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
            f"{row['variant']} | {row['timing_scope']} | {row['wall_clock']} | "
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
    lines.extend(
        [
            "",
            "Raw timing samples, correctness results, failures, timeouts, exclusions, "
            "and pairwise checks are retained in the JSON and CSV artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _plots(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if not summary["timings"]:
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
    axes = (
        ("degree", "degree", "degree"),
        (
            "log10_abs_discriminant",
            "log10-abs-discriminant",
            "log10 |D_K|",
        ),
    )
    with matplotlib.rc_context(
        {
            "svg.hashsalt": str(summary["run_fingerprint"]),
            "svg.fonttype": "none",
        }
    ):
        for workload in workloads:
            workload_rows = [
                row for row in summary["timings"] if row["workload"] == workload
            ]
            for axis_name, file_axis, axis_label in axes:
                rows = [
                    row
                    for row in workload_rows
                    if isinstance(row.get("metrics"), Mapping)
                    and isinstance(row["metrics"].get(axis_name), (int, float))
                    and not isinstance(row["metrics"].get(axis_name), bool)
                ]
                if not rows:
                    continue
                figure, axis = plt.subplots(figsize=(8, 5))
                series = sorted(
                    {
                        (
                            str(row["backend"]),
                            str(row.get("variant", "standard")),
                            str(row.get("timing_scope", "")),
                            str(row.get("wall_clock", "")),
                        )
                        for row in rows
                    }
                )
                base_counts: dict[tuple[str, str], int] = defaultdict(int)
                for backend, variant, _, _ in series:
                    base_counts[(backend, variant)] += 1
                for backend, variant, timing_scope, wall_clock in series:
                    points = sorted(
                        (
                            float(row["metrics"][axis_name]),
                            float(row["statistics"]["median"]),
                            row["statistics"]["bootstrap_95_low"],
                            row["statistics"]["bootstrap_95_high"],
                            str(row["case_key"]),
                        )
                        for row in rows
                        if row["backend"] == backend
                        and row.get("variant", "standard") == variant
                        and row.get("timing_scope") == timing_scope
                        and row.get("wall_clock") == wall_clock
                        and isinstance(row["statistics"].get("median"), (int, float))
                    )
                    if not points:
                        continue
                    x = [point[0] for point in points]
                    y = [point[1] for point in points]
                    low = [
                        0.0 if point[2] is None else point[1] - float(point[2])
                        for point in points
                    ]
                    high = [
                        0.0 if point[3] is None else float(point[3]) - point[1]
                        for point in points
                    ]
                    axis.errorbar(
                        x,
                        y,
                        yerr=[low, high],
                        fmt="o",
                        linestyle="none",
                        capsize=3,
                        label=(
                            _series_label(backend, variant)
                            if base_counts[(backend, variant)] == 1
                            else (
                                f"{_series_label(backend, variant)} "
                                f"[{timing_scope}; {wall_clock}]"
                            )
                        ),
                    )
                axis.set_yscale("log")
                axis.set_xlabel(axis_label)
                axis.set_ylabel("target wall time (ms; see row scope)")
                axis.set_title(workload)
                axis.legend()
                figure.tight_layout()
                name = f"runtime-{workload}-by-{file_axis}.svg"
                figure.savefig(
                    plot_dir / name,
                    format="svg",
                    metadata={"Date": None},
                )
                files.append(f"plots/{name}")
                plt.close(figure)
    if not files:
        return {
            "generated": False,
            "reason": "eligible timing data has no supported plot axes",
        }
    return {"generated": True, "files": sorted(files)}


def _renderer_identity(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary["timings"]:
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
                "source_ledger_schema_version": snapshot.get(
                    "source_ledger_schema_version", snapshot.get("schema_version")
                ),
                "observations_sha256": hashlib.sha256(
                    _canonical(snapshot["observations"])
                ).hexdigest(),
                "timing_samples_sha256": hashlib.sha256(
                    _canonical(_timing_samples(snapshot))
                ).hexdigest(),
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
        _write_jsonl(temporary / "timing_samples.jsonl", _timing_samples(snapshot))
        _write_csv(temporary / "timing_samples.csv", _timing_samples(snapshot))
        _write_csv(temporary / "timings.csv", summary["timings"])
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
