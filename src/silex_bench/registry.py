"""Built-in backend registry backed by the source-traced engine adapters."""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .backends import create_backend
from .backends.base import BackendAdapter
from .contracts import (
    BackendDescriptor,
    Case,
    EngineInfo,
    ImplementationAdapter,
    InvocationContext,
    Observation,
    ObservationStatus,
    Registry,
    TimingSample,
    ValidationResult,
    WorkloadContract,
)
from .model import BackendContext, FieldSpec, SampleRequest, silex_executable_keys
from .util import file_digest_nofollow
from .workloads import ALL_WORKLOADS, NUMBER_FIELD_WORKLOADS, SUNIT, builtin_workloads, sunit_module


def _milliseconds_to_ns(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(round(float(value) * 1_000_000))


def _legacy_context(
    context: InvocationContext, workloads: str | tuple[str, ...]
) -> BackendContext:
    selected = (workloads,) if isinstance(workloads, str) else workloads
    return BackendContext(
        workspace=context.workspace,
        bench_root=context.bench_root,
        silex_source=context.silex_source,
        silex_build_dir=context.silex_build_dir,
        tools=dict(context.tools),
        timeout_seconds=context.timeout_seconds,
        cpu=context.cpu,
        primary_clock="marked_wall",
        environment=dict(context.environment),
        selected_operations=selected,
        jit_repetitions=context.jit_repetitions,
        timeout_source=context.timeout_source,
    )


def _field(case: Case) -> FieldSpec:
    coefficients = tuple(int(value) for value in case.input["coefficients_low_to_high"])
    expected = case.expected
    invariants = expected.get("class_invariants")
    return FieldSpec(
        id=case.id,
        coefficients_low_to_high=coefficients,
        degree=len(coefficients) - 1,
        source=case.source,
        polynomial_discriminant=None,
        benchmark_role=next((tag for tag in ("core", "scale", "diversity", "holdout") if tag in case.tags), None),
        expected_class_order=(
            None if expected.get("class_order") is None else str(expected["class_order"])
        ),
        expected_class_invariants=(
            None if invariants is None else tuple(str(value) for value in invariants)
        ),
        expected_unit_rank=(
            expected.get("unit_rank") if type(expected.get("unit_rank")) is int else None
        ),
        expected_maximal_order_discriminant=(
            None
            if expected.get("maximal_order_discriminant") is None
            else str(expected["maximal_order_discriminant"])
        ),
        optimization_external_engines=(case.eligible_backends or None),
        timeout_seconds=(
            float(case.input["timeout_seconds"])
            if isinstance(case.input.get("timeout_seconds"), (int, float))
            else None
        ),
    )


def _status(payload: dict[str, Any]) -> ObservationStatus:
    if payload.get("success") is True and payload.get("timeout") is not True:
        return ObservationStatus.OK
    if payload.get("timeout") is True:
        return ObservationStatus.TIMEOUT
    if payload.get("available") is not True or payload.get("status") == "unavailable":
        return ObservationStatus.UNAVAILABLE
    return ObservationStatus.ERROR


def _timing_samples(
    payload: dict[str, Any],
    *,
    case: Case,
    backend: str,
    repetition: int,
    context: InvocationContext,
    default_scope: str,
) -> tuple[TimingSample, ...]:
    """Normalize adapter timing payloads at the campaign boundary."""

    raw_samples = payload.get("timing_samples")
    if not isinstance(raw_samples, list):
        timing = payload.get("timing")
        timing = timing if isinstance(timing, dict) else {}
        raw_samples = [
            {
                "variant": "standard",
                "sample_index": 0,
                "success": payload.get("success") is True,
                "timeout": payload.get("timeout") is True,
                "target_cpu_ms": payload.get(
                    "target_cpu_ms", timing.get("target_cpu_ms")
                ),
                "target_wall_ms": payload.get(
                    "target_wall_ms", timing.get("target_wall_ms")
                ),
                "process_wall_ms": payload.get("process_wall_ms"),
                "timing_scope": timing.get("scope", default_scope),
                "wall_clock": timing.get("wall_clock"),
                "cpu_clock": timing.get("algorithm_clock"),
                "internal_timing": timing,
                "diagnostics": payload.get("diagnostics", {}),
            }
        ]
    samples: list[TimingSample] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            continue
        timeout = raw.get("timeout") is True
        success = raw.get("success") is True and not timeout
        status = (
            ObservationStatus.OK
            if success
            else ObservationStatus.TIMEOUT
            if timeout
            else ObservationStatus.ERROR
        )
        internal = raw.get("internal_timing")
        diagnostics = raw.get("diagnostics")
        target_wall_ms = raw.get("target_wall_ms")
        if target_wall_ms is None and default_scope == "whole_process":
            target_wall_ms = raw.get("process_wall_ms")
        samples.append(
            TimingSample(
                case_key=case.key,
                workload=case.workload,
                backend=backend,
                repetition=repetition,
                variant=str(raw.get("variant", "standard")),
                sample_index=int(raw.get("sample_index", index)),
                status=status,
                timeout=timeout,
                target_wall_ns=_milliseconds_to_ns(target_wall_ms),
                target_cpu_ns=_milliseconds_to_ns(raw.get("target_cpu_ms")),
                process_wall_ns=_milliseconds_to_ns(raw.get("process_wall_ms")),
                timing_scope=str(raw.get("timing_scope") or default_scope),
                effective_timeout_seconds=context.timeout_seconds,
                timeout_source=context.timeout_source,
                wall_clock=(
                    str(raw["wall_clock"])
                    if raw.get("wall_clock") is not None
                    else None
                ),
                cpu_clock=(
                    str(raw["cpu_clock"])
                    if raw.get("cpu_clock") is not None
                    else None
                ),
                internal_timing=copy.deepcopy(internal)
                if isinstance(internal, dict)
                else {},
                diagnostics=copy.deepcopy(diagnostics)
                if isinstance(diagnostics, dict)
                else {},
            )
        )
    return tuple(samples)


def add_executable_digests(
    probe: dict[str, Any], *, selected_workloads: tuple[str, ...]
) -> dict[str, Any]:
    """Bind probe executable paths to bytes without changing adapter APIs."""
    enriched = copy.deepcopy(probe)
    identity = enriched.get("engine_identity")
    if not isinstance(identity, dict):
        return enriched
    required_silex = (
        set(silex_executable_keys(selected_workloads))
        if identity.get("engine") == "silex"
        else None
    )
    existing = identity.get("executable_sha256")
    existing = existing if isinstance(existing, dict) else {}
    digests: dict[str, str | None] = {}
    for key, value in identity.items():
        if "executable" not in key or not isinstance(value, str):
            continue
        if required_silex is not None and key not in required_silex:
            digests[key] = None
        elif isinstance(existing.get(key), str):
            digests[key] = existing[key]
        else:
            try:
                digests[key] = file_digest_nofollow(Path(value))
            except (OSError, ValueError):
                digests[key] = None
    if digests:
        identity["executable_sha256"] = digests
    return enriched


@dataclass(frozen=True)
class NativeImplementation(ImplementationAdapter):
    backend: str
    workload: str
    adapter: BackendAdapter

    def run(
        self,
        case: Case,
        *,
        repetition: int,
        order_index: int,
        warmup: Case | None,
        context: InvocationContext,
        contract: WorkloadContract,
    ) -> Observation:
        request = SampleRequest(
            field=_field(case),
            operation=self.workload,
            sample_kind="cold_algorithm",
            sample_index=repetition,
            warmup=None,
            seed=0,
            jit_repetitions=context.jit_repetitions,
        )
        try:
            payload = self.adapter.run(
                request, _legacy_context(context, self.workload)
            )
        except Exception as exc:
            payload = {
                "available": True,
                "success": False,
                "timeout": False,
                "status": "adapter_error",
                "error": f"{type(exc).__name__}: {exc}",
                "result": {},
                "proof": {},
                "timing": {},
            }
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        proof = payload.get("proof") if isinstance(payload.get("proof"), dict) else {}
        validation = (
            contract.validate_observation(case, self.backend, result, proof)
            if payload.get("success") is True
            else ValidationResult(False, (str(payload.get("error") or "backend execution failed"),), {})
        )
        timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
        samples = _timing_samples(
            payload,
            case=case,
            backend=self.backend,
            repetition=repetition,
            context=context,
            default_scope=contract.timing_scope,
        )
        command = payload.get("cmd")
        return Observation(
            case_key=case.key,
            workload=self.workload,
            backend=self.backend,
            repetition=repetition,
            order_index=order_index,
            status=_status(payload),
            success=payload.get("success") is True,
            timeout=payload.get("timeout") is True,
            result=dict(result),
            proof=dict(proof),
            validation=validation,
            target_wall_ns=_milliseconds_to_ns(
                timing.get("marked_target_wall_ms", payload.get("target_wall_ms"))
            ),
            process_wall_ns=_milliseconds_to_ns(payload.get("process_wall_ms")),
            internal_timing=copy.deepcopy(timing),
            engine_identity=copy.deepcopy(
                payload.get("engine_identity")
                if isinstance(payload.get("engine_identity"), dict)
                else {}
            ),
            command=tuple(str(item) for item in command) if isinstance(command, list) else (),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            error=(None if payload.get("error") is None else str(payload.get("error"))),
            timing_samples=samples,
        )


@dataclass(frozen=True)
class SUnitImplementation(ImplementationAdapter):
    backend: str
    workload: str = SUNIT

    def run(
        self,
        case: Case,
        *,
        repetition: int,
        order_index: int,
        warmup: Case | None,
        context: InvocationContext,
        contract: WorkloadContract,
    ) -> Observation:
        module = sunit_module()
        args = _sunit_arguments(context)
        runner = {
            "silex": module.run_silex,
            "pari": module.run_pari,
            "hecke": module.run_hecke,
        }.get(self.backend)
        if runner is None:
            payload = {
                "available": False,
                "success": False,
                "timeout": False,
                "failure_reason": f"{self.backend} does not implement {SUNIT}",
            }
        else:
            try:
                payload = runner(args, dict(case.input))
            except Exception as exc:
                payload = {
                    "available": True,
                    "success": False,
                    "timeout": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
        proof = {
            key: payload.get(key)
            for key in (
                "certification_status",
                "class_group_proof_status",
                "unit_group_proof_status",
                "regulator_proof_status",
                "s_class_proof_status",
                "s_unit_proof_status",
                "s_regulator_proof_status",
                "final_result_published",
            )
        }
        validation = (
            contract.validate_observation(case, self.backend, payload, proof)
            if payload.get("success") is True
            else ValidationResult(False, (str(payload.get("failure_reason") or "S-unit execution failed"),), {})
        )
        process_ms = payload.get("process_wall_ms")
        command = ("integrated-sunit", self.backend, case.id)
        timing = {
            "scope": "whole_process",
            "preparation_excluded": False,
            "jit_policy": "not_applicable_integrated_whole_process",
            "wall_clock": "python_perf_counter_monotonic",
            "phase_timing_ms": payload.get("phase_timing_ms", {}),
            "sunit_timing_ms": payload.get("sunit_timing_ms", {}),
            "component_timing_ms": payload.get("component_timing_ms", {}),
            "effective_affinity": payload.get("effective_affinity"),
            "launcher_executable": payload.get("launcher_executable"),
            "launcher_executable_sha256": payload.get(
                "launcher_executable_sha256"
            ),
        }
        samples = _timing_samples(
            {
                **payload,
                "target_wall_ms": process_ms,
                "timing": timing,
            },
            case=case,
            backend=self.backend,
            repetition=repetition,
            context=context,
            default_scope=contract.timing_scope,
        )
        return Observation(
            case_key=case.key,
            workload=SUNIT,
            backend=self.backend,
            repetition=repetition,
            order_index=order_index,
            status=_status(payload),
            success=payload.get("success") is True,
            timeout=payload.get("timeout") is True,
            result=copy.deepcopy(payload),
            proof=proof,
            validation=validation,
            target_wall_ns=_milliseconds_to_ns(process_ms),
            process_wall_ns=_milliseconds_to_ns(process_ms),
            internal_timing=timing,
            engine_identity=copy.deepcopy(
                payload.get("engine_identity")
                if isinstance(payload.get("engine_identity"), dict)
                else {}
            ),
            command=command,
            stdout="",
            stderr=str(payload.get("stderr", "")),
            error=(
                None
                if payload.get("failure_reason") is None
                else str(payload.get("failure_reason"))
            ),
            timing_samples=samples,
        )


def _sunit_arguments(context: InvocationContext) -> argparse.Namespace:
    return argparse.Namespace(
        silex_root=context.silex_source,
        build_dir=context.silex_build_dir,
        timeout=context.timeout_seconds,
        gp=context.tools.get("gp"),
        pari_version=context.tools.get("pari_version"),
        pari_source=(
            Path(context.tools["pari_source"])
            if context.tools.get("pari_source")
            else None
        ),
        julia=context.tools.get("julia"),
        hecke_project=context.tools.get("hecke_project"),
        magma=str(context.tools.get("magma", "magma")),
        cpu=context.cpu,
        environment=dict(context.environment),
    )


def _probe(
    context: InvocationContext,
    descriptor: BackendDescriptor,
    *,
    adapter: BackendAdapter,
) -> EngineInfo:
    selected_workloads = tuple(
        workload
        for workload in context.selected_workloads
        if workload in descriptor.implementations
    )
    if not context.selected_workloads:
        selected_workloads = descriptor.capabilities
    try:
        payload = adapter.probe(_legacy_context(context, selected_workloads))
        payload = add_executable_digests(
            payload, selected_workloads=selected_workloads
        )
    except Exception as exc:
        payload = {
            "available": False,
            "success": False,
            "engine_identity": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    identity = payload.get("engine_identity")
    return EngineInfo(
        backend=descriptor.id,
        display_name=descriptor.display_name,
        available=payload.get("available") is True and payload.get("success") is True,
        capabilities=descriptor.capabilities,
        identity=copy.deepcopy(identity) if isinstance(identity, dict) else {},
        error=None if payload.get("error") is None else str(payload.get("error")),
    )


def builtin_registry() -> Registry:
    displays = {
        "silex": "Silex",
        "pari": "PARI/GP",
        "hecke": "Hecke/OSCAR",
        "magma": "Magma",
    }
    descriptors: list[BackendDescriptor] = []
    for backend in ("silex", "pari", "hecke", "magma"):
        adapter = create_backend(backend)
        implementations: dict[str, ImplementationAdapter] = {
            workload: NativeImplementation(backend, workload, adapter)
            for workload in NUMBER_FIELD_WORKLOADS
        }
        if backend in {"silex", "pari", "hecke"}:
            implementations[SUNIT] = SUnitImplementation(backend)
        descriptors.append(
            BackendDescriptor(
                id=backend,
                display_name=displays[backend],
                implementations=implementations,
                probe_callback=partial(_probe, adapter=adapter),
            )
        )
    return Registry(builtin_workloads(), descriptors)
