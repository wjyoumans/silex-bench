"""Hecke benchmark adapter using fresh, uncached number fields.

The rigorous class/unit route follows Hecke's public API with
``class_group(...; GRH=false, redo=true)`` and
``unit_group(...; GRH=false)``.  Hecke's class-group call computes and stores
the shared class/unit context, so the immediately following unit-group call is
intentionally a same-sample lookup from that context.  Each JIT timing uses an
independently constructed field and order, preventing mathematical caches from
crossing the first-call/repeat-call boundary.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..model import (
    BackendContext,
    SampleRequest,
    canonical_invariants,
    polynomial_expr,
)
from ..process import (
    TARGET_NONCE_PLACEHOLDER,
    parse_key_values,
    resolve_executable,
    run_marked_process,
    run_process,
)
from .base import (
    BackendAdapter,
    identity_with_executed_digest,
    nonempty_diagnostic,
    parse_bool,
    parse_float,
    parse_int,
    process_state_is_valid,
    successful_probe,
    unavailable,
    unavailable_probe,
    unit_count_error,
)


_READY_MARKER = "__SILEX_BENCH_HECKE_READY__"
_TARGET_MARKER = "__SILEX_BENCH_HECKE_TARGET_DONE__"
_REQUESTED_THREADS = 1
_THREAD_COUNT_SOURCE = "julia_nthreads_and_blas_runtime_query_max"


def julia_environment(
    overrides: Mapping[str, Any] | None = None,
    *,
    julia: str | None = None,
) -> dict[str, str]:
    # Keep Julia's default depot unless the caller explicitly supplied an
    # environment override.  In particular, do not redirect JULIA_DEPOT_PATH.
    env = os.environ.copy()
    if overrides is not None:
        env.update({str(key): str(value) for key, value in overrides.items()})
    env["JULIA_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if julia is not None:
        julia_root = Path(julia).resolve().parent.parent
        library_paths = [julia_root / "lib", julia_root / "lib" / "julia"]
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(path) for path in library_paths]
            + ([existing] if existing else [])
        )
    return env


def _environment(
    context: BackendContext, *, julia: str | None = None
) -> dict[str, str]:
    return julia_environment(context.environment, julia=julia)


def _juliaup_candidates() -> list[str]:
    root = Path.home() / ".julia" / "juliaup"
    return [
        str(path.resolve())
        for path in sorted(root.glob("julia-*/bin/julia"), reverse=True)
        if path.is_file()
    ]


def resolve_julia_runtime(
    configured: Any = None,
    *,
    environment: Mapping[str, Any] | None = None,
) -> tuple[str | None, str]:
    if configured is not None and str(configured).strip():
        requested = str(configured).strip()
        return resolve_executable(requested), requested
    configured = (
        environment.get("SILEX_BENCH_JULIA")
        if environment is not None
        else None
    ) or os.environ.get("SILEX_BENCH_JULIA")
    if configured:
        requested = str(configured).strip()
        return resolve_executable(requested), requested

    path_julia = shutil.which("julia")
    resolved_path_julia = (
        Path(path_julia).resolve() if path_julia is not None else None
    )
    if (
        resolved_path_julia is not None
        and resolved_path_julia.name != "julialauncher"
    ):
        return str(resolved_path_julia), "julia"
    candidates = _juliaup_candidates()
    if candidates:
        return candidates[0], "default Julia depot"
    if path_julia is not None:
        return str(Path(path_julia).resolve()), "julia"
    return None, "julia"


def _resolve_julia(context: BackendContext) -> tuple[str | None, str]:
    return resolve_julia_runtime(
        context.tools.get("julia"), environment=context.environment
    )


def resolve_hecke_project(
    configured: Any = None,
    *,
    environment: Mapping[str, Any] | None = None,
) -> tuple[Path | None, str | None]:
    if configured is None:
        configured = (
            environment.get("SILEX_BENCH_HECKE_PROJECT")
            if environment is not None
            else None
        ) or os.environ.get("SILEX_BENCH_HECKE_PROJECT")
    if configured is None or not str(configured).strip():
        return None, None
    path = Path(str(configured)).expanduser().resolve()
    if not path.exists():
        return None, f"Hecke Julia project does not exist: {path}"
    if not path.is_dir():
        return None, f"Hecke Julia project must be a directory: {path}"
    if not any(
        (path / filename).is_file()
        for filename in ("Project.toml", "JuliaProject.toml")
    ):
        return None, f"Hecke Julia project has no Project.toml: {path}"
    return path, None


def _project(context: BackendContext) -> tuple[Path | None, str | None]:
    return resolve_hecke_project(
        context.tools.get("hecke_project"), environment=context.environment
    )


def julia_command(julia: str, project: Path | None) -> list[str]:
    command = [
        julia,
        "--startup-file=no",
        "--history-file=no",
        "--threads=1",
    ]
    if project is not None:
        command.append(f"--project={project}")
    return command


def _julia_command(julia: str, project: Path | None) -> list[str]:
    return julia_command(julia, project)


def hecke_load_error(process: Mapping[str, Any], project: Path | None) -> str:
    diagnostic = nonempty_diagnostic(
        process.get("error"),
        process.get("stderr"),
        fallback="Hecke version probe failed",
    )
    lowered = diagnostic.lower()
    missing_package = (
        "package hecke not found" in lowered
        or "required but does not seem to be installed" in lowered
        or "pkg.instantiate" in lowered
    )
    if project is not None and missing_package:
        return (
            f"configured Hecke Julia project {project} is not instantiated; "
            f"run `julia --project={project} -e 'using Pkg; Pkg.instantiate()'`"
        )
    if project is None and "package hecke" in lowered:
        return (
            "Hecke is not installed in the active Julia environment; run "
            "`julia -e 'using Pkg; Pkg.add(\"Hecke\")'` or configure "
            "hecke_project"
        )
    return diagnostic


def _parse_invariants(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    try:
        return canonical_invariants(
            part.strip() for part in value.split(",") if part.strip()
        )
    except ValueError:
        return None


_HELPERS = r"""
using Hecke
using Random

Qx, x = polynomial_ring(QQ, "x", cached = false)

function bench_process_cpu_ns()
  Sys.islinux() || return nothing
  value = Ref{NTuple{2, Clong}}()
  status = ccall(:clock_gettime, Cint,
                 (Cint, Ref{NTuple{2, Clong}}), 2, value)
  iszero(status) || return nothing
  seconds, nanoseconds = value[]
  return UInt64(seconds) * UInt64(1_000_000_000) + UInt64(nanoseconds)
end

function bench_field(f)
  return number_field(f, "a", cached = false)
end

# Hecke clears UnitGrpCtx.GRH only after its unconditional unit proof, which it
# skips when the unit rank is zero (_class_unit_group in Clgp.jl).
function bench_unit_group_grh_free(O, unit_ctx)
  return !unit_ctx.GRH || Hecke.unit_group_rank(O) == 0
end
"""


_CLASS_UNIT_OPERATIONS = ("class_unit_proven", "class_unit_grh")
_TIMING_SCOPES = {
    "class_unit_proven": "class_and_unit_group_only",
    "class_unit_grh": "class_and_unit_group_only",
    "maximal_order": "maximal_order_only",
    "ideal_multiply": "ideal_multiplication_only",
    "element_square_root": "number_field_element_is_square_only",
}

_CLASS_UNIT_CACHE_POLICY = (
    "class_group(redo=true) recomputes Hecke's shared class/unit context; "
    "unit_group then retrieves that same-sample context; each timing sample "
    "uses an independent fresh field and order"
)


def _sample_prefixes(request: SampleRequest) -> tuple[str, ...]:
    if request.jit_repetitions == 0:
        return ("target",)
    if request.jit_repetitions == 1:
        return ("first", "repeat")
    raise ValueError("Hecke JIT repetitions currently supports only zero or one")


def _preparation(operation: str, prefix: str, seed: int) -> str:
    common = f"""
Random.seed!({seed})
{prefix}_K, {prefix}_a = bench_field(P)
"""
    if operation in _CLASS_UNIT_OPERATIONS:
        return common + f"{prefix}_O = lll(maximal_order({prefix}_K))\n"
    if operation == "maximal_order":
        return common
    if operation == "ideal_multiply":
        return common + f"""
{prefix}_O = maximal_order({prefix}_K)
{prefix}_left_ideal = ideal({prefix}_O, ZZ(2))
{prefix}_right_ideal = ideal({prefix}_O, ZZ(3))
"""
    if operation == "element_square_root":
        return common + f"""
{prefix}_square = ({prefix}_K(1) + {prefix}_a)^2
"""
    raise ValueError(f"unsupported Hecke operation: {operation}")


def _timed_call(
    operation: str,
    prefix: str,
    *,
    output_prefix: str,
    seed: int,
) -> str:
    key = output_prefix
    # Conditional route: GRH = true skips Hecke's class-group and unit-group
    # proof phases; proven route: GRH = false runs them.
    grh_value = "true" if operation == "class_unit_grh" else "false"
    start = f"""
Random.seed!({seed})
{prefix}_cpu_t0 = bench_process_cpu_ns()
{prefix}_wall_t0 = time_ns()
"""
    if operation in _CLASS_UNIT_OPERATIONS:
        body = f"""
{prefix}_class_t0 = time_ns()
{prefix}_C, {prefix}_mC = class_group(
  {prefix}_O; GRH = {grh_value}, redo = true, do_lll = false
)
{prefix}_class_group_ms = (time_ns() - {prefix}_class_t0) / 1.0e6
{prefix}_unit_t0 = time_ns()
{prefix}_U, {prefix}_mU = unit_group({prefix}_O; GRH = {grh_value})
{prefix}_unit_group_ms = (time_ns() - {prefix}_unit_t0) / 1.0e6
"""
        timing_lines = f"""
println("{key}component_class_group_ms=", {prefix}_class_group_ms)
println("{key}component_unit_group_ms=", {prefix}_unit_group_ms)
"""
    elif operation == "maximal_order":
        body = f"""
{prefix}_O = maximal_order({prefix}_K)
{prefix}_maximal_order_ms = (time_ns() - {prefix}_wall_t0) / 1.0e6
"""
        timing_lines = (
            f'println("{key}component_maximal_order_ms=", '
            f"{prefix}_maximal_order_ms)\n"
        )
    elif operation == "ideal_multiply":
        body = f"""
{prefix}_product_ideal = {prefix}_left_ideal * {prefix}_right_ideal
{prefix}_ideal_multiply_ms = (time_ns() - {prefix}_wall_t0) / 1.0e6
"""
        timing_lines = (
            f'println("{key}component_ideal_multiply_ms=", '
            f"{prefix}_ideal_multiply_ms)\n"
        )
    elif operation == "element_square_root":
        body = f"""
{prefix}_root_found, {prefix}_square_root = is_square_with_sqrt({prefix}_square)
{prefix}_square_root_ms = (time_ns() - {prefix}_wall_t0) / 1.0e6
"""
        timing_lines = (
            f'println("{key}component_square_root_ms=", '
            f"{prefix}_square_root_ms)\n"
        )
    else:
        raise ValueError(f"unsupported Hecke operation: {operation}")
    finish = f"""
{prefix}_internal_target_wall_ms = (time_ns() - {prefix}_wall_t0) / 1.0e6
{prefix}_cpu_t1 = bench_process_cpu_ns()
{prefix}_internal_target_cpu_ms = (
  {prefix}_cpu_t0 === nothing || {prefix}_cpu_t1 === nothing ||
  {prefix}_cpu_t1 < {prefix}_cpu_t0
) ? nothing : ({prefix}_cpu_t1 - {prefix}_cpu_t0) / 1.0e6
{timing_lines}
println("{key}internal_target_cpu_ms=", {prefix}_internal_target_cpu_ms)
println("{key}internal_target_wall_ms=", {prefix}_internal_target_wall_ms)
flush(stdout)
"""
    return start + body + finish


def _standard_result(operation: str) -> str:
    if operation in _CLASS_UNIT_OPERATIONS:
        return """
target_signature = signature(target_K)
target_class_ctx = get_attribute(target_O, :ClassGrpCtx)
target_unit_ctx = get_attribute(target_O, :UnitGrpCtx)
println("class_order=", order(target_C))
println("class_invariants=", join(string.(elementary_divisors(target_C)), ","))
println("fundamental_unit_count=", length(target_unit_ctx.units))
println("signature_r1=", target_signature[1])
println("signature_r2=", target_signature[2])
println("polynomial_discriminant=", discriminant(P))
println("maximal_order_discriminant=", discriminant(target_O))
println("class_group_grh_free=", !target_class_ctx.GRH)
println("unit_group_grh_free=", bench_unit_group_grh_free(target_O, target_unit_ctx))
"""
    if operation == "maximal_order":
        return """
target_signature = signature(target_K)
println("polynomial_discriminant=", discriminant(P))
println("maximal_order_discriminant=", discriminant(target_O))
println("signature_r1=", target_signature[1])
println("signature_r2=", target_signature[2])
"""
    if operation == "ideal_multiply":
        return 'println("ideal_norm=", norm(target_product_ideal))\n'
    if operation == "element_square_root":
        return """
target_root_verified = target_root_found && target_square_root^2 == target_square
println("root_found=", target_root_found)
println("root_verified=", target_root_verified)
"""
    raise ValueError(f"unsupported Hecke operation: {operation}")


def _paired_result(operation: str) -> str:
    if operation in _CLASS_UNIT_OPERATIONS:
        return """
first_signature = signature(first_K)
repeat_signature = signature(repeat_K)
first_invariants = elementary_divisors(first_C)
repeat_invariants = elementary_divisors(repeat_C)
first_class_ctx = get_attribute(first_O, :ClassGrpCtx)
repeat_class_ctx = get_attribute(repeat_O, :ClassGrpCtx)
first_unit_ctx = get_attribute(first_O, :UnitGrpCtx)
repeat_unit_ctx = get_attribute(repeat_O, :UnitGrpCtx)
jit_results_agree = (
  order(first_C) == order(repeat_C) &&
  first_invariants == repeat_invariants &&
  length(first_unit_ctx.units) == length(repeat_unit_ctx.units) &&
  first_signature == repeat_signature &&
  discriminant(first_O) == discriminant(repeat_O)
)
println("class_order=", order(repeat_C))
println("class_invariants=", join(string.(repeat_invariants), ","))
println("fundamental_unit_count=", length(repeat_unit_ctx.units))
println("signature_r1=", repeat_signature[1])
println("signature_r2=", repeat_signature[2])
println("polynomial_discriminant=", discriminant(P))
println("maximal_order_discriminant=", discriminant(repeat_O))
println(
  "class_group_grh_free=",
  !first_class_ctx.GRH && !repeat_class_ctx.GRH,
)
println(
  "unit_group_grh_free=",
  bench_unit_group_grh_free(first_O, first_unit_ctx) &&
    bench_unit_group_grh_free(repeat_O, repeat_unit_ctx),
)
println("jit_results_agree=", jit_results_agree)
"""
    if operation == "maximal_order":
        return """
first_signature = signature(first_K)
repeat_signature = signature(repeat_K)
jit_results_agree = (
  discriminant(first_O) == discriminant(repeat_O) &&
  first_signature == repeat_signature
)
println("polynomial_discriminant=", discriminant(P))
println("maximal_order_discriminant=", discriminant(repeat_O))
println("signature_r1=", repeat_signature[1])
println("signature_r2=", repeat_signature[2])
println("jit_results_agree=", jit_results_agree)
"""
    if operation == "ideal_multiply":
        return """
first_ideal_norm = norm(first_product_ideal)
repeat_ideal_norm = norm(repeat_product_ideal)
jit_results_agree = first_ideal_norm == repeat_ideal_norm
println("ideal_norm=", repeat_ideal_norm)
println("jit_results_agree=", jit_results_agree)
"""
    if operation == "element_square_root":
        return """
first_root_verified = first_root_found && first_square_root^2 == first_square
repeat_root_verified = repeat_root_found && repeat_square_root^2 == repeat_square
jit_results_agree = (
  first_root_found == repeat_root_found &&
  first_root_verified == repeat_root_verified
)
println("root_found=", repeat_root_found)
println("root_verified=", repeat_root_verified)
println("jit_results_agree=", jit_results_agree)
"""
    raise ValueError(f"unsupported Hecke operation: {operation}")


def _programs(request: SampleRequest) -> tuple[str, str, str]:
    prefixes = _sample_prefixes(request)
    polynomial = polynomial_expr(request.field.coefficients_low_to_high)
    preparations = "".join(
        _preparation(request.operation, prefix, request.seed)
        for prefix in prefixes
    )
    ready = f"""
{_HELPERS}
P = {polynomial}
using LinearAlgebra
# Snapshot taken in the ready program, before the timed target; not a final-state query.
benchmark_reported_threads = max(
  Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads())
{preparations}
GC.gc()
println("{_READY_MARKER}")
flush(stdout)
"""

    paired = len(prefixes) == 2
    calls: list[str] = []
    for index, prefix in enumerate(prefixes):
        if index:
            calls.append("GC.gc()\n")
        output_prefix = f"{prefix}_" if paired else ""
        calls.append(
            _timed_call(
                request.operation,
                prefix,
                output_prefix=output_prefix,
                seed=request.seed,
            )
        )
    target = "".join(calls) + f"""
println("{_TARGET_MARKER}:" * benchmark_target_nonce)
flush(stdout)
"""
    result = (
        _paired_result(request.operation)
        if paired
        else _standard_result(request.operation)
    )
    final = result + """
println("reported_threads=", benchmark_reported_threads)
flush(stdout)
exit()
"""
    return ready, target, final


def _result(request: SampleRequest, values: dict[str, str]) -> dict[str, Any]:
    if request.operation in _CLASS_UNIT_OPERATIONS:
        r1 = parse_int(values, "signature_r1")
        r2 = parse_int(values, "signature_r2")
        return {
            "class_order": values.get("class_order"),
            "class_invariants": _parse_invariants(values.get("class_invariants")),
            "unit_rank": parse_int(values, "fundamental_unit_count"),
            "signature": [r1, r2] if r1 is not None and r2 is not None else None,
            "polynomial_discriminant": values.get("polynomial_discriminant"),
            "maximal_order_discriminant": values.get(
                "maximal_order_discriminant"
            ),
        }
    if request.operation == "maximal_order":
        r1 = parse_int(values, "signature_r1")
        r2 = parse_int(values, "signature_r2")
        return {
            "polynomial_discriminant": values.get("polynomial_discriminant"),
            "maximal_order_discriminant": values.get(
                "maximal_order_discriminant"
            ),
            "signature": [r1, r2] if r1 is not None and r2 is not None else None,
        }
    if request.operation == "ideal_multiply":
        return {"ideal_norm": values.get("ideal_norm")}
    if request.operation == "element_square_root":
        return {
            "root_found": parse_bool(values, "root_found"),
            "root_verified": parse_bool(values, "root_verified"),
        }
    return {}


def _components(
    request: SampleRequest,
    values: dict[str, str],
    *,
    key_prefix: str = "",
) -> dict[str, Any]:
    if request.operation in _CLASS_UNIT_OPERATIONS:
        return {
            "class_group": parse_float(
                values, f"{key_prefix}component_class_group_ms"
            ),
            "unit_group": parse_float(
                values, f"{key_prefix}component_unit_group_ms"
            ),
        }
    if request.operation == "maximal_order":
        return {
            "maximal_order": parse_float(
                values, f"{key_prefix}component_maximal_order_ms"
            ),
        }
    key = {
        "ideal_multiply": "component_ideal_multiply_ms",
        "element_square_root": "component_square_root_ms",
    }.get(request.operation)
    return (
        {request.operation: parse_float(values, f"{key_prefix}{key}")}
        if key is not None
        else {}
    )


def _complete(request: SampleRequest, result: dict[str, Any], proven: bool) -> bool:
    if request.operation in _CLASS_UNIT_OPERATIONS:
        return (
            (proven or request.operation == "class_unit_grh")
            and result.get("class_order") is not None
            and result.get("class_invariants") is not None
            and result.get("unit_rank") is not None
            and result.get("signature") is not None
            and result.get("maximal_order_discriminant") is not None
        )
    if request.operation == "maximal_order":
        return result.get("maximal_order_discriminant") is not None
    if request.operation == "ideal_multiply":
        return result.get("ideal_norm") is not None
    if request.operation == "element_square_root":
        return result.get("root_found") is True and result.get("root_verified") is True
    return False


def _timing_sample_payloads(
    request: SampleRequest,
    values: dict[str, str],
    process: dict[str, Any],
    *,
    observation_success: bool,
    results_agree: bool,
) -> list[dict[str, Any]]:
    paired = request.jit_repetitions == 1
    variants = (
        (("first", "first_call"), ("repeat", "repeat_call"))
        if paired
        else (("", "standard"),)
    )
    samples: list[dict[str, Any]] = []
    for sample_index, (prefix, variant) in enumerate(variants):
        key_prefix = f"{prefix}_" if prefix else ""
        target_cpu_ms = parse_float(
            values, f"{key_prefix}internal_target_cpu_ms"
        )
        target_wall_ms = parse_float(
            values, f"{key_prefix}internal_target_wall_ms"
        )
        target_completed = target_wall_ms is not None
        sample_timeout = (
            process.get("timeout") is True and not target_completed
        )
        sample_success = observation_success and target_completed
        sample_status = (
            "ok"
            if sample_success
            else "timeout"
            if sample_timeout
            else "compute_error"
        )
        internal_timing: dict[str, Any] = {
            "algorithm_clock": "clock_gettime_process_cpu",
            "wall_clock": "julia_time_ns_monotonic",
            "component_clock": "julia_time_ns_monotonic",
            "preparation_excluded": True,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "components_ms": _components(
                request, values, key_prefix=key_prefix
            ),
            "marked_process_affinity": process.get("effective_affinity"),
            "marked_process_supervisor_affinity": process.get(
                "supervisor_affinity"
            ),
            "marked_process_supervisor_isolation_tier": process.get(
                "supervisor_isolation_tier"
            ),
            "cpu_launcher_executable": process.get("launcher_executable"),
            "cpu_launcher_sha256": process.get("launcher_executable_sha256"),
        }
        if request.operation in _CLASS_UNIT_OPERATIONS:
            internal_timing.update(
                {
                    "class_unit_cache_policy": _CLASS_UNIT_CACHE_POLICY,
                    "cross_sample_mathematical_cache_reuse": False,
                }
            )
        if paired:
            internal_timing.update(
                {
                    "marked_pair_target_cpu_ms": process.get("target_cpu_ms"),
                    "marked_pair_target_wall_ms": process.get("target_wall_ms"),
                }
            )
        else:
            internal_timing.update(
                {
                    "marked_target_cpu_ms": process.get("target_cpu_ms"),
                    "marked_target_wall_ms": process.get("target_wall_ms"),
                }
            )
        samples.append(
            {
                "variant": variant,
                "sample_index": sample_index,
                "success": sample_success,
                "timeout": sample_timeout,
                "status": sample_status,
                "target_cpu_ms": target_cpu_ms,
                "target_wall_ms": target_wall_ms,
                "process_wall_ms": (
                    process.get("process_wall_ms") if not paired else None
                ),
                "timing_scope": _TIMING_SCOPES[request.operation],
                "wall_clock": "julia_time_ns_monotonic",
                "cpu_clock": "clock_gettime_process_cpu",
                "internal_timing": internal_timing,
                "diagnostics": (
                    {"results_agree": results_agree} if paired else {}
                ),
            }
        )
    return samples


class HeckeBackend(BackendAdapter):
    name = "hecke"

    def __init__(self) -> None:
        self._probe: dict[str, Any] | None = None

    def probe(self, context: BackendContext) -> dict[str, Any]:
        if self._probe is not None:
            return dict(self._probe)
        julia, requested = _resolve_julia(context)
        if julia is None:
            self._probe = unavailable_probe(
                self.name, f"Julia executable not found: {requested}"
            )
            return dict(self._probe)
        project, project_error = _project(context)
        if project_error is not None:
            self._probe = unavailable_probe(self.name, project_error)
            return dict(self._probe)
        code = r"""
using Hecke
println("julia_version=", VERSION)
println("hecke_version=", Base.pkgversion(Hecke))
println("hecke_source=", realpath(pathof(Hecke)))
"""
        process = run_process(
            [*_julia_command(julia, project), "-e", code],
            timeout=min(context.timeout_seconds, 120.0),
            cwd=context.bench_root,
            env=_environment(context, julia=julia),
        )
        values = parse_key_values(process.get("stdout", ""))
        identity = {
            "engine": self.name,
            "executable": str(Path(julia).resolve()),
            "version": values.get("julia_version"),
            "package_version": values.get("hecke_version"),
            "source": values.get("hecke_source"),
            "project": str(project) if project is not None else None,
        }
        identity = identity_with_executed_digest(identity, process)
        success = process_state_is_valid(process) and process["success"] and values.get(
            "hecke_version"
        ) is not None
        if success:
            self._probe = successful_probe(self.name, identity)
        else:
            self._probe = unavailable_probe(
                self.name,
                hecke_load_error(process, project),
                identity=identity,
                timeout=process.get("timeout") is True,
            )
        return dict(self._probe)

    def run(
        self, request: SampleRequest, context: BackendContext
    ) -> dict[str, Any]:
        probe = self.probe(context)
        if not probe.get("available"):
            payload = unavailable(self.name, str(probe.get("error")))
            payload["engine_identity"] = probe.get("engine_identity")
            payload["cmd"] = None
            return payload

        identity = probe["engine_identity"]
        julia = str(identity["executable"])
        project = Path(identity["project"]) if identity.get("project") else None
        ready, target, final = _programs(request)
        driver = f"""
readline(stdin)
{ready}
benchmark_target_nonce = strip(readline(stdin))
{target}
readline(stdin)
{final}
"""
        process = run_marked_process(
            [*_julia_command(julia, project), "-e", driver],
            ready_input="ready\n",
            target_input=TARGET_NONCE_PLACEHOLDER + "\n",
            final_input="final\n",
            ready_marker=_READY_MARKER,
            target_marker=_TARGET_MARKER,
            timeout=context.timeout_seconds,
            cwd=context.bench_root,
            cpu=context.cpu,
            env=_environment(context, julia=julia),
        )
        values = parse_key_values(process.get("stdout", ""))
        result = _result(request, values)
        class_group_proven = parse_bool(values, "class_group_grh_free") is True
        unit_group_proven = parse_bool(values, "unit_group_grh_free") is True
        class_group_flag_read = parse_bool(values, "class_group_grh_free") is not None
        unit_group_flag_read = parse_bool(values, "unit_group_grh_free") is not None
        paired = request.jit_repetitions == 1
        results_agree = (
            parse_bool(values, "jit_results_agree") is True
            if paired
            else True
        )
        unit_error = (
            unit_count_error("Hecke", result)
            if request.operation in _CLASS_UNIT_OPERATIONS
            else None
        )
        # A returned unit count that disagrees with r1 + r2 - 1 means the
        # `UnitGrpCtx` the adapter read the count from is not the full-rank
        # unit group Hecke's own class-group proof asserted and saturated
        # against (_class_unit_group asserts U.full_rank before
        # _class_group_proof runs; saturate!/simplify fold U.units into the
        # relation lattice they saturate). A count below r1 + r2 - 1 means
        # that assertion did not hold; a count above r1 + r2 - 1 means the
        # list the adapter read is longer than the group the proof asserted
        # and saturated against, so it is equally not that group. Either way
        # the returned count is not the full-rank group the class-group
        # proof used, so a mismatch also invalidates the class-group label:
        # it was proven using that same (possibly not-full-rank, or not the
        # same) unit group.
        class_group_proven_effective = class_group_proven and unit_error is None
        unit_group_proven_effective = unit_group_proven and unit_error is None
        proven = class_group_proven_effective and unit_group_proven_effective
        result_complete = (
            process_state_is_valid(process)
            and process["success"]
            and _complete(request, result, proven)
            and results_agree
            and unit_error is None
        )
        reported_threads = parse_int(values, "reported_threads")
        thread_count = {
            "requested": _REQUESTED_THREADS,
            "reported": reported_threads,
            "matches_requested": reported_threads == _REQUESTED_THREADS,
            "source": _THREAD_COUNT_SOURCE,
        }
        success = result_complete and thread_count["matches_requested"]
        if request.operation == "class_unit_grh":
            # Conditional route.  The class label is grh while
            # ClassGrpCtx.GRH holds; the unit label is grh while UnitGrpCtx.GRH
            # holds and the rank is positive (a rank-zero unit group is torsion
            # only, hence proven), the existing bench_unit_group_grh_free rule.
            # The overall label stays grh and is never relabelled proven.
            read_ok = class_group_flag_read and unit_group_flag_read
            conditional_ok = success and unit_error is None
            proof = {
                "certification_status": "grh" if conditional_ok else "unknown",
                "class_group_proof_status": (
                    "unknown"
                    if not read_ok or unit_error is not None
                    else "proven" if class_group_proven else "grh"
                ),
                "unit_group_proof_status": (
                    "unknown"
                    if not read_ok or unit_error is not None
                    else "proven" if unit_group_proven else "grh"
                ),
                "regulator_proof_status": "grh" if conditional_ok else "unknown",
                "proof_complete": False,
                "conditional_result_complete": conditional_ok,
                "final_result_published": success,
            }
        elif request.operation in _CLASS_UNIT_OPERATIONS:
            # `proven` already folds in `unit_error is None` via
            # `class_group_proven_effective`/`unit_group_proven_effective`, so
            # reuse it here instead of re-deriving that condition.
            if proven:
                certification_status = "proven"
            elif unit_error is not None:
                # A unit-count mismatch is a readback disagreement, not a
                # Hecke proof failure: `class_group_proven`/`unit_group_proven`
                # above (Hecke's own GRH flags) may still both be true. The
                # adapter cannot tell which unit group the class proof
                # actually used, so it reports unknown rather than claiming
                # the certification failed, matching Magma.
                certification_status = "unknown"
            else:
                certification_status = "failed"
            proof = {
                "certification_status": certification_status,
                "class_group_proof_status": (
                    "proven" if class_group_proven_effective else "unknown"
                ),
                "unit_group_proof_status": (
                    "proven" if unit_group_proven_effective else "unknown"
                ),
                "regulator_proof_status": (
                    "proven" if proven else "unknown"
                ),
                "proof_complete": proven,
                "final_result_published": success,
            }
        else:
            proof = {
                "certification_status": "not_applicable",
                "final_result_published": success,
            }
        marked_target_cpu_ms = process.get("target_cpu_ms")
        marked_target_wall_ms = process.get("target_wall_ms")
        target_cpu_ms = (
            None
            if paired
            else parse_float(values, "internal_target_cpu_ms")
        )
        target_wall_ms = (
            None
            if paired
            else parse_float(values, "internal_target_wall_ms")
        )
        timing = {
            "algorithm_clock": "clock_gettime_process_cpu",
            "wall_clock": "julia_time_ns_monotonic",
            "component_clock": "julia_time_ns_monotonic",
            "scope": _TIMING_SCOPES[request.operation],
            "preparation_excluded": True,
            "jit_pair": paired,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "marked_target_cpu_ms": marked_target_cpu_ms,
            "marked_target_wall_ms": marked_target_wall_ms,
            "marked_process_affinity": process.get("effective_affinity"),
            "marked_process_supervisor_affinity": process.get(
                "supervisor_affinity"
            ),
            "marked_process_supervisor_isolation_tier": process.get(
                "supervisor_isolation_tier"
            ),
            "cpu_launcher_executable": process.get("launcher_executable"),
            "cpu_launcher_sha256": process.get("launcher_executable_sha256"),
            "components_ms": (
                {}
                if paired
                else _components(request, values)
            ),
        }
        if request.operation in _CLASS_UNIT_OPERATIONS:
            timing.update(
                {
                    "class_unit_cache_policy": _CLASS_UNIT_CACHE_POLICY,
                    "cross_sample_mathematical_cache_reuse": False,
                }
            )
        error = process.get("error")
        if (
            not success
            and error is None
            and result_complete
            and not thread_count["matches_requested"]
        ):
            error = (
                "Hecke thread-count contract failed: requested "
                f"{_REQUESTED_THREADS}, reported {reported_threads!r}"
            )
        if not success and error is None and paired and not results_agree:
            error = "Hecke first-call and repeat-call results disagreed"
        if not success and error is None and unit_error is not None:
            error = unit_error
        if not success and error is None:
            error = "Hecke operation failed or returned incomplete output"
        timing_samples = _timing_sample_payloads(
            request,
            values,
            process,
            observation_success=success,
            results_agree=results_agree,
        )
        payload: dict[str, Any] = {
            "engine": self.name,
            "algorithm": "external",
            "available": process.get("available", True) is True,
            "success": success,
            "timeout": process.get("timeout") is True,
            "status": (
                "ok"
                if success
                else "timeout"
                if process.get("timeout") is True
                else "thread_contract"
                if result_complete and not thread_count["matches_requested"]
                else "compute_error"
            ),
            "error": error,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "process_wall_ms": process.get("process_wall_ms"),
            "engine_identity": identity_with_executed_digest(identity, process),
            "cmd": process.get("cmd"),
            "result": result,
            "proof": proof,
            "thread_count": thread_count,
            "timing": timing,
            "timing_samples": timing_samples,
        }
        if not success:
            payload["diagnostics"] = {
                "stdout": process.get("stdout", ""),
                "stderr": process.get("stderr", ""),
                "cmd": process.get("cmd"),
            }
        return payload
