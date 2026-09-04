"""Hecke benchmark adapter using fresh, uncached number fields.

The rigorous class/unit route follows Hecke's public API with
``class_group(...; GRH=false, redo=true)`` and
``unit_group(...; GRH=false)``.
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
)


_READY_MARKER = "__SILEX_BENCH_HECKE_READY__"
_TARGET_MARKER = "__SILEX_BENCH_HECKE_TARGET_DONE__"


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


def _warmup_call(request: SampleRequest) -> str:
    if request.warmup is None:
        return ""
    if request.warmup.degree != request.field.degree:
        raise ValueError("Hecke warmup field must have the target degree")
    if (
        request.warmup.id == request.field.id
        or request.warmup.coefficients_low_to_high
        == request.field.coefficients_low_to_high
    ):
        raise ValueError("Hecke warmup field must be distinct from the target")
    polynomial = polynomial_expr(request.warmup.coefficients_low_to_high)
    function = {
        "class_unit_proven": "bench_warm_class_unit",
        "maximal_order": "bench_warm_maximal_order",
        "ideal_multiply": "bench_warm_ideal_multiply",
        "element_square_root": "bench_warm_square_root",
    }.get(request.operation)
    if function is None:
        raise ValueError(f"unsupported Hecke operation: {request.operation}")
    return f"{function}({polynomial})"


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

function bench_warm_class_unit(f)
  K, a = bench_field(f)
  O = maximal_order(K)
  C, mC = class_group(O; GRH = false, redo = true)
  U, mU = unit_group(O; GRH = false)
  return order(C), rank(U)
end

function bench_warm_maximal_order(f)
  K, a = bench_field(f)
  O = maximal_order(K)
  return discriminant(O)
end

function bench_warm_ideal_multiply(f)
  K, a = bench_field(f)
  O = maximal_order(K)
  left = ideal(O, ZZ(2))
  right = ideal(O, ZZ(3))
  return norm(left * right)
end

function bench_warm_square_root(f)
  K, a = bench_field(f)
  value = (K(1) + a)^2
  found, root = is_square_with_sqrt(value)
  found && root^2 == value || error("warmup square-root verification failed")
  return root
end
"""


def _programs(request: SampleRequest) -> tuple[str, str, str]:
    polynomial = polynomial_expr(request.field.coefficients_low_to_high)
    warmup = _warmup_call(request)
    setup = ""
    if request.operation == "ideal_multiply":
        setup = f"""
P = {polynomial}
target_K, target_a = bench_field(P)
target_O = maximal_order(target_K)
left_ideal = ideal(target_O, ZZ(2))
right_ideal = ideal(target_O, ZZ(3))
"""
    elif request.operation == "element_square_root":
        setup = f"""
P = {polynomial}
target_K, target_a = bench_field(P)
square_target = (target_K(1) + target_a)^2
"""

    ready = f"""
{_HELPERS}
Random.seed!({request.seed})
{warmup}
Random.seed!({request.seed})
{setup}
GC.gc()
println("{_READY_MARKER}")
flush(stdout)
"""

    if request.operation == "class_unit_proven":
        target = f"""
P = {polynomial}
target_cpu_t0 = bench_process_cpu_ns()
total_t0 = time_ns()
field_t0 = time_ns()
target_K, target_a = bench_field(P)
target_signature = signature(target_K)
polynomial_discriminant = discriminant(P)
field_setup_ms = (time_ns() - field_t0) / 1.0e6
maximal_t0 = time_ns()
target_O = maximal_order(target_K)
maximal_order_discriminant = discriminant(target_O)
maximal_order_ms = (time_ns() - maximal_t0) / 1.0e6
class_t0 = time_ns()
target_C, target_mC = class_group(target_O; GRH = false, redo = true)
class_group_ms = (time_ns() - class_t0) / 1.0e6
unit_t0 = time_ns()
target_U, target_mU = unit_group(target_O; GRH = false)
unit_group_ms = (time_ns() - unit_t0) / 1.0e6
internal_target_wall_ms = (time_ns() - total_t0) / 1.0e6
target_cpu_t1 = bench_process_cpu_ns()
internal_target_cpu_ms = target_cpu_t0 === nothing || target_cpu_t1 === nothing ||
                         target_cpu_t1 < target_cpu_t0 ? nothing :
                         (target_cpu_t1 - target_cpu_t0) / 1.0e6
println("{_TARGET_MARKER}:" * benchmark_target_nonce)
flush(stdout)
"""
        final = """
println("component_field_setup_ms=", field_setup_ms)
println("component_maximal_order_ms=", maximal_order_ms)
println("component_class_group_ms=", class_group_ms)
println("component_unit_group_ms=", unit_group_ms)
println("internal_target_cpu_ms=", internal_target_cpu_ms)
println("internal_target_wall_ms=", internal_target_wall_ms)
println("class_order=", order(target_C))
println("class_invariants=", join(string.(elementary_divisors(target_C)), ","))
println("unit_rank=", target_signature[1] + target_signature[2] - 1)
println("signature_r1=", target_signature[1])
println("signature_r2=", target_signature[2])
println("polynomial_discriminant=", polynomial_discriminant)
println("maximal_order_discriminant=", maximal_order_discriminant)
println("proof_complete=true")
flush(stdout)
exit()
"""
        return ready, target, final

    if request.operation == "maximal_order":
        target = f"""
P = {polynomial}
target_cpu_t0 = bench_process_cpu_ns()
total_t0 = time_ns()
field_t0 = time_ns()
target_K, target_a = bench_field(P)
target_signature = signature(target_K)
polynomial_discriminant = discriminant(P)
field_setup_ms = (time_ns() - field_t0) / 1.0e6
maximal_t0 = time_ns()
target_O = maximal_order(target_K)
maximal_order_discriminant = discriminant(target_O)
maximal_order_ms = (time_ns() - maximal_t0) / 1.0e6
internal_target_wall_ms = (time_ns() - total_t0) / 1.0e6
target_cpu_t1 = bench_process_cpu_ns()
internal_target_cpu_ms = target_cpu_t0 === nothing || target_cpu_t1 === nothing ||
                         target_cpu_t1 < target_cpu_t0 ? nothing :
                         (target_cpu_t1 - target_cpu_t0) / 1.0e6
println("{_TARGET_MARKER}:" * benchmark_target_nonce)
flush(stdout)
"""
        final = """
println("component_field_setup_ms=", field_setup_ms)
println("component_maximal_order_ms=", maximal_order_ms)
println("internal_target_cpu_ms=", internal_target_cpu_ms)
println("internal_target_wall_ms=", internal_target_wall_ms)
println("polynomial_discriminant=", polynomial_discriminant)
println("maximal_order_discriminant=", maximal_order_discriminant)
println("signature_r1=", target_signature[1])
println("signature_r2=", target_signature[2])
flush(stdout)
exit()
"""
        return ready, target, final

    if request.operation == "ideal_multiply":
        target = f"""
target_cpu_t0 = bench_process_cpu_ns()
operation_t0 = time_ns()
product_ideal = left_ideal * right_ideal
ideal_multiply_ms = (time_ns() - operation_t0) / 1.0e6
target_cpu_t1 = bench_process_cpu_ns()
internal_target_cpu_ms = target_cpu_t0 === nothing || target_cpu_t1 === nothing ||
                         target_cpu_t1 < target_cpu_t0 ? nothing :
                         (target_cpu_t1 - target_cpu_t0) / 1.0e6
println("{_TARGET_MARKER}:" * benchmark_target_nonce)
flush(stdout)
"""
        final = """
println("component_ideal_multiply_ms=", ideal_multiply_ms)
println("internal_target_cpu_ms=", internal_target_cpu_ms)
println("internal_target_wall_ms=", ideal_multiply_ms)
println("ideal_norm=", norm(product_ideal))
flush(stdout)
exit()
"""
        return ready, target, final

    if request.operation == "element_square_root":
        target = f"""
target_cpu_t0 = bench_process_cpu_ns()
operation_t0 = time_ns()
root_found, square_root = is_square_with_sqrt(square_target)
square_root_ms = (time_ns() - operation_t0) / 1.0e6
target_cpu_t1 = bench_process_cpu_ns()
internal_target_cpu_ms = target_cpu_t0 === nothing || target_cpu_t1 === nothing ||
                         target_cpu_t1 < target_cpu_t0 ? nothing :
                         (target_cpu_t1 - target_cpu_t0) / 1.0e6
println("{_TARGET_MARKER}:" * benchmark_target_nonce)
flush(stdout)
"""
        final = """
root_verified = root_found && square_root^2 == square_target
println("component_square_root_ms=", square_root_ms)
println("internal_target_cpu_ms=", internal_target_cpu_ms)
println("internal_target_wall_ms=", square_root_ms)
println("root_found=", root_found)
println("root_verified=", root_verified)
flush(stdout)
exit()
"""
        return ready, target, final

    raise ValueError(f"unsupported Hecke operation: {request.operation}")


def _result(request: SampleRequest, values: dict[str, str]) -> dict[str, Any]:
    if request.operation == "class_unit_proven":
        r1 = parse_int(values, "signature_r1")
        r2 = parse_int(values, "signature_r2")
        return {
            "class_order": values.get("class_order"),
            "class_invariants": _parse_invariants(values.get("class_invariants")),
            "unit_rank": parse_int(values, "unit_rank"),
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


def _components(request: SampleRequest, values: dict[str, str]) -> dict[str, Any]:
    if request.operation == "class_unit_proven":
        return {
            "field_setup": parse_float(values, "component_field_setup_ms"),
            "maximal_order": parse_float(values, "component_maximal_order_ms"),
            "class_group": parse_float(values, "component_class_group_ms"),
            "unit_group": parse_float(values, "component_unit_group_ms"),
        }
    if request.operation == "maximal_order":
        return {
            "field_setup": parse_float(values, "component_field_setup_ms"),
            "maximal_order": parse_float(values, "component_maximal_order_ms"),
        }
    key = {
        "ideal_multiply": "component_ideal_multiply_ms",
        "element_square_root": "component_square_root_ms",
    }.get(request.operation)
    return (
        {request.operation: parse_float(values, key)} if key is not None else {}
    )


def _complete(request: SampleRequest, result: dict[str, Any], proven: bool) -> bool:
    if request.operation == "class_unit_proven":
        return (
            proven
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
        proven = parse_bool(values, "proof_complete") is True
        success = (
            process_state_is_valid(process)
            and process["success"]
            and _complete(request, result, proven)
        )
        if request.operation == "class_unit_proven":
            proof = {
                "certification_status": "proven" if proven else "failed",
                "class_group_proof_status": (
                    "proven" if proven else "unknown"
                ),
                "unit_group_proof_status": (
                    "proven" if proven else "unknown"
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
        internal_target_cpu_ms = parse_float(
            values, "internal_target_cpu_ms"
        )
        internal_target_wall_ms = parse_float(
            values, "internal_target_wall_ms"
        )
        target_cpu_ms = internal_target_cpu_ms
        target_wall_ms = internal_target_wall_ms
        timing = {
            "algorithm_clock": "clock_gettime_process_cpu",
            "wall_clock": "julia_time_ns_wall",
            "component_clock": "julia_time_ns_wall",
            "scope": (
                "field_setup_through_result"
                if request.operation in {"class_unit_proven", "maximal_order"}
                else "named_operation_only"
            ),
            "warmup_excluded": True,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "marked_target_cpu_ms": marked_target_cpu_ms,
            "marked_target_wall_ms": marked_target_wall_ms,
            "marked_process_affinity": process.get("effective_affinity"),
            "cpu_launcher_executable": process.get("launcher_executable"),
            "cpu_launcher_sha256": process.get("launcher_executable_sha256"),
            "components_ms": _components(request, values),
        }
        error = process.get("error")
        if not success and error is None:
            error = "Hecke operation failed or returned incomplete output"
        payload: dict[str, Any] = {
            "engine": self.name,
            "algorithm": "external",
            "available": process.get("available", True) is True,
            "success": success,
            "timeout": process.get("timeout") is True,
            "status": "ok" if success else (
                "timeout" if process.get("timeout") is True else "compute_error"
            ),
            "error": error,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "process_wall_ms": process.get("process_wall_ms"),
            "engine_identity": identity_with_executed_digest(identity, process),
            "cmd": process.get("cmd"),
            "result": result,
            "proof": proof,
            "timing": timing,
        }
        if not success:
            payload["diagnostics"] = {
                "stdout": process.get("stdout", ""),
                "stderr": process.get("stderr", ""),
                "cmd": process.get("cmd"),
            }
        return payload
