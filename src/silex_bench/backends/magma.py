"""Magma backend with rigorous, isolated benchmark operations.

The class/unit proof flags and operation boundaries in this adapter follow the
Magma V2.28 handbook.  In particular, the proven route deliberately uses
``Proof := "Full"`` and ``GRH := false``; it does not select or tune Magma's
underlying algorithms.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..model import (
    BackendContext,
    FieldSpec,
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
    parse_bool,
    parse_float,
    parse_int,
    process_state_is_valid,
    successful_probe,
    unavailable,
    unavailable_probe,
    unit_count_error,
)


_READY_MARKER = "__SILEX_BENCH_MAGMA_READY__"
_TARGET_MARKER = "__SILEX_BENCH_MAGMA_TARGET_DONE__"
_REQUESTED_THREADS = 1
_THREAD_COUNT_SOURCE = "magma_getnthreads_runtime_query"
_LICENSE_FAILURE_PATTERNS = (
    "couldn't create socket for mac address startup",
    "could not create socket for mac address startup",
    "unable to find a valid magma license",
    "no valid magma license",
    "this machine is not licensed",
    "magma license manager",
)


def _environment(context: BackendContext) -> dict[str, str]:
    return {**os.environ, **context.environment}


def _configured_executable(context: BackendContext) -> tuple[str, str]:
    configured = context.tools.get("magma")
    if configured is not None and str(configured).strip():
        return str(configured), "tools.magma"
    environment = context.environment.get("SILEX_BENCH_MAGMA")
    if not environment:
        environment = os.environ.get("SILEX_BENCH_MAGMA")
    if environment:
        return environment, "SILEX_BENCH_MAGMA"
    return "magma", "PATH"


def _resolve_magma(context: BackendContext) -> tuple[str | None, str | None]:
    candidate, source = _configured_executable(context)
    executable = resolve_executable(candidate)
    if executable is None:
        return None, f"Magma executable from {source} was not found: {candidate}"
    return executable, None


def _license_error(text: str) -> str | None:
    lowered = text.lower()
    if any(pattern in lowered for pattern in _LICENSE_FAILURE_PATTERNS):
        return (
            "Magma could not initialize its host-bound license/socket. "
            "Run silex-bench outside the filesystem sandbox or grant host "
            "execution for the configured Magma batch command."
        )
    return None


def _failure_detail(raw: dict[str, Any]) -> str:
    output = "\n".join(
        str(raw.get(key, "")) for key in ("stdout", "stderr")
    ).strip()
    license_error = _license_error(output)
    if license_error is not None:
        return license_error
    message = str(raw.get("error") or "Magma execution failed")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # Magma prints a resource-usage trailer on exit; it is not the error.
    informative = [
        line for line in lines if not line.startswith("Total time:")
    ]
    lines = informative or lines
    if lines:
        message += f": {lines[-1][:500]}"
    return message


def _is_license_failure(raw: dict[str, Any]) -> bool:
    output = "\n".join(
        str(raw.get(key, "")) for key in ("stdout", "stderr")
    )
    return _license_error(output) is not None


def _parse_invariants(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return canonical_invariants(re.findall(r"-?\d+", value))


def _field_program(field: FieldSpec, suffix: str) -> str:
    variable = f"x_{suffix}"
    polynomial = polynomial_expr(field.coefficients_low_to_high, variable)
    return f"""
Qx_{suffix}<{variable}> := PolynomialRing(Rationals());
f_{suffix} := {polynomial};
K_{suffix}<a_{suffix}> := NumberField(f_{suffix});
"""


_CLASS_UNIT_OPERATIONS = ("class_unit_proven", "class_unit_grh")


def _program_parts(request: SampleRequest) -> tuple[str, str, str]:
    ready = f"""
SetNthreads(1);
benchmark_reported_threads := GetNthreads();
SetSeed({request.seed});
"""

    setup = _field_program(request.field, "target")
    # Proven route: Proof := "Full" and GRH := false.  Conditional route:
    # Proof := "GRH" (a GRH-based bound replaces the Minkowski bound) and
    # UnitGroup GRH := true (skips the unit proof phase), per the V2.28
    # handbook ClassGroup and UnitGroup entries.
    class_proof = "GRH" if request.operation == "class_unit_grh" else "Full"
    unit_grh = "true" if request.operation == "class_unit_grh" else "false"
    if request.operation in _CLASS_UNIT_OPERATIONS:
        ready += setup + "O_target := MaximalOrder(K_target);\n"
        target = f"""
target_cpu_start := Cputime();
target_wall_start := Realtime();
class_cpu_start := Cputime();
class_wall_start := Realtime();
C_target, class_map_target := ClassGroup(O_target : Proof := "{class_proof}");
class_cpu_seconds := Cputime(class_cpu_start);
class_wall_seconds := Realtime(class_wall_start);
unit_cpu_start := Cputime();
unit_wall_start := Realtime();
U_target, unit_map_target := UnitGroup(O_target : GRH := {unit_grh});
unit_cpu_seconds := Cputime(unit_cpu_start);
unit_wall_seconds := Realtime(unit_wall_start);
target_internal_cpu_seconds := Cputime(target_cpu_start);
target_internal_wall_seconds := Realtime(target_wall_start);
printf "{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}\\n";
"""
        final = """
r1_target, r2_target := Signature(K_target);
printf "class_order=%o\\n", Order(C_target);
printf "class_invariants=%o\\n", AbelianInvariants(C_target);
fundamental_units_target := [
    unit_map_target(U_target.i) : i in [1..Ngens(U_target)]
    | Order(U_target.i) eq 0
];
printf "fundamental_unit_count=%o\\n", #fundamental_units_target;
printf "signature_r1=%o\\n", r1_target;
printf "signature_r2=%o\\n", r2_target;
printf "polynomial_discriminant=%o\\n", Discriminant(f_target);
printf "maximal_order_discriminant=%o\\n", Discriminant(O_target);
printf "class_cpu_seconds=%o\\n", class_cpu_seconds;
printf "class_wall_seconds=%o\\n", class_wall_seconds;
printf "unit_cpu_seconds=%o\\n", unit_cpu_seconds;
printf "unit_wall_seconds=%o\\n", unit_wall_seconds;
printf "target_internal_cpu_seconds=%o\\n", target_internal_cpu_seconds;
printf "target_internal_wall_seconds=%o\\n", target_internal_wall_seconds;
quit;
"""
    elif request.operation == "maximal_order":
        ready += setup
        target = f"""
target_cpu_start := Cputime();
target_wall_start := Realtime();
O_target := MaximalOrder(K_target);
target_internal_cpu_seconds := Cputime(target_cpu_start);
target_internal_wall_seconds := Realtime(target_wall_start);
printf "{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}\\n";
"""
        final = """
r1_target, r2_target := Signature(K_target);
printf "signature_r1=%o\\n", r1_target;
printf "signature_r2=%o\\n", r2_target;
printf "polynomial_discriminant=%o\\n", Discriminant(f_target);
printf "maximal_order_discriminant=%o\\n", Discriminant(O_target);
printf "target_internal_cpu_seconds=%o\\n", target_internal_cpu_seconds;
printf "target_internal_wall_seconds=%o\\n", target_internal_wall_seconds;
quit;
"""
    elif request.operation == "ideal_multiply":
        ready += setup + """
O_target := MaximalOrder(K_target);
I2_target := ideal<O_target | 2>;
I3_target := ideal<O_target | 3>;
"""
        target = f"""
target_cpu_start := Cputime();
target_wall_start := Realtime();
product_target := I2_target * I3_target;
target_internal_cpu_seconds := Cputime(target_cpu_start);
target_internal_wall_seconds := Realtime(target_wall_start);
printf "{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}\\n";
"""
        final = """
printf "ideal_norm=%o\\n", Norm(product_target);
printf "target_internal_cpu_seconds=%o\\n", target_internal_cpu_seconds;
printf "target_internal_wall_seconds=%o\\n", target_internal_wall_seconds;
quit;
"""
    elif request.operation == "element_square_root":
        ready += setup + "square_target := (K_target!1 + a_target)^2;\n"
        target = f"""
target_cpu_start := Cputime();
target_wall_start := Realtime();
root_found, root_target := IsSquare(square_target);
target_internal_cpu_seconds := Cputime(target_cpu_start);
target_internal_wall_seconds := Realtime(target_wall_start);
printf "{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}\\n";
"""
        final = """
root_verified := false;
if root_found then
    root_verified := root_target^2 eq square_target;
end if;
printf "root_found=%o\\n", root_found;
printf "root_verified=%o\\n", root_verified;
printf "target_internal_cpu_seconds=%o\\n", target_internal_cpu_seconds;
printf "target_internal_wall_seconds=%o\\n", target_internal_wall_seconds;
quit;
"""
    else:
        raise ValueError(f"unsupported Magma operation: {request.operation}")
    ready += f'printf "{_READY_MARKER}\\n";\n'
    if final.count("quit;\n") != 1:
        raise ValueError("Magma final program must end with exactly one quit")
    final = final.replace(
        "quit;\n",
        'printf "reported_threads=%o\\n", benchmark_reported_threads;\nquit;\n',
    )
    return ready, target, final


def _milliseconds(values: dict[str, str], key: str) -> float | None:
    seconds = parse_float(values, key)
    return None if seconds is None else seconds * 1000.0


def _normalized_result(
    operation: str, values: dict[str, str]
) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any]
    required: list[str]
    if operation in _CLASS_UNIT_OPERATIONS:
        r1 = parse_int(values, "signature_r1")
        r2 = parse_int(values, "signature_r2")
        result = {
            "class_order": values.get("class_order"),
            "class_invariants": _parse_invariants(
                values.get("class_invariants")
            ),
            "unit_rank": parse_int(values, "fundamental_unit_count"),
            "signature": [r1, r2]
            if r1 is not None and r2 is not None
            else None,
            "polynomial_discriminant": values.get("polynomial_discriminant"),
            "maximal_order_discriminant": values.get(
                "maximal_order_discriminant"
            ),
        }
        required = [
            "class_order",
            "class_invariants",
            "unit_rank",
            "signature",
            "maximal_order_discriminant",
        ]
    elif operation == "maximal_order":
        r1 = parse_int(values, "signature_r1")
        r2 = parse_int(values, "signature_r2")
        result = {
            "polynomial_discriminant": values.get("polynomial_discriminant"),
            "maximal_order_discriminant": values.get(
                "maximal_order_discriminant"
            ),
            "signature": [r1, r2]
            if r1 is not None and r2 is not None
            else None,
        }
        required = ["maximal_order_discriminant"]
    elif operation == "ideal_multiply":
        result = {"ideal_norm": values.get("ideal_norm")}
        required = ["ideal_norm"]
    elif operation == "element_square_root":
        result = {
            "root_found": parse_bool(values, "root_found"),
            "root_verified": parse_bool(values, "root_verified"),
        }
        required = ["root_found", "root_verified"]
    else:
        raise ValueError(f"unsupported Magma operation: {operation}")
    missing = [key for key in required if result.get(key) is None]
    if operation == "element_square_root":
        missing.extend(
            key
            for key in required
            if result.get(key) is not True and key not in missing
        )
    return result, missing


def _timing(
    operation: str,
    values: dict[str, str],
    *,
    target_cpu_ms: float | None,
    target_wall_ms: float | None,
    marked_target_cpu_ms: float | None,
    marked_target_wall_ms: float | None,
    marked_process_affinity: list[int] | None,
    marked_process_supervisor_affinity: list[int] | None = None,
    marked_process_supervisor_isolation_tier: int | None = None,
) -> dict[str, Any]:
    scopes = {
        "class_unit_proven": "class_and_unit_group_only",
        "class_unit_grh": "class_and_unit_group_only",
        "maximal_order": "maximal_order_only",
        "ideal_multiply": "ideal_multiplication_only",
        "element_square_root": "number_field_element_is_square_only",
    }
    timing: dict[str, Any] = {
        "scope": scopes[operation],
        "algorithm_clock": "magma_cputime",
        "wall_clock": "magma_realtime",
        "component_clock": "magma_cputime_and_realtime",
        "preparation_excluded": True,
        "target_cpu_ms": target_cpu_ms,
        "target_wall_ms": target_wall_ms,
        "internal_cpu_ms": _milliseconds(
            values, "target_internal_cpu_seconds"
        ),
        "internal_wall_ms": _milliseconds(
            values, "target_internal_wall_seconds"
        ),
        "marked_target_cpu_ms": marked_target_cpu_ms,
        "marked_target_wall_ms": marked_target_wall_ms,
        "marked_process_affinity": marked_process_affinity,
        "marked_process_supervisor_affinity": marked_process_supervisor_affinity,
        "marked_process_supervisor_isolation_tier": (
            marked_process_supervisor_isolation_tier
        ),
    }
    if operation in _CLASS_UNIT_OPERATIONS:
        timing["components_ms"] = {
            component: {
                "cpu_ms": _milliseconds(values, f"{component}_cpu_seconds"),
                "wall_ms": _milliseconds(
                    values, f"{component}_wall_seconds"
                ),
            }
            for component in ("class", "unit")
        }
    return timing


class MagmaBackend(BackendAdapter):
    """Run Magma V2.28 operations in a fresh marked process per sample."""

    name = "magma"

    def __init__(self) -> None:
        self._probe: dict[str, Any] | None = None

    def probe(self, context: BackendContext) -> dict[str, Any]:
        if self._probe is not None:
            return dict(self._probe)
        executable, error = _resolve_magma(context)
        if executable is None:
            self._probe = unavailable_probe(
                self.name, error or "Magma is unavailable"
            )
            return dict(self._probe)
        program = """
version_major, version_minor, version_patch := GetVersion();
printf "magma_version=%o.%o-%o\\n", version_major, version_minor, version_patch;
quit;
"""
        raw = run_process(
            [executable, "-bn"],
            timeout=min(15.0, max(1.0, context.timeout_seconds)),
            cwd=context.bench_root,
            stdin=program,
            env=_environment(context),
        )
        values = parse_key_values(str(raw.get("stdout", "")))
        if (
            not process_state_is_valid(raw)
            or not raw["success"]
            or "magma_version" not in values
        ):
            detail = _failure_detail(raw)
            self._probe = unavailable_probe(
                self.name,
                detail,
                timeout=raw.get("timeout") is True,
            )
            return dict(self._probe)
        identity = {
            "engine": self.name,
            "executable": str(Path(executable).resolve()),
            "version": values["magma_version"],
        }
        identity = identity_with_executed_digest(identity, raw)
        self._probe = successful_probe(self.name, identity)
        return dict(self._probe)

    def run(
        self, request: SampleRequest, context: BackendContext
    ) -> dict[str, Any]:
        probe = self.probe(context)
        if not probe.get("available"):
            payload = unavailable(self.name, str(probe.get("error")))
            payload["engine_identity"] = probe.get("engine_identity")
            return payload
        executable = str(probe["engine_identity"]["executable"])
        try:
            ready, target, final = _program_parts(request)
        except ValueError as exc:
            payload = unavailable(self.name, str(exc))
            payload.update(
                {
                    "available": True,
                    "status": "invalid_request",
                    "executable": executable,
                }
            )
            return payload

        raw = run_marked_process(
            [executable, "-bn"],
            ready_input=ready,
            target_input=target,
            final_input=final,
            ready_marker=_READY_MARKER,
            target_marker=_TARGET_MARKER,
            timeout=context.timeout_seconds,
            cwd=context.bench_root,
            cpu=context.cpu,
            env=_environment(context),
        )
        values = parse_key_values(str(raw.get("stdout", "")))
        result, missing = _normalized_result(request.operation, values)
        process_success = process_state_is_valid(raw) and raw["success"]
        unit_error = (
            unit_count_error("Magma", result)
            if request.operation in _CLASS_UNIT_OPERATIONS and not missing
            else None
        )
        reported_threads = parse_int(values, "reported_threads")
        thread_count = {
            "requested": _REQUESTED_THREADS,
            "reported": reported_threads,
            "matches_requested": reported_threads == _REQUESTED_THREADS,
            "source": _THREAD_COUNT_SOURCE,
        }
        result_complete = (
            process_success and not missing and unit_error is None
        )
        success = result_complete and thread_count["matches_requested"]
        license_failure = _is_license_failure(raw)
        if success:
            status = "ok"
            run_error = None
        elif result_complete and not thread_count["matches_requested"]:
            status = "thread_contract"
            run_error = (
                "Magma thread-count contract failed: requested "
                f"{_REQUESTED_THREADS}, reported {reported_threads!r}"
            )
        elif license_failure:
            status = "unavailable"
            run_error = _failure_detail(raw)
        elif raw.get("timeout") is True:
            status = "timeout"
            run_error = str(raw.get("error") or "Magma process timed out")
        else:
            status = "compute_error"
            run_error = _failure_detail(raw)
            if process_success and missing:
                run_error = (
                    "Magma output omitted required values: "
                    + ", ".join(missing)
                )
            elif process_success and unit_error is not None:
                run_error = unit_error

        proof: dict[str, Any] = {}
        if request.operation == "class_unit_grh":
            # Labels come from the call contract, as for the proven route:
            # ClassGroup with Proof := "GRH" is "correct under the GRH" and
            # UnitGroup with GRH := true has "the same level of rigour"
            # (V2.28 handbook).  Magma reports no proof state and Bench has
            # not verified it, so no unconditional proof is claimed.
            proof = {
                "certification_status": "grh" if success else "unknown",
                "class_group_proof_status": "grh" if success else "unknown",
                "unit_group_proof_status": "grh" if success else "unknown",
                "regulator_proof_status": "grh" if success else "unknown",
                "proof_complete": False,
                "conditional_result_complete": success,
                "final_result_published": success,
            }
        elif request.operation in _CLASS_UNIT_OPERATIONS:
            # The V2.28 handbook documents no intrinsic that reports the
            # proof state of a computed class or unit group.  The labels
            # therefore come from the call contract: ClassGroup with
            # Proof := "Full" returns a guaranteed result and UnitGroup with
            # GRH := false runs its rigorous proof phase, so a call that
            # returns a complete, count-checked result is proven.
            proof = {
                "certification_status": "proven" if success else "unknown",
                "class_group_proof_status": (
                    "proven" if success else "unknown"
                ),
                "unit_group_proof_status": (
                    "proven" if success else "unknown"
                ),
                "regulator_proof_status": (
                    "proven" if success else "unknown"
                ),
                "proof_complete": success,
                "final_result_published": success,
            }
        else:
            proof = {
                "certification_status": "not_applicable",
                "final_result_published": success,
            }
        marked_target_cpu_ms = raw.get("target_cpu_ms")
        marked_target_wall_ms = raw.get("target_wall_ms")
        internal_target_cpu_ms = _milliseconds(
            values, "target_internal_cpu_seconds"
        )
        internal_target_wall_ms = _milliseconds(
            values, "target_internal_wall_seconds"
        )
        target_cpu_ms = internal_target_cpu_ms
        target_wall_ms = internal_target_wall_ms
        identity = identity_with_executed_digest(
            probe.get("engine_identity"), raw
        )
        payload = {
            "engine": self.name,
            "algorithm": "external",
            "available": not license_failure,
            "success": success,
            "timeout": raw.get("timeout") is True,
            "status": status,
            "error": run_error,
            "engine_identity": identity,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "process_wall_ms": raw.get("process_wall_ms"),
            "cmd": raw.get("cmd"),
            "result": result,
            "proof": proof,
            "thread_count": thread_count,
            "timing": _timing(
                request.operation,
                values,
                target_cpu_ms=target_cpu_ms,
                target_wall_ms=target_wall_ms,
                marked_target_cpu_ms=marked_target_cpu_ms,
                marked_target_wall_ms=marked_target_wall_ms,
                marked_process_affinity=raw.get("effective_affinity"),
                marked_process_supervisor_affinity=raw.get("supervisor_affinity"),
                marked_process_supervisor_isolation_tier=raw.get(
                    "supervisor_isolation_tier"
                ),
            ),
        }
        payload["timing"]["cpu_launcher_executable"] = raw.get(
            "launcher_executable"
        )
        payload["timing"]["cpu_launcher_sha256"] = raw.get(
            "launcher_executable_sha256"
        )
        if not success:
            payload["diagnostics"] = {
                "stdout": raw.get("stdout", ""),
                "stderr": raw.get("stderr", ""),
                "cmd": raw.get("cmd"),
            }
        return payload
