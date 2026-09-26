"""Silex benchmark adapter."""

from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

from ..model import (
    MAX_SILEX_CMAKE_CACHE_BYTES,
    BackendContext,
    SampleRequest,
    backend_result_contract_errors,
    coeffs_arg,
    silex_executable_keys,
)
from ..process import TARGET_NONCE_PLACEHOLDER, run_marked_process
from ..util import (
    file_digest_nofollow,
    open_absolute_directory_nofollow,
    parse_bounded_json_bytes,
    read_bytes_nofollow,
)
from .base import (
    BackendAdapter,
    identity_with_executed_digest,
    nonempty_diagnostic,
    process_state_is_valid,
    successful_probe,
    unavailable,
    unavailable_probe,
)


_READY_MARKER = "__SILEX_BENCH_SILEX_READY__"
_TARGET_MARKER = "__SILEX_BENCH_SILEX_TARGET_DONE__"
_REQUESTED_THREADS = 1
_THREAD_COUNT_SOURCE = "silex_flint_get_num_threads"
_TIMING_SCOPES = {
    "class_unit_proven": "class_and_unit_group_only",
    "maximal_order": "maximal_order_only",
    "ideal_multiply": "ideal_multiplication_only",
    "element_square_root": "number_field_element_is_square_only",
}
SILEX_BENCHMARK_CMAKE_CACHE_REQUIREMENTS = (
    ("CMAKE_BUILD_TYPE:STRING", "Release"),
    ("SILEX_BUILD_BENCHMARK_ADAPTERS:BOOL", "ON"),
    ("SILEX_BUILD_TESTS:BOOL", "OFF"),
    ("SILEX_BUILD_BENCHMARKS:BOOL", "OFF"),
    ("SILEX_BUILD_EXAMPLES:BOOL", "OFF"),
    ("SILEX_BUILD_DOCS:BOOL", "OFF"),
    ("SILEX_ENABLE_LOGGING:BOOL", "OFF"),
    ("SILEX_ENABLE_DEBUG_CHECKS:BOOL", "OFF"),
    ("SILEX_ENABLE_PROFILING:BOOL", "OFF"),
    ("SILEX_ENABLE_SANITIZERS:BOOL", "OFF"),
    ("SILEX_ENABLE_FRAME_POINTERS:BOOL", "OFF"),
)
SILEX_BENCHMARK_CMAKE_ARGUMENTS = tuple(
    f"-D{key.split(':', 1)[0]}={value}"
    for key, value in SILEX_BENCHMARK_CMAKE_CACHE_REQUIREMENTS
)


def _thread_count(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("engine_thread_count")
    reported = value if isinstance(value, int) and not isinstance(value, bool) else None
    return {
        "requested": _REQUESTED_THREADS,
        "reported": reported,
        "matches_requested": reported == _REQUESTED_THREADS,
        "source": _THREAD_COUNT_SOURCE,
    }


def _json_payload(output: str) -> dict[str, Any]:
    if "\ufffd" in output:
        raise ValueError("Silex output contains invalid UTF-8 replacement evidence")
    start = output.find("{")
    if start < 0:
        raise ValueError("Silex output did not contain a JSON object")
    payload = parse_bounded_json_bytes(
        output[start:].encode("utf-8"),
        source="Silex output",
    )
    if not isinstance(payload, dict):
        raise ValueError("Silex JSON output was not an object")
    return payload


def _native_payload_type_error(
    payload: dict[str, Any],
    *,
    required_objects: tuple[str, ...] = (),
    required_booleans: tuple[str, ...] = (),
    optional_booleans: tuple[str, ...] = (),
) -> str | None:
    for key in required_objects:
        if type(payload.get(key)) is not dict:
            return f"native Silex field {key!r} must be an object"
    for key in required_booleans:
        if type(payload.get(key)) is not bool:
            return f"native Silex field {key!r} must be a Boolean"
    for key in optional_booleans:
        if key in payload and type(payload[key]) is not bool:
            return f"native Silex field {key!r} must be a Boolean when present"
    return None


def _native_failure_payload(
    process: dict[str, Any],
    identity: dict[str, Any],
    payload: dict[str, Any],
    *,
    operation: str,
    algorithm: str,
) -> dict[str, Any]:
    timeout = process.get("timeout") is True or payload.get("timeout") is True
    failure_reason = nonempty_diagnostic(
        payload.get("failure_reason"),
        payload.get("error"),
        process.get("error"),
        process.get("stderr"),
        fallback=f"Silex {operation} computation failed",
    )
    failure_stage = payload.get("failure_stage")
    if type(failure_stage) is not str or not failure_stage.strip():
        failure_stage = "native_execution"

    if operation == "class_unit_proven":
        native_timing = payload.get("measurement_timing")
        native_timing = native_timing if isinstance(native_timing, dict) else {}
        timing = {
            **native_timing,
            "native_scope": native_timing.get(
                "scope", native_timing.get("algorithm_scope")
            ),
            "scope": _TIMING_SCOPES[operation],
            "preparation_excluded": True,
            "component_timing_ms": (
                payload.get("component_timing_ms")
                if isinstance(payload.get("component_timing_ms"), dict)
                else {}
            ),
        }
        proof = {
            "certification_status": payload.get("certification_status"),
            "class_group_proof_status": payload.get("class_group_proof_status"),
            "unit_group_proof_status": payload.get("unit_group_proof_status"),
            "regulator_proof_status": payload.get("regulator_proof_status"),
            "final_result_published": False,
        }
    else:
        clocks = payload.get("timing_clock")
        clocks = clocks if isinstance(clocks, dict) else {}
        timing = {
            "native_scope": payload.get("timing_scope"),
            "scope": _TIMING_SCOPES[operation],
            "algorithm_clock": clocks.get("cpu"),
            "wall_clock": clocks.get("wall"),
            "target_cpu_ms": payload.get("target_cpu_ms"),
            "target_wall_ms": payload.get("target_wall_ms"),
            "source": payload.get("source"),
        }
        proof = {}
    timing.update(
        {
            "preparation_excluded": True,
            "marked_target_cpu_ms": process.get("target_cpu_ms"),
            "marked_target_wall_ms": process.get("target_wall_ms"),
            "marked_process_affinity": process.get("effective_affinity"),
            "cpu_launcher_executable": process.get("launcher_executable"),
            "cpu_launcher_sha256": process.get("launcher_executable_sha256"),
        }
    )
    return {
        "engine": "silex",
        "algorithm": algorithm,
        "available": process.get("available") is True,
        "success": False,
        "timeout": timeout,
        "status": "timeout" if timeout else "compute_error",
        "error": failure_reason,
        "target_cpu_ms": payload.get("target_cpu_ms"),
        "target_wall_ms": payload.get("target_wall_ms"),
        "process_wall_ms": process.get("process_wall_ms"),
        "engine_identity": identity,
        "result": {},
        "proof": proof,
        "thread_count": _thread_count(payload),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "equation_order_index": payload.get("equation_order_index"),
        "timing": timing,
        "cmd": process.get("cmd"),
        "stdout": process.get("stdout", ""),
        "stderr": process.get("stderr", ""),
        "diagnostics": {
            "stdout": process.get("stdout", ""),
            "stderr": process.get("stderr", ""),
            "cmd": process.get("cmd"),
        },
    }


class SilexBackend(BackendAdapter):
    name = "silex"

    def __init__(self) -> None:
        self._probe: dict[str, Any] | None = None

    @staticmethod
    def _executables(context: BackendContext) -> tuple[Path, Path]:
        class_unit = context.silex_build_dir / "silex-class-unit-instance"
        operation = context.silex_build_dir / "silex-operation-instance"
        return class_unit, operation

    @staticmethod
    def _cache_error(context: BackendContext) -> str | None:
        cache = context.silex_build_dir / "CMakeCache.txt"
        try:
            text = read_bytes_nofollow(
                cache,
                root=context.silex_build_dir,
                max_bytes=MAX_SILEX_CMAKE_CACHE_BYTES,
            ).decode("utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return f"could not read benchmark CMake cache {cache}: {exc}"
        required = dict(SILEX_BENCHMARK_CMAKE_CACHE_REQUIREMENTS)
        values: dict[str, str] = {}
        required_names = {key.split(":", 1)[0] for key in required}
        seen_required: dict[str, str] = {}
        for line in text.splitlines():
            if "=" not in line or line.startswith("//") or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            name = key.split(":", 1)[0]
            if name in required_names:
                previous = seen_required.get(name)
                if previous is not None:
                    return (
                        f"duplicate required CMake cache variable {name!r}: "
                        f"{previous!r} and {key!r}"
                    )
                seen_required[name] = key
            values[key] = value
        mismatches = [
            f"{key}={values.get(key)!r} (expected {expected!r})"
            for key, expected in required.items()
            if values.get(key) != expected
        ]
        return "; ".join(mismatches) if mismatches else None

    def probe(self, context: BackendContext) -> dict[str, Any]:
        if self._probe is None:
            self._probe = self._probe_uncached(context)
        return copy.deepcopy(self._probe)

    def _probe_uncached(self, context: BackendContext) -> dict[str, Any]:
        for label, root in (
            ("source", context.silex_source),
            ("build", context.silex_build_dir),
        ):
            try:
                descriptor = open_absolute_directory_nofollow(root)
            except (OSError, ValueError) as exc:
                return unavailable_probe(
                    "silex",
                    f"invalid configured Silex {label} directory: {exc}",
                )
            else:
                os.close(descriptor)
        class_unit, operation = self._executables(context)
        executables = {
            "class_unit_executable": class_unit,
            "operation_executable": operation,
        }
        required_keys = set(silex_executable_keys(context.selected_operations))
        required = [
            path for key, path in executables.items() if key in required_keys
        ]
        metadata: dict[Path, os.stat_result] = {}
        missing: list[str] = []
        for path in required:
            try:
                info = path.lstat()
            except OSError:
                missing.append(str(path))
                continue
            if not stat.S_ISREG(info.st_mode):
                missing.append(str(path))
                continue
            metadata[path] = info
        if missing:
            return unavailable_probe(
                "silex", "missing Silex executables: " + ", ".join(missing)
            )
        nonexecutable = [
            str(path) for path in required if not metadata[path].st_mode & 0o111
        ]
        if nonexecutable:
            return unavailable_probe(
                "silex",
                "Silex executables are not executable: "
                + ", ".join(nonexecutable),
            )
        cache_error = self._cache_error(context)
        if cache_error:
            return unavailable_probe(
                "silex", "invalid Silex benchmark build: " + cache_error
            )
        try:
            executable_digests = {
                "class_unit_executable": (
                    file_digest_nofollow(class_unit)
                    if "class_unit_executable" in required_keys
                    else None
                ),
                "operation_executable": (
                    file_digest_nofollow(operation)
                    if "operation_executable" in required_keys
                    else None
                ),
            }
        except (OSError, ValueError) as exc:
            return unavailable_probe(
                "silex", f"could not hash Silex benchmark executable: {exc}"
            )
        return successful_probe(
            "silex",
            {
                "engine": "silex",
                "class_unit_executable": str(class_unit),
                "operation_executable": str(operation),
                "source": str(context.silex_source),
                "build_dir": str(context.silex_build_dir),
                "executable_sha256": executable_digests,
            },
        )

    def run(
        self, request: SampleRequest, context: BackendContext
    ) -> dict[str, Any]:
        probe = self.probe(context)
        if not probe.get("available"):
            payload = unavailable("silex", str(probe.get("error")))
            payload["engine_identity"] = probe.get("engine_identity")
            payload["cmd"] = None
            return payload
        if request.operation == "class_unit_proven":
            return self._run_class_unit(request, context, probe["engine_identity"])
        return self._run_operation(request, context, probe["engine_identity"])

    def _run_class_unit(
        self,
        request: SampleRequest,
        context: BackendContext,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        executable, _ = self._executables(context)
        cmd = [
            str(executable),
            "--marked-protocol",
            "--coeffs",
            coeffs_arg(request.field.coefficients_low_to_high),
            "--mode",
            "proven",
        ]
        process = run_marked_process(
            cmd,
            ready_input="",
            target_input=TARGET_NONCE_PLACEHOLDER + "\n",
            final_input="result\n",
            ready_marker=_READY_MARKER,
            target_marker=_TARGET_MARKER,
            timeout=context.timeout_seconds,
            cwd=context.silex_source,
            cpu=context.cpu,
            env={**os.environ, **context.environment},
        )
        identity = identity_with_executed_digest(
            identity, process, executable_key="class_unit_executable"
        )
        if not process_state_is_valid(process):
            return self._process_failure(
                process, identity, "invalid Silex process state"
            )
        try:
            payload = _json_payload(process.get("stdout", ""))
        except (json.JSONDecodeError, ValueError) as exc:
            return self._process_failure(
                process, identity, f"invalid Silex class/unit JSON: {exc}"
            )
        envelope_error = _native_payload_type_error(
            payload,
            required_booleans=("success",),
            optional_booleans=("timeout",),
        )
        if envelope_error is not None:
            return self._process_failure(process, identity, envelope_error)
        if payload["success"] is False:
            return _native_failure_payload(
                process,
                identity,
                payload,
                operation=request.operation,
                algorithm=context.silex_backend,
            )
        payload_error = _native_payload_type_error(
            payload,
            required_objects=(
                "class_group",
                "unit_group",
                "measurement_timing",
                "component_timing_ms",
            ),
            required_booleans=(
                "success",
                "expectations_passed",
                "final_result_published",
            ),
            optional_booleans=("timeout",),
        )
        if payload_error is not None:
            return self._process_failure(process, identity, payload_error)
        timeout = process["timeout"] is True or payload.get("timeout") is True
        result = {
            "class_order": payload["class_group"].get("order"),
            "class_invariants": payload["class_group"].get("invariants"),
            "unit_rank": payload["unit_group"].get("free_rank"),
            "signature": payload.get("signature"),
            "maximal_order_discriminant": payload.get(
                "maximal_order_discriminant"
            ),
        }
        if (
            process["success"] is True
            and payload["success"] is True
            and timeout is False
        ):
            result_errors = backend_result_contract_errors(
                result,
                operation=request.operation,
                expected_field_degree=request.field.degree,
            )
            if result_errors:
                return self._process_failure(
                    process,
                    identity,
                    "invalid native Silex result: " + "; ".join(result_errors),
                )
        computation_complete = (
            process.get("success") is True
            and payload.get("success") is True
            and payload.get("expectations_passed", True) is True
            and timeout is False
        )
        published = payload.get("final_result_published") is True
        proof = {
            "certification_status": payload.get("certification_status"),
            "class_group_proof_status": payload.get("class_group_proof_status"),
            "unit_group_proof_status": payload.get("unit_group_proof_status"),
            "regulator_proof_status": payload.get("regulator_proof_status"),
            "final_result_published": published,
        }
        proof_complete = (
            proof["certification_status"] == "proven"
            and proof["class_group_proof_status"] == "proven"
            and proof["unit_group_proof_status"] == "proven"
            and proof["regulator_proof_status"] == "verified"
            and published
        )
        result_complete = computation_complete and proof_complete
        thread_count = _thread_count(payload)
        success = result_complete and thread_count["matches_requested"]
        timing = payload.get("measurement_timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = {
            **timing,
            "native_scope": timing.get(
                "scope", timing.get("algorithm_scope")
            ),
            "scope": _TIMING_SCOPES[request.operation],
            "wall_clock": timing.get("wall_clock", timing.get("component_clock")),
            "preparation_excluded": True,
            "marked_target_cpu_ms": process.get("target_cpu_ms"),
            "marked_target_wall_ms": process.get("target_wall_ms"),
            "marked_process_affinity": process.get("effective_affinity"),
            "cpu_launcher_executable": process.get("launcher_executable"),
            "cpu_launcher_sha256": process.get("launcher_executable_sha256"),
        }
        error = None
        if not success:
            error = nonempty_diagnostic(
                (
                    "Silex native class/unit payload reported a timeout"
                    if timeout is True
                    else None
                ),
                (
                    "Silex proven-publication contract failed: expected proven "
                    "class/unit labels, a verified regulator, and published result"
                    if computation_complete and not proof_complete
                    else None
                ),
                (
                    "Silex thread-count contract failed: requested "
                    f"{_REQUESTED_THREADS}, reported {thread_count['reported']!r}"
                    if result_complete and not thread_count["matches_requested"]
                    else None
                ),
                payload.get("failure_reason"),
                payload.get("error"),
                process.get("stderr"),
                fallback="Silex class/unit computation failed",
            )
        failure_stage = payload.get("failure_stage")
        if not success and (type(failure_stage) is not str or not failure_stage.strip()):
            failure_stage = "native_execution"
        failure_reason = payload.get("failure_reason")
        if not success:
            failure_reason = nonempty_diagnostic(
                failure_reason,
                error,
                fallback="Silex class/unit computation failed",
            )
        return {
            "engine": "silex",
            "algorithm": context.silex_backend,
            "available": True,
            "success": success,
            "timeout": timeout is True,
            "status": (
                "ok"
                if success
                else "timeout"
                if timeout is True
                else "thread_contract"
                if result_complete and not thread_count["matches_requested"]
                else "proof_contract"
                if computation_complete and not proof_complete
                else "compute_error"
            ),
            "error": error,
            "target_cpu_ms": timing.get("target_cpu_ms"),
            "target_wall_ms": timing.get("target_wall_ms"),
            "process_wall_ms": process.get("process_wall_ms"),
            "engine_identity": identity,
            "result": result,
            "proof": proof,
            "thread_count": thread_count,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "equation_order_index": payload.get("equation_order_index"),
            "timing": {
                **timing,
                "component_timing_ms": payload.get("component_timing_ms", {}),
            },
            "cmd": process.get("cmd"),
            "stdout": process.get("stdout", ""),
            "stderr": process.get("stderr", ""),
        }

    def _run_operation(
        self,
        request: SampleRequest,
        context: BackendContext,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        _, executable = self._executables(context)
        cmd = [
            str(executable),
            "--marked-protocol",
            "--coeffs",
            coeffs_arg(request.field.coefficients_low_to_high),
            "--operation",
            request.operation,
        ]
        process = run_marked_process(
            cmd,
            ready_input="",
            target_input=TARGET_NONCE_PLACEHOLDER + "\n",
            final_input="result\n",
            ready_marker=_READY_MARKER,
            target_marker=_TARGET_MARKER,
            timeout=context.timeout_seconds,
            cwd=context.silex_source,
            cpu=context.cpu,
            env={**os.environ, **context.environment},
        )
        identity = identity_with_executed_digest(
            identity, process, executable_key="operation_executable"
        )
        if not process_state_is_valid(process):
            return self._process_failure(
                process, identity, "invalid Silex process state"
            )
        try:
            payload = _json_payload(process.get("stdout", ""))
        except (json.JSONDecodeError, ValueError) as exc:
            return self._process_failure(
                process, identity, f"invalid Silex operation JSON: {exc}"
            )
        envelope_error = _native_payload_type_error(
            payload,
            required_booleans=("success",),
            optional_booleans=("timeout",),
        )
        if envelope_error is not None:
            return self._process_failure(process, identity, envelope_error)
        if payload["success"] is False:
            return _native_failure_payload(
                process,
                identity,
                payload,
                operation=request.operation,
                algorithm=context.silex_backend,
            )
        payload_error = _native_payload_type_error(
            payload,
            required_objects=("timing_clock",),
            required_booleans=("success",),
            optional_booleans=("timeout",),
        )
        if payload_error is not None:
            return self._process_failure(process, identity, payload_error)
        timeout = process["timeout"] is True or payload.get("timeout") is True
        result = {
            key: payload.get(key)
            for key in (
                "maximal_order_discriminant",
                "ideal_norm",
                "root_found",
                "root_verified",
            )
            if key in payload
        }
        if (
            process["success"] is True
            and payload["success"] is True
            and timeout is False
        ):
            result_errors = backend_result_contract_errors(
                result,
                operation=request.operation,
                expected_field_degree=request.field.degree,
            )
            if request.operation == "element_square_root" and (
                result.get("root_found") is not True
                or result.get("root_verified") is not True
            ):
                result_errors.append(
                    "a successful square-root result must be found and verified"
                )
            if result_errors:
                return self._process_failure(
                    process,
                    identity,
                    "invalid native Silex result: " + "; ".join(result_errors),
                )
        result_complete = (
            process.get("success") is True and payload.get("success") is True
            and timeout is False
        )
        thread_count = _thread_count(payload)
        success = result_complete and thread_count["matches_requested"]
        error = None
        if not success:
            error = nonempty_diagnostic(
                (
                    "Silex native operation payload reported a timeout"
                    if timeout is True
                    else None
                ),
                (
                    "Silex thread-count contract failed: requested "
                    f"{_REQUESTED_THREADS}, reported {thread_count['reported']!r}"
                    if result_complete and not thread_count["matches_requested"]
                    else None
                ),
                payload.get("error"),
                process.get("error"),
                process.get("stderr"),
                fallback="Silex operation failed",
            )
        return {
            "engine": "silex",
            "algorithm": context.silex_backend,
            "available": True,
            "success": success,
            "timeout": timeout is True,
            "status": (
                "ok"
                if success
                else "timeout"
                if timeout is True
                else "thread_contract"
                if result_complete and not thread_count["matches_requested"]
                else "compute_error"
            ),
            "error": error,
            "target_cpu_ms": payload.get("target_cpu_ms"),
            "target_wall_ms": payload.get("target_wall_ms"),
            "process_wall_ms": process.get("process_wall_ms"),
            "engine_identity": identity,
            "result": result,
            "proof": {},
            "thread_count": thread_count,
            "timing": {
                "native_scope": payload.get("timing_scope"),
                "scope": _TIMING_SCOPES[request.operation],
                "algorithm_clock": payload.get("timing_clock", {}).get("cpu"),
                "wall_clock": payload.get("timing_clock", {}).get("wall"),
                "target_cpu_ms": payload.get("target_cpu_ms"),
                "target_wall_ms": payload.get("target_wall_ms"),
                "marked_target_cpu_ms": process.get("target_cpu_ms"),
                "marked_target_wall_ms": process.get("target_wall_ms"),
                "marked_process_affinity": process.get("effective_affinity"),
                "cpu_launcher_executable": process.get("launcher_executable"),
                "cpu_launcher_sha256": process.get("launcher_executable_sha256"),
                "source": payload.get("source"),
                "preparation_excluded": True,
            },
            "cmd": process.get("cmd"),
            "stdout": process.get("stdout", ""),
            "stderr": process.get("stderr", ""),
        }

    @staticmethod
    def _process_failure(
        process: dict[str, Any], identity: dict[str, Any], error: str
    ) -> dict[str, Any]:
        diagnostic = nonempty_diagnostic(
            error,
            process.get("error"),
            process.get("stderr"),
            fallback="invalid Silex process output",
        )
        return {
            "engine": "silex",
            "algorithm": "default",
            "available": process.get("available") is True,
            "success": False,
            "timeout": process.get("timeout") is True,
            "status": (
                "timeout" if process.get("timeout") is True else "invalid_output"
            ),
            "error": diagnostic,
            "target_cpu_ms": None,
            "target_wall_ms": None,
            "process_wall_ms": process.get("process_wall_ms"),
            "engine_identity": identity,
            "result": {},
            "proof": {},
            "thread_count": _thread_count({}),
            "timing": {
                "marked_process_affinity": process.get("effective_affinity"),
            },
            "cmd": process.get("cmd"),
            "stdout": process.get("stdout", ""),
            "stderr": process.get("stderr", ""),
        }
