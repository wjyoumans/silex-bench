"""Shared benchmark data contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .process import trusted_cpu_launcher_identity
from .util import exact_json_equal, file_digest_nofollow


BACKENDS = ("silex", "pari", "hecke", "magma")
EXTERNAL_BACKENDS = ("pari", "hecke", "magma")
OPERATIONS = (
    "class_unit_proven",
    "maximal_order",
    "ideal_multiply",
    "element_square_root",
)
SAMPLE_KINDS = ("warm_algorithm", "cold_process")
PRIMARY_CLOCKS = ("cpu", "wall", "marked_cpu", "marked_wall")
SILEX_BACKENDS = ("default",)
MAX_EXPECTED_SAMPLE_GROUPS = 100_000
MAX_EXPECTED_SAMPLE_RECORDS = 100_000
MAX_SILEX_CMAKE_CACHE_BYTES = 1 << 20

_INTEGER_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_TEXT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RUN_FINGERPRINT_KEYS = (
    "config_path",
    "config_sha256",
    "resolved_config",
    "fields",
    "warmup_fields",
    "engine_probes",
    "machine",
    "sources",
    "silex_benchmark_build",
    "sample_policy",
)
_RUN_FINGERPRINT_COMPONENT_TYPES = {
    "config_path": str,
    "config_sha256": str,
    "resolved_config": dict,
    "fields": list,
    "warmup_fields": list,
    "engine_probes": dict,
    "machine": dict,
    "sources": dict,
    "silex_benchmark_build": dict,
    "sample_policy": dict,
}
_THREAD_COUNT_CONTRACTS = {
    "silex": "silex_flint_get_num_threads",
    "pari": "pari_default_nbthreads_runtime_query",
}


@dataclass(frozen=True)
class FieldSpec:
    id: str
    coefficients_low_to_high: tuple[int, ...]
    degree: int
    source: str
    polynomial_discriminant: str | None = None
    benchmark_role: str | None = None
    expected_class_order: str | None = None
    expected_class_invariants: tuple[str, ...] | None = None
    expected_unit_rank: int | None = None
    expected_maximal_order_discriminant: str | None = None
    optimization_external_engines: tuple[str, ...] | None = None
    timeout_seconds: float | None = None
    equation_order_index: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coefficients_low_to_high"] = list(
            self.coefficients_low_to_high
        )
        if self.optimization_external_engines is not None:
            payload["optimization_external_engines"] = list(
                self.optimization_external_engines
            )
        if self.expected_class_invariants is not None:
            payload["expected_class_invariants"] = list(
                self.expected_class_invariants
            )
        return payload


@dataclass(frozen=True)
class SampleRequest:
    field: FieldSpec
    operation: str
    sample_kind: str
    sample_index: int
    warmup: FieldSpec | None
    seed: int
    jit_repetitions: int = 0


@dataclass
class BackendContext:
    workspace: Path
    bench_root: Path
    silex_source: Path
    silex_build_dir: Path
    tools: dict[str, Any]
    timeout_seconds: float
    cpu: int | None
    primary_clock: str
    silex_backend: str = "default"
    environment: dict[str, str] = field(default_factory=dict)
    selected_operations: tuple[str, ...] = ()
    jit_repetitions: int = 0
    timeout_source: str = "campaign_ceiling"


def run_fingerprint_payload(manifest: Any) -> dict[str, Any]:
    """Return the exact persisted payload bound by a run fingerprint."""
    if not isinstance(manifest, dict):
        raise ValueError("run fingerprint manifest must be an object")
    missing = [key for key in _RUN_FINGERPRINT_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            "run fingerprint payload is missing: " + ", ".join(missing)
        )
    malformed = [
        key
        for key, expected_type in _RUN_FINGERPRINT_COMPONENT_TYPES.items()
        if type(manifest[key]) is not expected_type
    ]
    if malformed:
        raise ValueError(
            "run fingerprint payload has malformed components: "
            + ", ".join(malformed)
        )
    if not manifest["config_path"]:
        raise ValueError("run fingerprint config_path must be nonempty")
    if _SHA256_TEXT.fullmatch(manifest["config_sha256"]) is None:
        raise ValueError(
            "run fingerprint config_sha256 must be a lowercase SHA-256 value"
        )
    return {
        "config_path": manifest["config_path"],
        "config_sha256": manifest["config_sha256"],
        "config": manifest["resolved_config"],
        "fields": manifest["fields"],
        "warmup_fields": manifest["warmup_fields"],
        "engine_probes": manifest["engine_probes"],
        "machine": manifest["machine"],
        "sources": manifest["sources"],
        "silex_benchmark_build": manifest["silex_benchmark_build"],
        "sample_policy": manifest["sample_policy"],
    }


def run_environment_contract_errors(manifest: Any) -> list[str]:
    """Return schema errors for fingerprinted machine, source, and build identity."""
    if type(manifest) is not dict:
        return ["environment identity manifest must be an object"]
    errors: list[str] = []

    machine = manifest.get("machine")
    machine_keys = {
        "hostname",
        "platform",
        "architecture",
        "python",
        "cpu_model",
        "available_affinity",
        "requested_cpu",
    }
    if type(machine) is not dict:
        errors.append("machine must be an object")
    else:
        if set(machine) != machine_keys:
            errors.append("machine must contain exactly the persisted identity fields")
        for key in ("hostname", "platform", "architecture", "python"):
            if type(machine.get(key)) is not str:
                errors.append(f"machine.{key} must be a string")
        cpu_model = machine.get("cpu_model")
        if cpu_model is not None and type(cpu_model) is not str:
            errors.append("machine.cpu_model must be null or a string")
        affinity = machine.get("available_affinity")
        if affinity is not None and (
            type(affinity) is not list
            or any(type(value) is not int or value < 0 for value in affinity)
            or affinity != sorted(set(affinity))
        ):
            errors.append(
                "machine.available_affinity must be null or a sorted unique "
                "nonnegative integer array"
            )
        requested_cpu = machine.get("requested_cpu")
        if requested_cpu is not None and (
            type(requested_cpu) is not int or requested_cpu < 0
        ):
            errors.append("machine.requested_cpu must be null or nonnegative integer")
        elif (
            requested_cpu is not None
            and type(affinity) is list
            and requested_cpu not in affinity
        ):
            errors.append("machine.requested_cpu must be in available_affinity")

    policy = manifest.get("sample_policy")
    resolved_config = manifest.get("resolved_config")
    execution = (
        resolved_config.get("execution")
        if type(resolved_config) is dict
        else None
    )
    cpu_claims = {
        "sample_policy.cpu": policy.get("cpu") if type(policy) is dict else None,
        "resolved_config.execution.cpu": (
            execution.get("cpu") if type(execution) is dict else None
        ),
        "machine.requested_cpu": (
            machine.get("requested_cpu") if type(machine) is dict else None
        ),
    }
    valid_cpu_claims = True
    for label, value in cpu_claims.items():
        if value is not None and (type(value) is not int or value < 0):
            errors.append(f"{label} must be null or a nonnegative integer")
            valid_cpu_claims = False
    if valid_cpu_claims and len(
        {(type(value), value) for value in cpu_claims.values()}
    ) != 1:
        errors.append(
            "resolved configuration, sample policy, and machine requested CPU "
            "must match exactly"
        )

    sources = manifest.get("sources")
    source_keys = {"silex_bench", "silex"}
    git_identity_keys = {
        "path",
        "revision",
        "dirty",
        "status",
        "worktree_sha256",
    }
    if type(sources) is not dict:
        errors.append("sources must be an object")
    else:
        if set(sources) != source_keys:
            errors.append("sources must contain exactly silex_bench and silex")
        for source in source_keys:
            identity = sources.get(source)
            if type(identity) is not dict:
                errors.append(f"sources.{source} must be an object")
                continue
            if set(identity) != git_identity_keys:
                errors.append(
                    f"sources.{source} must contain exactly the Git identity fields"
                )
            path = identity.get("path")
            if type(path) is not str or not path.strip():
                errors.append(f"sources.{source}.path must be a nonempty string")
            revision = identity.get("revision")
            if revision is not None and (
                type(revision) is not str
                or _GIT_REVISION_TEXT.fullmatch(revision) is None
            ):
                errors.append(
                    f"sources.{source}.revision must be null or a lowercase Git hash"
                )
            dirty = identity.get("dirty")
            if dirty is not None and type(dirty) is not bool:
                errors.append(f"sources.{source}.dirty must be null or boolean")
            status = identity.get("status")
            if status is not None and type(status) is not str:
                errors.append(f"sources.{source}.status must be null or a string")
            worktree_digest = identity.get("worktree_sha256")
            if worktree_digest is not None and (
                type(worktree_digest) is not str
                or _SHA256_TEXT.fullmatch(worktree_digest) is None
            ):
                errors.append(
                    f"sources.{source}.worktree_sha256 must be null or a "
                    "lowercase SHA-256 value"
                )
            if status is None:
                if any(
                    value is not None
                    for value in (revision, dirty, worktree_digest)
                ):
                    errors.append(
                        f"sources.{source} must be wholly unavailable when Git "
                        "status is unavailable"
                    )
            elif type(status) is str:
                if type(dirty) is bool and dirty is not bool(status):
                    errors.append(
                        f"sources.{source}.dirty must exactly match whether Git "
                        "status is nonempty"
                    )
                if worktree_digest is None:
                    errors.append(
                        f"sources.{source}.worktree_sha256 is required when Git "
                        "status is available"
                    )

    build = manifest.get("silex_benchmark_build")
    build_keys = {
        "class_unit_executable",
        "operation_executable",
        "cmake_cache",
    }
    if type(build) is not dict:
        errors.append("silex_benchmark_build must be an object")
    else:
        if set(build) != build_keys:
            errors.append(
                "silex_benchmark_build must contain exactly the two adapters and "
                "CMake cache"
            )
        for key in build_keys:
            identity = build.get(key)
            if type(identity) is not dict:
                errors.append(f"silex_benchmark_build.{key} must be an object")
                continue
            if set(identity) != {"path", "sha256"}:
                errors.append(
                    f"silex_benchmark_build.{key} must contain exactly path and sha256"
                )
            path = identity.get("path")
            if type(path) is not str or not path.strip():
                errors.append(
                    f"silex_benchmark_build.{key}.path must be a nonempty string"
                )
            digest = identity.get("sha256")
            if digest is not None and (
                type(digest) is not str or _SHA256_TEXT.fullmatch(digest) is None
            ):
                errors.append(
                    f"silex_benchmark_build.{key}.sha256 must be null or a "
                    "lowercase SHA-256 value"
                )
    return errors


def canonical_invariants(values: Any) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, dict)):
        return None
    try:
        normalized = [str(value).strip() for value in values]
        normalized = [
            value for value in normalized if value and int(value) > 1
        ]
    except (TypeError, ValueError):
        return None
    return sorted(normalized, key=int)


def _integer_text_is_at_least(value: Any, minimum: int) -> bool:
    parsed = integer_text_value(value)
    return parsed is not None and parsed >= minimum


def integer_text_value(value: Any) -> int | None:
    if type(value) is not str or _INTEGER_TEXT.fullmatch(value) is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _finite_nonnegative_number_or_none(value: Any) -> bool:
    if value is None:
        return True
    if type(value) is int:
        try:
            return value >= 0 and math.isfinite(float(value))
        except OverflowError:
            return False
    return type(value) is float and math.isfinite(value) and value >= 0


def silex_executable_keys(
    selected_operations: Iterable[str] | None,
) -> tuple[str, ...]:
    """Return the Silex adapters required by the selected operations."""
    if selected_operations is None:
        return ("class_unit_executable", "operation_executable")
    operations = set(selected_operations)
    if not operations:
        return ("class_unit_executable", "operation_executable")
    class_unit_operations = {"class_unit_proven", "sunit_proven"}
    required: list[str] = []
    if operations & class_unit_operations:
        required.append("class_unit_executable")
    if operations - class_unit_operations:
        required.append("operation_executable")
    return tuple(required)


def engine_identity_contract_errors(
    identity: Any,
    *,
    expected_backend: str,
    selected_operations: Iterable[str] | None = None,
) -> list[str]:
    """Return backend-specific schema errors for a persisted engine identity."""
    if not isinstance(identity, dict) or not identity:
        return ["must be a nonempty object"]
    if expected_backend not in BACKENDS:
        return [f"has unknown expected backend {expected_backend!r}"]

    errors: list[str] = []
    expected_keys = {
        "silex": {
            "engine",
            "class_unit_executable",
            "operation_executable",
            "source",
            "build_dir",
            "executable_sha256",
        },
        "pari": {
            "engine",
            "executable",
            "version",
            "required_version",
            "source",
            "source_version",
            "source_version_file",
            "source_version_file_sha256",
            "executable_sha256",
        },
        "hecke": {
            "engine",
            "executable",
            "version",
            "package_version",
            "source",
            "project",
            "executable_sha256",
        },
        "magma": {
            "engine",
            "executable",
            "version",
            "executable_sha256",
        },
    }[expected_backend]
    if set(identity) != expected_keys:
        errors.append("must contain exactly the backend-specific identity fields")
    if identity.get("engine") != expected_backend:
        errors.append(
            f"engine must be {expected_backend!r}, not {identity.get('engine')!r}"
        )

    required_strings = {
        "silex": (
            "class_unit_executable",
            "operation_executable",
            "source",
            "build_dir",
        ),
        "pari": ("executable", "version"),
        "hecke": ("executable", "version", "package_version", "source"),
        "magma": ("executable", "version"),
    }[expected_backend]
    for key in required_strings:
        value = identity.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a nonempty string")

    executable_keys = {
        "silex": ("class_unit_executable", "operation_executable"),
        "pari": ("executable",),
        "hecke": ("executable",),
        "magma": ("executable",),
    }[expected_backend]
    required_executable_keys = set(executable_keys)
    if expected_backend == "silex":
        required_executable_keys = set(silex_executable_keys(selected_operations))
    executable_digests = identity.get("executable_sha256")
    if type(executable_digests) is not dict:
        errors.append("executable_sha256 must be an object")
    elif set(executable_digests) != set(executable_keys):
        errors.append(
            "executable_sha256 must contain exactly: "
            + ", ".join(executable_keys)
        )
    else:
        for key in executable_keys:
            digest = executable_digests[key]
            if key not in required_executable_keys:
                if digest is not None:
                    errors.append(
                        f"executable_sha256.{key} must be null for an unselected "
                        "Silex adapter"
                    )
                continue
            if type(digest) is not str or _SHA256_TEXT.fullmatch(digest) is None:
                errors.append(
                    f"executable_sha256.{key} must be a lowercase SHA-256 value"
                )

    nullable_strings = {
        "pari": (
            "required_version",
            "source",
            "source_version",
            "source_version_file",
            "source_version_file_sha256",
        ),
        "hecke": ("project",),
    }.get(expected_backend, ())
    for key in nullable_strings:
        if key not in identity:
            errors.append(f"must declare {key}")
            continue
        value = identity[key]
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            errors.append(f"{key} must be null or a nonempty string")

    if expected_backend == "pari":
        source_digest = identity.get("source_version_file_sha256")
        if source_digest is not None and (
            type(source_digest) is not str
            or _SHA256_TEXT.fullmatch(source_digest) is None
        ):
            errors.append(
                "source_version_file_sha256 must be null or a lowercase SHA-256 value"
            )

    if expected_backend == "pari":
        version = identity.get("version")
        required_version = identity.get("required_version")
        source_version = identity.get("source_version")
        if required_version is not None and required_version != version:
            errors.append("required_version must match version when present")
        if source_version is not None and source_version != version:
            errors.append("source_version must match executable version when present")
    return errors


def engine_identity_core(identity: Any) -> Any:
    """Return identity fields emitted at runtime before digest enrichment."""
    if type(identity) is not dict:
        return identity
    return {
        key: value
        for key, value in identity.items()
        if key != "executable_sha256"
    }


def engine_probe_contract_errors(
    probe: Any,
    *,
    expected_backend: str,
    selected_operations: Iterable[str] | None = None,
) -> list[str]:
    """Return errors for one canonical persisted backend probe envelope."""
    if type(probe) is not dict:
        return ["probe must be an object"]
    errors: list[str] = []
    expected_keys = {
        "engine",
        "available",
        "success",
        "timeout",
        "status",
        "error",
        "engine_identity",
    }
    if set(probe) != expected_keys:
        errors.append("probe must contain exactly the canonical state fields")
    if probe.get("engine") != expected_backend:
        errors.append(f"probe.engine must be {expected_backend!r}")
    for key in ("available", "success", "timeout"):
        if type(probe.get(key)) is not bool:
            errors.append(f"probe.{key} must be a boolean")
    status = probe.get("status")
    if type(status) is not str or not status:
        errors.append("probe.status must be a nonempty string")
    error = probe.get("error")
    if error is not None and (type(error) is not str or not error):
        errors.append("probe.error must be null or a nonempty string")

    success = probe.get("success") is True
    if success:
        if probe.get("available") is not True:
            errors.append("a successful probe must be available")
        if probe.get("timeout") is not False:
            errors.append("a successful probe must not be timed out")
        if status != "ok":
            errors.append("a successful probe must have status 'ok'")
        if error is not None:
            errors.append("a successful probe must not contain an error")
    else:
        if probe.get("available") is not False:
            errors.append("an unsuccessful probe must be unavailable")
        expected_status = "timeout" if probe.get("timeout") is True else "unavailable"
        if status != expected_status:
            errors.append(
                "probe.status must be 'timeout' exactly when probe.timeout is true; "
                f"expected {expected_status!r}"
            )
        if type(error) is not str or not error:
            errors.append("an unsuccessful probe must contain an error")

    identity = probe.get("engine_identity")
    if identity is None:
        if success:
            errors.append("a successful probe must contain an engine identity")
    else:
        errors.extend(
            "engine_identity " + message
            for message in engine_identity_contract_errors(
                identity,
                expected_backend=expected_backend,
                selected_operations=selected_operations,
            )
        )
    return errors


def run_identity_coupling_errors(manifest: Any) -> list[str]:
    """Cross-bind persisted source, build, probe, and configured tool identity."""
    if type(manifest) is not dict:
        return ["identity coupling manifest must be an object"]
    errors: list[str] = []
    selected = manifest.get("selected_backends")
    operations = manifest.get("selected_operations")
    probes = manifest.get("engine_probes")
    sources = manifest.get("sources")
    build = manifest.get("silex_benchmark_build")
    resolved = manifest.get("resolved_config")
    if type(selected) is not list or type(operations) is not list:
        return errors
    if type(probes) is not dict or type(sources) is not dict:
        return errors
    if type(build) is not dict or type(resolved) is not dict:
        return errors

    if "silex" in selected:
        probe = probes.get("silex")
        identity = probe.get("engine_identity") if type(probe) is dict else None
        source_identity = sources.get("silex")
        if type(identity) is dict and type(source_identity) is dict:
            if source_identity.get("path") != identity.get("source"):
                errors.append("sources.silex.path must match the Silex probe source")
            build_dir = identity.get("build_dir")
            if type(build_dir) is str:
                expected_paths = {
                    "class_unit_executable": str(
                        Path(build_dir) / "silex-class-unit-instance"
                    ),
                    "operation_executable": str(
                        Path(build_dir) / "silex-operation-instance"
                    ),
                    "cmake_cache": str(Path(build_dir) / "CMakeCache.txt"),
                }
                digests = identity.get("executable_sha256")
                required = set(silex_executable_keys(operations))
                for key, expected_path in expected_paths.items():
                    artifact = build.get(key)
                    if type(artifact) is not dict:
                        continue
                    if artifact.get("path") != expected_path:
                        errors.append(
                            f"silex_benchmark_build.{key}.path must match Silex build_dir"
                        )
                    if key in required and type(digests) is dict and (
                        artifact.get("path") != identity.get(key)
                        or artifact.get("sha256") != digests.get(key)
                    ):
                        errors.append(
                            f"silex_benchmark_build.{key} must match the selected "
                            "Silex probe executable identity"
                        )

    if "pari" in selected:
        probe = probes.get("pari")
        identity = probe.get("engine_identity") if type(probe) is dict else None
        tools = resolved.get("tools")
        if type(identity) is dict and type(tools) is dict:
            for tool_key, identity_key in (
                ("pari_version", "required_version"),
                ("pari_source", "source"),
                ("gp", "executable"),
            ):
                configured = tools.get(tool_key)
                if configured is not None and configured != identity.get(identity_key):
                    errors.append(
                        f"PARI identity {identity_key} must match resolved tools.{tool_key}"
                    )
            source = identity.get("source")
            source_fields = (
                identity.get("source_version"),
                identity.get("source_version_file"),
                identity.get("source_version_file_sha256"),
            )
            if source is None and any(value is not None for value in source_fields):
                errors.append("PARI source provenance fields must be coherently absent")
            if source is not None and (type(source) is not str or not source):
                errors.append("PARI source must be null or a nonempty string")
            elif source is not None:
                if any(value is None for value in source_fields):
                    errors.append("PARI source provenance fields must be coherently present")
                if identity.get("source_version_file") != str(
                    Path(source) / "config" / "version"
                ):
                    errors.append(
                        "PARI source_version_file must be <source>/config/version"
                    )
    return errors


def live_engine_identity_contract_errors(identity: Any) -> list[str]:
    """Cross-check persisted digests against files still present at pinned paths."""
    if type(identity) is not dict:
        return ["live engine identity must be an object"]
    engine = identity.get("engine")
    executable_keys = (
        ("class_unit_executable", "operation_executable")
        if engine == "silex"
        else ("executable",)
    )
    digest_map = identity.get("executable_sha256")
    errors: list[str] = []
    if type(digest_map) is dict:
        for key in executable_keys:
            path_value = identity.get(key)
            expected = digest_map.get(key)
            if type(path_value) is not str or type(expected) is not str:
                continue
            try:
                actual = file_digest_nofollow(Path(path_value))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                errors.append(f"live executable {key} could not be verified: {exc}")
            else:
                if actual != expected:
                    errors.append(f"live executable {key} digest does not match identity")
    source_file = identity.get("source_version_file")
    expected_source_digest = identity.get("source_version_file_sha256")
    if type(source_file) is str and type(expected_source_digest) is str:
        try:
            actual_source_digest = file_digest_nofollow(Path(source_file))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            errors.append(f"live PARI source version file could not be verified: {exc}")
        else:
            if actual_source_digest != expected_source_digest:
                errors.append("live PARI source version file digest does not match identity")
    return errors


def field_payload_contract_errors(payload: Any) -> list[str]:
    """Return schema errors for a persisted field or warmup payload."""
    if not isinstance(payload, dict):
        return ["field payload must be an object"]
    errors: list[str] = []
    field_id = payload.get("id")
    if not isinstance(field_id, str) or not field_id:
        errors.append("id must be a nonempty string")
    coefficients = payload.get("coefficients_low_to_high")
    if (
        type(coefficients) is not list
        or len(coefficients) < 2
        or any(type(value) is not int for value in coefficients)
        or coefficients[-1] != 1
    ):
        errors.append(
            "coefficients_low_to_high must be a monic nonconstant integer array"
        )
    degree = payload.get("degree")
    if (
        type(degree) is not int
        or degree < 1
        or type(coefficients) is not list
        or degree != len(coefficients) - 1
    ):
        errors.append("degree must match the defining polynomial")
    if not isinstance(payload.get("source"), str) or not payload["source"]:
        errors.append("source must be a nonempty string")
    role = payload.get("benchmark_role")
    if role is not None and (not isinstance(role, str) or not role):
        errors.append("benchmark_role must be null or a nonempty string")
    parsed_exact: dict[str, int | None] = {}
    for key, minimum, allow_negative in (
        ("polynomial_discriminant", 1, True),
        ("expected_class_order", 1, False),
        ("expected_maximal_order_discriminant", 1, True),
        ("equation_order_index", 1, False),
    ):
        value = payload.get(key)
        if value is None:
            parsed_exact[key] = None
            continue
        parsed = integer_text_value(value)
        parsed_exact[key] = parsed
        if parsed is None or parsed == 0 or (not allow_negative and parsed < minimum):
            errors.append(f"{key} must be null or a valid integer string")
    polynomial_discriminant = parsed_exact.get("polynomial_discriminant")
    maximal_order_discriminant = parsed_exact.get(
        "expected_maximal_order_discriminant"
    )
    equation_order_index = parsed_exact.get("equation_order_index")
    if (
        polynomial_discriminant is not None
        and maximal_order_discriminant is not None
        and equation_order_index is not None
        and polynomial_discriminant
        != maximal_order_discriminant * equation_order_index**2
    ):
        errors.append(
            "polynomial_discriminant must equal expected maximal-order "
            "discriminant times equation-order index squared"
        )
    invariants = payload.get("expected_class_invariants")
    if invariants is not None:
        if type(invariants) is not list or any(
            not _integer_text_is_at_least(value, 2) for value in invariants
        ):
            errors.append(
                "expected_class_invariants must be null or an array of integer "
                "strings >= 2"
            )
        else:
            parsed_invariants = [int(value) for value in invariants]
            if parsed_invariants != sorted(parsed_invariants):
                errors.append("expected_class_invariants must be nondecreasing")
            expected_class_order = integer_text_value(
                payload.get("expected_class_order")
            )
            if (
                expected_class_order is not None
                and math.prod(parsed_invariants) != expected_class_order
            ):
                errors.append(
                    "expected_class_invariants product must equal expected_class_order"
                )
    unit_rank = payload.get("expected_unit_rank")
    if unit_rank is not None and (type(unit_rank) is not int or unit_rank < 0):
        errors.append("expected_unit_rank must be null or a nonnegative integer")
    external = payload.get("optimization_external_engines")
    if external is not None and (
        type(external) is not list
        or any(engine not in EXTERNAL_BACKENDS for engine in external)
        or len(set(external)) != len(external)
    ):
        errors.append(
            "optimization_external_engines must be null or a unique known external-backend array"
        )
    timeout = payload.get("timeout_seconds")
    if timeout is not None and (
        not _finite_nonnegative_number_or_none(timeout)
        or timeout <= 0
    ):
        errors.append("timeout_seconds must be null or a finite positive number")
    return errors


def backend_result_contract_errors(
    result: Any,
    *,
    operation: str,
    expected_field_degree: int | None = None,
) -> list[str]:
    """Return the operation-specific errors for one successful result object."""
    if type(result) is not dict:
        return ["backend.result must be an object"]
    if operation not in OPERATIONS:
        return [f"unknown operation {operation!r}"]

    errors: list[str] = []
    if operation == "class_unit_proven":
        class_order = integer_text_value(result.get("class_order"))
        if class_order is None or class_order < 1:
            errors.append("backend.result.class_order must be a positive integer string")
        invariants = result.get("class_invariants")
        if type(invariants) is not list or any(
            not _integer_text_is_at_least(value, 2) for value in invariants
        ):
            errors.append(
                "backend.result.class_invariants must be an array of integer strings >= 2"
            )
        else:
            parsed_invariants = [int(value) for value in invariants]
            if parsed_invariants != sorted(parsed_invariants):
                errors.append("backend.result.class_invariants must be nondecreasing")
            if class_order is not None and math.prod(parsed_invariants) != class_order:
                errors.append(
                    "backend.result.class_invariants product must equal class_order"
                )
        unit_rank = result.get("unit_rank")
        if type(unit_rank) is not int or unit_rank < 0:
            errors.append("backend.result.unit_rank must be a nonnegative integer")
        signature = result.get("signature")
        if (
            type(signature) is not list
            or len(signature) != 2
            or any(type(value) is not int or value < 0 for value in signature)
        ):
            errors.append(
                "backend.result.signature must contain two nonnegative integers"
            )
        elif type(unit_rank) is int and unit_rank >= 0:
            expected_unit_rank = signature[0] + signature[1] - 1
            if expected_unit_rank < 0 or unit_rank != expected_unit_rank:
                errors.append(
                    "backend.result.unit_rank must equal r1 + r2 - 1 from signature"
                )
            if (
                type(expected_field_degree) is int
                and signature[0] + 2 * signature[1] != expected_field_degree
            ):
                errors.append(
                    "backend.result.signature must satisfy r1 + 2*r2 = field degree"
                )
        discriminant = integer_text_value(
            result.get("maximal_order_discriminant")
        )
        if discriminant is None or discriminant == 0:
            errors.append(
                "backend.result.maximal_order_discriminant must be a nonzero integer string"
            )
        elif (
            type(signature) is list
            and len(signature) == 2
            and all(type(value) is int and value >= 0 for value in signature)
            and (discriminant < 0) != (signature[1] % 2 == 1)
        ):
            errors.append(
                "backend.result maximal-order discriminant sign must match "
                "the field signature"
            )
    elif operation == "maximal_order":
        discriminant = integer_text_value(
            result.get("maximal_order_discriminant")
        )
        if discriminant is None or discriminant == 0:
            errors.append(
                "backend.result.maximal_order_discriminant must be a nonzero integer string"
            )
    elif operation == "ideal_multiply":
        if not _integer_text_is_at_least(result.get("ideal_norm"), 1):
            errors.append("backend.result.ideal_norm must be a positive integer string")
    elif operation == "element_square_root":
        for key in ("root_found", "root_verified"):
            if type(result.get(key)) is not bool:
                errors.append(f"backend.result.{key} must be a boolean")
    return errors


def backend_payload_contract_errors(
    payload: Any,
    *,
    operation: str,
    expected_backend: str,
    expected_field_degree: int | None = None,
    expected_field: dict[str, Any] | None = None,
    expected_cpu: int | None = None,
    machine_affinity: list[int] | None = None,
    selected_operations: Iterable[str] | None = None,
) -> list[str]:
    """Return fail-closed schema errors for one persisted backend payload."""
    if not isinstance(payload, dict):
        return ["backend payload must be an object"]

    errors: list[str] = []
    engine = payload.get("engine")
    if engine != expected_backend:
        errors.append(
            "requested_backend does not match backend.engine: "
            f"{expected_backend!r} != {engine!r}"
        )
    elif engine not in BACKENDS:
        errors.append(f"backend.engine is not known: {engine!r}")
    for key in ("available", "success", "timeout"):
        if type(payload.get(key)) is not bool:
            errors.append(f"backend.{key} must be a boolean")
    if not isinstance(payload.get("status"), str) or not payload["status"]:
        errors.append("backend.status must be a nonempty string")
    error = payload.get("error")
    if error is not None and (type(error) is not str or not error.strip()):
        errors.append("backend.error must be null or a nonempty string")
    errors.extend(
        "backend.engine_identity " + error
        for error in engine_identity_contract_errors(
            payload.get("engine_identity"),
            expected_backend=expected_backend,
            selected_operations=(
                (operation,) if selected_operations is None else selected_operations
            ),
        )
    )
    for key in ("target_cpu_ms", "target_wall_ms", "process_wall_ms"):
        if not _finite_nonnegative_number_or_none(payload.get(key)):
            errors.append(f"backend.{key} must be null or a finite nonnegative number")

    result = payload.get("result")
    proof = payload.get("proof")
    timing = payload.get("timing")
    if not isinstance(result, dict):
        errors.append("backend.result must be an object")
        result = {}
    if not isinstance(proof, dict):
        errors.append("backend.proof must be an object")
        proof = {}
    if not isinstance(timing, dict):
        errors.append("backend.timing must be an object")
        timing = {}
    for key in ("marked_target_cpu_ms", "marked_target_wall_ms"):
        if key in timing and not _finite_nonnegative_number_or_none(timing[key]):
            errors.append(
                f"backend.timing.{key} must be null or a finite nonnegative number"
            )
    marked_affinity = timing.get("marked_process_affinity")
    marked_affinity_valid = marked_affinity is None or (
        type(marked_affinity) is list
        and bool(marked_affinity)
        and all(type(value) is int and value >= 0 for value in marked_affinity)
        and marked_affinity == sorted(set(marked_affinity))
    )
    if not marked_affinity_valid:
        errors.append(
            "backend.timing.marked_process_affinity must be null or a sorted "
            "unique nonempty array of nonnegative integers"
        )

    if payload.get("success") is not True:
        return errors
    if payload.get("available") is not True:
        errors.append("a successful backend must be available")
    if payload.get("timeout") is not False:
        errors.append("a successful backend must not be timed out")
    if payload.get("status") != "ok":
        errors.append("a successful backend must have status 'ok'")
    if payload.get("error") is not None:
        errors.append("a successful backend must not contain an error")
    if expected_backend == "silex" and payload.get("algorithm") != "default":
        errors.append("a successful Silex backend must use algorithm 'default'")
    launcher_path = timing.get("cpu_launcher_executable")
    launcher_digest = timing.get("cpu_launcher_sha256")
    if expected_cpu is None:
        if launcher_path is not None or launcher_digest is not None:
            errors.append(
                "an unpinned successful backend must not record a CPU launcher identity"
            )
    else:
        try:
            trusted_launcher = trusted_cpu_launcher_identity()
        except (OSError, ValueError) as exc:
            errors.append(f"trusted CPU launcher identity is unavailable: {exc}")
        else:
            if (
                launcher_path != trusted_launcher["executable"]
                or launcher_digest != trusted_launcher["sha256"]
            ):
                errors.append(
                    "a CPU-pinned successful backend must bind the trusted launcher "
                    "path and SHA-256"
                )
    if expected_cpu is not None and marked_affinity != [expected_cpu]:
        errors.append(
            "a successful backend must record the requested singleton process affinity"
        )
    if (
        marked_affinity_valid
        and type(marked_affinity) is list
        and type(machine_affinity) is list
        and any(value not in machine_affinity for value in marked_affinity)
    ):
        errors.append(
            "backend.timing.marked_process_affinity must be within machine affinity"
        )
    adapter_identity = payload.get("adapter_engine_identity")
    if adapter_identity is None:
        errors.append(
            "a successful backend must contain backend.adapter_engine_identity"
        )
    else:
        errors.extend(
            "backend.adapter_engine_identity " + error
            for error in engine_identity_contract_errors(
                adapter_identity,
                expected_backend=expected_backend,
                selected_operations=(
                    (operation,) if selected_operations is None else selected_operations
                ),
            )
        )
        if not exact_json_equal(adapter_identity, payload.get("engine_identity")):
            errors.append(
                "backend.adapter_engine_identity must match backend.engine_identity"
            )
    thread_count_source = _THREAD_COUNT_CONTRACTS.get(expected_backend)
    if thread_count_source is not None:
        thread_count = payload.get("thread_count")
        if (
            type(thread_count) is not dict
            or set(thread_count)
            != {"requested", "reported", "matches_requested", "source"}
            or type(thread_count.get("requested")) is not int
            or thread_count.get("requested") != 1
            or type(thread_count.get("reported")) is not int
            or thread_count.get("reported") != 1
            or thread_count.get("matches_requested") is not True
            or thread_count.get("source") != thread_count_source
        ):
            errors.append(
                "a successful backend must record the exact one-thread runtime contract"
            )
    if operation not in OPERATIONS:
        errors.append(f"unknown operation {operation!r}")
        return errors

    errors.extend(
        backend_result_contract_errors(
            result,
            operation=operation,
            expected_field_degree=expected_field_degree,
        )
    )
    if operation == "class_unit_proven":
        required_proof = {
            "certification_status": "proven",
            "class_group_proof_status": "proven",
            "unit_group_proof_status": "proven",
            "regulator_proof_status": (
                "verified" if expected_backend == "silex" else "proven"
            ),
            "final_result_published": True,
        }
        for key, expected in required_proof.items():
            if (
                proof.get(key) is not True
                if expected is True
                else proof.get(key) != expected
            ):
                errors.append(f"backend.proof.{key} must be {expected!r}")
        if expected_backend in EXTERNAL_BACKENDS and proof.get("proof_complete") is not True:
            errors.append("backend.proof.proof_complete must be True")
    field_payload = expected_field if type(expected_field) is dict else {}
    if operation == "class_unit_proven":
        expected_values = (
            ("class_order", field_payload.get("expected_class_order")),
            ("unit_rank", field_payload.get("expected_unit_rank")),
            (
                "maximal_order_discriminant",
                field_payload.get("expected_maximal_order_discriminant"),
            ),
        )
        for key, expected in expected_values:
            if expected is not None and result.get(key) != expected:
                errors.append(f"backend.result.{key} must match field expectation")
        expected_invariants = field_payload.get("expected_class_invariants")
        if expected_invariants is not None and result.get(
            "class_invariants"
        ) != expected_invariants:
            errors.append(
                "backend.result.class_invariants must match field expectation"
            )
        expected_equation_order_index = field_payload.get(
            "equation_order_index"
        )
        if (
            expected_backend == "silex"
            and expected_equation_order_index is not None
            and payload.get("equation_order_index")
            != expected_equation_order_index
        ):
            errors.append(
                "backend.equation_order_index must match field expectation"
            )
    elif operation == "maximal_order":
        expected = field_payload.get("expected_maximal_order_discriminant")
        if expected is not None and result.get("maximal_order_discriminant") != expected:
            errors.append(
                "backend.result.maximal_order_discriminant must match field expectation"
            )
    elif operation == "ideal_multiply" and type(expected_field_degree) is int:
        if result.get("ideal_norm") != str(6**expected_field_degree):
            errors.append("backend.result.ideal_norm must equal 6^field_degree")
    elif operation == "element_square_root" and (
        result.get("root_found") is not True
        or result.get("root_verified") is not True
    ):
        errors.append("a successful square-root result must be found and verified")
    return errors


def coeffs_arg(coefficients: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(value) for value in coefficients)


def polynomial_expr(
    coefficients: tuple[int, ...] | list[int], variable: str = "x"
) -> str:
    pieces: list[tuple[int, str]] = []
    for power, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        if power == 0:
            term = str(magnitude)
        else:
            monomial = variable if power == 1 else f"{variable}^{power}"
            term = monomial if magnitude == 1 else f"{magnitude}*{monomial}"
        pieces.append((1 if coefficient > 0 else -1, term))
    if not pieces:
        return "0"
    first_sign, first_term = pieces[-1]
    output = first_term if first_sign > 0 else f"-{first_term}"
    for sign, term in reversed(pieces[:-1]):
        output += f" {'+' if sign > 0 else '-'} {term}"
    return output
