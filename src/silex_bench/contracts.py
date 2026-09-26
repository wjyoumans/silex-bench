"""Typed extension contracts for comparative benchmark workloads.

The campaign runner keeps backend mechanics and mathematical comparison policy
on opposite sides of these interfaces. Backends produce observations; workload
contracts decide whether those observations are mathematically usable.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


CONFIG_SCHEMA_VERSION = 1
CAMPAIGN_SCHEMA_VERSION = 2
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ObservationStatus(str, enum.Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"
    INVALID = "invalid"


class AgreementStatus(str, enum.Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"


@dataclasses.dataclass(frozen=True)
class Case:
    id: str
    workload: str
    input: dict[str, Any]
    tags: tuple[str, ...]
    metrics: dict[str, int | float]
    expected: dict[str, Any]
    expected_status: str = "success"
    performance_eligible: bool = True
    eligible_backends: tuple[str, ...] = ()
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.workload}:{self.id}"

    def to_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["tags"] = list(self.tags)
        payload["eligible_backends"] = list(self.eligible_backends)
        return payload


@dataclasses.dataclass(frozen=True)
class EngineInfo:
    backend: str
    display_name: str
    available: bool
    capabilities: tuple[str, ...]
    identity: dict[str, Any]
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "display_name": self.display_name,
            "available": self.available,
            "capabilities": list(self.capabilities),
            "identity": self.identity,
            "error": self.error,
        }


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    success: bool
    errors: tuple[str, ...] = ()
    checks: Mapping[str, bool] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "errors": list(self.errors),
            "checks": dict(self.checks),
        }


@dataclasses.dataclass(frozen=True)
class AgreementResult:
    success: bool
    differences: tuple[str, ...] = ()
    checks: Mapping[str, bool] = dataclasses.field(default_factory=dict)
    status: AgreementStatus | None = None

    def to_json(self) -> dict[str, Any]:
        status = self.status or (
            AgreementStatus.AGREE if self.success else AgreementStatus.DISAGREE
        )
        return {
            "success": self.success,
            "status": status.value,
            "differences": list(self.differences),
            "checks": dict(self.checks),
        }


@dataclasses.dataclass(frozen=True)
class TimingSample:
    """One independently reportable timing within an observation process."""

    case_key: str
    workload: str
    backend: str
    repetition: int
    variant: str
    sample_index: int
    status: ObservationStatus
    timeout: bool
    target_wall_ns: int | None
    target_cpu_ns: int | None
    process_wall_ns: int | None
    timing_scope: str
    effective_timeout_seconds: float
    timeout_source: str
    wall_clock: str | None
    cpu_clock: str | None
    internal_timing: dict[str, Any]
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def timing_eligible(self) -> bool:
        return (
            self.status is ObservationStatus.OK
            and not self.timeout
            and self.target_wall_ns is not None
            and self.target_wall_ns > 0
        )

    def to_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclasses.dataclass(frozen=True)
class Observation:
    case_key: str
    workload: str
    backend: str
    repetition: int
    order_index: int
    status: ObservationStatus
    success: bool
    timeout: bool
    result: dict[str, Any]
    proof: dict[str, Any]
    validation: ValidationResult
    target_wall_ns: int | None
    process_wall_ns: int | None
    internal_timing: dict[str, Any]
    engine_identity: dict[str, Any]
    command: tuple[str, ...]
    stdout: str
    stderr: str
    error: str | None = None
    timing_samples: tuple[TimingSample, ...] = ()

    @property
    def correctness_eligible(self) -> bool:
        return (
            self.status is ObservationStatus.OK
            and self.success
            and not self.timeout
            and self.validation.success
        )

    @property
    def timing_eligible(self) -> bool:
        return self.correctness_eligible and self.target_wall_ns is not None and self.target_wall_ns > 0

    def to_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        # Timing rows are normalized into their own ledger table.  Keeping
        # them out of the parent JSON makes the parent the sole source for
        # correctness, command, and engine identity.
        payload.pop("timing_samples", None)
        payload["status"] = self.status.value
        payload["validation"] = self.validation.to_json()
        payload["command"] = list(self.command)
        return payload


@dataclasses.dataclass(frozen=True)
class InvocationContext:
    bench_root: Path
    workspace: Path
    silex_source: Path
    silex_build_dir: Path
    tools: Mapping[str, Any]
    timeout_seconds: float
    cpu: int | None
    environment: Mapping[str, str]
    selected_workloads: tuple[str, ...] = ()
    jit_repetitions: int = 0
    timeout_source: str = "campaign_ceiling"


class ImplementationAdapter(Protocol):
    backend: str
    workload: str

    def run(
        self,
        case: Case,
        *,
        repetition: int,
        order_index: int,
        warmup: Case | None,
        context: InvocationContext,
        contract: "WorkloadContract",
    ) -> Observation:
        """Execute one isolated observation."""


@dataclasses.dataclass(frozen=True)
class BackendDescriptor:
    id: str
    display_name: str
    implementations: Mapping[str, ImplementationAdapter]
    probe_callback: Any

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self.implementations))

    def probe(self, context: InvocationContext) -> EngineInfo:
        return self.probe_callback(context, self)


class WorkloadContract(ABC):
    id: str
    display_name: str
    timing_scope: str
    scale_axes: tuple[str, ...]

    @abstractmethod
    def validate_case(self, case: Case) -> ValidationResult:
        """Validate one fully materialized case."""

    @abstractmethod
    def validate_observation(
        self,
        case: Case,
        backend: str,
        result: Mapping[str, Any],
        proof: Mapping[str, Any],
    ) -> ValidationResult:
        """Validate one backend's canonical output and proof state."""

    @abstractmethod
    def compare(
        self,
        case: Case,
        lhs_backend: str,
        lhs: Mapping[str, Any],
        rhs_backend: str,
        rhs: Mapping[str, Any],
    ) -> AgreementResult:
        """Compare two already validated canonical results."""


class Registry:
    """Validated in-tree registry for workloads and backend descriptors."""

    def __init__(
        self,
        workloads: Sequence[WorkloadContract],
        backends: Sequence[BackendDescriptor],
    ) -> None:
        self.workloads = self._unique("workload", workloads)
        self.backends = self._unique("backend", backends)
        errors: list[str] = []
        for backend in self.backends.values():
            for workload_id, implementation in backend.implementations.items():
                if workload_id not in self.workloads:
                    errors.append(
                        f"backend {backend.id!r} declares unknown workload {workload_id!r}"
                    )
                if implementation.backend != backend.id:
                    errors.append(
                        f"implementation backend mismatch for {backend.id!r}/{workload_id!r}"
                    )
                if implementation.workload != workload_id:
                    errors.append(
                        f"implementation workload mismatch for {backend.id!r}/{workload_id!r}"
                    )
        if errors:
            raise ValueError("invalid registry: " + "; ".join(errors))

    @staticmethod
    def _unique(kind: str, values: Sequence[Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for value in values:
            identifier = getattr(value, "id", None)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{kind} identifiers must be nonempty strings")
            if identifier in result:
                raise ValueError(f"duplicate {kind} identifier: {identifier}")
            result[identifier] = value
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "workloads": [
                {
                    "id": item.id,
                    "display_name": item.display_name,
                    "timing_scope": item.timing_scope,
                    "scale_axes": list(item.scale_axes),
                }
                for item in self.workloads.values()
            ],
            "backends": [
                {
                    "id": item.id,
                    "display_name": item.display_name,
                    "capabilities": list(item.capabilities),
                }
                for item in self.backends.values()
            ],
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def engine_identity_binding(
    backend: str, workload: str, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize probe and execution identities to their stable engine binding."""

    executable_key = "executable"
    if backend == "silex":
        executable_key = (
            "class_unit_executable"
            if workload in {"class_unit_proven", "sunit_proven"}
            else "operation_executable"
        )
    executable = identity.get(executable_key)
    if executable is None and executable_key != "executable":
        executable = identity.get("executable")
    digests = identity.get("executable_sha256")
    if isinstance(digests, Mapping):
        executable_sha256 = digests.get(executable_key)
        if executable_sha256 is None and executable_key != "executable":
            executable_sha256 = digests.get("executable")
    else:
        executable_sha256 = digests
    declared_engine = identity.get("engine")
    return {
        "engine": backend if declared_engine is None else declared_engine,
        "executable": executable,
        "executable_sha256": executable_sha256,
        "version": identity.get("version"),
        "package_version": identity.get("package_version"),
        "source": identity.get("source"),
        "source_version": identity.get("source_version"),
        "required_version": identity.get("required_version"),
        "project": identity.get("project"),
    }


def _manifest_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _manifest_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "text" if allow_empty else "nonempty text"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _manifest_strings(
    value: Any, label: str, *, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a string array" if allow_empty else "a nonempty string array"
        raise ValueError(f"{label} must be {qualifier}")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def _manifest_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _manifest_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label} must be a finite {qualifier}number")
    return result


def _manifest_digest(
    value: Any, label: str, *, optional: bool = False
) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        qualifier = " or null" if optional else ""
        raise ValueError(f"{label} must be a lowercase SHA-256 digest{qualifier}")
    return value


def _manifest_pairs(
    value: Any,
    label: str,
    *,
    backends: set[str] | None = None,
) -> list[list[str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array of backend pairs")
    seen: set[frozenset[str]] = set()
    for index, pair in enumerate(value):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, str) or not item for item in pair)
        ):
            raise ValueError(f"{label}[{index}] must contain two backend IDs")
        if pair[0] == pair[1]:
            raise ValueError(f"{label}[{index}] must select distinct backends")
        if backends is not None and any(item not in backends for item in pair):
            raise ValueError(f"{label}[{index}] references an unplanned backend")
        unordered = frozenset(pair)
        if unordered in seen:
            raise ValueError(f"{label} must not contain duplicate backend pairs")
        seen.add(unordered)
    return value


def _validate_execution(
    value: Any, *, profile: str, mode: str, schema_version: int
) -> dict[str, Any]:
    execution = _manifest_object(value, "campaign manifest.plan.execution")
    if _manifest_text(
        execution.get("profile"), "campaign manifest.plan.execution.profile"
    ) != profile:
        raise ValueError("campaign manifest plan execution profile does not match plan.profile")
    for key in ("include_tags", "exclude_tags", "case_ids"):
        _manifest_strings(
            execution.get(key),
            f"campaign manifest.plan.execution.{key}",
            allow_empty=True,
        )
    required_tags = execution.get("required_tags")
    if not (schema_version == 1 and required_tags is None):
        _manifest_strings(
            required_tags,
            "campaign manifest.plan.execution.required_tags",
            allow_empty=True,
        )
    repetitions = _manifest_int(
        execution.get("repetitions"),
        "campaign manifest.plan.execution.repetitions",
        minimum=1,
    )
    if schema_version == 1:
        jit_repetitions = _manifest_int(
            execution.get("warmups"),
            "campaign manifest.plan.execution.warmups",
        )
        if jit_repetitions > 1:
            raise ValueError(
                "campaign manifest.plan.execution.warmups must be zero or one"
            )
    else:
        jit_repetitions = _manifest_int(
            execution.get("jit_repetitions"),
            "campaign manifest.plan.execution.jit_repetitions",
        )
        if jit_repetitions > 1:
            raise ValueError(
                "campaign manifest.plan.execution.jit_repetitions must be zero or one"
            )
        exclusions = execution.get("backend_exclusions")
        if not isinstance(exclusions, list):
            raise ValueError(
                "campaign manifest.plan.execution.backend_exclusions must be an array"
            )
        seen_exclusions: set[tuple[str, str]] = set()
        for index, exclusion_value in enumerate(exclusions):
            label = (
                "campaign manifest.plan.execution.backend_exclusions"
                f"[{index}]"
            )
            exclusion = _manifest_object(exclusion_value, label)
            if set(exclusion) != {"backend", "workload", "reason"}:
                raise ValueError(
                    f"{label} must contain exactly backend, workload, and reason"
                )
            backend = _manifest_text(exclusion.get("backend"), f"{label}.backend")
            workload = _manifest_text(
                exclusion.get("workload"), f"{label}.workload"
            )
            _manifest_text(exclusion.get("reason"), f"{label}.reason")
            cell = (backend, workload)
            if cell in seen_exclusions:
                raise ValueError(
                    "campaign manifest backend exclusions must not contain duplicates"
                )
            seen_exclusions.add(cell)
    _manifest_number(
        execution.get("timeout_seconds"),
        "campaign manifest.plan.execution.timeout_seconds",
        positive=True,
    )
    budget = execution.get("budget_seconds")
    if budget is not None:
        _manifest_number(
            budget,
            "campaign manifest.plan.execution.budget_seconds",
            positive=True,
        )
    cpu = execution.get("cpu")
    if cpu is not None:
        _manifest_int(cpu, "campaign manifest.plan.execution.cpu")
    _manifest_int(
        execution.get("threads"),
        "campaign manifest.plan.execution.threads",
        minimum=1,
    )
    minimum_repetitions = _manifest_int(
        execution.get("minimum_repetitions"),
        "campaign manifest.plan.execution.minimum_repetitions",
    )
    for key in ("publication", "require_clean_sources", "require_all_adapters"):
        if type(execution.get(key)) is not bool:
            raise ValueError(f"campaign manifest.plan.execution.{key} must be a boolean")
    if (
        schema_version == CAMPAIGN_SCHEMA_VERSION
        and execution["publication"]
        and not execution["require_clean_sources"]
    ):
        raise ValueError(
            "campaign manifest publication execution requires clean-source enforcement"
        )
    publication_floor = 9 if schema_version == 1 else 3
    if execution["publication"] and (
        repetitions < minimum_repetitions
        or minimum_repetitions < publication_floor
    ):
        raise ValueError(
            "campaign manifest publication execution requires at least "
            f"{publication_floor} repetitions"
        )
    if mode == "correctness" and (
        repetitions != 1
        or jit_repetitions != 0
        or execution["publication"]
    ):
        raise ValueError(
            "campaign manifest correctness execution must use one cold, non-publication repetition"
        )
    metrics = _manifest_object(
        execution.get("metrics"), "campaign manifest.plan.execution.metrics"
    )
    for name, bounds in metrics.items():
        _manifest_text(name, "campaign manifest plan metric name")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(
                f"campaign manifest.plan.execution.metrics.{name} must contain [min, max]"
            )
        normalized: list[float | None] = []
        for index, bound in enumerate(bounds):
            normalized.append(
                None
                if bound is None
                else _manifest_number(
                    bound,
                    f"campaign manifest.plan.execution.metrics.{name}[{index}]",
                )
            )
        if (
            normalized[0] is not None
            and normalized[1] is not None
            and normalized[0] > normalized[1]
        ):
            raise ValueError(
                f"campaign manifest.plan.execution.metrics.{name} has min greater than max"
            )
    return execution


def _validate_overrides(
    value: Any,
    *,
    workloads: list[str],
    backends: list[str],
    required_pairs: list[list[str]],
    execution: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    overrides = _manifest_object(value, "campaign manifest.plan.overrides")
    override_workloads = _manifest_strings(
        overrides.get("workloads"),
        "campaign manifest.plan.overrides.workloads",
        allow_empty=True,
    )
    override_backends = _manifest_strings(
        overrides.get("backends"),
        "campaign manifest.plan.overrides.backends",
        allow_empty=True,
    )
    case_ids = _manifest_strings(
        overrides.get("case_ids"),
        "campaign manifest.plan.overrides.case_ids",
        allow_empty=True,
    )
    tags = _manifest_strings(
        overrides.get("tags"),
        "campaign manifest.plan.overrides.tags",
        allow_empty=True,
    )
    if override_workloads and override_workloads != workloads:
        raise ValueError("campaign manifest workload override does not match planned workloads")
    if override_backends and override_backends != backends:
        raise ValueError("campaign manifest backend override does not match planned backends")
    if case_ids != execution["case_ids"] or tags != execution.get("required_tags", []):
        raise ValueError("campaign manifest selection overrides do not match plan execution")
    override_pairs = _manifest_pairs(
        overrides.get("required_pairs"),
        "campaign manifest.plan.overrides.required_pairs",
        backends=set(backends),
    )
    if override_pairs and override_pairs != required_pairs:
        raise ValueError("campaign manifest required-pair override does not match plan")
    for key in ("metric_minima", "metric_maxima"):
        bounds = _manifest_object(
            overrides.get(key), f"campaign manifest.plan.overrides.{key}"
        )
        for name, bound in bounds.items():
            _manifest_text(name, f"campaign manifest.plan.overrides.{key} metric name")
            _manifest_number(bound, f"campaign manifest.plan.overrides.{key}.{name}")
    repetitions = overrides.get("repetitions")
    if repetitions is not None:
        _manifest_int(
            repetitions,
            "campaign manifest.plan.overrides.repetitions",
            minimum=1,
        )
        if mode == "performance" and repetitions != execution["repetitions"]:
            raise ValueError("campaign manifest repetition override does not match execution")
    for key in ("timeout_seconds", "budget_seconds"):
        item = overrides.get(key)
        if item is not None:
            number = _manifest_number(
                item, f"campaign manifest.plan.overrides.{key}", positive=True
            )
            if number != float(execution[key]):
                raise ValueError(f"campaign manifest {key} override does not match execution")
    cpu = overrides.get("cpu")
    if cpu is not None:
        _manifest_int(cpu, "campaign manifest.plan.overrides.cpu")
        if cpu != execution["cpu"]:
            raise ValueError("campaign manifest CPU override does not match execution")
    require_all = overrides.get("require_all_adapters")
    if type(require_all) is not bool:
        raise ValueError("campaign manifest.plan.overrides.require_all_adapters must be a boolean")
    if require_all != execution["require_all_adapters"]:
        raise ValueError("campaign manifest adapter policy override does not match execution")
    return overrides


def _validate_case_payload(value: Any, index: int, workloads: set[str]) -> tuple[str, dict[str, Any]]:
    label = f"campaign manifest.plan.cases[{index}]"
    case = _manifest_object(value, label)
    identifier = _manifest_text(case.get("id"), f"{label}.id")
    workload = _manifest_text(case.get("workload"), f"{label}.workload")
    if workload not in workloads:
        raise ValueError(f"{label}.workload references an unplanned workload")
    _manifest_object(case.get("input"), f"{label}.input")
    _manifest_strings(case.get("tags"), f"{label}.tags", allow_empty=True)
    metrics = _manifest_object(case.get("metrics"), f"{label}.metrics")
    for name, metric in metrics.items():
        _manifest_text(name, f"{label} metric name")
        _manifest_number(metric, f"{label}.metrics.{name}")
    _manifest_object(case.get("expected"), f"{label}.expected")
    if case.get("expected_status") not in {"success", "failure"}:
        raise ValueError(f"{label}.expected_status must be success or failure")
    if type(case.get("performance_eligible")) is not bool:
        raise ValueError(f"{label}.performance_eligible must be a boolean")
    _manifest_strings(
        case.get("eligible_backends"), f"{label}.eligible_backends", allow_empty=True
    )
    _manifest_text(case.get("source"), f"{label}.source", allow_empty=True)
    return f"{workload}:{identifier}", case


def _validate_source_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _manifest_object(value, label)
    _manifest_text(identity.get("path"), f"{label}.path")
    revision = identity.get("revision")
    if revision is not None:
        _manifest_text(revision, f"{label}.revision")
    dirty = identity.get("dirty")
    if dirty is not None and type(dirty) is not bool:
        raise ValueError(f"{label}.dirty must be a boolean or null")
    status = identity.get("status")
    if status is not None and not isinstance(status, str):
        raise ValueError(f"{label}.status must be text or null")
    _manifest_digest(
        identity.get("worktree_sha256"), f"{label}.worktree_sha256", optional=True
    )
    return identity


def campaign_manifest_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash the immutable manifest body, excluding its self-identifying field."""

    payload = dict(value)
    payload.pop("run_fingerprint", None)
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def validate_campaign_manifest(value: Any) -> dict[str, Any]:
    """Validate current manifests and frozen report-only schema-v1 manifests."""

    manifest = _manifest_object(value, "campaign manifest")
    version = manifest.get("schema_version")
    if type(version) is not int or version not in {1, CAMPAIGN_SCHEMA_VERSION}:
        raise ValueError(
            "campaign manifest schema_version must be 1 or "
            f"{CAMPAIGN_SCHEMA_VERSION}"
        )

    plan = _manifest_object(manifest.get("plan"), "campaign manifest.plan")
    if plan.get("schema_version") != version or type(
        plan.get("schema_version")
    ) is not int:
        raise ValueError(
            f"campaign manifest.plan.schema_version must be {version}"
        )
    invocation_root = plan.get("invocation_root")
    if not (version == 1 and invocation_root is None):
        _manifest_text(
            invocation_root,
            "campaign manifest.plan.invocation_root",
            allow_empty=version == 1,
        )
    for key in (
        "suite",
        "suite_path",
        "suite_sha256",
        "profile",
        "profile_path",
        "profile_sha256",
    ):
        _manifest_text(plan.get(key), f"campaign manifest.plan.{key}")
    tools_path = plan.get("tools_path")
    tools_sha256 = plan.get("tools_sha256")
    if tools_path is not None:
        _manifest_text(tools_path, "campaign manifest.plan.tools_path")
    if tools_sha256 is not None:
        _manifest_text(tools_sha256, "campaign manifest.plan.tools_sha256")
    if (tools_path is None) != (tools_sha256 is None):
        raise ValueError("campaign manifest plan tool path and digest must both be null or present")

    workloads = _manifest_strings(
        plan.get("workloads"), "campaign manifest.plan.workloads"
    )
    backends = _manifest_strings(plan.get("backends"), "campaign manifest.plan.backends")
    backend_set = set(backends)
    raw_capabilities = plan.get("adapter_capabilities")
    early_schema_v1 = version == 1 and raw_capabilities is None
    if not early_schema_v1 and (
        not isinstance(raw_capabilities, list)
        or len(raw_capabilities) != len(backends)
    ):
        raise ValueError(
            "campaign manifest.plan.adapter_capabilities must match planned backends"
        )
    adapter_capabilities: list[dict[str, Any]] = []
    for index, (backend, raw_capability) in enumerate(
        zip(backends, raw_capabilities or [])
    ):
        label = f"campaign manifest.plan.adapter_capabilities[{index}]"
        capability = _manifest_object(raw_capability, label)
        if capability.get("backend") != backend:
            raise ValueError(f"{label}.backend must match planned backend order")
        supported = _manifest_strings(
            capability.get("capabilities"),
            f"{label}.capabilities",
            allow_empty=True,
        )
        if any(workload not in workloads for workload in supported):
            raise ValueError(f"{label}.capabilities references an unplanned workload")
        adapter_capabilities.append(capability)
    required_pairs = _manifest_pairs(
        plan.get("required_pairs"),
        "campaign manifest.plan.required_pairs",
        backends=backend_set,
    )
    mode = plan.get("mode")
    if mode not in {"correctness", "performance"}:
        raise ValueError("campaign manifest.plan.mode must be correctness or performance")
    execution = _validate_execution(
        plan.get("execution"),
        profile=plan["profile"],
        mode=mode,
        schema_version=version,
    )
    if version == CAMPAIGN_SCHEMA_VERSION:
        for exclusion in execution["backend_exclusions"]:
            if exclusion["backend"] not in backend_set:
                raise ValueError(
                    "campaign manifest backend exclusion references an unplanned backend"
                )
            if exclusion["workload"] not in workloads:
                raise ValueError(
                    "campaign manifest backend exclusion references an unplanned workload"
                )
    overrides = _validate_overrides(
        plan.get("overrides"),
        workloads=workloads,
        backends=backends,
        required_pairs=required_pairs,
        execution=execution,
        mode=mode,
    )

    raw_cases = plan.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("campaign manifest.plan.cases must be a nonempty array")
    cases: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        key, case = _validate_case_payload(raw_case, index, set(workloads))
        if key in cases:
            raise ValueError(f"campaign manifest plan contains duplicate case {key}")
        cases[key] = case
    case_count = _manifest_int(
        plan.get("case_count"), "campaign manifest.plan.case_count", minimum=1
    )
    if case_count != len(cases):
        raise ValueError("campaign manifest plan case_count does not match plan.cases")
    exclusions = {
        (item["backend"], item["workload"])
        for item in execution.get("backend_exclusions", [])
    }
    selected_counts: dict[str, int] = {}
    for key, case in cases.items():
        eligible = set(case["eligible_backends"])
        selected = (
            [
                backend
                for backend in backends
                if backend == "silex" or backend in eligible
            ]
            if eligible
            else list(backends)
        )
        selected_counts[key] = sum(
            (backend, case["workload"]) not in exclusions
            for backend in selected
        )
    expected_samples = int(execution["repetitions"]) * sum(
        selected_counts.values()
    )
    sample_count = _manifest_int(
        plan.get("sample_count"), "campaign manifest.plan.sample_count"
    )
    if sample_count != expected_samples:
        raise ValueError(
            "campaign manifest plan sample_count does not match cases, backends, and repetitions"
        )
    nominal = _manifest_number(
        plan.get("nominal_timeout_product_seconds"),
        "campaign manifest.plan.nominal_timeout_product_seconds",
    )
    if version == 1:
        expected_nominal = sample_count * float(execution["timeout_seconds"])
    else:
        ceiling = float(execution["timeout_seconds"])
        expected_nominal = int(execution["repetitions"]) * sum(
            selected_counts[key]
            * min(
                ceiling,
                float(case["input"].get("timeout_seconds")),
            )
            if isinstance(case["input"].get("timeout_seconds"), (int, float))
            and not isinstance(case["input"].get("timeout_seconds"), bool)
            and float(case["input"]["timeout_seconds"]) > 0
            else selected_counts[key] * ceiling
            for key, case in cases.items()
        )
    if nominal != expected_nominal:
        raise ValueError(
            "campaign manifest nominal timeout product does not match sample_count and timeout"
        )

    plan_fingerprint = _manifest_digest(
        plan.get("plan_fingerprint"), "campaign manifest.plan.plan_fingerprint"
    )
    cases_digest = hashlib.sha256(canonical_json(raw_cases).encode()).hexdigest()
    expected_plan_fingerprint = hashlib.sha256(
        canonical_json(
            {
                "schema_version": version,
                "suite_sha256": plan["suite_sha256"],
                "profile_sha256": plan["profile_sha256"],
                "tools_sha256": tools_sha256,
                "workloads": workloads,
                "backends": backends,
                "adapter_capabilities": adapter_capabilities,
                "required_pairs": required_pairs,
                "execution": execution,
                "cases_sha256": cases_digest,
                "overrides": overrides,
            }
        ).encode()
    ).hexdigest()
    if not early_schema_v1 and plan_fingerprint != expected_plan_fingerprint:
        raise ValueError("campaign manifest plan_fingerprint does not match the resolved plan")

    probes = manifest.get("engine_probes")
    if not isinstance(probes, list):
        raise ValueError("campaign manifest.engine_probes must be an array")
    probe_backends: list[str] = []
    for index, raw_probe in enumerate(probes):
        label = f"campaign manifest.engine_probes[{index}]"
        probe = _manifest_object(raw_probe, label)
        backend = _manifest_text(probe.get("backend"), f"{label}.backend")
        if backend not in backend_set:
            raise ValueError(f"{label}.backend references an unplanned backend")
        probe_backends.append(backend)
        _manifest_text(probe.get("display_name"), f"{label}.display_name")
        if type(probe.get("available")) is not bool:
            raise ValueError(f"{label}.available must be a boolean")
        _manifest_strings(
            probe.get("capabilities"), f"{label}.capabilities", allow_empty=True
        )
        _manifest_object(probe.get("identity"), f"{label}.identity")
        error = probe.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"{label}.error must be text or null")
    if len(set(probe_backends)) != len(probe_backends):
        raise ValueError("campaign manifest engine probes must not repeat backends")
    if probe_backends != backends:
        raise ValueError("campaign manifest engine probes must match planned backends in order")

    sources = _manifest_object(manifest.get("sources"), "campaign manifest.sources")
    bench_source = _validate_source_identity(
        sources.get("silex_bench"), "campaign manifest.sources.silex_bench"
    )
    if not early_schema_v1:
        if bench_source.get("provenance") not in {
            "source_checkout",
            "installed_distribution",
        }:
            raise ValueError(
                "campaign manifest.sources.silex_bench.provenance must identify source or installed bytes"
            )
        _manifest_text(
            bench_source.get("package_version"),
            "campaign manifest.sources.silex_bench.package_version",
        )
        _manifest_digest(
            bench_source.get("package_sha256"),
            "campaign manifest.sources.silex_bench.package_sha256",
            optional=True,
        )
    _validate_source_identity(sources.get("silex"), "campaign manifest.sources.silex")

    machine = _manifest_object(manifest.get("machine"), "campaign manifest.machine")
    for key in ("hostname", "system", "platform", "architecture", "python"):
        _manifest_text(
            machine.get(key), f"campaign manifest.machine.{key}", allow_empty=True
        )
    cpu_model = machine.get("cpu_model")
    if cpu_model is not None and not isinstance(cpu_model, str):
        raise ValueError("campaign manifest.machine.cpu_model must be text or null")
    affinity = machine.get("affinity")
    if affinity is not None:
        if not isinstance(affinity, list):
            raise ValueError("campaign manifest.machine.affinity must be an integer array or null")
        for cpu in affinity:
            _manifest_int(cpu, "campaign manifest.machine.affinity CPU")
        if len(set(affinity)) != len(affinity):
            raise ValueError("campaign manifest.machine.affinity must not contain duplicates")
    requested_cpu = machine.get("requested_cpu")
    if requested_cpu is not None:
        _manifest_int(requested_cpu, "campaign manifest.machine.requested_cpu")
    if requested_cpu != execution["cpu"]:
        raise ValueError("campaign manifest requested CPU does not match plan execution")

    timing = _manifest_object(manifest.get("timing"), "campaign manifest.timing")
    expected_timing = (
        {
            "primary_clock": "target_wall_ns",
            "scope": "workload contract: supervisor_marked_target or whole_process",
            "ratio_definition": "baseline_over_candidate",
            "performance_execution": "serial_paired_blocks",
        }
        if version == 1
        else {
            "primary_clock": "timing_samples.target_wall_ns",
            "clock_policy": "backend_internal_target_wall",
            "scope": "sample-local intended operation; supervisor markers audit boundaries",
            "deadline_policy": "min(campaign_ceiling, case_timeout_hint)",
            "jit_policy": "Hecke first_call/repeat_call on fresh uncached inputs",
            "performance_execution": "serial_paired_blocks",
        }
    )
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            raise ValueError(f"campaign manifest.timing.{key} must be {expected}")

    run_fingerprint = _manifest_digest(
        manifest.get("run_fingerprint"), "campaign manifest.run_fingerprint"
    )
    try:
        expected_run_fingerprint = campaign_manifest_fingerprint(manifest)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"campaign manifest is not canonical JSON: {exc}") from exc
    if run_fingerprint != expected_run_fingerprint:
        raise ValueError("campaign manifest run_fingerprint does not match its contents")
    return manifest
