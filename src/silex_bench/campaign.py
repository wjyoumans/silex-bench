"""Campaign planning, probing, execution, and resume."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import platform
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import __version__
from .configuration import (
    ProfileConfig,
    RunOverrides,
    SuiteConfig,
    ToolConfig,
    effective_execution,
    resolved_fingerprint,
)
from .contracts import (
    AgreementResult,
    AgreementStatus,
    Case,
    EngineInfo,
    InvocationContext,
    Observation,
    ObservationStatus,
    Registry,
    CAMPAIGN_SCHEMA_VERSION,
    ValidationResult,
    engine_identity_binding,
)
from .ledger import RunLedger
from .util import file_digest_nofollow, git_identity, machine_identity
from .workloads import cases_digest, load_cases, select_cases


def _adapter_capabilities(
    registry: Registry,
    backends: tuple[str, ...],
    workloads: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "backend": backend,
            "capabilities": [
                workload
                for workload in workloads
                if workload in registry.backends[backend].implementations
            ],
        }
        for backend in backends
    ]


@dataclasses.dataclass(frozen=True)
class CampaignPlan:
    bench_root: Path
    suite: SuiteConfig
    profile: ProfileConfig
    tools: ToolConfig
    overrides: RunOverrides
    registry: Registry
    performance: bool
    execution: dict[str, Any]
    backends: tuple[str, ...]
    cases: tuple[Case, ...]
    plan_fingerprint: str
    required_pairs: tuple[tuple[str, str], ...]

    @property
    def workloads(self) -> tuple[str, ...]:
        return self.overrides.workloads or self.suite.workloads

    @property
    def sample_count(self) -> int:
        backend_count = sum(
            len(
                self.backends
                if not case.eligible_backends
                else tuple(
                    backend
                    for backend in self.backends
                    if backend in set(case.eligible_backends) | {"silex"}
                )
            )
            for case in self.cases
        )
        return backend_count * int(self.execution["repetitions"])

    @property
    def nominal_timeout_product_seconds(self) -> float:
        return self.sample_count * float(self.execution["timeout_seconds"])

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "invocation_root": str(self.bench_root),
            "suite": self.suite.id,
            "suite_path": str(self.suite.path),
            "suite_sha256": self.suite.sha256,
            "profile": self.profile.id,
            "profile_path": str(self.profile.path),
            "profile_sha256": self.profile.sha256,
            "tools_path": None if self.tools.path is None else str(self.tools.path),
            "tools_sha256": self.tools.sha256,
            "workloads": list(self.workloads),
            "backends": list(self.backends),
            "adapter_capabilities": _adapter_capabilities(
                self.registry, self.backends, self.workloads
            ),
            "required_pairs": [list(pair) for pair in self.required_pairs],
            "mode": "performance" if self.performance else "correctness",
            "execution": self.execution,
            "overrides": self.overrides.to_json(),
            "cases": [case.to_json() for case in self.cases],
            "case_count": len(self.cases),
            "sample_count": self.sample_count,
            "nominal_timeout_product_seconds": self.nominal_timeout_product_seconds,
            "plan_fingerprint": self.plan_fingerprint,
        }


def build_plan(
    bench_root: Path,
    suite: SuiteConfig,
    profile: ProfileConfig,
    tools: ToolConfig,
    overrides: RunOverrides,
    registry: Registry,
    *,
    performance: bool,
) -> CampaignPlan:
    selected_workloads = overrides.workloads or suite.workloads
    unknown_workloads = sorted(set(selected_workloads) - set(registry.workloads))
    selected_backends = overrides.backends or suite.backends
    unknown_backends = sorted(set(selected_backends) - set(registry.backends))
    if unknown_workloads:
        raise ValueError("unknown workloads: " + ", ".join(unknown_workloads))
    if unknown_backends:
        raise ValueError("unknown backends: " + ", ".join(unknown_backends))
    if len(set(selected_backends)) != len(selected_backends):
        raise ValueError("selected backends must not contain duplicates")
    required_pairs = overrides.required_pairs or suite.required_pairs
    for candidate, baseline in required_pairs:
        if candidate not in selected_backends or baseline not in selected_backends:
            raise ValueError(
                f"selected backends omit required pair {candidate}/{baseline}; "
                "use --require-pair CANDIDATE:BASELINE to replace the suite default"
            )
        for workload in selected_workloads:
            missing = [
                backend
                for backend in (candidate, baseline)
                if workload not in registry.backends[backend].implementations
            ]
            if missing:
                raise ValueError(
                    f"required pair {candidate}/{baseline} does not implement "
                    f"{workload}: {', '.join(missing)}"
                )
    execution = effective_execution(profile, overrides)
    if not performance:
        execution = {
            **execution,
            "repetitions": 1,
            "warmups": 0,
            "publication": False,
        }
    materialized = load_cases(suite.corpora, selected_workloads)
    selected_cases = select_cases(materialized, execution, performance=performance)
    if not selected_cases:
        raise ValueError("campaign selection produced no cases")
    for case in selected_cases:
        contract = registry.workloads[case.workload]
        validation = contract.validate_case(case)
        if not validation.success:
            raise ValueError(
                f"invalid case {case.key}: " + "; ".join(validation.errors)
            )
    base_payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "suite_sha256": suite.sha256,
        "profile_sha256": profile.sha256,
        "tools_sha256": tools.sha256,
        "workloads": list(selected_workloads),
        "backends": list(selected_backends),
        "adapter_capabilities": _adapter_capabilities(
            registry, tuple(selected_backends), tuple(selected_workloads)
        ),
        "required_pairs": [list(pair) for pair in required_pairs],
        "execution": execution,
        "cases_sha256": cases_digest(selected_cases),
        "overrides": overrides.to_json(),
    }
    fingerprint = resolved_fingerprint(base_payload)
    return CampaignPlan(
        bench_root=bench_root,
        suite=suite,
        profile=profile,
        tools=tools,
        overrides=overrides,
        registry=registry,
        performance=performance,
        execution=execution,
        backends=tuple(selected_backends),
        cases=tuple(selected_cases),
        plan_fingerprint=fingerprint,
        required_pairs=tuple(required_pairs),
    )


def invocation_context(plan: CampaignPlan) -> InvocationContext:
    values = dict(plan.tools.values)
    default_workspace = plan.bench_root
    if (
        plan.bench_root.name == "silex-bench"
        and (plan.bench_root.parent / "silex").is_dir()
    ):
        default_workspace = plan.bench_root.parent
    workspace = Path(values.get("workspace", default_workspace)).expanduser().absolute()
    silex_source = Path(values.get("silex_source", workspace / "silex")).expanduser().absolute()
    silex_build = Path(
        values.get("silex_build_dir", silex_source / "build" / "benchmark-adapters")
    ).expanduser().absolute()
    environment: dict[str, str] = {}
    for name in (
        "GP",
        "JULIA",
        "MAGMA",
        "PARI_SOURCE",
        "PARI_VERSION",
        "HECKE_PROJECT",
    ):
        key = f"SILEX_BENCH_{name}"
        if key in os.environ:
            environment[key] = os.environ[key]
    return InvocationContext(
        bench_root=plan.bench_root,
        workspace=workspace,
        silex_source=silex_source,
        silex_build_dir=silex_build,
        tools=values,
        timeout_seconds=float(plan.execution["timeout_seconds"]),
        cpu=plan.execution.get("cpu"),
        environment=environment,
        selected_workloads=plan.workloads,
    )


def prepare_silex(plan: CampaignPlan) -> list[list[str]]:
    from .backends.silex import SILEX_BENCHMARK_CMAKE_ARGUMENTS

    context = invocation_context(plan)
    if "silex" not in plan.backends:
        return []
    context.silex_build_dir.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(context.silex_source),
        "-B",
        str(context.silex_build_dir),
        *SILEX_BENCHMARK_CMAKE_ARGUMENTS,
    ]
    build = [
        "cmake",
        "--build",
        str(context.silex_build_dir),
        "--target",
        "silex-class-unit-instance",
        "silex-operation-instance",
    ]
    for command in (configure, build):
        process = subprocess.run(
            command,
            cwd=context.silex_source,
            check=False,
            text=True,
            capture_output=True,
            timeout=900,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Silex adapter build failed ({process.returncode}): "
                f"{process.stderr[-4000:] or process.stdout[-4000:]}"
            )
    return [configure, build]


def _package_tree_sha256(package_root: Path) -> str | None:
    """Bind the importable harness code and packaged data, excluding bytecode."""

    digest = hashlib.sha256()
    digest.update(b"silex-bench-package-v1\0")
    try:
        entries = sorted(package_root.rglob("*"), key=lambda path: path.as_posix())
        for path in entries:
            relative = path.relative_to(package_root)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            metadata = path.lstat()
            encoded = relative.as_posix().encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            if stat.S_ISDIR(metadata.st_mode):
                digest.update(b"directory\0")
            elif stat.S_ISLNK(metadata.st_mode):
                digest.update(b"symlink\0")
                target = os.readlink(path).encode("utf-8")
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
            elif stat.S_ISREG(metadata.st_mode):
                digest.update(b"file\0")
                digest.update(file_digest_nofollow(path).encode("ascii"))
            else:
                return None
    except (OSError, UnicodeError, ValueError):
        return None
    return digest.hexdigest()


def _benchmark_source_root(module_path: Path | None = None) -> Path | None:
    """Return the checkout that actually supplied this imported module, if any."""

    module = (module_path or Path(__file__)).resolve()
    try:
        candidate = module.parents[2]
    except IndexError:
        return None
    expected = candidate / "src" / "silex_bench" / module.name
    try:
        if expected.resolve(strict=True) != module:
            return None
    except OSError:
        return None
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _benchmark_identity(module_path: Path | None = None) -> dict[str, Any]:
    """Describe harness bytes without ever attributing the caller's repository."""

    module = (module_path or Path(__file__)).resolve()
    package_root = module.parent
    source_root = _benchmark_source_root(module)
    identity = (
        git_identity(source_root)
        if source_root is not None
        else {
            "path": str(package_root),
            "revision": None,
            "dirty": None,
            "status": None,
            "worktree_sha256": None,
        }
    )
    return {
        **identity,
        "provenance": "source_checkout" if source_root is not None else "installed_distribution",
        "package_version": __version__,
        "package_sha256": _package_tree_sha256(package_root),
    }


def _machine() -> dict[str, Any]:
    hardware = machine_identity(None)
    return {
        "hostname": hardware["hostname"],
        "system": platform.system(),
        "platform": hardware["platform"],
        "architecture": hardware["architecture"],
        "python": hardware["python"],
        "cpu_model": hardware["cpu_model"],
        "affinity": hardware["available_affinity"],
        "requested_cpu": None,
    }


def probe_engines(plan: CampaignPlan) -> list[EngineInfo]:
    context = invocation_context(plan)
    return [plan.registry.backends[name].probe(context) for name in plan.backends]


def doctor(plan: CampaignPlan) -> dict[str, Any]:
    probes = probe_engines(plan)
    by_name = {probe.backend: probe for probe in probes}
    required_errors: list[str] = []
    required_backends = {backend for pair in plan.required_pairs for backend in pair}
    applicable_backends = {
        backend
        for backend in plan.backends
        if any(
            workload in plan.registry.backends[backend].implementations
            for workload in plan.workloads
        )
    }
    if plan.overrides.require_all_adapters:
        required_backends |= applicable_backends
    for backend in sorted(required_backends):
        if not by_name[backend].available:
            required_errors.append(
                f"required backend {backend} is unavailable: {by_name[backend].error}"
            )
    optional_errors: list[str] = []
    for backend in sorted(applicable_backends - required_backends):
        if not by_name[backend].available:
            optional_errors.append(
                f"optional backend {backend} is unavailable: {by_name[backend].error}"
            )
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "success": not required_errors,
        "plan_fingerprint": plan.plan_fingerprint,
        "required_errors": required_errors,
        "optional_errors": optional_errors,
        "engines": [probe.to_json() for probe in probes],
        "case_count": len(plan.cases),
        "sample_count": plan.sample_count,
    }


def _manifest(plan: CampaignPlan, probes: list[EngineInfo]) -> dict[str, Any]:
    context = invocation_context(plan)
    plan_payload = plan.to_json()
    machine = _machine()
    machine["requested_cpu"] = context.cpu
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "plan": plan_payload,
        "engine_probes": [probe.to_json() for probe in probes],
        "sources": {
            "silex_bench": _benchmark_identity(),
            "silex": git_identity(context.silex_source),
        },
        "machine": machine,
        "timing": {
            "primary_clock": "target_wall_ns",
            "scope": "workload contract: supervisor_marked_target or whole_process",
            "ratio_definition": "baseline_over_candidate",
            "performance_execution": "serial_paired_blocks",
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _warmup(case: Case, cases: tuple[Case, ...]) -> Case | None:
    matches = [
        candidate
        for candidate in cases
        if candidate.workload == case.workload
        and candidate.id != case.id
        and candidate.metrics.get("degree") == case.metrics.get("degree")
    ]
    return matches[0] if matches else case


def _backend_order(case: Case, repetition: int, backends: tuple[str, ...]) -> list[str]:
    if not backends:
        return []
    seed = int.from_bytes(hashlib.sha256(case.key.encode()).digest()[:8], "big")
    offset = (seed + repetition) % len(backends)
    return list(backends[offset:] + backends[:offset])


def _placeholder(
    case: Case,
    backend: str,
    repetition: int,
    order_index: int,
    status: ObservationStatus,
    error: str,
    identity: dict[str, Any],
) -> Observation:
    return Observation(
        case_key=case.key,
        workload=case.workload,
        backend=backend,
        repetition=repetition,
        order_index=order_index,
        status=status,
        success=False,
        timeout=False,
        result={},
        proof={},
        validation=ValidationResult(False, (error,), {}),
        target_wall_ns=None,
        process_wall_ns=None,
        internal_timing={},
        engine_identity=identity,
        command=(),
        stdout="",
        stderr="",
        error=error,
    )


def _enforce_engine_binding(
    observation: Observation, probe: EngineInfo
) -> Observation:
    if not observation.correctness_eligible:
        return observation
    expected = engine_identity_binding(
        observation.backend, observation.workload, probe.identity
    )
    actual = engine_identity_binding(
        observation.backend, observation.workload, observation.engine_identity
    )
    if expected == actual:
        return observation
    error = (
        "engine identity changed after the initial probe: "
        f"expected {json.dumps(expected, sort_keys=True)}, "
        f"observed {json.dumps(actual, sort_keys=True)}"
    )
    return dataclasses.replace(
        observation,
        status=ObservationStatus.ERROR,
        success=False,
        validation=ValidationResult(False, (error,), {}),
        error=error,
    )


def _observation_from_json(payload: Mapping[str, Any]) -> Observation:
    validation_raw = payload.get("validation", {})
    status = ObservationStatus(str(payload["status"]))
    return Observation(
        case_key=str(payload["case_key"]),
        workload=str(payload["workload"]),
        backend=str(payload["backend"]),
        repetition=int(payload["repetition"]),
        order_index=int(payload["order_index"]),
        status=status,
        success=payload.get("success") is True,
        timeout=payload.get("timeout") is True,
        result=dict(payload.get("result", {})),
        proof=dict(payload.get("proof", {})),
        validation=ValidationResult(
            validation_raw.get("success") is True,
            tuple(validation_raw.get("errors", [])),
            dict(validation_raw.get("checks", {})),
        ),
        target_wall_ns=payload.get("target_wall_ns"),
        process_wall_ns=payload.get("process_wall_ns"),
        internal_timing=dict(payload.get("internal_timing", {})),
        engine_identity=dict(payload.get("engine_identity", {})),
        command=tuple(payload.get("command", [])),
        stdout=str(payload.get("stdout", "")),
        stderr=str(payload.get("stderr", "")),
        error=payload.get("error"),
    )


def _selected_for_case(case: Case, backends: tuple[str, ...]) -> tuple[str, ...]:
    if not case.eligible_backends:
        return backends
    allowed = set(case.eligible_backends)
    allowed.add("silex")
    return tuple(backend for backend in backends if backend in allowed)


def _pairs(
    backends: tuple[str, ...], required_pairs: tuple[tuple[str, str], ...]
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [
        pair for pair in required_pairs if pair[0] in backends and pair[1] in backends
    ]
    present = {frozenset(pair) for pair in pairs}
    for lhs, rhs in itertools.combinations(backends, 2):
        if frozenset((lhs, rhs)) in present:
            continue
        if rhs == "silex":
            lhs, rhs = rhs, lhs
        pairs.append((lhs, rhs))
    return pairs


def _compute_agreements(
    ledger: RunLedger,
    plan: CampaignPlan,
    case: Case,
    repetition: int,
    backends: tuple[str, ...],
) -> None:
    contract = plan.registry.workloads[case.workload]
    observations: dict[str, Observation] = {}
    for backend in backends:
        payload = ledger.observation(case.key, backend, repetition)
        if payload is not None:
            observations[backend] = _observation_from_json(payload)
    for candidate, baseline in _pairs(backends, plan.required_pairs):
        lhs = observations.get(candidate)
        rhs = observations.get(baseline)
        if lhs is None or rhs is None:
            result = AgreementResult(
                False,
                ("missing paired observation",),
                {"present": False},
                AgreementStatus.INCOMPLETE,
            )
            eligible = False
            ratio = None
        elif ObservationStatus.UNSUPPORTED in {lhs.status, rhs.status}:
            result = AgreementResult(
                False,
                ("one or both observations are unsupported",),
                {"supported": False},
                AgreementStatus.UNSUPPORTED,
            )
            eligible = False
            ratio = None
        elif ObservationStatus.UNAVAILABLE in {lhs.status, rhs.status}:
            result = AgreementResult(
                False,
                ("one or both implementations are unavailable",),
                {"available": False},
                AgreementStatus.UNAVAILABLE,
            )
            eligible = False
            ratio = None
        elif not lhs.correctness_eligible or not rhs.correctness_eligible:
            result = AgreementResult(
                False,
                ("one or both observations failed validation",),
                {"validated": False},
                AgreementStatus.INVALID,
            )
            eligible = False
            ratio = None
        else:
            result = contract.compare(case, candidate, lhs.result, baseline, rhs.result)
            eligible = bool(
                plan.performance
                and case.performance_eligible
                and result.success
                and lhs.timing_eligible
                and rhs.timing_eligible
            )
            ratio = (
                rhs.target_wall_ns / lhs.target_wall_ns
                if eligible and lhs.target_wall_ns and rhs.target_wall_ns
                else None
            )
        ledger.put_agreement(
            case.key,
            repetition,
            candidate,
            baseline,
            result,
            timing_eligible=eligible,
            ratio=ratio,
        )


def _required_pair_success(snapshot: dict[str, Any], plan: CampaignPlan) -> bool:
    required = set(plan.required_pairs)
    expected: set[tuple[str, int, str, str]] = set()
    for case in plan.cases:
        selected = _selected_for_case(case, plan.backends)
        for pair in required:
            if pair[0] in selected and pair[1] in selected:
                for repetition in range(int(plan.execution["repetitions"])):
                    expected.add((case.key, repetition, pair[0], pair[1]))
    actual: dict[tuple[str, int, str, str], bool] = {
        (
            row["case_key"],
            int(row["repetition"]),
            row["lhs_backend"],
            row["rhs_backend"],
        ): row.get("success") is True
        for row in snapshot["agreements"]
    }
    return bool(expected) and all(actual.get(key) is True for key in expected)


def _strict_success(snapshot: dict[str, Any], plan: CampaignPlan) -> bool:
    """Enforce every selected adapter capability without penalizing unsupported cells."""

    if not plan.overrides.require_all_adapters:
        return True
    probes = {row["backend"]: row for row in snapshot["engines"]}
    applicable = {
        (case.key, backend, repetition)
        for case in plan.cases
        for backend in _selected_for_case(case, plan.backends)
        if case.workload in plan.registry.backends[backend].implementations
        for repetition in range(int(plan.execution["repetitions"]))
    }
    applicable_backends = {backend for _, backend, _ in applicable}
    if any(probes.get(backend, {}).get("available") is not True for backend in applicable_backends):
        return False
    observations = {
        (row["case_key"], row["backend"], int(row["repetition"])): row
        for row in snapshot["observations"]
    }
    if any(
        observations.get(key, {}).get("status") != ObservationStatus.OK.value
        or observations.get(key, {}).get("success") is not True
        or observations.get(key, {}).get("validation", {}).get("success") is not True
        for key in applicable
    ):
        return False
    applicable_pairs: set[tuple[str, int, str, str]] = set()
    for case in plan.cases:
        selected = _selected_for_case(case, plan.backends)
        supported = tuple(
            backend
            for backend in selected
            if case.workload in plan.registry.backends[backend].implementations
        )
        for lhs, rhs in _pairs(supported, plan.required_pairs):
            for repetition in range(int(plan.execution["repetitions"])):
                applicable_pairs.add((case.key, repetition, lhs, rhs))
    agreements = {
        (
            row["case_key"],
            int(row["repetition"]),
            row["lhs_backend"],
            row["rhs_backend"],
        ): row
        for row in snapshot["agreements"]
    }
    for key in applicable_pairs:
        row = agreements.get(key, {})
        if row.get("status") != AgreementStatus.AGREE.value:
            return False
        case = next(item for item in plan.cases if item.key == key[0])
        if plan.performance and case.performance_eligible and row.get("timing_eligible") is not True:
            return False
    return True


def run_campaign(
    plan: CampaignPlan,
    run_dir: Path,
    *,
    resume: bool = False,
    progress: Callable[[Observation, int, int], None] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().absolute()
    if run_dir.exists() and not resume and any(run_dir.iterdir()):
        raise ValueError(f"run directory is nonempty: {run_dir}")
    probes = probe_engines(plan)
    manifest = _manifest(plan, probes)
    run_fingerprint = resolved_fingerprint(manifest)
    manifest["run_fingerprint"] = run_fingerprint
    by_probe = {probe.backend: probe for probe in probes}
    started = time.monotonic()
    budget = plan.execution.get("budget_seconds")
    with RunLedger(run_dir, create=not resume) as ledger:
        ledger.initialize(run_fingerprint, manifest)
        if resume and ledger.fingerprint() != run_fingerprint:
            raise ValueError("resume fingerprint does not match live config, cases, engines, sources, or machine")
        ledger.put_engines(probes)
        ledger.put_cases(plan.cases)
        _atomic_json(run_dir / "manifest.json", manifest)
        context = invocation_context(plan)
        exhausted = False
        for case in plan.cases:
            selected = _selected_for_case(case, plan.backends)
            for repetition in range(int(plan.execution["repetitions"])):
                order = _backend_order(case, repetition, selected)
                for order_index, backend in enumerate(order):
                    if ledger.has_observation(case.key, backend, repetition):
                        continue
                    if budget is not None and time.monotonic() - started >= float(budget):
                        exhausted = True
                        break
                    probe = by_probe[backend]
                    descriptor = plan.registry.backends[backend]
                    implementation = descriptor.implementations.get(case.workload)
                    if implementation is None:
                        observation = _placeholder(
                            case,
                            backend,
                            repetition,
                            order_index,
                            ObservationStatus.UNSUPPORTED,
                            f"{backend} does not support {case.workload}",
                            probe.identity,
                        )
                    elif not probe.available:
                        observation = _placeholder(
                            case,
                            backend,
                            repetition,
                            order_index,
                            ObservationStatus.UNAVAILABLE,
                            probe.error or f"{backend} is unavailable",
                            probe.identity,
                        )
                    else:
                        observation = implementation.run(
                            case,
                            repetition=repetition,
                            order_index=order_index,
                            warmup=(
                                _warmup(case, plan.cases)
                                if int(plan.execution["warmups"]) > 0
                                else None
                            ),
                            context=context,
                            contract=plan.registry.workloads[case.workload],
                        )
                        observation = _enforce_engine_binding(observation, probe)
                    ledger.put_observation(observation)
                    if progress is not None:
                        progress(observation, len(ledger.observations()), plan.sample_count)
                if exhausted:
                    _compute_agreements(ledger, plan, case, repetition, selected)
                    break
                _compute_agreements(ledger, plan, case, repetition, selected)
            if exhausted:
                break
        if exhausted:
            ledger.set_state("budget_exhausted")
        else:
            snapshot = ledger.snapshot()
            ledger.set_state(
                "complete"
                if _required_pair_success(snapshot, plan) and _strict_success(snapshot, plan)
                else "failed"
            )
        snapshot = ledger.snapshot()
    return snapshot


def default_run_dir(plan: CampaignPlan) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return plan.bench_root / "runs" / f"{stamp}-{plan.suite.id}-{plan.profile.id}"
