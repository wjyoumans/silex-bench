"""Strict suite, profile, and local-tool configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any

from .contracts import CAMPAIGN_SCHEMA_VERSION


MAX_CONFIG_BYTES = 1 << 20
_SUITE_KEYS = {
    "schema_version",
    "id",
    "title",
    "workloads",
    "backends",
    "required_pairs",
    "corpora",
    "reports",
}
_PROFILE_KEYS = {
    "schema_version",
    "id",
    "description",
    "include_tags",
    "exclude_tags",
    "repetitions",
    "warmups",
    "timeout_seconds",
    "budget_seconds",
    "cpu",
    "threads",
    "publication",
    "minimum_repetitions",
    "require_clean_sources",
    "metrics",
}
_TOOLS_KEYS = {
    "schema_version",
    "workspace",
    "silex_source",
    "silex_build_dir",
    "gp",
    "pari_version",
    "pari_source",
    "julia",
    "hecke_project",
    "magma",
}
_REPORT_KEYS = {"primary_clock", "speedup"}


@dataclasses.dataclass(frozen=True)
class SuiteConfig:
    path: Path
    id: str
    title: str
    workloads: tuple[str, ...]
    backends: tuple[str, ...]
    required_pairs: tuple[tuple[str, str], ...]
    corpora: dict[str, Path]
    reports: dict[str, Any]
    sha256: str


@dataclasses.dataclass(frozen=True)
class ProfileConfig:
    path: Path
    id: str
    description: str
    include_tags: tuple[str, ...]
    exclude_tags: tuple[str, ...]
    repetitions: int
    warmups: int
    timeout_seconds: float
    budget_seconds: float | None
    cpu: int | None
    threads: int
    publication: bool
    minimum_repetitions: int
    require_clean_sources: bool
    metrics: dict[str, tuple[float | None, float | None]]
    sha256: str


@dataclasses.dataclass(frozen=True)
class ToolConfig:
    path: Path | None
    values: dict[str, Any]
    sha256: str | None


@dataclasses.dataclass(frozen=True)
class RunOverrides:
    workloads: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metric_minima: tuple[tuple[str, float], ...] = ()
    metric_maxima: tuple[tuple[str, float], ...] = ()
    repetitions: int | None = None
    timeout_seconds: float | None = None
    budget_seconds: float | None = None
    cpu: int | None = None
    required_pairs: tuple[tuple[str, str], ...] = ()
    require_all_adapters: bool = False

    def __post_init__(self) -> None:
        for label, values in (
            ("workloads", self.workloads),
            ("backends", self.backends),
            ("case IDs", self.case_ids),
            ("tags", self.tags),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"override {label} must contain nonempty text")
        for label, bounds in (
            ("minimum", self.metric_minima),
            ("maximum", self.metric_maxima),
        ):
            for name, value in bounds:
                if (
                    not isinstance(name, str)
                    or not name
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"metric {label} overrides must be finite NAME/value pairs")
        if self.repetitions is not None and self.repetitions <= 0:
            raise ValueError("repetition override must be positive")
        for label, value in (
            ("timeout", self.timeout_seconds),
            ("budget", self.budget_seconds),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{label} override must be a finite positive number")
        if self.cpu is not None and self.cpu < 0:
            raise ValueError("CPU override must be nonnegative")
        seen_pairs: set[tuple[str, str]] = set()
        seen_unordered: set[frozenset[str]] = set()
        for pair in self.required_pairs:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(not isinstance(value, str) or not value for value in pair)
            ):
                raise ValueError("required pair overrides must contain two backend IDs")
            if pair[0] == pair[1]:
                raise ValueError("required pair overrides must select distinct backends")
            unordered = frozenset(pair)
            if pair in seen_pairs or unordered in seen_unordered:
                raise ValueError("required pair overrides must not contain duplicates")
            seen_pairs.add(pair)
            seen_unordered.add(unordered)
        if type(self.require_all_adapters) is not bool:
            raise ValueError("require_all_adapters must be a boolean")

    def to_json(self) -> dict[str, Any]:
        return {
            "workloads": list(self.workloads),
            "backends": list(self.backends),
            "case_ids": list(self.case_ids),
            "tags": list(self.tags),
            "metric_minima": dict(self.metric_minima),
            "metric_maxima": dict(self.metric_maxima),
            "repetitions": self.repetitions,
            "timeout_seconds": self.timeout_seconds,
            "budget_seconds": self.budget_seconds,
            "cpu": self.cpu,
            "required_pairs": [list(pair) for pair in self.required_pairs],
            "require_all_adapters": self.require_all_adapters,
        }


def _read_toml(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().absolute()
    data = path.read_bytes()
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes: {path}")
    try:
        decoded = data.decode("utf-8")
        payload = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a table: {path}")
    return payload, hashlib.sha256(data).hexdigest()


def _exact_keys(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(unknown)}")


def _schema(payload: dict[str, Any], label: str) -> None:
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != CAMPAIGN_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{label} schema_version must be {CAMPAIGN_SCHEMA_VERSION}"
        )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _strings(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be {'an' if allow_empty else 'a nonempty'} string array")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _resolve(base: Path, value: Any, label: str) -> Path:
    text = _identifier(value, label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(os.fspath(path)))


def load_suite(path: Path) -> SuiteConfig:
    payload, digest = _read_toml(path)
    _exact_keys(payload, _SUITE_KEYS, "suite")
    _schema(payload, "suite")
    identifier = _identifier(payload.get("id"), "suite.id")
    title = _identifier(payload.get("title", identifier), "suite.title")
    workloads = _strings(payload.get("workloads"), "suite.workloads")
    backends = _strings(payload.get("backends"), "suite.backends")
    raw_pairs = payload.get("required_pairs", [])
    if not isinstance(raw_pairs, list):
        raise ValueError("suite.required_pairs must be an array of backend pairs")
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, list) or len(pair) != 2 or not all(
            isinstance(item, str) and item for item in pair
        ):
            raise ValueError(f"suite.required_pairs[{index}] must contain two backend IDs")
        lhs, rhs = pair
        if lhs == rhs or lhs not in backends or rhs not in backends:
            raise ValueError(f"suite.required_pairs[{index}] must select two distinct configured backends")
        normalized = (lhs, rhs)
        if normalized in pairs:
            raise ValueError("suite.required_pairs must not contain duplicates")
        pairs.append(normalized)
    corpora_raw = payload.get("corpora")
    if not isinstance(corpora_raw, dict) or not corpora_raw:
        raise ValueError("suite.corpora must be a nonempty table")
    corpora = {
        _identifier(name, "suite corpus name"): _resolve(path.parent, value, f"suite.corpora.{name}")
        for name, value in corpora_raw.items()
    }
    reports = payload.get("reports", {})
    if not isinstance(reports, dict):
        raise ValueError("suite.reports must be a table")
    _exact_keys(reports, _REPORT_KEYS, "suite report")
    if reports.get("primary_clock", "target_wall_ns") != "target_wall_ns":
        raise ValueError("suite.reports.primary_clock must be target_wall_ns")
    if reports.get("speedup", "baseline_over_candidate") != "baseline_over_candidate":
        raise ValueError("suite.reports.speedup must be baseline_over_candidate")
    return SuiteConfig(
        path=path.expanduser().absolute(),
        id=identifier,
        title=title,
        workloads=workloads,
        backends=backends,
        required_pairs=tuple(pairs),
        corpora=corpora,
        reports=dict(reports),
        sha256=digest,
    )


def load_profile(path: Path) -> ProfileConfig:
    payload, digest = _read_toml(path)
    _exact_keys(payload, _PROFILE_KEYS, "profile")
    _schema(payload, "profile")
    identifier = _identifier(payload.get("id"), "profile.id")
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ValueError("profile.description must be text")
    include_tags = _strings(payload.get("include_tags", []), "profile.include_tags", allow_empty=True)
    exclude_tags = _strings(payload.get("exclude_tags", []), "profile.exclude_tags", allow_empty=True)
    repetitions = _nonnegative_int(payload.get("repetitions", 1), "profile.repetitions")
    if repetitions == 0:
        raise ValueError("profile.repetitions must be positive")
    warmups = _nonnegative_int(payload.get("warmups", 0), "profile.warmups")
    if warmups > 1:
        raise ValueError("profile.warmups currently supports only zero or one")
    timeout = _positive_number(payload.get("timeout_seconds", 60), "profile.timeout_seconds")
    budget_raw = payload.get("budget_seconds")
    budget = None if budget_raw is None else _positive_number(budget_raw, "profile.budget_seconds")
    cpu_raw = payload.get("cpu")
    cpu = None if cpu_raw is None else _nonnegative_int(cpu_raw, "profile.cpu")
    threads = _nonnegative_int(payload.get("threads", 1), "profile.threads")
    if threads == 0:
        raise ValueError("profile.threads must be positive")
    publication = payload.get("publication", False)
    clean = payload.get("require_clean_sources", publication)
    if type(publication) is not bool or type(clean) is not bool:
        raise ValueError("profile publication flags must be booleans")
    minimum = _nonnegative_int(
        payload.get("minimum_repetitions", 9 if publication else 1),
        "profile.minimum_repetitions",
    )
    if publication and (repetitions < minimum or minimum < 9):
        raise ValueError("publication profiles require at least nine repetitions")
    metrics_raw = payload.get("metrics", {})
    if not isinstance(metrics_raw, dict):
        raise ValueError("profile.metrics must be a table")
    metrics: dict[str, tuple[float | None, float | None]] = {}
    for name, bounds in metrics_raw.items():
        if not isinstance(name, str) or not name or not isinstance(bounds, dict):
            raise ValueError("profile metric bounds must be named tables")
        unknown = sorted(set(bounds) - {"min", "max"})
        if unknown:
            raise ValueError(f"unknown bounds for metric {name}: {', '.join(unknown)}")
        minimum_value = bounds.get("min")
        maximum_value = bounds.get("max")
        for label, value in (("min", minimum_value), ("max", maximum_value)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"profile.metrics.{name}.{label} must be finite")
        if minimum_value is not None and maximum_value is not None and float(minimum_value) > float(maximum_value):
            raise ValueError(f"profile metric {name} has min greater than max")
        metrics[name] = (
            None if minimum_value is None else float(minimum_value),
            None if maximum_value is None else float(maximum_value),
        )
    return ProfileConfig(
        path=path.expanduser().absolute(),
        id=identifier,
        description=description,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        repetitions=repetitions,
        warmups=warmups,
        timeout_seconds=timeout,
        budget_seconds=budget,
        cpu=cpu,
        threads=threads,
        publication=publication,
        minimum_repetitions=minimum,
        require_clean_sources=clean,
        metrics=metrics,
        sha256=digest,
    )


def load_tools(path: Path | None) -> ToolConfig:
    if path is None:
        return ToolConfig(path=None, values={}, sha256=None)
    payload, digest = _read_toml(path)
    _exact_keys(payload, _TOOLS_KEYS, "tools")
    _schema(payload, "tools")
    values = dict(payload)
    values.pop("schema_version", None)
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"tools.{key} must be nonempty text")
    for key in ("workspace", "silex_source", "silex_build_dir", "pari_source", "hecke_project"):
        if key in values:
            values[key] = str(_resolve(path.parent, values[key], f"tools.{key}"))
    return ToolConfig(path=path.expanduser().absolute(), values=values, sha256=digest)


def effective_execution(profile: ProfileConfig, overrides: RunOverrides) -> dict[str, Any]:
    metrics = {name: [bounds[0], bounds[1]] for name, bounds in profile.metrics.items()}
    for name, value in overrides.metric_minima:
        metrics.setdefault(name, [None, None])[0] = value
    for name, value in overrides.metric_maxima:
        metrics.setdefault(name, [None, None])[1] = value
    for name, bounds in metrics.items():
        if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            raise ValueError(f"effective metric {name} has min greater than max")
    return {
        "profile": profile.id,
        "include_tags": list(profile.include_tags),
        "required_tags": list(overrides.tags),
        "exclude_tags": list(profile.exclude_tags),
        "repetitions": (
            profile.repetitions
            if overrides.repetitions is None
            else overrides.repetitions
        ),
        "warmups": profile.warmups,
        "timeout_seconds": (
            profile.timeout_seconds
            if overrides.timeout_seconds is None
            else overrides.timeout_seconds
        ),
        "budget_seconds": (
            overrides.budget_seconds
            if overrides.budget_seconds is not None
            else profile.budget_seconds
        ),
        "cpu": overrides.cpu if overrides.cpu is not None else profile.cpu,
        "threads": profile.threads,
        "publication": profile.publication,
        "minimum_repetitions": profile.minimum_repetitions,
        "require_clean_sources": profile.require_clean_sources,
        "metrics": metrics,
        "case_ids": list(overrides.case_ids),
        "require_all_adapters": overrides.require_all_adapters,
    }


def resolved_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
