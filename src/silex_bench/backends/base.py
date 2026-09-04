"""Backend adapter interface and normalization helpers."""

from __future__ import annotations

import copy
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from ..model import BackendContext, SampleRequest


_CANONICAL_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")


class BackendAdapter(ABC):
    name: str

    @abstractmethod
    def probe(self, context: BackendContext) -> dict[str, Any]:
        """Return availability and engine identity without running a sample."""

    @abstractmethod
    def run(
        self, request: SampleRequest, context: BackendContext
    ) -> dict[str, Any]:
        """Run one isolated sample and return a normalized backend payload."""


def process_state_is_valid(process: Any) -> bool:
    """Return whether a subprocess result has an exact, consistent state."""
    return (
        type(process) is dict
        and type(process.get("available")) is bool
        and type(process.get("success")) is bool
        and type(process.get("timeout")) is bool
        and not (process["success"] and not process["available"])
        and not (process["success"] and process["timeout"])
    )


def nonempty_diagnostic(*values: Any, fallback: str) -> str:
    """Return the first nonempty text diagnostic, or a fixed fallback."""
    for value in values:
        if type(value) is str and value.strip():
            return value
    return fallback


def identity_with_executed_digest(
    identity: Any,
    process: Any,
    *,
    executable_key: str = "executable",
) -> Any:
    """Return an identity bound to the immutable bytes executed by the process."""
    enriched = copy.deepcopy(identity)
    if type(enriched) is not dict or type(process) is not dict:
        return enriched
    digest = process.get("executable_sha256")
    if type(digest) is not str:
        return enriched
    digests = enriched.get("executable_sha256")
    if type(digests) is not dict:
        digests = {}
    else:
        digests = copy.deepcopy(digests)
    digests[executable_key] = digest
    enriched["executable_sha256"] = digests
    return enriched


def successful_probe(engine: str, identity: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical persisted envelope for a successful probe."""
    return {
        "engine": engine,
        "available": True,
        "success": True,
        "timeout": False,
        "status": "ok",
        "error": None,
        "engine_identity": identity,
    }


def unavailable_probe(
    engine: str,
    error: str,
    *,
    identity: dict[str, Any] | None = None,
    timeout: bool = False,
) -> dict[str, Any]:
    """Return the canonical persisted envelope for an unsuccessful probe."""
    return {
        "engine": engine,
        "available": False,
        "success": False,
        "timeout": timeout,
        "status": "timeout" if timeout else "unavailable",
        "error": error,
        "engine_identity": identity,
    }


def unavailable(engine: str, error: str) -> dict[str, Any]:
    return {
        "engine": engine,
        "algorithm": "external" if engine != "silex" else "default",
        "available": False,
        "success": False,
        "timeout": False,
        "status": "unavailable",
        "error": error,
        "target_cpu_ms": None,
        "target_wall_ms": None,
        "process_wall_ms": None,
        "result": {},
        "proof": {},
        "timing": {},
    }


def parse_int(values: dict[str, str], key: str) -> int | None:
    value = values.get(key)
    if value is None or _CANONICAL_INTEGER.fullmatch(value) is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(values: dict[str, str], key: str) -> float | None:
    value = values.get(key)
    if value is None or value in {"", "nothing", "None"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_bool(values: dict[str, str], key: str) -> bool | None:
    value = values.get(key)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    return None
