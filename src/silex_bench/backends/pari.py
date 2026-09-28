"""PARI/GP benchmark adapter.

The class/unit proof contract follows PARI's ``bnfinit(nf, 1)`` followed by a
checked ``bnfcertify``.  Result extraction deliberately uses GP's named BNF
members instead of representation-dependent vector offsets.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any

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
from ..util import read_bytes_nofollow
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


_READY_MARKER = "__SILEX_BENCH_PARI_READY__"
_TARGET_MARKER = "__SILEX_BENCH_PARI_TARGET_DONE__"
_REQUESTED_THREADS = 1
_THREAD_COUNT_SOURCE = "pari_default_nbthreads_runtime_query"
MAX_PARI_VERSION_BYTES = 64 << 10
_SOURCE_VERSION_KEYS = ("VersionMajor", "VersionMinor", "patch")
_SOURCE_VERSION_ASSIGNMENT = re.compile(
    r"^(VersionMajor|VersionMinor|patch)='([0-9]+)'$"
)
_SOURCE_VERSION_ASSIGNMENT_MENTION = re.compile(
    r"(?<![A-Za-z0-9_])(VersionMajor|VersionMinor|patch)[ \t]*(?:\+?=|:=)"
)
_SOURCE_VERSION_KEY_MENTION = re.compile(
    r"(?<![A-Za-z0-9_])(VersionMajor|VersionMinor|patch)(?![A-Za-z0-9_])"
)
_SOURCE_VERSION_INDIRECT_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_])(?:eval|printf|read)(?![A-Za-z0-9_])"
)
_SOURCE_VERSION_ANSI_C_QUOTING = re.compile(r"\$'")
_SOURCE_VERSION_SHELL_TOKEN_MARKUP = re.compile(r"[\\'\"]")
_SOURCE_VERSION_LINE_CONTINUATION = re.compile(r"\\\r?\n")
_SOURCE_VERSION_SHELL_ASSIGNMENT_WORD = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*="
)
_SOURCE_VERSION_SHELL_PUNCTUATION = "();<>|&`"
_SOURCE_VERSION_SHELL_ARITHMETIC_EXPANSION = "$(("
_SOURCE_VERSION_SHELL_COMMAND_PREFIXES = frozenset(
    {"!", "do", "elif", "else", "if", "then", "time", "until", "while", "{"}
)
_SOURCE_VERSION_SHELL_NONCOMMAND_WORDS = frozenset(
    {"case", "done", "esac", "fi", "for", "in", "select", "}"}
)
_SOURCE_VERSION_UNSAFE_SHELL_COMMANDS = frozenset(
    {
        ".",
        "alias",
        "builtin",
        "command",
        "declare",
        "enable",
        "eval",
        "exec",
        "export",
        "fc",
        "getopts",
        "let",
        "local",
        "mapfile",
        "printf",
        "read",
        "readarray",
        "readonly",
        "set",
        "source",
        "trap",
        "typeset",
        "unalias",
        "unset",
        "wait",
    }
)


def _has_ambiguous_shell_command(line: str) -> bool:
    try:
        lexer = shlex.shlex(
            line,
            posix=True,
            punctuation_chars=_SOURCE_VERSION_SHELL_PUNCTUATION,
        )
        lexer.commenters = "#"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return True

    command_expected = True
    for token in tokens:
        if _SOURCE_VERSION_SHELL_ARITHMETIC_EXPANSION in token:
            return True
        if token == "(" and not command_expected:
            continue
        if token and all(
            character in _SOURCE_VERSION_SHELL_PUNCTUATION for character in token
        ):
            command_expected = True
            continue
        if token in _SOURCE_VERSION_SHELL_COMMAND_PREFIXES:
            command_expected = True
            continue
        if not command_expected:
            continue
        if _SOURCE_VERSION_SHELL_ASSIGNMENT_WORD.match(token):
            continue
        if token in _SOURCE_VERSION_SHELL_NONCOMMAND_WORDS:
            command_expected = False
            continue
        if "$" in token or "`" in token:
            return True
        if token in _SOURCE_VERSION_UNSAFE_SHELL_COMMANDS:
            return True
        command_expected = False
    return False


def _environment(context: BackendContext) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in context.environment.items()})
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def _configured_tool(context: BackendContext, key: str, env_key: str) -> str:
    configured = context.tools.get(key)
    if configured is not None and str(configured).strip():
        return str(configured)
    configured = context.environment.get(env_key) or os.environ.get(env_key)
    return str(configured) if configured else key


def _configured_value(
    context: BackendContext, key: str, env_key: str
) -> str | None:
    configured = context.tools.get(key)
    if configured is not None and str(configured).strip():
        return str(configured).strip()
    configured = context.environment.get(env_key) or os.environ.get(env_key)
    if configured is None or not str(configured).strip():
        return None
    return str(configured).strip()


def parse_pari_source_version_bytes(version_bytes: bytes, version_file: Path) -> str:
    """Parse one exact PARI config/version generation without ambiguity."""
    try:
        version_text = version_bytes.decode("utf-8")
        lines = version_text.splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"could not read PARI source version file {version_file}: {exc}"
        ) from exc
    values: dict[str, str] = {}
    assignment_mentions = {key: 0 for key in _SOURCE_VERSION_KEYS}
    ambiguous_indirect_syntax = False
    noncanonical_preamble = False
    logical_text = _SOURCE_VERSION_LINE_CONTINUATION.sub("", version_text)
    for line in logical_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for mention in _SOURCE_VERSION_ASSIGNMENT_MENTION.finditer(stripped):
            assignment_mentions[mention.group(1)] += 1
    for line in lines:
        stripped = line.strip()
        match = _SOURCE_VERSION_ASSIGNMENT.fullmatch(stripped)
        if match:
            key = match.group(1)
            if key in values:
                raise ValueError(
                    f"PARI source version file {version_file} contains duplicate "
                    f"assignment for {key}"
                )
            values[key] = match.group(2)
        elif stripped and not stripped.startswith("#"):
            if len(values) != len(_SOURCE_VERSION_KEYS):
                noncanonical_preamble = True
            ambiguous = None
            key_scan_text = _SOURCE_VERSION_SHELL_TOKEN_MARKUP.sub("", stripped)
            for mention in _SOURCE_VERSION_KEY_MENTION.finditer(key_scan_text):
                start = mention.start()
                shell_expansion = (
                    start > 0 and key_scan_text[start - 1] == "$"
                ) or (start > 1 and key_scan_text[start - 2 : start] == "${")
                if not shell_expansion:
                    ambiguous = mention
                    break
            if ambiguous:
                raise ValueError(
                    f"PARI source version file {version_file} contains ambiguous "
                    f"assignment for {ambiguous.group(1)}"
                )
            if (
                _SOURCE_VERSION_ANSI_C_QUOTING.search(stripped)
                or _SOURCE_VERSION_INDIRECT_ASSIGNMENT.search(key_scan_text)
                or _has_ambiguous_shell_command(stripped)
            ):
                ambiguous_indirect_syntax = True
    missing = [key for key in _SOURCE_VERSION_KEYS if key not in values]
    if missing:
        raise ValueError(
            f"PARI source version file {version_file} is missing "
            + ", ".join(missing)
        )
    ambiguous = [
        key for key in _SOURCE_VERSION_KEYS if assignment_mentions[key] != 1
    ]
    if ambiguous:
        raise ValueError(
            f"PARI source version file {version_file} contains ambiguous "
            f"assignment for {ambiguous[0]}"
        )
    if ambiguous_indirect_syntax:
        raise ValueError(
            f"PARI source version file {version_file} contains ambiguous "
            "indirect assignment syntax"
        )
    if noncanonical_preamble:
        raise ValueError(
            f"PARI source version file {version_file} contains "
            "noncanonical content before all required assignments"
        )
    return ".".join(values[key] for key in _SOURCE_VERSION_KEYS)


def _source_release(source: Path) -> tuple[str, str, str]:
    version_file = source / "config" / "version"
    try:
        version_bytes = read_bytes_nofollow(
            version_file,
            root=source,
            max_bytes=MAX_PARI_VERSION_BYTES,
        )
    except OSError as exc:
        raise ValueError(
            f"could not read PARI source version file {version_file}: {exc}"
        ) from exc
    return (
        parse_pari_source_version_bytes(version_bytes, version_file),
        os.path.abspath(os.fspath(version_file)),
        hashlib.sha256(version_bytes).hexdigest(),
    )


def _parse_invariants(value: str | None) -> list[str] | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return []
    try:
        return canonical_invariants(
            part.strip() for part in stripped.split(",") if part.strip()
        )
    except ValueError:
        return None


def _programs(request: SampleRequest) -> tuple[str, str, str]:
    polynomial = polynomial_expr(request.field.coefficients_low_to_high)
    setup = ""
    if request.operation == "class_unit_proven":
        setup = f"""
P = {polynomial};
nf = nfinit(P);
"""
    elif request.operation == "maximal_order":
        setup = f"P = {polynomial};\n"
    elif request.operation == "ideal_multiply":
        setup = f"""
P = {polynomial};
nf = nfinit(P);
left_ideal = idealhnf(nf, 2);
right_ideal = idealhnf(nf, 3);
"""
    elif request.operation == "element_square_root":
        setup = f"""
P = {polynomial};
nf = nfinit(P);
square_base = nfalgtobasis(nf, x + 1);
square_target = nfeltmul(nf, square_base, square_base);
square_root = 0;
"""

    ready = f"""
default(nbthreads, {_REQUESTED_THREADS});
benchmark_reported_threads = default(nbthreads);
setrand({request.seed});
{setup}
print("{_READY_MARKER}");
"""

    if request.operation == "class_unit_proven":
        target = f"""
target_cpu_start_ms = getabstime();
target_wall_start_ms = getwalltime();
gettime();
b = bnfinit(nf, 1);
bnfinit_ms = gettime();
gettime();
certified = bnfcertify(b);
certification_ms = gettime();
if (!certified, error("bnfcertify failed"));
target_internal_cpu_ms = getabstime() - target_cpu_start_ms;
target_internal_wall_ms = getwalltime() - target_wall_start_ms;
print("{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}");
"""
        final = """
print("target_internal_cpu_ms=", target_internal_cpu_ms);
print("target_internal_wall_ms=", target_internal_wall_ms);
print("component_bnfinit_ms=", bnfinit_ms);
print("component_certification_ms=", certification_ms);
print("class_order=", b.no);
print("class_invariants=", b.cyc);
print("fundamental_unit_count=", #b.fu);
print("polynomial_discriminant=", poldisc(P));
print("maximal_order_discriminant=", b.disc);
print("signature_r1=", b.r1);
print("signature_r2=", b.r2);
print("certified=", certified);
print("reported_threads=", benchmark_reported_threads);
quit
"""
        return ready, target, final

    if request.operation == "maximal_order":
        target = f"""
target_cpu_start_ms = getabstime();
target_wall_start_ms = getwalltime();
gettime();
nf = nfinit(P);
maximal_order_ms = gettime();
target_internal_cpu_ms = getabstime() - target_cpu_start_ms;
target_internal_wall_ms = getwalltime() - target_wall_start_ms;
print("{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}");
"""
        final = """
print("target_internal_cpu_ms=", target_internal_cpu_ms);
print("target_internal_wall_ms=", target_internal_wall_ms);
print("component_maximal_order_ms=", maximal_order_ms);
print("polynomial_discriminant=", poldisc(P));
print("maximal_order_discriminant=", nf.disc);
print("signature_r1=", nf.r1);
print("signature_r2=", nf.r2);
print("reported_threads=", benchmark_reported_threads);
quit
"""
        return ready, target, final

    if request.operation == "ideal_multiply":
        target = f"""
target_cpu_start_ms = getabstime();
target_wall_start_ms = getwalltime();
gettime();
product_ideal = idealmul(nf, left_ideal, right_ideal);
ideal_multiply_ms = gettime();
target_internal_cpu_ms = getabstime() - target_cpu_start_ms;
target_internal_wall_ms = getwalltime() - target_wall_start_ms;
print("{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}");
"""
        final = """
print("target_internal_cpu_ms=", target_internal_cpu_ms);
print("target_internal_wall_ms=", target_internal_wall_ms);
print("component_ideal_multiply_ms=", ideal_multiply_ms);
print("ideal_norm=", idealnorm(nf, product_ideal));
print("reported_threads=", benchmark_reported_threads);
quit
"""
        return ready, target, final

    if request.operation == "element_square_root":
        target = f"""
target_cpu_start_ms = getabstime();
target_wall_start_ms = getwalltime();
gettime();
root_found = nfeltissquare(nf, square_target, &square_root);
square_root_ms = gettime();
target_internal_cpu_ms = getabstime() - target_cpu_start_ms;
target_internal_wall_ms = getwalltime() - target_wall_start_ms;
print("{_TARGET_MARKER}:{TARGET_NONCE_PLACEHOLDER}");
"""
        final = """
root_verified = root_found && nfeltmul(nf, square_root, square_root) == square_target;
print("target_internal_cpu_ms=", target_internal_cpu_ms);
print("target_internal_wall_ms=", target_internal_wall_ms);
print("component_square_root_ms=", square_root_ms);
print("root_found=", root_found);
print("root_verified=", root_verified);
print("reported_threads=", benchmark_reported_threads);
quit
"""
        return ready, target, final

    raise ValueError(f"unsupported PARI operation: {request.operation}")


def _result(request: SampleRequest, values: dict[str, str]) -> dict[str, Any]:
    if request.operation == "class_unit_proven":
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


def _components(request: SampleRequest, values: dict[str, str]) -> dict[str, Any]:
    if request.operation == "class_unit_proven":
        return {
            "bnfinit": parse_float(values, "component_bnfinit_ms"),
            "certification": parse_float(
                values, "component_certification_ms"
            ),
        }
    key = {
        "maximal_order": "component_maximal_order_ms",
        "ideal_multiply": "component_ideal_multiply_ms",
        "element_square_root": "component_square_root_ms",
    }.get(request.operation)
    return (
        {request.operation: parse_float(values, key)} if key is not None else {}
    )


def _complete(request: SampleRequest, result: dict[str, Any], certified: bool) -> bool:
    if request.operation == "class_unit_proven":
        return (
            certified
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


class PariBackend(BackendAdapter):
    name = "pari"

    def __init__(self) -> None:
        self._probe: dict[str, Any] | None = None

    def probe(self, context: BackendContext) -> dict[str, Any]:
        if self._probe is not None:
            return dict(self._probe)
        requested = _configured_tool(context, "gp", "SILEX_BENCH_GP")
        gp = resolve_executable(requested)
        if gp is None:
            self._probe = unavailable_probe(
                self.name, f"PARI/GP executable not found: {requested}"
            )
            return dict(self._probe)
        process = run_process(
            [gp, "--version-short"],
            timeout=min(context.timeout_seconds, 30.0),
            cwd=context.bench_root,
            env=_environment(context),
        )
        version = process.get("stdout", "").strip().splitlines()
        observed_version = version[-1].strip() if version else None
        source = _configured_value(
            context, "pari_source", "SILEX_BENCH_PARI_SOURCE"
        )
        source_path = (
            Path(os.path.abspath(os.fspath(Path(source).expanduser())))
            if source
            else None
        )
        source_version = None
        source_version_file = None
        source_version_file_sha256 = None
        provenance_error = None
        if source_path is not None:
            try:
                (
                    source_version,
                    source_version_file,
                    source_version_file_sha256,
                ) = _source_release(source_path)
            except ValueError as exc:
                provenance_error = nonempty_diagnostic(
                    str(exc), fallback="PARI source provenance validation failed"
                )
        required_version = _configured_value(
            context, "pari_version", "SILEX_BENCH_PARI_VERSION"
        )
        if required_version is None:
            required_version = source_version
        if (
            provenance_error is None
            and source_version is not None
            and required_version is not None
            and source_version != required_version
        ):
            provenance_error = (
                f"PARI source version {source_version} does not match required "
                f"version {required_version}"
            )
        if (
            provenance_error is None
            and required_version is not None
            and observed_version != required_version
        ):
            provenance_error = (
                f"PARI/GP executable version {observed_version!r} does not match "
                f"required source-traced version {required_version}"
            )
        identity = {
            "engine": self.name,
            "executable": str(Path(gp).resolve()),
            "version": observed_version,
            "required_version": required_version,
            "source": str(source_path) if source_path else None,
            "source_version": source_version,
            "source_version_file": source_version_file,
            "source_version_file_sha256": source_version_file_sha256,
        }
        identity = identity_with_executed_digest(identity, process)
        process_success = process_state_is_valid(process) and process["success"]
        if not process_success:
            probe_error = (
                process.get("error")
                or process.get("stderr")
                or "PARI/GP version probe failed"
            )
        elif not observed_version:
            probe_error = "PARI/GP version probe returned no version"
        else:
            probe_error = provenance_error
        if process_success and observed_version and provenance_error is None:
            self._probe = successful_probe(self.name, identity)
        else:
            self._probe = unavailable_probe(
                self.name,
                str(probe_error),
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

        gp = str(probe["engine_identity"]["executable"])
        ready, target, final = _programs(request)
        process = run_marked_process(
            [gp, "-q", "-f"],
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
        values = parse_key_values(process.get("stdout", ""))
        result = _result(request, values)
        certified = parse_bool(values, "certified") is True
        process_success = process_state_is_valid(process) and process["success"]
        unit_error = (
            unit_count_error("PARI", result)
            if request.operation == "class_unit_proven"
            else None
        )
        # bnfcertify proves the whole bnf structure at once, so a returned
        # unit count that disagrees with r1 + r2 - 1 means the certified bnf
        # is not the one the adapter read back; the proof labels drop to
        # unknown together with certification, matching the other adapters.
        certified_effective = certified and unit_error is None
        result_complete = (
            process_success
            and _complete(request, result, certified)
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
        if request.operation == "class_unit_proven":
            proof = {
                "certification_status": "proven" if certified_effective else "failed",
                "class_group_proof_status": (
                    "proven" if certified_effective else "unknown"
                ),
                "unit_group_proof_status": (
                    "proven" if certified_effective else "unknown"
                ),
                "regulator_proof_status": (
                    "proven" if certified_effective else "unknown"
                ),
                "proof_complete": certified_effective,
                "final_result_published": result_complete,
            }
        else:
            proof = {
                "certification_status": "not_applicable",
                "final_result_published": result_complete,
            }
        marked_target_cpu_ms = process.get("target_cpu_ms")
        marked_target_wall_ms = process.get("target_wall_ms")
        internal_target_cpu_ms = parse_float(
            values, "target_internal_cpu_ms"
        )
        internal_target_wall_ms = parse_float(
            values, "target_internal_wall_ms"
        )
        target_cpu_ms = internal_target_cpu_ms
        target_wall_ms = internal_target_wall_ms
        timing = {
            "algorithm_clock": "pari_getabstime_ms",
            "wall_clock": "pari_getwalltime_ms",
            "component_clock": "pari_gettime_ms",
            "scope": {
                "class_unit_proven": "class_and_unit_group_only",
                "maximal_order": "maximal_order_only",
                "ideal_multiply": "ideal_multiplication_only",
                "element_square_root": "number_field_element_is_square_only",
            }[request.operation],
            "preparation_excluded": True,
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
        if (
            not success
            and error is None
            and result_complete
            and not thread_count["matches_requested"]
        ):
            error = (
                "PARI thread-count contract failed: requested "
                f"{_REQUESTED_THREADS}, reported {reported_threads!r}"
            )
        if not success and error is None and unit_error is not None:
            error = unit_error
        if not success and error is None:
            error = "PARI operation failed or returned incomplete output"
        if success:
            status = "ok"
        elif process.get("timeout") is True:
            status = "timeout"
        elif result_complete and not thread_count["matches_requested"]:
            status = "thread_contract"
        else:
            status = "compute_error"
        payload: dict[str, Any] = {
            "engine": self.name,
            "algorithm": "external",
            "available": process.get("available", True) is True,
            "success": success,
            "timeout": process.get("timeout") is True,
            "status": status,
            "error": error,
            "target_cpu_ms": target_cpu_ms,
            "target_wall_ms": target_wall_ms,
            "process_wall_ms": process.get("process_wall_ms"),
            "engine_identity": identity_with_executed_digest(
                probe.get("engine_identity"), process
            ),
            "cmd": process.get("cmd"),
            "result": result,
            "proof": proof,
            "thread_count": thread_count,
            "timing": timing,
        }
        if not success:
            payload["diagnostics"] = {
                "stdout": process.get("stdout", ""),
                "stderr": process.get("stderr", ""),
                "cmd": process.get("cmd"),
            }
        return payload
