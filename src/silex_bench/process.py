"""Subprocess helpers with bounded marked target timing."""

from __future__ import annotations

import fcntl
import functools
import hashlib
import math
import os
import secrets
import select
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .util import read_bytes_nofollow, trusted_system_executable


MAX_CAPTURE_BYTES = 1 << 20
MAX_PROCESS_INPUT_BYTES = 1 << 20
MAX_PROTOCOL_INPUT_BYTES = 1 << 20
MAX_PROTOCOL_MARKER_BYTES = 256
MAX_IMMUTABLE_INPUT_BYTES = 16 << 20
TARGET_NONCE_PLACEHOLDER = "__SILEX_BENCH_TARGET_NONCE__"
_READ_CHUNK_BYTES = 65536
_SUPERVISOR_CONTROL_BYTES = 128
_SUPERVISOR_HANDSHAKE_SECONDS = 5.0
_SUPERVISOR_FDS_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_FDS"
_SUPERVISOR_EXECUTABLE_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_EXECUTABLE"
_SUPERVISOR_CONTROL_FD_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_CONTROL_FD"
_SUPERVISOR_CPU_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_CPU"
_SUPERVISOR_ENV_KEYS = {
    _SUPERVISOR_FDS_ENV,
    _SUPERVISOR_EXECUTABLE_ENV,
    _SUPERVISOR_CONTROL_FD_ENV,
    _SUPERVISOR_CPU_ENV,
}

# Python 3.11 exposes ``os.memfd_create`` but not these fcntl names.  The
# fallback values are part of Linux's stable userspace ABI (linux/fcntl.h), not
# CPython implementation details.
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
_IMMUTABLE_MEMFD_SEALS = (
    _F_SEAL_WRITE | _F_SEAL_GROW | _F_SEAL_SHRINK | _F_SEAL_SEAL
)

_SUPERVISOR_SOURCE = r"""
import ctypes
import os
import signal
import subprocess
import sys
import time

PR_SET_PDEATHSIG = 1
PR_SET_CHILD_SUBREAPER = 36
MAX_DESCENDANTS = 100000
MAX_CHILDREN_BYTES = 1 << 20
FDS_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_FDS"
EXECUTABLE_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_EXECUTABLE"
CONTROL_FD_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_CONTROL_FD"
CPU_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_CPU"

stop_requested = False


def request_stop(signum, frame):
    del signum, frame
    global stop_requested
    stop_requested = True


def report(control_fd, message):
    payload = (message + "\n").encode("ascii", errors="replace")[:128]
    offset = 0
    while offset < len(payload):
        offset += os.write(control_fd, payload[offset:])


def direct_children(pid):
    path = f"/proc/{pid}/task/{pid}/children"
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        payload = os.read(descriptor, MAX_CHILDREN_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_CHILDREN_BYTES:
        raise RuntimeError("process descendant list exceeds its byte limit")
    result = []
    for token in payload.split():
        value = int(token)
        if value > 0:
            result.append(value)
    return result


def descendants():
    pending = [os.getpid()]
    seen = set()
    while pending:
        parent = pending.pop()
        try:
            children = direct_children(parent)
        except (FileNotFoundError, ProcessLookupError):
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            if len(seen) > MAX_DESCENDANTS:
                raise RuntimeError("process descendant count exceeds its limit")
            pending.append(child)
    return seen


def signal_all(pids, signum):
    for pid in pids:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def terminate_descendants():
    observed = False
    frozen = set()
    for _ in range(32):
        current = descendants()
        if not current:
            break
        observed = True
        frozen.update(current)
        signal_all(current, signal.SIGSTOP)
        time.sleep(0.002)
        following = descendants()
        frozen.update(following)
        if following.issubset(current):
            break
    frozen.update(descendants())
    signal_all(frozen, signal.SIGKILL)

    deadline = time.monotonic() + 0.75
    while time.monotonic() < deadline:
        while True:
            try:
                waited, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                waited = 0
                break
            if waited <= 0:
                break
        remaining = descendants()
        if not remaining:
            try:
                os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return observed
        signal_all(remaining, signal.SIGKILL)
        time.sleep(0.005)
    if descendants():
        raise RuntimeError("process descendants survived bounded cleanup")
    return observed


def finish(returncode):
    if returncode < 0:
        signum = -returncode
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        time.sleep(0.1)
        os._exit(128 + signum)
    os._exit(min(returncode, 255))


def main():
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)

    control_fd = int(os.environ[CONTROL_FD_ENV])
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        report(control_fd, f"ERROR child subreaper setup failed: {ctypes.get_errno()}")
        os._exit(126)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        report(control_fd, f"ERROR parent-death signal setup failed: {ctypes.get_errno()}")
        os._exit(126)
    if os.getppid() != parent_pid:
        report(control_fd, "ERROR parent exited during supervisor setup")
        os._exit(126)

    fd_text = os.environ.get(FDS_ENV, "")
    inherited_fds = tuple(int(value) for value in fd_text.split(",") if value)
    target_executable = os.environ[EXECUTABLE_ENV]
    cpu_text = os.environ.get(CPU_ENV, "")
    if cpu_text:
        os.sched_setaffinity(0, {int(cpu_text)})

    target_env = dict(os.environ)
    for key in (FDS_ENV, EXECUTABLE_ENV, CONTROL_FD_ENV, CPU_ENV):
        target_env.pop(key, None)
    if not sys.argv[1:]:
        report(control_fd, "ERROR supervisor target command is empty")
        os._exit(126)
    try:
        target = subprocess.Popen(
            sys.argv[1:],
            executable=target_executable,
            pass_fds=inherited_fds,
            env=target_env,
            close_fds=True,
        )
    except BaseException as exc:
        report(control_fd, f"ERROR target spawn failed: {exc}")
        os._exit(126)
    report(control_fd, f"PID {target.pid}")
    os.close(control_fd)

    while True:
        if stop_requested:
            terminate_descendants()
            os._exit(124)
        returncode = target.poll()
        if returncode is not None:
            while True:
                try:
                    waited, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if waited <= 0:
                    break
            if not descendants():
                finish(returncode)
        time.sleep(0.005)


try:
    main()
except BaseException as exc:
    try:
        terminate_descendants()
    except BaseException:
        pass
    try:
        control = int(os.environ.get(CONTROL_FD_ENV, "-1"))
        if control >= 0:
            report(control, f"ERROR supervisor failed: {exc}")
    except BaseException:
        pass
    os._exit(126)
"""


class _OutputLimitExceeded(Exception):
    pass


def _command_error(cmd: Any) -> str | None:
    if type(cmd) is not list or not cmd:
        return "command must contain an executable"
    if type(cmd[0]) is not str or not cmd[0]:
        return "command executable must be a nonempty string"
    if any(type(argument) is not str for argument in cmd):
        return "command arguments must be strings"
    if any("\0" in argument for argument in cmd):
        return "command arguments must not contain null bytes"
    return None


def _command_failure(cmd: Any, error: str) -> dict[str, Any]:
    safe_command = (
        list(cmd)
        if type(cmd) is list and all(type(argument) is str for argument in cmd)
        else None
    )
    return {
        "available": False,
        "success": False,
        "timeout": False,
        "cmd": safe_command,
        "error": error,
    }


def _duration_error(value: Any, label: str) -> str | None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        return f"{label} must be a finite positive number"
    return None


def _process_input_bytes(value: Any) -> tuple[bytes | None, str | None]:
    if value is None:
        return None, None
    if type(value) is not str:
        return None, "stdin must be a string"
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return None, "stdin must be valid UTF-8"
    if b"\0" in encoded:
        return None, "stdin must not contain null bytes"
    if len(encoded) > MAX_PROCESS_INPUT_BYTES:
        return None, f"stdin exceeds the {MAX_PROCESS_INPUT_BYTES}-byte input limit"
    return encoded, None


def _environment_error(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not dict:
        return "environment must be an object"
    for key, item in value.items():
        if type(key) is not str or not key or "=" in key or "\0" in key:
            return "environment keys must be nonempty strings without '=' or null bytes"
        if type(item) is not str or "\0" in item:
            return "environment values must be strings without null bytes"
        if key in _SUPERVISOR_ENV_KEYS:
            return f"environment key {key!r} is reserved for process containment"
    return None


def _protocol_error(
    ready_input: Any,
    target_input: Any,
    final_input: Any,
    ready_marker: Any,
    target_marker: Any,
) -> str | None:
    for label, value in (
        ("ready_input", ready_input),
        ("target_input", target_input),
        ("final_input", final_input),
    ):
        if type(value) is not str:
            return f"{label} must be a string"
        try:
            encoded = value.encode()
        except UnicodeEncodeError:
            return f"{label} must be valid UTF-8"
        if b"\0" in encoded:
            return f"{label} must not contain null bytes"
        if len(encoded) > MAX_PROTOCOL_INPUT_BYTES:
            return (
                f"{label} exceeds the {MAX_PROTOCOL_INPUT_BYTES}-byte "
                "protocol input limit"
            )
    if target_input.count(TARGET_NONCE_PLACEHOLDER) != 1:
        return "target_input must contain the target nonce placeholder exactly once"
    expanded_target = target_input.replace(TARGET_NONCE_PLACEHOLDER, "0" * 32)
    if len(expanded_target.encode("utf-8")) > MAX_PROTOCOL_INPUT_BYTES:
        return (
            f"target_input exceeds the {MAX_PROTOCOL_INPUT_BYTES}-byte protocol "
            "input limit after nonce expansion"
        )
    for label, marker in (
        ("ready_marker", ready_marker),
        ("target_marker", target_marker),
    ):
        if type(marker) is not str or not marker:
            return f"{label} must be a nonempty printable ASCII string"
        if not marker.isascii() or any(not 0x21 <= ord(ch) <= 0x7E for ch in marker):
            return f"{label} must be a nonempty printable ASCII string"
        if len(marker.encode("ascii")) > MAX_PROTOCOL_MARKER_BYTES:
            return (
                f"{label} exceeds the {MAX_PROTOCOL_MARKER_BYTES}-byte "
                "protocol marker limit"
            )
    return None


def _target_nonce_is_valid(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 32
        and all(ch in "0123456789abcdef" for ch in value)
    )


def process_cpu_runtime_ns(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/schedstat").read_text().split()[0])
    except (IndexError, OSError, ValueError):
        return None


@functools.lru_cache(maxsize=1)
def _trusted_cpu_launcher_identity() -> tuple[str, str]:
    path = trusted_system_executable("taskset")
    descriptor, digest, resolved = _snapshot_executable(
        path,
        cwd=Path.cwd(),
        env=None,
    )
    os.close(descriptor)
    return resolved, digest


def trusted_cpu_launcher_identity() -> dict[str, str]:
    """Return a fresh copy of the trusted CPU-launcher path/digest identity."""
    path, digest = _trusted_cpu_launcher_identity()
    return {"executable": path, "sha256": digest}


def _with_cpu(cmd: list[str], cpu: int | None) -> list[str]:
    if cpu is None:
        return cmd
    try:
        available_cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise OSError(
            "CPU affinity requested but the current process affinity "
            f"is unavailable: {exc}"
        ) from exc
    if cpu not in available_cpus:
        raise OSError(
            f"requested CPU {cpu} is outside the current process affinity "
            f"{available_cpus}"
        )
    taskset = trusted_system_executable("taskset")
    return [taskset, "-c", str(cpu), *cmd]


def _resolved_command_path(
    value: str,
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> Path:
    if os.sep not in value:
        search_path = (os.environ if env is None else env).get("PATH", os.defpath)
        resolved = shutil.which(value, path=search_path)
        if resolved is None:
            raise FileNotFoundError(f"executable not found: {value}")
        return Path(resolved).resolve(strict=True)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve(strict=True)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("could not snapshot executable bytes")
        view = view[written:]


def _seal_memfd_immutable(descriptor: int) -> None:
    fcntl.fcntl(descriptor, _F_ADD_SEALS, _IMMUTABLE_MEMFD_SEALS)
    applied = fcntl.fcntl(descriptor, _F_GET_SEALS)
    if applied & _IMMUTABLE_MEMFD_SEALS != _IMMUTABLE_MEMFD_SEALS:
        raise OSError("kernel did not apply every required immutable memfd seal")


def _snapshot_executable(
    value: str,
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> tuple[int, str, str]:
    if not hasattr(os, "memfd_create"):
        raise OSError("immutable executable snapshots require Linux memfd_create")
    path = _resolved_command_path(value, cwd=cwd, env=env)
    source = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    snapshot: int | None = None
    try:
        opened = os.fstat(source)
        if not stat.S_ISREG(opened.st_mode) or not opened.st_mode & 0o111:
            raise PermissionError(f"executable is not a regular executable file: {path}")
        snapshot = os.memfd_create(
            "silex-bench-executable",
            getattr(os, "MFD_CLOEXEC", 0)
            | getattr(os, "MFD_ALLOW_SEALING", 0),
        )
        digest = hashlib.sha256()
        while block := os.read(source, 1024 * 1024):
            _write_all(snapshot, block)
            digest.update(block)
        finished = os.fstat(source)
        signature = lambda metadata: (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if signature(opened) != signature(finished):
            raise OSError(f"executable changed while being snapshotted: {path}")
        os.fchmod(snapshot, 0o500)
        os.lseek(snapshot, 0, os.SEEK_SET)
        _seal_memfd_immutable(snapshot)
        result = snapshot, digest.hexdigest(), str(path)
        snapshot = None
        return result
    finally:
        if snapshot is not None:
            os.close(snapshot)
        os.close(source)


def _snapshot_regular_argument(
    value: str,
    *,
    cwd: Path,
) -> tuple[int, str, str]:
    if not hasattr(os, "memfd_create"):
        raise OSError("immutable input snapshots require Linux memfd_create")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    path = Path(os.path.abspath(os.fspath(candidate)))
    payload = read_bytes_nofollow(
        path,
        root=path.parent,
        max_bytes=MAX_IMMUTABLE_INPUT_BYTES,
    )
    snapshot = os.memfd_create(
        "silex-bench-input",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    try:
        _write_all(snapshot, payload)
        os.fchmod(snapshot, 0o400)
        os.lseek(snapshot, 0, os.SEEK_SET)
        _seal_memfd_immutable(snapshot)
        result = snapshot, hashlib.sha256(payload).hexdigest(), str(path)
        snapshot = -1
        return result
    finally:
        if snapshot >= 0:
            os.close(snapshot)


def _immutable_argument_error(value: Any, cmd: list[str]) -> str | None:
    if type(value) is not tuple:
        return "immutable_path_arguments must be a tuple of argument indices"
    if any(type(index) is not int for index in value):
        return "immutable_path_arguments must contain exact integer indices"
    if len(set(value)) != len(value):
        return "immutable_path_arguments must not contain duplicate indices"
    if any(index < 1 or index >= len(cmd) for index in value):
        return "immutable_path_arguments must identify non-executable command arguments"
    return None


def _directory_argument_error(
    directory_descriptors: Any,
    directory_argument_descriptors: Any,
    cmd: list[str],
    immutable_path_arguments: tuple[int, ...],
) -> str | None:
    if directory_descriptors is None:
        directory_descriptors = {}
    if type(directory_descriptors) is not dict:
        return "directory_descriptors must be an object"
    for target_descriptor, supplied_descriptor in directory_descriptors.items():
        if type(target_descriptor) is not int or type(supplied_descriptor) is not int:
            return (
                "directory_descriptors must map exact integer target descriptors "
                "to exact integer open descriptors"
            )
        if target_descriptor < 3:
            return "directory target descriptors must be at least 3"
        if supplied_descriptor < 0:
            return "directory_descriptors must contain open descriptors"
    if directory_argument_descriptors is None:
        directory_argument_descriptors = {}
    if type(directory_argument_descriptors) is not dict:
        return "directory_argument_descriptors must be an object"
    for index, target_descriptor in directory_argument_descriptors.items():
        if type(index) is not int or type(target_descriptor) is not int:
            return (
                "directory_argument_descriptors must map exact integer indices "
                "to exact integer target descriptors"
            )
        if index < 1 or index >= len(cmd):
            return (
                "directory_argument_descriptors must identify non-executable "
                "command arguments"
            )
        if target_descriptor not in directory_descriptors:
            return (
                "directory arguments must reference a declared directory target "
                "descriptor"
            )
        if index in immutable_path_arguments:
            return (
                "directory arguments must not overlap immutable regular-file "
                "arguments"
            )
    return None


def _pinned_command(
    cmd: list[str],
    *,
    cpu: int | None,
    cwd: Path,
    env: dict[str, str] | None,
    immutable_path_arguments: tuple[int, ...],
    directory_descriptors: dict[int, int],
    directory_argument_descriptors: dict[int, int],
) -> dict[str, Any]:
    display = _with_cpu(cmd, cpu)
    backend_fd, backend_digest, backend_path = _snapshot_executable(
        cmd[0], cwd=cwd, env=env
    )
    descriptors = [backend_fd]
    immutable_inputs: list[dict[str, Any]] = []
    try:
        backend_command = [backend_path, *cmd[1:]]
        for index in immutable_path_arguments:
            descriptor, digest, path = _snapshot_regular_argument(
                cmd[index],
                cwd=cwd,
            )
            descriptors.append(descriptor)
            backend_command[index] = f"/proc/self/fd/{descriptor}"
            immutable_inputs.append(
                {
                    "argument_index": index,
                    "path": path,
                    "sha256": digest,
                }
            )
        for target_descriptor, supplied_descriptor in sorted(
            directory_descriptors.items()
        ):
            descriptor = fcntl.fcntl(
                supplied_descriptor,
                fcntl.F_DUPFD_CLOEXEC,
                target_descriptor,
            )
            try:
                if descriptor != target_descriptor:
                    raise ValueError(
                        f"directory target descriptor {target_descriptor} is unavailable"
                    )
                metadata = os.fstat(descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(
                        f"directory target descriptor {target_descriptor} must identify "
                        "a directory"
                    )
            except BaseException:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)
        for index, target_descriptor in sorted(
            directory_argument_descriptors.items()
        ):
            backend_command[index] = f"/proc/self/fd/{target_descriptor}"
        backend_snapshot_path = f"/proc/self/fd/{backend_fd}"
        launcher_path: str | None = None
        launcher_digest: str | None = None
        if cpu is None:
            spawn = backend_command
            executable = backend_snapshot_path
        else:
            taskset_fd, launcher_digest, launcher_path = _snapshot_executable(
                display[0], cwd=cwd, env=env
            )
            descriptors.append(taskset_fd)
            spawn = [
                launcher_path,
                "-c",
                str(cpu),
                backend_snapshot_path,
                *backend_command[1:],
            ]
            executable = f"/proc/self/fd/{taskset_fd}"
        return {
            "display": display,
            "spawn": spawn,
            "executable": executable,
            "descriptors": tuple(descriptors),
            "backend_digest": backend_digest,
            "backend_path": backend_path,
            "launcher_path": launcher_path,
            "launcher_digest": launcher_digest,
            "immutable_inputs": immutable_inputs,
        }
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def _extend_limited(output: bytearray, chunk: bytes) -> bool:
    remaining = MAX_CAPTURE_BYTES - len(output)
    if len(chunk) <= remaining:
        output.extend(chunk)
        return True
    output.extend(chunk[:remaining])
    return False


def _close_stdin(process: subprocess.Popen[Any]) -> None:
    if process.stdin is not None:
        process.stdin.close()
        process.stdin = None


def _close_output_pipes(process: subprocess.Popen[Any]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _bounded_communicate(
    process: subprocess.Popen[bytes],
    input_data: bytes | None,
    timeout: float,
) -> tuple[bytearray, bytearray, str | None]:
    stdout = bytearray()
    stderr = bytearray()
    readers: dict[int, bytearray] = {}
    if process.stdout is not None:
        readers[process.stdout.fileno()] = stdout
    if process.stderr is not None:
        readers[process.stderr.fileno()] = stderr
    for fd in readers:
        os.set_blocking(fd, False)

    stdin_fd: int | None = None
    input_offset = 0
    pending_input = input_data or b""
    if process.stdin is not None:
        if pending_input:
            stdin_fd = process.stdin.fileno()
            os.set_blocking(stdin_fd, False)
        else:
            _close_stdin(process)

    deadline = time.monotonic() + timeout
    while readers or stdin_fd is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return stdout, stderr, "process timed out"
        readable, writable, _ = select.select(
            list(readers),
            [stdin_fd] if stdin_fd is not None else [],
            [],
            remaining,
        )
        if not readable and not writable:
            return stdout, stderr, "process timed out"
        if stdin_fd is not None and stdin_fd in writable:
            try:
                written = os.write(stdin_fd, pending_input[input_offset:])
            except BrokenPipeError:
                written = len(pending_input) - input_offset
            input_offset += written
            if input_offset >= len(pending_input):
                _close_stdin(process)
                stdin_fd = None
        for fd in readable:
            output = readers[fd]
            capacity = MAX_CAPTURE_BYTES - len(output)
            try:
                chunk = os.read(fd, min(_READ_CHUNK_BYTES, capacity + 1))
            except BlockingIOError:
                continue
            if not chunk:
                del readers[fd]
                continue
            if not _extend_limited(output, chunk):
                return stdout, stderr, "process output limit exceeded"

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return stdout, stderr, "process timed out"
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        return stdout, stderr, "process timed out"
    return stdout, stderr, None


def _stop_process(
    process: subprocess.Popen[Any],
) -> tuple[int | None, bytes, bytes]:
    _close_stdin(process)
    # Descendants that escaped the process group can retain inherited pipes.
    # Closing our read ends before reaping prevents communicate() from buffering
    # an unbounded tail outside the per-stream capture contract.
    _close_output_pipes(process)
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            if process.poll() is None:
                process.kill()
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            return None, b"", b""
    return process.returncode, b"", b""


def _supervised_popen(
    pinned: dict[str, Any],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    stdin: Any,
    cpu: int | None,
    bufsize: int = -1,
) -> tuple[subprocess.Popen[bytes], int]:
    control_read, control_write = os.pipe2(
        getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        supervisor_environment = dict(os.environ if env is None else env)
        supervisor_environment[_SUPERVISOR_FDS_ENV] = ",".join(
            str(descriptor) for descriptor in pinned["descriptors"]
        )
        supervisor_environment[_SUPERVISOR_EXECUTABLE_ENV] = pinned["executable"]
        supervisor_environment[_SUPERVISOR_CONTROL_FD_ENV] = str(control_write)
        supervisor_environment[_SUPERVISOR_CPU_ENV] = "" if cpu is None else str(cpu)
        supervisor_command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _SUPERVISOR_SOURCE,
            *pinned["spawn"],
        ]
        process = subprocess.Popen(
            supervisor_command,
            executable="/proc/self/exe",
            pass_fds=(*pinned["descriptors"], control_write),
            cwd=cwd,
            env=supervisor_environment,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bufsize,
            start_new_session=True,
        )
        os.close(control_write)
        control_write = -1

        response = bytearray()
        deadline = time.monotonic() + _SUPERVISOR_HANDSHAKE_SECONDS
        while b"\n" not in response:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OSError("process supervisor handshake timed out")
            readable, _, _ = select.select([control_read], [], [], remaining)
            if control_read not in readable:
                raise OSError("process supervisor handshake timed out")
            chunk = os.read(
                control_read,
                _SUPERVISOR_CONTROL_BYTES + 1 - len(response),
            )
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > _SUPERVISOR_CONTROL_BYTES:
                raise OSError("process supervisor response exceeds its size limit")
        line = bytes(response).splitlines()[0] if response else b""
        fields = line.decode("ascii", errors="replace").split()
        if len(fields) != 2 or fields[0] != "PID" or not fields[1].isdigit():
            detail = line.decode("ascii", errors="replace") or "no response"
            raise OSError(f"process supervisor could not start target: {detail}")
        target_pid = int(fields[1])
        if target_pid <= 0:
            raise OSError("process supervisor returned an invalid target PID")
        return process, target_pid
    except BaseException:
        if process is not None:
            _stop_process(process)
        raise
    finally:
        os.close(control_read)
        if control_write >= 0:
            os.close(control_write)


def run_process(
    cmd: list[str],
    *,
    timeout: float,
    cwd: Path,
    stdin: str | None = None,
    cpu: int | None = None,
    env: dict[str, str] | None = None,
    immutable_path_arguments: tuple[int, ...] = (),
    directory_descriptors: dict[int, int] | None = None,
    directory_argument_descriptors: dict[int, int] | None = None,
) -> dict[str, Any]:
    if (command_error := _command_error(cmd)) is not None:
        return _command_failure(cmd, command_error)
    if (duration_error := _duration_error(timeout, "timeout")) is not None:
        return _command_failure(cmd, duration_error)
    input_bytes, input_error = _process_input_bytes(stdin)
    if input_error is not None:
        return _command_failure(cmd, input_error)
    if (environment_error := _environment_error(env)) is not None:
        return _command_failure(cmd, environment_error)
    if (argument_error := _immutable_argument_error(immutable_path_arguments, cmd)) is not None:
        return _command_failure(cmd, argument_error)
    if (
        directory_error := _directory_argument_error(
            directory_descriptors,
            directory_argument_descriptors,
            cmd,
            immutable_path_arguments,
        )
    ) is not None:
        return _command_failure(cmd, directory_error)
    try:
        pinned = _pinned_command(
            cmd,
            cpu=cpu,
            cwd=cwd,
            env=env,
            immutable_path_arguments=immutable_path_arguments,
            directory_descriptors=directory_descriptors or {},
            directory_argument_descriptors=directory_argument_descriptors or {},
        )
    except (OSError, ValueError) as exc:
        return _command_failure(cmd, str(exc))
    execution_evidence = {
        "cmd": pinned["display"],
        "executable_sha256": pinned["backend_digest"],
        "launcher_executable": pinned["launcher_path"],
        "launcher_executable_sha256": pinned["launcher_digest"],
        "immutable_inputs": pinned["immutable_inputs"],
    }
    start = time.perf_counter_ns()
    process: subprocess.Popen[bytes] | None = None
    try:
        try:
            process, _ = _supervised_popen(
                pinned,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE if stdin is not None else None,
                cpu=cpu,
            )
        finally:
            for descriptor in pinned["descriptors"]:
                os.close(descriptor)
        stdout, stderr, capture_error = _bounded_communicate(
            process,
            input_bytes,
            timeout,
        )
        if capture_error is not None:
            returncode, stdout_tail, stderr_tail = _stop_process(process)
            _extend_limited(stdout, stdout_tail)
            _extend_limited(stderr, stderr_tail)
            return {
                "available": True,
                "success": False,
                "timeout": capture_error == "process timed out",
                "returncode": returncode,
                "process_wall_ms":
                    (time.perf_counter_ns() - start) / 1_000_000,
                **execution_evidence,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "error": capture_error,
            }
    except subprocess.TimeoutExpired:
        if process is None:
            raise
        returncode, stdout_tail, stderr_tail = _stop_process(process)
        return {
            "available": True,
            "success": False,
            "timeout": True,
            "returncode": returncode,
            "process_wall_ms": (time.perf_counter_ns() - start) / 1_000_000,
            **execution_evidence,
            "stdout": stdout_tail.decode(errors="replace"),
            "stderr": stderr_tail.decode(errors="replace"),
            "error": "process timed out",
        }
    except (OSError, OverflowError, ValueError) as exc:
        if process is not None:
            _stop_process(process)
        return {
            "available": False,
            "success": False,
            "timeout": False,
            **execution_evidence,
            "error": str(exc),
        }
    _close_output_pipes(process)
    return {
        "available": True,
        "success": process.returncode == 0,
        "timeout": False,
        "returncode": process.returncode,
        "process_wall_ms": (time.perf_counter_ns() - start) / 1_000_000,
        **execution_evidence,
        "stdout": bytes(stdout).decode(errors="replace"),
        "stderr": bytes(stderr).decode(errors="replace"),
    }


def run_marked_process(
    cmd: list[str],
    *,
    ready_input: str,
    target_input: str,
    final_input: str,
    ready_marker: str,
    target_marker: str,
    timeout: float,
    exit_grace: float = 5.0,
    cwd: Path,
    cpu: int | None = None,
    env: dict[str, str] | None = None,
    immutable_path_arguments: tuple[int, ...] = (),
    directory_descriptors: dict[int, int] | None = None,
    directory_argument_descriptors: dict[int, int] | None = None,
) -> dict[str, Any]:
    if (command_error := _command_error(cmd)) is not None:
        return _command_failure(cmd, command_error)
    for label, value in (("timeout", timeout), ("exit_grace", exit_grace)):
        if (duration_error := _duration_error(value, label)) is not None:
            return _command_failure(cmd, duration_error)
    protocol_error = _protocol_error(
        ready_input, target_input, final_input, ready_marker, target_marker
    )
    if protocol_error is not None:
        return _command_failure(cmd, protocol_error)
    if (environment_error := _environment_error(env)) is not None:
        return _command_failure(cmd, environment_error)
    if (argument_error := _immutable_argument_error(immutable_path_arguments, cmd)) is not None:
        return _command_failure(cmd, argument_error)
    if (
        directory_error := _directory_argument_error(
            directory_descriptors,
            directory_argument_descriptors,
            cmd,
            immutable_path_arguments,
        )
    ) is not None:
        return _command_failure(cmd, directory_error)
    try:
        pinned = _pinned_command(
            cmd,
            cpu=cpu,
            cwd=cwd,
            env=env,
            immutable_path_arguments=immutable_path_arguments,
            directory_descriptors=directory_descriptors or {},
            directory_argument_descriptors=directory_argument_descriptors or {},
        )
    except (OSError, ValueError) as exc:
        return _command_failure(cmd, str(exc))
    execution_evidence = {
        "cmd": pinned["display"],
        "executable_sha256": pinned["backend_digest"],
        "launcher_executable": pinned["launcher_path"],
        "launcher_executable_sha256": pinned["launcher_digest"],
        "immutable_inputs": pinned["immutable_inputs"],
    }

    stdout_output = bytearray()
    stderr_output = bytearray()
    process_start = time.perf_counter_ns()
    deadline = time.monotonic() + timeout
    process: subprocess.Popen[bytes] | None = None
    target_pid: int | None = None
    effective_affinity: list[int] | None = None
    open_streams: dict[int, bytearray] = {}

    def consume_readable(readable: list[int]) -> bool:
        consumed = False
        for descriptor in readable:
            output = open_streams[descriptor]
            capacity = MAX_CAPTURE_BYTES - len(output)
            try:
                chunk = os.read(
                    descriptor, min(_READ_CHUNK_BYTES, capacity + 1)
                )
            except (BlockingIOError, InterruptedError):
                continue
            consumed = True
            if not chunk:
                del open_streams[descriptor]
                continue
            if not _extend_limited(output, chunk):
                raise _OutputLimitExceeded
        return consumed

    def read_streams(wait: float) -> bool:
        if not open_streams:
            return False
        readable, _, _ = select.select(list(open_streams), [], [], wait)
        return bool(readable) and consume_readable(readable)

    def write_input(value: str) -> bool:
        if process is None or process.stdin is None:
            return False
        descriptor = process.stdin.fileno()
        data = value.encode()
        offset = 0
        while offset < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            readable, writable, _ = select.select(
                list(open_streams), [descriptor], [], remaining
            )
            if readable:
                consume_readable(readable)
            if descriptor not in writable:
                continue
            try:
                written = os.write(
                    descriptor, data[offset : offset + _READ_CHUNK_BYTES]
                )
            except (BlockingIOError, InterruptedError):
                continue
            except BrokenPipeError:
                return False
            if written <= 0:
                return False
            offset += written
        return True

    def complete_lines(*, start_offset: int = 0) -> list[bytes]:
        segment = bytes(stdout_output[start_offset:])
        return [
            line.rstrip(b"\r\n")
            for line in segment.splitlines(keepends=True)
            if line.endswith((b"\n", b"\r"))
        ]

    def has_complete_line(marker: str, *, start_offset: int = 0) -> bool:
        return marker.encode("ascii") in complete_lines(start_offset=start_offset)

    def has_marker_namespace(marker: str, *, start_offset: int = 0) -> bool:
        marker_bytes = marker.encode("ascii")
        prefix = marker_bytes + b":"
        return any(
            line == marker_bytes or line.startswith(prefix)
            for line in complete_lines(start_offset=start_offset)
        )

    def read_until(marker: str, *, start_offset: int = 0) -> bool:
        while not has_complete_line(marker, start_offset=start_offset):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not read_streams(remaining):
                return False
        return True

    def read_until_target(
        marker: str, expected: str, *, start_offset: int
    ) -> str:
        marker_bytes = marker.encode("ascii")
        expected_bytes = expected.encode("ascii")
        prefix = marker_bytes + b":"
        while True:
            for line in complete_lines(start_offset=start_offset):
                if line == expected_bytes:
                    return "matched"
                if line == marker_bytes or line.startswith(prefix):
                    return "invalid"
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not read_streams(remaining):
                return "missing"

    def drain_until_exit() -> bool:
        if process is None:
            return False
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not read_streams(remaining):
                return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return False
        return True

    def failure(message: str) -> dict[str, Any]:
        if process is None:
            returncode = None
        else:
            returncode, _, _ = _stop_process(process)
        return {
            "available": process is not None,
            "success": False,
            "timeout": time.monotonic() >= deadline,
            "returncode": returncode,
            "process_wall_ms":
                (time.perf_counter_ns() - process_start) / 1_000_000,
            **execution_evidence,
            "effective_affinity": effective_affinity,
            "stdout": stdout_output.decode(errors="replace"),
            "stderr": stderr_output.decode(errors="replace"),
            "error": message,
        }

    try:
        try:
            process, target_pid = _supervised_popen(
                pinned,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE,
                cpu=cpu,
                bufsize=0,
            )
        finally:
            for descriptor in pinned["descriptors"]:
                os.close(descriptor)
        if process.stdin is None or process.stdout is None or process.stderr is None:
            return failure("marked process pipes are unavailable")
        open_streams[process.stdout.fileno()] = stdout_output
        open_streams[process.stderr.fileno()] = stderr_output
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        if not write_input(ready_input):
            return failure("marked process timed out writing ready input")
        if not read_until(ready_marker):
            return failure("marked process did not reach ready marker")
        if has_marker_namespace(target_marker):
            return failure(
                "marked process reached target marker before target dispatch"
            )
        try:
            effective_affinity = sorted(os.sched_getaffinity(target_pid))
        except (AttributeError, OSError) as exc:
            if cpu is not None:
                return failure(
                    "could not verify marked process CPU affinity after "
                    f"readiness: {exc}"
                )
        if cpu is not None and effective_affinity != [cpu]:
            return failure(
                "marked process CPU affinity does not match the requested "
                f"CPU {cpu}: {effective_affinity}"
            )

        # Preparation and target each get the configured budget. Generate the
        # nonce only after readiness so pre-dispatch output cannot predict it.
        deadline = time.monotonic() + timeout
        try:
            target_nonce = secrets.token_hex(16)
        except Exception as exc:
            return failure(f"could not generate target nonce: {exc}")
        if not _target_nonce_is_valid(target_nonce):
            return failure("generated target nonce is invalid")
        nonce_target_input = target_input.replace(
            TARGET_NONCE_PLACEHOLDER, target_nonce
        )
        nonce_target_marker = f"{target_marker}:{target_nonce}"
        target_cpu_start = process_cpu_runtime_ns(target_pid)
        target_start = time.perf_counter_ns()
        target_output_start = len(stdout_output)
        if not write_input(nonce_target_input):
            return failure("marked process timed out writing target input")
        target_marker_state = read_until_target(
            target_marker,
            nonce_target_marker,
            start_offset=target_output_start,
        )
        if target_marker_state == "invalid":
            return failure("marked process emitted an invalid target marker")
        if target_marker_state != "matched":
            return failure("marked process did not reach target marker")
        target_wall_ms = (time.perf_counter_ns() - target_start) / 1_000_000
        target_cpu_end = process_cpu_runtime_ns(target_pid)
        target_cpu_ms = None
        if (
            target_cpu_start is not None
            and target_cpu_end is not None
            and target_cpu_end >= target_cpu_start
        ):
            target_cpu_ms = (target_cpu_end - target_cpu_start) / 1_000_000

        deadline = time.monotonic() + exit_grace
        if not write_input(final_input):
            return failure("marked process timed out writing final input")
        _close_stdin(process)
        if not drain_until_exit():
            return failure("marked process timed out after target marker")
    except _OutputLimitExceeded:
        return failure("marked process output limit exceeded")
    except (OSError, OverflowError, ValueError) as exc:
        return failure(str(exc))

    _close_output_pipes(process)
    return {
        "available": True,
        "success": process.returncode == 0,
        "timeout": False,
        "returncode": process.returncode,
        "process_wall_ms":
            (time.perf_counter_ns() - process_start) / 1_000_000,
        "target_cpu_ms": target_cpu_ms,
        "target_wall_ms": target_wall_ms,
        **execution_evidence,
        "effective_affinity": effective_affinity,
        "stdout": stdout_output.decode(errors="replace"),
        "stderr": stderr_output.decode(errors="replace"),
    }


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.replace("_", "").isalnum():
            if key in values:
                return {}
            values[key] = value.strip()
    return values


def resolve_executable(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(value)
