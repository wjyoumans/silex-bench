"""Subprocess helpers with bounded marked target timing."""

from __future__ import annotations

import fcntl
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
# Upper bound on what the harness reads from the post-handshake control
# channel. The supervisor writes at most a few bounded (128-byte) lines
# there.
_SUPERVISOR_CHANNEL_MAX_BYTES = 4096
# Once every captured stream has reached EOF, the supervisor has already begun
# exiting (it holds both pipes open until then), so only kernel exit
# bookkeeping remains before it can be reaped. This bounds that final wait
# when the observation deadline has already passed.
_SUPERVISOR_EXIT_AFTER_EOF_SECONDS = 1.0
# On a stop request, the embedded supervisor now spends its bounded cleanup
# budget at most once (see handle_stop_request/terminate_descendants in
# _SUPERVISOR_SOURCE): up to 32 rounds of a 2 ms freeze-and-confirm loop
# (a nominal ~0.07 s), then a single 0.75 s hard-kill loop, for a nominal
# sleep budget of about 0.82 s. That figure counts only the sleeps: each
# round also pays for a full /proc walk, and the loop can overshoot its
# deadline by up to one iteration, so actual worst-case latency (large
# descendant tree, or under load) runs somewhat higher. This wait must clear
# that worst case with margin so the harness does not give up, and SIGKILL
# the supervisor's process group directly, before the supervisor's own
# bounded cleanup has had a chance to finish.
_SUPERVISOR_STOP_WAIT_SECONDS = 1.5
_SUPERVISOR_FDS_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_FDS"
_SUPERVISOR_EXECUTABLE_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_EXECUTABLE"
_SUPERVISOR_CONTROL_FD_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_CONTROL_FD"
# Carries the supervisor's own housekeeping CPU set (comma-separated, may be
# empty), not the target CPU: the target is pinned separately through
# taskset. See _supervisor_housekeeping_cpus for how this set is chosen.
_SUPERVISOR_AFFINITY_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_AFFINITY"
_SUPERVISOR_ENV_KEYS = {
    _SUPERVISOR_FDS_ENV,
    _SUPERVISOR_EXECUTABLE_ENV,
    _SUPERVISOR_CONTROL_FD_ENV,
    _SUPERVISOR_AFFINITY_ENV,
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
import select
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
AFFINITY_ENV = "SILEX_BENCH_INTERNAL_SUPERVISOR_AFFINITY"
# Only used as a last-resort heartbeat when os.pidfd_open is unavailable
# (pre-5.3 kernel or pre-3.9 Python): the primary wait is event-driven via
# the SIGCHLD/SIGTERM/SIGINT/SIGHUP self-pipe below.
FALLBACK_WAIT_SECONDS = 0.2

stop_requested = False
# The control pipe stays open after the PID handshake, so the supervisor
# can still report later errors and, just before it exits,
# whether its exit status is the target's own or one it produced itself (124
# after a stop request, 126 after an internal failure). Set by main().
control_channel_fd = None


def request_stop(signum, frame):
    del signum, frame
    global stop_requested
    stop_requested = True


def ignore_child_exit(signum, frame):
    # No-op: registering a handler (instead of leaving SIGCHLD at its default
    # disposition) makes Python write to the wakeup self-pipe on child exit,
    # which is all this supervisor uses SIGCHLD for. Reaping still happens
    # explicitly via os.waitpid.
    del signum, frame


def report(control_fd, message):
    payload = (message + "\n").encode("ascii", errors="replace")[:128]
    offset = 0
    while offset < len(payload):
        offset += os.write(control_fd, payload[offset:])


def report_post_handshake(message):
    # Best effort: a lost report must never change how the supervisor exits.
    # The harness treats a missing EXIT line as an unknown origin.
    if control_channel_fd is None:
        return
    try:
        report(control_channel_fd, message)
    except BaseException:
        pass


def report_exit(origin, returncode):
    report_post_handshake(f"EXIT {origin} {returncode}")


def has_children_support():
    # /proc/<pid>/task/<pid>/children only exists with CONFIG_PROC_CHILDREN.
    # Checked once at startup so the supervisor fails closed instead of
    # silently treating every process as childless for its whole lifetime.
    pid = os.getpid()
    path = f"/proc/{pid}/task/{pid}/children"
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except (FileNotFoundError, ProcessLookupError):
        return False
    os.close(descriptor)
    return True


def direct_children(pid):
    # Reads every thread's children file, not only the thread-group leader's
    # (task/<pid>/children): a child forked by a non-leader thread is
    # otherwise invisible until it is reparented.
    task_root = f"/proc/{pid}/task"
    try:
        thread_ids = os.listdir(task_root)
    except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
        return []
    result = []
    for thread_id in thread_ids:
        path = f"/proc/{pid}/task/{thread_id}/children"
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            payload = os.read(descriptor, MAX_CHILDREN_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(payload) > MAX_CHILDREN_BYTES:
            raise RuntimeError("process descendant list exceeds its byte limit")
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


def reap_children():
    while True:
        try:
            waited, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if waited <= 0:
            return


def process_state(pid):
    # State letter from /proc/<pid>/stat, or None when the process is gone.
    # The command name is parenthesised and may contain ")" or spaces, so the
    # state is the first field after the last ")".
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            payload = handle.read(4096)
    except (FileNotFoundError, ProcessLookupError):
        return None
    tail = payload.rpartition(b")")[2].split()
    return tail[0].decode("ascii", "replace") if tail else None


def live_descendants():
    # Descendants that are not exited zombies. Only the outlived check uses
    # this; descendants() still returns every descendant for cleanup/kill.
    return {pid for pid in descendants() if process_state(pid) not in (None, "Z")}


def outlived_descendants():
    # Reap, then scan. A descendant that exits between the reap and the scan
    # stays a zombie child of this supervisor and would be counted as alive,
    # so a non-empty scan is followed by a second reap and scan. A descendant
    # that exits after the second reap is still a zombie at the second scan;
    # live_descendants() skips those (state Z), so only live processes count.
    reap_children()
    found = live_descendants()
    if found:
        reap_children()
        found = live_descendants()
    return found


def signal_all(pids, signum):
    for pid in pids:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def kill_target_directly(pid, pidfd=None):
    # Fallback for the stop path, independent of /proc descendant
    # enumeration: SIGKILL the known target PID directly, unconditionally, so
    # the target itself cannot outlive a gap in descendant enumeration.
    # Also SIGKILL the target's own process group, but only when it differs
    # from the supervisor's own group: absent a setsid call, the target
    # shares the supervisor's group, and killing that group here would kill
    # the supervisor itself before terminate_descendants() has a chance to
    # walk /proc and reach descendants that escaped into their own session
    # (which do not share the target's group either). terminate_descendants()
    # still runs afterwards to confirm and reap the wider tree.
    #
    # Callers must only invoke this while the target is confirmed unreaped
    # (Popen.returncode is still None); see handle_stop_request. That makes
    # this pid race-free to read and signal here, since this process is the
    # target's only reaper and does so solely through the explicit
    # waitpid() calls elsewhere in this file, none of which run before this
    # function returns. Reading the pgid up front, before sending any
    # signal, keeps that guarantee even across the pidfd/kill calls below.
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = None
    signaled_by_pidfd = False
    if pidfd is not None:
        # signal.pidfd_send_signal targets the specific process the pidfd
        # was opened for, not a pid number, so it cannot hit a
        # process that has reused a recycled pid. Prefer it whenever a
        # pidfd is available (Linux 5.1+, Python 3.9+); os.kill(pid, ...)
        # below is the fallback for older kernels/interpreters.
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            signaled_by_pidfd = True
        except (AttributeError, ProcessLookupError, OSError):
            signaled_by_pidfd = False
    if not signaled_by_pidfd:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if pgid is not None and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def handle_stop_request(target, target_pidfd):
    # Only signal the target while it is confirmed unreaped
    # (Popen.poll()/Popen.wait() has not yet set target.returncode).  Once
    # the target has been reaped, its pid is no longer held by this
    # process's own child and the kernel is free to recycle it; signalling
    # it (directly or via its process group) at that point could hit an
    # unrelated same-UID process instead. Skip the direct
    # kill in that case; terminate_descendants() below still runs to finish
    # any orphan cleanup.
    if target.returncode is None:
        kill_target_directly(target.pid, target_pidfd)
    try:
        terminate_descendants()
    except BaseException as exc:
        # terminate_descendants() already spent its full bounded budget once;
        # do not retry it here (the top-level handler below would otherwise
        # spend that budget a second time).
        report_post_handshake(f"ERROR cleanup after stop request failed: {exc}")
        report_exit("supervisor", 126)
        os._exit(126)
    report_exit("supervisor", 124)
    os._exit(124)


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
    # The reported value is the status the harness will observe: the target's
    # own exit code, or the negated signal number the supervisor re-raises on
    # itself below.
    report_exit("target", returncode if returncode < 0 else min(returncode, 255))
    if returncode < 0:
        signum = -returncode
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        time.sleep(0.1)
        os._exit(128 + signum)
    os._exit(min(returncode, 255))


def main():
    global control_channel_fd
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)
    # SIGCHLD needs a registered handler (not the default disposition) for
    # Python to write to the wakeup self-pipe when a child exits.
    signal.signal(signal.SIGCHLD, ignore_child_exit)
    wakeup_read, wakeup_write = os.pipe()
    os.set_blocking(wakeup_read, False)
    os.set_blocking(wakeup_write, False)
    # The pipe only needs to be readable to wake the select() below; it does
    # not need to hold every queued signal byte. Without
    # warn_on_full_buffer=False, more than 64 KiB of queued signals (for
    # example, a target that forks many short-lived children) would make
    # Python write a warning to this process's stderr, which is the captured
    # backend stderr.
    signal.set_wakeup_fd(wakeup_write, warn_on_full_buffer=False)

    control_fd = int(os.environ[CONTROL_FD_ENV])
    if not has_children_support():
        report(
            control_fd,
            "ERROR /proc children enumeration unavailable (CONFIG_PROC_CHILDREN)",
        )
        os._exit(126)
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
    # This pins the supervisor itself, away from the target CPU (chosen by
    # _supervisor_housekeeping_cpus in the parent); the target is pinned to
    # the requested CPU separately, through taskset. The value is empty only
    # when no CPU pinning was requested at all (cpu is None): in that case
    # there is no target CPU to keep the supervisor off of, so it keeps its
    # inherited affinity. When a CPU was requested, _supervisor_housekeeping_
    # cpus always returns a non-empty set (its fallback tiers include the
    # target CPU itself as a last resort), so this value is never empty in
    # that case.
    affinity_text = os.environ.get(AFFINITY_ENV, "")
    housekeeping_cpus = {
        int(value) for value in affinity_text.split(",") if value
    }
    if housekeeping_cpus:
        os.sched_setaffinity(0, housekeeping_cpus)

    target_env = dict(os.environ)
    for key in (FDS_ENV, EXECUTABLE_ENV, CONTROL_FD_ENV, AFFINITY_ENV):
        target_env.pop(key, None)
    if not sys.argv[1:]:
        report(control_fd, "ERROR supervisor target command is empty")
        os._exit(126)
    # Start of the whole-process envelope: taken immediately before the spawn so
    # the supervisor's own startup, handshake and taskset exec are excluded.
    spawn_ns = time.monotonic_ns()
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
    control_channel_fd = control_fd

    try:
        target_pidfd = os.pidfd_open(target.pid, 0)
    except (AttributeError, OSError):
        target_pidfd = None

    def wait_for_event():
        # Blocks until the target exits (pidfd readable), a signal we
        # registered fires (self-pipe readable via set_wakeup_fd), or, only
        # when pidfd_open is unavailable, the bounded fallback elapses. This
        # replaces the fixed-rate poll that previously ran for the target's
        # entire lifetime, including the timed interval.
        wait_fds = [wakeup_read]
        if target_pidfd is not None:
            wait_fds.append(target_pidfd)
            timeout = None
        else:
            timeout = FALLBACK_WAIT_SECONDS
        select.select(wait_fds, [], [], timeout)
        try:
            while os.read(wakeup_read, 4096):
                pass
        except BlockingIOError:
            pass

    while True:
        if stop_requested:
            handle_stop_request(target, target_pidfd)
        returncode = target.poll()
        if returncode is not None:
            # End of the envelope: the target is reaped. CLOCK_MONOTONIC
            # spawn-to-reap, reported over the control channel.
            reaped_ns = time.monotonic_ns()
            report_post_handshake(f"WALL {reaped_ns - spawn_ns}")
            # Descendants still alive now outlived the target. They are killed
            # at once (no polling drain) and the harness fails the sample.
            if outlived_descendants():
                report_post_handshake("OUTLIVED")
                terminate_descendants()
            finish(returncode)
        wait_for_event()


if __name__ == "__main__":
    # Guarded so tests can exec() this source into a fresh namespace (with a
    # different __name__) and call its functions directly, without also
    # running the live supervisor. Run via "python -c" as intended, __name__
    # is "__main__" and behavior is unchanged.
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
                if control_channel_fd is not None:
                    report(control, "EXIT supervisor 126")
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
    """Whole-thread-group user+system CPU time of ``pid`` in nanoseconds.

    Reads utime+stime from ``/proc/<pid>/stat``, which accumulates every
    thread of the process, including threads that have already exited. The
    resolution is one clock tick. Diagnostic only; backends report their own
    CPU value.
    """
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        # The command name (field 2) may contain spaces and parentheses, so
        # split after its closing parenthesis; utime and stime are fields
        # 14 and 15, i.e. indexes 11 and 12 of the remainder (field 3 first).
        fields = text[text.rindex(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        return ticks * 1_000_000_000 // os.sysconf("SC_CLK_TCK")
    except (IndexError, OSError, ValueError):
        return None


def _thread_group_affinities(pid: int) -> dict[int, list[int]]:
    """Return the CPU affinity of every thread of ``pid``, keyed by TID.

    A thread that exits between listing and reading is skipped. A thread
    that has just exited but is still listed in ``/proc/<pid>/task`` is read
    and counted, so the check is best-effort for drifted-then-exited threads:
    every thread inherits the taskset affinity, so only a thread that changed
    its own affinity can trip it. The leader (TID == ``pid``) must be present
    or the read fails.
    """
    affinities: dict[int, list[int]] = {}
    for entry in os.listdir(f"/proc/{pid}/task"):
        try:
            tid = int(entry)
        except ValueError:
            continue
        try:
            affinities[tid] = sorted(os.sched_getaffinity(tid))
        except ProcessLookupError:
            continue
    if pid not in affinities:
        raise OSError(f"could not read the affinity of process {pid}")
    return affinities


def _check_thread_affinities(
    pid: int, cpu: int | None
) -> tuple[list[int], str | None]:
    """Return the leader affinity and a mismatch message for ``cpu``."""
    affinities = _thread_group_affinities(pid)
    leader = affinities[pid]
    if cpu is not None:
        for tid, affinity in sorted(affinities.items()):
            if affinity != [cpu]:
                return leader, f"thread {tid} has affinity {affinity}"
    return leader, None


def _current_available_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise OSError(
            "CPU affinity requested but the current process affinity "
            f"is unavailable: {exc}"
        ) from exc


def _with_cpu(cmd: list[str], cpu: int | None) -> list[str]:
    if cpu is None:
        return cmd
    available_cpus = _current_available_cpus()
    if cpu not in available_cpus:
        raise OSError(
            f"requested CPU {cpu} is outside the current process affinity "
            f"{available_cpus}"
        )
    taskset = trusted_system_executable("taskset")
    return [taskset, "-c", str(cpu), *cmd]


def _read_thread_siblings(cpu: int) -> set[int] | None:
    """Read a CPU's SMT sibling set from sysfs topology.

    Returns None when the topology file is absent or unreadable (for
    example, a non-Linux-standard sysfs layout, a restricted mount, or a
    virtualized CPU topology that does not expose it), so callers can fall
    back to treating the CPU as having no known siblings.
    """
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        text = path.read_text()
    except OSError:
        return None
    siblings: set[int] = set()
    for token in text.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                continue
            siblings.update(range(start, end + 1))
        else:
            try:
                siblings.add(int(token))
            except ValueError:
                continue
    return siblings or None


def _supervisor_housekeeping_cpus(
    cpu: int, available_cpus: list[int]
) -> tuple[int, ...]:
    """Choose CPUs for the supervisor process that avoid the measured CPU.

    The supervisor's own scheduling activity (its blocking wait, cleanup,
    and control-channel I/O) must not compete with the pinned target for the
    target CPU's execution resources, including its SMT sibling(s): sharing
    a physical core with an SMT sibling still contends for its execution
    units and caches. `cpu` is the target CPU; `available_cpus` is the set
    the launching process may itself be pinned to (for example, under an
    external cpuset).

    Falls back in stages when the available set is too small to exclude the
    full sibling set:
      1. available_cpus minus (cpu and its SMT siblings), the normal case;
      2. available_cpus minus cpu alone, if (1) is empty: this still shares
         SMT execution resources with the target (a 2-CPU SMT pair is the
         common case here), but keeps the supervisor off the exact target
         logical CPU;
      3. available_cpus unchanged, only when cpu is the sole available CPU:
         the supervisor necessarily shares it with the target, as it did
         before this housekeeping split existed.
    Returns a sorted tuple; an empty result never occurs because tier 3
    always returns at least {cpu}.
    """
    available = set(available_cpus)
    siblings = _read_thread_siblings(cpu)
    excluded = {cpu} | (siblings or set())
    tier1 = available - excluded
    if tier1:
        return tuple(sorted(tier1))
    tier2 = available - {cpu}
    if tier2:
        return tuple(sorted(tier2))
    return tuple(sorted(available))


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
    supervisor_cpus: tuple[int, ...] | None = None
    if cpu is not None:
        supervisor_cpus = _supervisor_housekeeping_cpus(
            cpu, _current_available_cpus()
        )
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
            "supervisor_cpus": supervisor_cpus,
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


def _split_complete_lines(segment: bytes, *, eof: bool) -> tuple[list[bytes], bytes]:
    """Split `segment` into confirmed-complete lines and a held-back tail.

    A line counts as complete once it ends in `\\n` or `\\r` (matching the
    whole-buffer `splitlines()` semantics this incremental scan replaces),
    except for one case: a segment that ends in a bare `\\r` (not
    `\\r\\n`) is ambiguous when more input can still arrive, since the very
    next byte read separately may be the `\\n` that completes a `\\r\\n`
    terminator. That trailing bare `\\r` is therefore held back in the
    returned tail, exactly like a terminator-less tail, until either `eof` is
    set (no further byte can ever arrive, so it is resolved as its own bare
    `\\r` line) or a caller re-invokes this with more bytes appended ahead of
    it. Committing it immediately and scanning the next call's segment from
    just past it, as an earlier incremental rewrite did, mistook a `\\n`
    arriving in a later read for its own empty line.
    """
    pieces = segment.splitlines(keepends=True)
    incomplete = b""
    if pieces:
        last = pieces[-1]
        if not last.endswith(b"\n") and not (last.endswith(b"\r") and eof):
            incomplete = pieces.pop()
    complete = [line.rstrip(b"\r\n") for line in pieces]
    return complete, incomplete


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

    def read_ready(readable: list[int]) -> bool:
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
                return False
        return True

    deadline = time.monotonic() + timeout
    while readers or stdin_fd is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable, writable, _ = select.select(
            list(readers),
            [stdin_fd] if stdin_fd is not None else [],
            [],
            remaining,
        )
        if not readable and not writable:
            break
        if stdin_fd is not None and stdin_fd in writable:
            try:
                written = os.write(stdin_fd, pending_input[input_offset:])
            except BrokenPipeError:
                written = len(pending_input) - input_offset
            input_offset += written
            if input_offset >= len(pending_input):
                _close_stdin(process)
                stdin_fd = None
        if not read_ready(readable):
            return stdout, stderr, "process output limit exceeded"

    if readers:
        # The deadline passed with output still open. Timeout classification
        # uses what has already happened, not when this harness got to look:
        # drain what is readable without waiting, and an EOF on every stream
        # that was already there counts as an exit.
        while readers:
            readable, _, _ = select.select(list(readers), [], [], 0)
            if not readable:
                return stdout, stderr, "process timed out"
            if not read_ready(readable):
                return stdout, stderr, "process output limit exceeded"
    if stdin_fd is not None:
        # Every output stream is at EOF, so the supervisor is exiting and
        # can no longer consume input.
        _close_stdin(process)
    # Every stream is at EOF, and the supervisor holds both pipes until it
    # exits, so it is already exiting; a deadline that passed meanwhile does
    # not make that exit a timeout.
    remaining = deadline - time.monotonic()
    try:
        process.wait(timeout=max(remaining, _SUPERVISOR_EXIT_AFTER_EOF_SECONDS))
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
        process.wait(timeout=_SUPERVISOR_STOP_WAIT_SECONDS)
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


class _ObservationDeadlineExpired(TimeoutError):
    """An observation deadline expired before supervisor cleanup began."""


class _SupervisorChannel:
    """The supervisor's control pipe after the PID handshake.

    The supervisor reports post-handshake errors as ``ERROR <text>`` lines and,
    just before exiting, one ``EXIT <origin> <status>`` line naming whether its
    exit status is the target's own (``target``) or one it produced itself
    (``supervisor``: 124 after a stop request, 126 after an internal failure).
    Read only after the supervisor has exited, when the pipe is at EOF.
    """

    def __init__(self, descriptor: int, pending: bytes) -> None:
        self._descriptor = descriptor
        self._buffer = bytearray(pending)
        self._report: tuple[str | None, int | None, str | None] | None = None
        self.envelope_ns: int | None = None
        self.descendants_outlived = False

    def _read_available(self) -> None:
        if self._descriptor < 0:
            return
        while len(self._buffer) <= _SUPERVISOR_CHANNEL_MAX_BYTES:
            try:
                chunk = os.read(
                    self._descriptor,
                    _SUPERVISOR_CHANNEL_MAX_BYTES + 1 - len(self._buffer),
                )
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            if not chunk:
                return
            self._buffer.extend(chunk)

    def exit_report(self) -> tuple[str | None, int | None, str | None]:
        """Return ``(origin, status, supervisor_error)`` from the channel."""
        if self._report is not None:
            return self._report
        self._read_available()
        origin: str | None = None
        status: int | None = None
        errors: list[str] = []
        # Only complete lines are trusted; a truncated tail is ignored.
        for raw_line in bytes(self._buffer).split(b"\n")[:-1]:
            line = raw_line.decode("ascii", errors="replace")
            if line.startswith("ERROR "):
                errors.append(line[len("ERROR "):])
                continue
            fields = line.split()
            if len(fields) == 2 and fields[0] == "WALL" and fields[1].isdigit():
                self.envelope_ns = int(fields[1])
                continue
            if fields == ["OUTLIVED"]:
                self.descendants_outlived = True
                continue
            if (
                len(fields) == 3
                and fields[0] == "EXIT"
                and fields[1] in {"target", "supervisor"}
            ):
                try:
                    status = int(fields[2])
                except ValueError:
                    continue
                origin = fields[1]
        self._report = (origin, status, "; ".join(errors) or None)
        return self._report

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1


def _with_failure_origin(
    result: dict[str, Any], channel: _SupervisorChannel | None
) -> dict[str, Any]:
    """Record whether a failed run's exit status came from the target.

    ``failure_origin`` is ``"target"`` when the status is the target's own,
    ``"supervisor"`` when the supervisor produced it (124 after a stop
    request, 126 after an internal failure), and ``None`` for a successful
    run, a run without an exit status, or when the supervisor did not report
    the status it exited with. Closes ``channel``.
    """
    origin, status, supervisor_error = (None, None, None)
    if channel is not None:
        origin, status, supervisor_error = channel.exit_report()
        channel.close()
    if channel is not None:
        if channel.envelope_ns is not None and "process_wall_ms" in result:
            # Supervisor-measured spawn-to-reap CLOCK_MONOTONIC envelope; it
            # excludes harness and supervisor startup overhead.
            result["process_wall_ms"] = channel.envelope_ns / 1_000_000
        if channel.descendants_outlived:
            result["success"] = False
            result["descendants_outlived_target"] = True
            existing = result.get("error")
            result["error"] = (
                f"{existing}; descendants_outlived_target"
                if existing
                else "descendants_outlived_target"
            )
    returncode = result.get("returncode")
    if result.get("success") is True or returncode is None or status != returncode:
        origin = None
    result["failure_origin"] = origin
    if origin == "supervisor" and supervisor_error:
        error = result.get("error")
        result["error"] = (
            f"process supervisor failed: {supervisor_error}"
            if not error
            else f"{error} (process supervisor: {supervisor_error})"
        )
    return result


def _supervised_popen(
    pinned: dict[str, Any],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    stdin: Any,
    bufsize: int = -1,
    deadline: float | None = None,
) -> tuple[subprocess.Popen[bytes], int, _SupervisorChannel]:
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
        supervisor_cpus = pinned.get("supervisor_cpus")
        supervisor_environment[_SUPERVISOR_AFFINITY_ENV] = (
            ""
            if not supervisor_cpus
            else ",".join(str(value) for value in supervisor_cpus)
        )
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
        handshake_deadline = time.monotonic() + _SUPERVISOR_HANDSHAKE_SECONDS
        if deadline is not None:
            handshake_deadline = min(handshake_deadline, deadline)

        def handshake_timeout() -> OSError:
            # The handshake has its own launch limit. Only expiration of the
            # observation cutoff is a benchmark timeout; cleanup cannot change
            # either classification after this exception has been created.
            error_type = (
                _ObservationDeadlineExpired
                if deadline is not None and handshake_deadline == deadline
                else OSError
            )
            return error_type("process supervisor handshake timed out")

        while b"\n" not in response:
            remaining = handshake_deadline - time.monotonic()
            if remaining <= 0:
                raise handshake_timeout()
            readable, _, _ = select.select([control_read], [], [], remaining)
            if control_read not in readable:
                raise handshake_timeout()
            chunk = os.read(
                control_read,
                _SUPERVISOR_CONTROL_BYTES + 1 - len(response),
            )
            if not chunk:
                break
            response.extend(chunk)
            # Only the handshake line itself is bounded here; bytes after its
            # newline already belong to the post-handshake channel.
            if b"\n" not in response and len(response) > _SUPERVISOR_CONTROL_BYTES:
                raise OSError("process supervisor response exceeds its size limit")
        line, _, pending = bytes(response).partition(b"\n")
        fields = line.decode("ascii", errors="replace").split()
        if len(fields) != 2 or fields[0] != "PID" or not fields[1].isdigit():
            detail = line.decode("ascii", errors="replace") or "no response"
            raise OSError(f"process supervisor could not start target: {detail}")
        target_pid = int(fields[1])
        if target_pid <= 0:
            raise OSError("process supervisor returned an invalid target PID")
        channel = _SupervisorChannel(control_read, pending)
        control_read = -1
        return process, target_pid, channel
    except BaseException:
        if process is not None:
            _stop_process(process)
        raise
    finally:
        if control_read >= 0:
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
    deadline = time.monotonic() + timeout
    process: subprocess.Popen[bytes] | None = None
    channel: _SupervisorChannel | None = None
    try:
        try:
            process, _, channel = _supervised_popen(
                pinned,
                cwd=cwd,
                env=env,
                # DEVNULL, not None, when there is no caller-supplied input:
                # None would let the target inherit this harness's own stdin,
                # so it could consume a user's terminal or piped input.
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                deadline=deadline,
            )
        finally:
            for descriptor in pinned["descriptors"]:
                os.close(descriptor)
        # A deadline that already passed still gets one observation of the
        # supervised process before it is classified.
        remaining = max(deadline - time.monotonic(), 0.0)
        stdout, stderr, capture_error = _bounded_communicate(
            process,
            input_bytes,
            remaining,
        )
        if capture_error is not None:
            # Capture the elapsed wall time at classification, before
            # _stop_process spends its own cleanup budget (up to about 2 s):
            # otherwise a timeout row would report more than the deadline.
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            returncode, stdout_tail, stderr_tail = _stop_process(process)
            _extend_limited(stdout, stdout_tail)
            _extend_limited(stderr, stderr_tail)
            return _with_failure_origin({
                "available": True,
                "success": False,
                "timeout": capture_error == "process timed out",
                "returncode": returncode,
                "process_wall_ms": elapsed_ms,
                **execution_evidence,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "error": capture_error,
            }, channel)
    except (subprocess.TimeoutExpired, _ObservationDeadlineExpired) as exc:
        # Same rationale as above: classify the wall time before cleanup.
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        if process is None:
            returncode, stdout_tail, stderr_tail = None, b"", b""
        else:
            returncode, stdout_tail, stderr_tail = _stop_process(process)
        return _with_failure_origin({
            "available": process is not None,
            "success": False,
            "timeout": True,
            "returncode": returncode,
            "process_wall_ms": elapsed_ms,
            **execution_evidence,
            "stdout": stdout_tail.decode(errors="replace"),
            "stderr": stderr_tail.decode(errors="replace"),
            "error": str(exc) if isinstance(exc, _ObservationDeadlineExpired) else "process timed out",
        }, channel)
    except (OSError, OverflowError, ValueError) as exc:
        if process is not None:
            _stop_process(process)
        return _with_failure_origin({
            "available": process is not None,
            "success": False,
            "timeout": False,
            **execution_evidence,
            "error": str(exc),
        }, channel)
    except BaseException:
        # The supervisor is started in a separate session, so a Ctrl-C sent to
        # the harness does not reach it.  Reap the full supervised process tree
        # before allowing the campaign to checkpoint its interrupted state.
        if process is not None:
            _stop_process(process)
        if channel is not None:
            channel.close()
        raise
    _close_output_pipes(process)
    return _with_failure_origin({
        "available": True,
        "success": process.returncode == 0,
        "timeout": False,
        "returncode": process.returncode,
        "process_wall_ms": (time.perf_counter_ns() - start) / 1_000_000,
        **execution_evidence,
        "stdout": bytes(stdout).decode(errors="replace"),
        "stderr": bytes(stderr).decode(errors="replace"),
    }, channel)


def run_marked_process(
    cmd: list[str],
    *,
    ready_input: str,
    target_input: str,
    final_input: str,
    ready_marker: str,
    target_marker: str,
    timeout: float,
    cwd: Path,
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
    # complete_lines() incremental scan state: rather than re-copying and
    # re-splitting the whole segment since start_offset on
    # every call (quadratic in the number of reads for a chatty target),
    # cache the lines already confirmed complete and only re-scan the
    # unterminated tail since the last call, appending any newly-arrived
    # bytes to it. Reset whenever start_offset itself changes (a new phase).
    line_scan_offset = 0
    line_scan_tail_start = 0
    line_scan_complete: list[bytes] = []
    # Set once a 0-byte read shows stdout itself is closed, so
    # complete_lines() knows a held-back trailing bare "\r" can never still
    # be joined by a later "\n" and may be resolved as its own line. See
    # _split_complete_lines().
    stdout_closed = False
    process_start = time.perf_counter_ns()
    deadline = time.monotonic() + timeout
    process: subprocess.Popen[bytes] | None = None
    channel: _SupervisorChannel | None = None
    target_pid: int | None = None
    effective_affinity: list[int] | None = None
    # Re-read after the target marker: the readiness check below only shows
    # the affinity a backend or runtime started with, and a
    # backend or its runtime (OpenMP, Julia) could still change it during the
    # measured interval. When a CPU was requested, a mismatch here fails the
    # sample exactly as a readiness mismatch does.
    effective_affinity_after_target: list[int] | None = None
    open_streams: dict[int, bytearray] = {}

    def consume_readable(readable: list[int]) -> bool:
        nonlocal stdout_closed
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
                if output is stdout_output:
                    stdout_closed = True
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

    def write_input(value: str) -> str:
        # Returns "ok", "timeout", or "broken_pipe": distinct outcomes, so a
        # closed pipe is reported for what it is rather than folded into the
        # generic "timed out" reason.
        if process is None or process.stdin is None:
            return "timeout"
        descriptor = process.stdin.fileno()
        data = value.encode()
        offset = 0
        while offset < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
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
                return "broken_pipe"
            if written <= 0:
                return "broken_pipe"
            offset += written
        return "ok"

    def write_input_or_fail(value: str, phase: str) -> dict[str, Any] | None:
        status = write_input(value)
        if status == "ok":
            return None
        if status == "broken_pipe":
            return failure(
                f"marked process input pipe closed while writing {phase} input",
                timed_out=False,
            )
        return failure(f"marked process timed out writing {phase} input")

    def complete_lines(*, start_offset: int = 0) -> list[bytes]:
        nonlocal line_scan_offset, line_scan_tail_start, line_scan_complete
        if start_offset != line_scan_offset:
            line_scan_offset = start_offset
            line_scan_tail_start = start_offset
            line_scan_complete = []
        if line_scan_tail_start < len(stdout_output):
            segment = bytes(stdout_output[line_scan_tail_start:])
            newly_complete, incomplete = _split_complete_lines(
                segment, eof=stdout_closed
            )
            line_scan_complete.extend(newly_complete)
            line_scan_tail_start = len(stdout_output) - len(incomplete)
        # Callers only read this list (membership test or iteration); they
        # must not mutate it, since it is the live cache, not a fresh copy.
        return line_scan_complete

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
                # The exact nonce marker preceded by other bytes on its line:
                # an unterminated line (such as a leftover prompt) ran into
                # the marker. A protocol error, not a missing marker. The
                # nonce is unpredictable, so ordinary output cannot end with
                # it by accident.
                if line.endswith(expected_bytes):
                    return "unterminated"
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not read_streams(remaining):
                return "missing"

    def observed_exit() -> bool:
        # Timeout classification uses what has already happened, not when
        # this harness got to look. Drain whatever is readable without
        # waiting: EOF on every stream shows the supervisor
        # is already exiting, since it holds both pipes until it exits.
        if process is None:
            return False
        try:
            while open_streams and read_streams(0):
                pass
        except _OutputLimitExceeded:
            pass
        return not open_streams or process.poll() is not None

    def drain_until_exit() -> bool:
        if process is None:
            return False
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not read_streams(remaining):
                break
        if open_streams and not observed_exit():
            return False
        remaining = deadline - time.monotonic()
        try:
            process.wait(
                timeout=max(remaining, _SUPERVISOR_EXIT_AFTER_EOF_SECONDS)
            )
        except subprocess.TimeoutExpired:
            return False
        return True

    def failure(message: str, *, timed_out: bool | None = None) -> dict[str, Any]:
        # Classify at the failure boundary, before process-tree cleanup adds
        # elapsed time (both the timeout/success classification and the
        # reported wall time: cleanup can take up to about 2 s, and a
        # timeout row must not report more than the deadline. Launch
        # exceptions carry their classification separately. A run whose exit
        # or EOF was already observable when the deadline was noticed exited
        # rather than timed out.
        if timed_out is None:
            timed_out = time.monotonic() >= deadline and not observed_exit()
        elapsed_ms = (time.perf_counter_ns() - process_start) / 1_000_000
        if process is None:
            returncode = None
        else:
            returncode, _, _ = _stop_process(process)
        return _with_failure_origin({
            "available": process is not None,
            "success": False,
            "timeout": timed_out,
            "returncode": returncode,
            "process_wall_ms": elapsed_ms,
            **execution_evidence,
            "effective_affinity": effective_affinity,
            "effective_affinity_after_target": effective_affinity_after_target,
            "stdout": stdout_output.decode(errors="replace"),
            "stderr": stderr_output.decode(errors="replace"),
            "error": message,
        }, channel)

    try:
        try:
            process, target_pid, channel = _supervised_popen(
                pinned,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE,
                bufsize=0,
                deadline=deadline,
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
        if (write_failure := write_input_or_fail(ready_input, "ready")) is not None:
            return write_failure
        if not read_until(ready_marker):
            return failure("marked process did not reach ready marker")
        if has_marker_namespace(target_marker):
            return failure(
                "marked process reached target marker before target dispatch"
            )
        affinity_mismatch: str | None = None
        try:
            effective_affinity, affinity_mismatch = _check_thread_affinities(
                target_pid, cpu
            )
        except (AttributeError, OSError) as exc:
            if cpu is not None:
                return failure(
                    "could not verify marked process CPU affinity after "
                    f"readiness: {exc}"
                )
        if cpu is not None and affinity_mismatch is not None:
            return failure(
                "marked process CPU affinity does not match the requested "
                f"CPU {cpu}: {effective_affinity} ({affinity_mismatch})"
            )

        # One non-resetting deadline covers process launch, preparation, the
        # measured target, result extraction, and orderly shutdown.  This is
        # the observation cutoff recorded by the campaign; resetting it at a
        # phase boundary could let one observation consume several cutoffs.
        # Generate the nonce only after readiness so pre-dispatch output
        # cannot predict it.
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
        # Scan from the start of the line in progress at dispatch, not from
        # the dispatch offset itself, so a leftover unterminated pre-dispatch
        # line (for example a prompt) is seen joined to whatever completes
        # it whether it was read before or after dispatch.
        pre_dispatch = bytes(stdout_output)
        target_line_start = (
            max(pre_dispatch.rfind(b"\n"), pre_dispatch.rfind(b"\r")) + 1
        )
        if (
            write_failure := write_input_or_fail(nonce_target_input, "target")
        ) is not None:
            return write_failure
        target_marker_state = read_until_target(
            target_marker,
            nonce_target_marker,
            start_offset=target_line_start,
        )
        if target_marker_state == "invalid":
            return failure("marked process emitted an invalid target marker")
        if target_marker_state == "unterminated":
            return failure(
                "marked process emitted the target marker after unterminated "
                "output on the same line",
                timed_out=False,
            )
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

        if cpu is not None:
            # Unlike the readiness-time effective_affinity read above (always
            # attempted), this re-read is skipped entirely when there is no
            # requested cpu to compare it against. With a requested cpu it
            # has the readiness check's severity: an unverifiable or changed
            # affinity fails the sample.
            try:
                (
                    effective_affinity_after_target,
                    affinity_mismatch,
                ) = _check_thread_affinities(target_pid, cpu)
            except (AttributeError, OSError) as exc:
                return failure(
                    "could not verify marked process CPU affinity after "
                    f"the target marker: {exc}"
                )
            if affinity_mismatch is not None:
                return failure(
                    "marked process CPU affinity after the target marker does "
                    f"not match the requested CPU {cpu}: "
                    f"{effective_affinity_after_target} ({affinity_mismatch})"
                )

        if (write_failure := write_input_or_fail(final_input, "final")) is not None:
            return write_failure
        _close_stdin(process)
        if not drain_until_exit():
            return failure("marked process timed out after target marker")
    except _OutputLimitExceeded:
        return failure("marked process output limit exceeded", timed_out=False)
    except _ObservationDeadlineExpired as exc:
        return failure(str(exc), timed_out=True)
    except (OSError, OverflowError, ValueError) as exc:
        return failure(str(exc), timed_out=False)
    except BaseException:
        # In particular, Ctrl-C must not leave the supervised backend or any
        # descendant running after the campaign records an interruption.
        if process is not None:
            _stop_process(process)
        if channel is not None:
            channel.close()
        raise

    _close_output_pipes(process)
    return _with_failure_origin({
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
        "effective_affinity_after_target": effective_affinity_after_target,
        "stdout": stdout_output.decode(errors="replace"),
        "stderr": stderr_output.decode(errors="replace"),
    }, channel)


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
