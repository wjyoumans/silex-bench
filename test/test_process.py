from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from silex_bench import process as process_module
from silex_bench.process import (
    MAX_PROTOCOL_INPUT_BYTES,
    TARGET_NONCE_PLACEHOLDER,
    parse_key_values,
    run_marked_process,
    run_process,
)


def _parse_cpu_list(text: str) -> set[int]:
    """Parse a Linux cpulist (e.g. "1-3,7") as read from /proc or sysfs."""
    values: set[int] = set()
    for token in text.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(token))
    return values


def _load_supervisor_namespace() -> dict[str, Any]:
    """Exec the embedded supervisor source into a fresh namespace.

    ``__name__`` is deliberately not ``"__main__"``, so the module-level
    ``main()`` call at the bottom of the source does not run; the namespace's
    functions (``direct_children``, ``handle_stop_request``, and so on) can
    then be called and monkeypatched directly, without spawning a real
    supervisor subprocess. This exec() alone never touches this test
    process's own signal state, since it only defines functions. A test
    that goes on to call ``namespace["main"]()`` itself must still mock
    ``signal.signal``/``signal.set_wakeup_fd``/``os.pipe`` (see
    ``test_missing_own_children_file_fails_closed_at_startup``), since
    ``main()`` installs real handlers and a real wakeup fd otherwise.
    """
    namespace: dict[str, Any] = {"__name__": "test_supervisor_source"}
    exec(compile(process_module._SUPERVISOR_SOURCE, "<supervisor>", "exec"), namespace)
    return namespace


class ProcessTests(unittest.TestCase):
    def test_process_error_classification_precedes_cleanup(self) -> None:
        now = [0.0]
        supervisor = mock.Mock()

        def cleanup(process):
            now[0] = 11.0
            return -15, b"", b""

        with mock.patch("silex_bench.process.time.monotonic", side_effect=lambda: now[0]), mock.patch(
            "silex_bench.process._supervised_popen", return_value=(supervisor, 12345, None)
        ), mock.patch(
            "silex_bench.process._bounded_communicate", side_effect=OSError("capture failed")
        ), mock.patch("silex_bench.process._stop_process", side_effect=cleanup):
            result = run_process([sys.executable, "-c", "pass"], timeout=10, cwd=Path.cwd())

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["error"], "capture failed")

    def test_marked_protocol_error_classification_precedes_cleanup(self) -> None:
        now = [0.0]
        supervisor = mock.Mock(stdin=None)

        def cleanup(process):
            now[0] = 11.0
            return -15, b"", b""

        with mock.patch("silex_bench.process.time.monotonic", side_effect=lambda: now[0]), mock.patch(
            "silex_bench.process._supervised_popen", return_value=(supervisor, 12345, None)
        ), mock.patch("silex_bench.process._stop_process", side_effect=cleanup):
            result = run_marked_process(
                [sys.executable, "-c", "pass"], timeout=10, cwd=Path.cwd(),
                ready_input="ready\n", target_input=f"{TARGET_NONCE_PLACEHOLDER}\n",
                final_input="finish\n", ready_marker="READY", target_marker="TARGET",
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["error"], "marked process pipes are unavailable")

    def test_launch_failure_classification_survives_supervisor_cleanup(self) -> None:
        # Exercise the real handshake helper, whose cleanup runs before either
        # public runner receives the failure. All processes and I/O are mocked.
        for marked in (False, True):
            for outcome in ("launch_error", "handshake_limit", "observation_deadline"):
                with self.subTest(marked=marked, outcome=outcome):
                    now = [0.0]
                    timeout = 2.0 if outcome == "observation_deadline" else 10.0
                    supervisor = mock.Mock()
                    pinned = {
                        "descriptors": (), "executable": sys.executable,
                        "spawn": [sys.executable], "display": [sys.executable],
                        "backend_digest": "fixture", "launcher_path": None,
                        "launcher_digest": None, "immutable_inputs": [],
                    }

                    def select_handshake(readable, writable, exceptional, wait):
                        if outcome == "launch_error":
                            return readable, [], []
                        now[0] += wait
                        return [], [], []

                    def cleanup(process):
                        now[0] = 11.0
                        return -15, b"", b""

                    with mock.patch("silex_bench.process._pinned_command", return_value=pinned), mock.patch(
                        "silex_bench.process.subprocess.Popen", return_value=supervisor
                    ), mock.patch("silex_bench.process.time.monotonic", side_effect=lambda: now[0]), mock.patch(
                        "silex_bench.process.select.select", side_effect=select_handshake
                    ), mock.patch("silex_bench.process.os.read", return_value=b"ERROR launch failed\n"), mock.patch(
                        "silex_bench.process._stop_process", side_effect=cleanup
                    ) as stop:
                        kwargs = dict(timeout=timeout, cwd=Path.cwd())
                        if marked:
                            result = run_marked_process(
                                [sys.executable], **kwargs, ready_input="ready\n",
                                target_input=f"{TARGET_NONCE_PLACEHOLDER}\n", final_input="finish\n",
                                ready_marker="READY", target_marker="TARGET",
                            )
                        else:
                            result = run_process([sys.executable], **kwargs)

                    self.assertFalse(result["success"])
                    self.assertEqual(result["timeout"], outcome == "observation_deadline")
                    self.assertIn("launch failed" if outcome == "launch_error" else "timed out", result["error"])
                    stop.assert_called_once_with(supervisor)

    def test_process_interrupt_reaps_the_separate_session_supervisor(self) -> None:
        supervisor = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.process._supervised_popen",
            return_value=(supervisor, 12345, None),
        ), mock.patch(
            "silex_bench.process._bounded_communicate",
            side_effect=KeyboardInterrupt,
        ), mock.patch("silex_bench.process._stop_process") as stop_process:
            with self.assertRaises(KeyboardInterrupt):
                run_process(
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    timeout=1.0,
                    cwd=Path(temporary),
                )

        stop_process.assert_called_once_with(supervisor)

    def test_immutable_memfd_sealing_supports_python_without_fcntl_names(
        self,
    ) -> None:
        from silex_bench import process as process_module

        descriptor = os.memfd_create(
            "silex-bench-seal-test",
            getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
        )
        try:
            os.write(descriptor, b"immutable")
            process_module._seal_memfd_immutable(descriptor)
            applied = fcntl.fcntl(
                descriptor,
                getattr(fcntl, "F_GET_SEALS", 1034),
            )
            self.assertEqual(
                applied & process_module._IMMUTABLE_MEMFD_SEALS,
                process_module._IMMUTABLE_MEMFD_SEALS,
            )
            with self.assertRaises(OSError):
                os.pwrite(descriptor, b"X", 0)
            with self.assertRaises(OSError):
                os.ftruncate(descriptor, 0)
        finally:
            os.close(descriptor)

    def test_process_executes_an_immutable_executable_snapshot(self) -> None:
        from silex_bench import process as process_module

        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "tool"
            original = b"#!/usr/bin/env python3\nprint('ORIGINAL')\n"
            replacement = b"#!/usr/bin/env python3\nprint('REPLACEMENT')\n"
            executable.write_bytes(original)
            executable.chmod(0o755)
            real_popen = subprocess.Popen

            def replace_during_spawn(command: list[str], **kwargs: Any):
                executable.write_bytes(replacement)
                try:
                    return real_popen(command, **kwargs)
                finally:
                    executable.write_bytes(original)

            with mock.patch.object(
                process_module.subprocess,
                "Popen",
                side_effect=replace_during_spawn,
            ):
                result = run_process(
                    [str(executable)],
                    timeout=2.0,
                    cwd=Path(temporary),
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"].strip(), "ORIGINAL")
        self.assertEqual(
            result["executable_sha256"], hashlib.sha256(original).hexdigest()
        )

    def test_process_executes_immutable_path_argument_snapshots(self) -> None:
        from silex_bench import process as process_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = root / "program.py"
            manifest = root / "manifest.json"
            original_program = (
                b"import json, pathlib, sys\n"
                b"print(json.dumps({'program': 'original', "
                b"'manifest': pathlib.Path(sys.argv[1]).read_text()}))\n"
            )
            replacement_program = b"print('replacement')\n"
            original_manifest = b"original manifest\n"
            replacement_manifest = b"replacement manifest\n"
            program.write_bytes(original_program)
            manifest.write_bytes(original_manifest)
            real_popen = subprocess.Popen

            def replace_during_spawn(command: list[str], **kwargs: Any):
                program.write_bytes(replacement_program)
                manifest.write_bytes(replacement_manifest)
                try:
                    return real_popen(command, **kwargs)
                finally:
                    program.write_bytes(original_program)
                    manifest.write_bytes(original_manifest)

            with mock.patch.object(
                process_module.subprocess,
                "Popen",
                side_effect=replace_during_spawn,
            ):
                result = run_process(
                    [sys.executable, str(program), str(manifest)],
                    timeout=2.0,
                    cwd=root,
                    immutable_path_arguments=(1, 2),
                )

        self.assertTrue(result["success"])
        payload = json.loads(result["stdout"])
        self.assertEqual(payload["program"], "original")
        self.assertEqual(payload["manifest"], original_manifest.decode())
        self.assertEqual(
            result["immutable_inputs"],
            [
                {
                    "argument_index": 1,
                    "path": str(program),
                    "sha256": hashlib.sha256(original_program).hexdigest(),
                },
                {
                    "argument_index": 2,
                    "path": str(manifest),
                    "sha256": hashlib.sha256(original_manifest).hexdigest(),
                },
            ],
        )

    def test_process_anchors_directory_arguments_across_path_replacement(self) -> None:
        from silex_bench import process as process_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "build"
            directory.mkdir()
            moved = root / "original-build"
            outside = root / "outside-build"
            outside.mkdir()
            descriptor = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            real_popen = subprocess.Popen

            def replace_during_spawn(command: list[str], **kwargs: Any):
                directory.rename(moved)
                directory.symlink_to(outside, target_is_directory=True)
                return real_popen(command, **kwargs)

            try:
                with mock.patch.object(
                    process_module.subprocess,
                    "Popen",
                    side_effect=replace_during_spawn,
                ):
                    result = run_process(
                        [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; import sys; "
                            "(Path(sys.argv[1]) / 'marker').write_text('anchored')",
                            str(directory),
                        ],
                        timeout=2.0,
                        cwd=root,
                        directory_descriptors={100: descriptor},
                        directory_argument_descriptors={3: 100},
                    )
            finally:
                os.close(descriptor)

            self.assertTrue(result["success"], result)
            self.assertEqual((moved / "marker").read_text(), "anchored")
            self.assertFalse((outside / "marker").exists())

    def test_directory_descriptor_path_is_stable_across_process_invocations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            build = root / "build"
            source.mkdir()
            build.mkdir()
            source_descriptor = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            build_descriptor = os.open(
                build,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            first = root / "first.txt"
            second = root / "second.txt"
            writer = (
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1]).write_text("
                f"sys.argv[-1] + '|' + str(Path('/proc/self/fd/100').samefile({str(source)!r})))"
            )
            try:
                configured = run_process(
                    [
                        sys.executable,
                        "-c",
                        writer,
                        str(first),
                        str(source),
                        str(build),
                    ],
                    timeout=2.0,
                    cwd=root,
                    directory_descriptors={
                        100: source_descriptor,
                        101: build_descriptor,
                    },
                    directory_argument_descriptors={
                        4: 100,
                        5: 101,
                    },
                )
                built = run_process(
                    [
                        sys.executable,
                        "-c",
                        writer,
                        str(second),
                        str(build),
                    ],
                    timeout=2.0,
                    cwd=root,
                    directory_descriptors={
                        100: source_descriptor,
                        101: build_descriptor,
                    },
                    directory_argument_descriptors={4: 101},
                )
            finally:
                os.close(build_descriptor)
                os.close(source_descriptor)

            self.assertTrue(configured["success"], configured)
            self.assertTrue(built["success"], built)
            expected = "/proc/self/fd/101|True"
            self.assertEqual(first.read_text(), expected)
            self.assertEqual(second.read_text(), expected)

    def test_process_refuses_an_occupied_directory_target_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            occupied = root / "occupied"
            source.mkdir()
            occupied.mkdir()
            source_descriptor = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            occupied_descriptor = os.open(
                occupied,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            target_descriptor = fcntl.fcntl(
                occupied_descriptor,
                fcntl.F_DUPFD_CLOEXEC,
                100,
            )
            before = os.fstat(target_descriptor)
            try:
                result = run_process(
                    [sys.executable, "-c", "raise SystemExit(99)", str(source)],
                    timeout=2.0,
                    cwd=root,
                    directory_descriptors={target_descriptor: source_descriptor},
                    directory_argument_descriptors={3: target_descriptor},
                )
                after = os.fstat(target_descriptor)
            finally:
                os.close(target_descriptor)
                os.close(occupied_descriptor)
                os.close(source_descriptor)

            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertFalse(result["timeout"])
            self.assertIn("target descriptor", result["error"])
            self.assertIn("unavailable", result["error"])
            self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))

    def test_key_value_parser_rejects_duplicate_fields(self) -> None:
        self.assertEqual(parse_key_values("proof=false\nproof=true\n"), {})

    def test_process_empty_environment_does_not_use_ambient_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "hostile-tool-ran"
            hostile = root / "ambient-only-tool"
            hostile.write_text(
                f"#!{sys.executable}\n"
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            hostile.chmod(0o755)
            ambient = dict(os.environ)
            ambient["PATH"] = str(root)

            with mock.patch.dict(os.environ, ambient, clear=True):
                result = run_process(
                    ["ambient-only-tool"],
                    timeout=2.0,
                    cwd=root,
                    env={},
                )

        self.assertFalse(result["available"])
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertFalse(marker.exists())

    def test_process_rejects_fifo_executable_before_timeout_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "input-fifo"
            os.mkfifo(fifo, 0o755)
            program = (
                "from pathlib import Path; "
                "from silex_bench.process import run_process; "
                "result=run_process(['./input-fifo'], timeout=0.1, cwd=Path('.')); "
                "\nif not (result['available'] is False "
                "and result['success'] is False "
                "and result['timeout'] is False): "
                "raise SystemExit(repr(result))\n"
                "print('rejected')"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            completed = subprocess.run(
                [sys.executable, "-O", "-B", "-c", program],
                cwd=temporary,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=0.5,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "rejected")

    def test_process_output_is_bounded(self) -> None:
        from silex_bench import process as process_module

        limit = getattr(process_module, "MAX_CAPTURE_BYTES", 1 << 20)
        with tempfile.TemporaryDirectory() as temporary:
            result = run_process(
                [
                    sys.executable,
                    "-c",
                    f"import os; os.write(1, b'x' * ({limit} + 1))",
                ],
                timeout=2.0,
                cwd=Path(temporary),
            )
        self.assertFalse(result["success"])
        self.assertIn("output limit", result["error"])
        self.assertLessEqual(len(result["stdout"].encode()), limit)

    def test_process_success_and_failure_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cwd = Path(temporary)
            success = run_process(
                [sys.executable, "-c", "print('value=17')"],
                timeout=2.0,
                cwd=cwd,
            )
            self.assertTrue(success["available"])
            self.assertTrue(success["success"])
            self.assertFalse(success["timeout"])
            self.assertEqual(parse_key_values(success["stdout"]), {"value": "17"})

            failure = run_process(
                [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"],
                timeout=2.0,
                cwd=cwd,
            )
            self.assertTrue(failure["available"])
            self.assertFalse(failure["success"])
            self.assertFalse(failure["timeout"])
            self.assertEqual(failure["returncode"], 7)
            self.assertIn("bad", failure["stdout"])

    def test_process_without_stdin_does_not_inherit_the_harness_stdin(self) -> None:
        # Guards against passing stdin=None when the caller supplies no
        # input, which would let the target read from this test process's
        # own stdin. Reading it here would hang if that were still true
        # (nothing writes to this test's stdin); with the fix (DEVNULL), the
        # target's stdin.read() returns an immediate EOF.
        with tempfile.TemporaryDirectory() as temporary:
            result = run_process(
                [sys.executable, "-c", "import sys; print('data=' + repr(sys.stdin.read()))"],
                timeout=5.0,
                cwd=Path(temporary),
            )
        self.assertTrue(result["success"])
        self.assertEqual(parse_key_values(result["stdout"]), {"data": "''"})

    def test_process_malformed_argv_is_structured(self) -> None:
        cases = (
            ([], "contain an executable"),
            (["bad\0executable"], "null"),
            ([b"/bin/true"], "nonempty string"),
        )
        for command, expected_error in cases:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                result = run_process(
                    command,
                    timeout=1.0,
                    cwd=Path(temporary),
                )

            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertFalse(result["timeout"])
            self.assertIn(expected_error, result["error"])
            json.dumps(result)

    def test_process_rejects_malformed_input_before_spawn(self) -> None:
        cases: tuple[tuple[str, Any, Any], ...] = (
            ("stdin object", object(), None),
            ("stdin surrogate", "\ud800", None),
            ("environment key", None, {1: "value"}),
            ("environment value", None, {"KEY": 1}),
            ("environment null", None, {"KEY": "bad\0value"}),
        )
        for label, stdin, environment in cases:
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as temporary,
                mock.patch("silex_bench.process.subprocess.Popen") as popen,
            ):
                result = run_process(
                    [sys.executable, "-c", "raise SystemExit(99)"],
                    timeout=1.0,
                    cwd=Path(temporary),
                    stdin=cast(Any, stdin),
                    env=cast(Any, environment),
                )

            popen.assert_not_called()
            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertFalse(result["timeout"])
            self.assertTrue(result["error"])
            json.dumps(result)

    def test_process_rejects_nonfinite_or_nonpositive_timeout(self) -> None:
        for timeout in (float("inf"), float("nan"), 0.0, -1.0, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary:
                result = run_process(
                    [sys.executable, "-c", "raise SystemExit(99)"],
                    timeout=timeout,
                    cwd=Path(temporary),
                )

            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertFalse(result["timeout"])
            self.assertIn("finite positive", result["error"])
            json.dumps(result)

    def test_process_timeout_preserves_partial_output_and_kills_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            result = run_process(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys,time; print('partial', flush=True); "
                    "print('error-partial', file=sys.stderr, flush=True); "
                    "time.sleep(5)",
                ],
                timeout=0.1,
                cwd=Path(temporary),
            )
            elapsed = time.monotonic() - started
            self.assertTrue(result["available"])
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            self.assertEqual(result["error"], "process timed out")
            self.assertIn("partial", result["stdout"])
            self.assertIn("error-partial", result["stderr"])
            self.assertLess(elapsed, 2.0)

    def test_process_timeout_kills_group_after_leader_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "descendant-survived"
            descendant = (
                "import pathlib,time; "
                "time.sleep(0.8); "
                f"pathlib.Path({str(marker)!r}).write_text('survived')"
            )
            leader = (
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable, '-c', "
                f"{descendant!r}]); "
                "print('leader-exited', flush=True)"
            )
            result = run_process(
                [sys.executable, "-u", "-c", leader],
                timeout=0.1,
                cwd=Path(temporary),
            )

            self.assertTrue(result["available"])
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            time.sleep(0.9)
            self.assertFalse(marker.exists())

    def test_process_timeout_kills_descendant_that_escaped_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "setsid-descendant-survived"
            descendant = (
                "import pathlib,time; "
                "time.sleep(0.4); "
                f"pathlib.Path({str(marker)!r}).write_text('survived')"
            )
            leader = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', "
                f"{descendant!r}], start_new_session=True, "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL, close_fds=True); "
                "time.sleep(5)"
            )
            result = run_process(
                [sys.executable, "-u", "-c", leader],
                timeout=0.1,
                cwd=Path(temporary),
            )

            self.assertTrue(result["available"])
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            time.sleep(0.5)
            self.assertFalse(marker.exists())

    def test_process_target_cannot_kill_containment_supervisor(self) -> None:
        self.skipTest(
            "out of scope for the trusted-backend harness; testing hostile "
            "code requires a disposable VM or independently isolated UID/PID "
            "worker, and supervisor-kill probes are forbidden in a live user "
            "session"
        )

    def test_process_timeout_kills_reparented_double_fork_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "double-fork-descendant-survived"
            leader = (
                "import os,pathlib,time; "
                "child=os.fork(); "
                "exec(\"if child == 0:\\n"
                " os.setsid()\\n"
                " if os.fork() != 0: os._exit(0)\\n"
                " time.sleep(0.4)\\n"
                f" pathlib.Path({str(marker)!r}).write_text('survived')\\n\"); "
                "os.waitpid(child, 0); time.sleep(5)"
            )
            result = run_process(
                [sys.executable, "-u", "-c", leader],
                timeout=0.1,
                cwd=Path(temporary),
            )

            self.assertTrue(result["available"])
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            time.sleep(0.5)
            self.assertFalse(marker.exists())

    def test_marked_process_measures_only_target_segment(self) -> None:
        # A bare ">= 0.0" assertion on target_wall_ms would not show that the
        # ready and final phases are excluded from it. A
        # sleep in each phase, well outside the target segment itself, lets
        # target_wall_ms and process_wall_ms be told apart: target_wall_ms
        # must stay far below either sleep, while process_wall_ms (the
        # whole-process envelope) must exceed their sum.
        child = """
import sys
import time
sys.stdin.readline()
time.sleep(0.3)
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
time.sleep(0.3)
print("answer=42", flush=True)
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=5.0,
                cwd=Path(temporary),
            )
            self.assertTrue(result["available"])
            self.assertTrue(result["success"])
            self.assertFalse(result["timeout"])
            self.assertIsNotNone(result["target_wall_ms"])
            self.assertGreaterEqual(result["target_wall_ms"], 0.0)
            self.assertLess(result["target_wall_ms"], 150.0)
            self.assertGreaterEqual(result["process_wall_ms"], 550.0)
            self.assertEqual(parse_key_values(result["stdout"])["answer"], "42")

    def test_marked_process_handles_many_lines_across_both_phases(self) -> None:
        # complete_lines() used to re-copy and re-split the whole captured
        # segment since start_offset on every read, which is quadratic in
        # the number of reads. This exercises its incremental
        # rewrite with several thousand lines before and after the ready
        # marker (a phase-offset change partway through), including a mix of
        # \n, \r and \r\n terminators, and confirms every line still arrives
        # intact and in order and that marker detection is unaffected.
        line_count = 3000
        child = f"""
import sys
sys.stdin.readline()
for i in range({line_count}):
    end = "\\r\\n" if i % 3 == 0 else ("\\r" if i % 3 == 1 else "\\n")
    sys.stdout.write("before" + str(i) + end)
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
for i in range({line_count}):
    end = "\\r\\n" if i % 3 == 0 else ("\\r" if i % 3 == 1 else "\\n")
    sys.stdout.write("after" + str(i) + end)
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=10.0,
                cwd=Path(temporary),
            )
        self.assertTrue(result["success"], result)
        self.assertFalse(result["timeout"])
        lines = result["stdout"].splitlines()
        before_lines = [line for line in lines if line.startswith("before")]
        after_lines = [line for line in lines if line.startswith("after")]
        self.assertEqual(before_lines, [f"before{i}" for i in range(line_count)])
        self.assertEqual(after_lines, [f"after{i}" for i in range(line_count)])

    def test_split_complete_lines_holds_back_cr_split_from_its_newline(self) -> None:
        # A whole-buffer re-scan always sees a "\r" and a later "\n"
        # together, so it merges them into one "\r\n" line terminator. An
        # earlier incremental rewrite instead committed a trailing bare "\r"
        # as complete the moment it saw it, then treated a "\n" arriving in a
        # later read as its own, spurious empty line. This drives
        # _split_complete_lines() directly with that terminator split across
        # every call boundary the fix must resolve without a phantom line.
        complete, tail = process_module._split_complete_lines(
            b"line1\r", eof=False
        )
        self.assertEqual(complete, [])
        self.assertEqual(tail, b"line1\r")

        more_complete, tail = process_module._split_complete_lines(
            tail + b"\nline2\r\n", eof=False
        )
        self.assertEqual(more_complete, [b"line1", b"line2"])
        self.assertEqual(tail, b"")

    def test_split_complete_lines_resolves_trailing_cr_at_eof(self) -> None:
        # A trailing bare "\r" with nothing after it, and no more input
        # possible, is a complete one-line-terminated-by-\r line: the same
        # content a whole-buffer re-scan would report immediately, since it
        # never has more input to wait for either.
        complete, tail = process_module._split_complete_lines(
            b"line1\r", eof=True
        )
        self.assertEqual(complete, [b"line1"])
        self.assertEqual(tail, b"")

    def test_split_complete_lines_matches_whole_buffer_rescan(self) -> None:
        # Equivalence test: for a fixed final buffer mixing \n, \r and \r\n
        # terminators, split arbitrarily into chunks that may or may not land
        # inside a "\r\n" pair, the incremental scan's cumulative result once
        # every chunk has arrived (eof=True on the last one) must match a
        # single whole-buffer re-scan of the same bytes -- the property the
        # old, whole-buffer-rescan implementation had simply by re-deriving
        # the line list from scratch on every call.
        def whole_buffer_rescan(buffer: bytes) -> list[bytes]:
            return [
                line.rstrip(b"\r\n")
                for line in buffer.splitlines(keepends=True)
                if line.endswith((b"\n", b"\r"))
            ]

        terminators = (b"\n", b"\r", b"\r\n")
        rng = random.Random(20260927)
        for trial in range(200):
            line_count = rng.randint(1, 12)
            buffer = b"".join(
                f"line{i}".encode() + rng.choice(terminators)
                for i in range(line_count)
            )
            # Optionally leave a final unterminated tail, which never
            # resolves under either algorithm.
            unterminated = b""
            if rng.random() < 0.5:
                unterminated = b"tail"
                buffer += unterminated

            # Split the buffer at random byte offsets, including offsets
            # that fall inside a "\r\n" pair.
            cut_count = rng.randint(0, len(buffer))
            cuts = sorted(rng.sample(range(len(buffer) + 1), cut_count))
            offsets = sorted({0, len(buffer), *cuts})
            chunks = [
                buffer[start:end]
                for start, end in zip(offsets, offsets[1:])
                if start != end
            ]

            complete: list[bytes] = []
            pending = b""
            for index, chunk in enumerate(chunks):
                pending += chunk
                at_eof = index == len(chunks) - 1
                newly_complete, pending = process_module._split_complete_lines(
                    pending, eof=at_eof
                )
                complete.extend(newly_complete)

            with self.subTest(trial=trial, buffer=buffer, offsets=offsets):
                self.assertEqual(complete, whole_buffer_rescan(buffer))
                self.assertEqual(pending, unterminated)

    def test_marked_process_rejects_target_marker_before_dispatch(self) -> None:
        child = """
import sys
import time
print("READY", flush=True)
print("TARGET", flush=True)
sys.stdin.readline()
sys.stdin.readline()
time.sleep(0.25)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("before target dispatch", result["error"])

    def test_marked_process_rejects_nonce_marker_before_dispatch(self) -> None:
        child = """
import os
import sys
sys.stdin.readline()
os.write(1, b"READY\\nTARGET:forged\\n")
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("before target dispatch", result["error"])

    def test_marked_process_rejects_malformed_generated_nonce(self) -> None:
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
sys.stdin.readline()
"""
        for nonce in (None, "a" * 31, "A" * 32, "g" * 32, "a" * 32 + "\n"):
            with (
                self.subTest(nonce=nonce),
                tempfile.TemporaryDirectory() as temporary,
                mock.patch("silex_bench.process.secrets.token_hex", return_value=nonce),
            ):
                result = run_marked_process(
                    [sys.executable, "-u", "-c", child],
                    ready_input="start\n",
                    target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                    final_input="finish\n",
                    ready_marker="READY",
                    target_marker="TARGET",
                    timeout=1.0,
                    cwd=Path(temporary),
                )
                self.assertFalse(result["success"])
                self.assertFalse(result["timeout"])
                self.assertIn("generated target nonce is invalid", result["error"])

    def test_marked_process_rejects_mismatched_target_nonce(self) -> None:
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
replacement = "0" if nonce[0] != "0" else "1"
print("TARGET:" + replacement + nonce[1:], flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=1.0,
                cwd=Path(temporary),
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("invalid target marker", result["error"])

    def test_marked_process_bounds_target_input_after_nonce_expansion(self) -> None:
        target_input = TARGET_NONCE_PLACEHOLDER + (
            "x"
            * (
                MAX_PROTOCOL_INPUT_BYTES
                - len(TARGET_NONCE_PLACEHOLDER.encode("utf-8"))
            )
        )
        self.assertEqual(len(target_input.encode("utf-8")), MAX_PROTOCOL_INPUT_BYTES)
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("silex_bench.process.subprocess.Popen") as popen,
        ):
            result = run_marked_process(
                [sys.executable, "-c", "raise SystemExit(99)"],
                ready_input="start\n",
                target_input=target_input,
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=1.0,
                cwd=Path(temporary),
            )

        popen.assert_not_called()
        self.assertFalse(result["success"])
        self.assertIn("protocol input limit", result["error"])

    def test_marked_process_rejects_malformed_protocol_text(self) -> None:
        cases = (
            ({"ready_input": "\ud800"}, "ready_input must be valid UTF-8"),
            ({"target_input": TARGET_NONCE_PLACEHOLDER + "\0"}, "target_input"),
            ({"final_input": "finish\0"}, "final_input"),
            ({"ready_marker": "READY\0"}, "ready_marker"),
            ({"target_marker": "TARGÉT"}, "target_marker"),
        )
        defaults = {
            "ready_input": "start\n",
            "target_input": TARGET_NONCE_PLACEHOLDER + "\n",
            "final_input": "finish\n",
            "ready_marker": "READY",
            "target_marker": "TARGET",
        }
        for mutation, expected_error in cases:
            arguments = {**defaults, **mutation}
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                result = run_marked_process(
                    [sys.executable, "-c", "raise SystemExit(99)"],
                    ready_input=arguments["ready_input"],
                    target_input=arguments["target_input"],
                    final_input=arguments["final_input"],
                    ready_marker=arguments["ready_marker"],
                    target_marker=arguments["target_marker"],
                    timeout=1.0,
                    cwd=Path(temporary),
                )
                self.assertFalse(result["available"])
                self.assertFalse(result["success"])
                self.assertFalse(result["timeout"])
                self.assertIn(expected_error, result["error"])
                json.dumps(result)

    def test_marked_process_malformed_argv_is_structured(self) -> None:
        cases = (
            ([], "contain an executable"),
            (["bad\0executable"], "null"),
            ([b"/bin/true"], "nonempty string"),
        )
        for command, expected_error in cases:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                result = run_marked_process(
                    command,
                    ready_input="start\n",
                    target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                    final_input="finish\n",
                    ready_marker="READY",
                    target_marker="TARGET",
                    timeout=1.0,
                    cwd=Path(temporary),
                )

            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertFalse(result["timeout"])
            self.assertIn(expected_error, result["error"])
            json.dumps(result)

    def test_marked_process_rejects_nonfinite_deadlines(self) -> None:
        for timeout in (float("inf"), float("nan"), 0.0, False):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary:
                result = run_marked_process(
                    [sys.executable, "-c", "raise SystemExit(99)"],
                    ready_input="start\n",
                    target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                    final_input="finish\n",
                    ready_marker="READY",
                    target_marker="TARGET",
                    timeout=timeout,
                    cwd=Path(temporary),
                )

            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertIn("finite positive", result["error"])
            json.dumps(result)

    def test_marked_process_keeps_stderr_separate(self) -> None:
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
print("answer=42", flush=True)
print("diagnostic", file=sys.stderr, flush=True)
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
            )
        self.assertTrue(result["success"])
        self.assertIn("answer=42", result["stdout"])
        self.assertNotIn("diagnostic", result["stdout"])
        self.assertIn("diagnostic", result["stderr"])

    def test_marked_process_output_is_bounded(self) -> None:
        from silex_bench import process as process_module

        limit = getattr(process_module, "MAX_CAPTURE_BYTES", 1 << 20)
        child = f"""
import os
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
os.write(1, b'x' * ({limit} + 1))
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
            )
        self.assertFalse(result["success"])
        self.assertIn("output limit", result["error"])
        self.assertLessEqual(len(result["stdout"].encode()), limit)

    def test_marked_process_timeout_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", "import time; time.sleep(5)"],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=1.0,
                cwd=Path(temporary),
            )
            self.assertTrue(result["available"])
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            self.assertEqual(result["error"], "marked process did not reach ready marker")

    def test_marked_process_uses_one_nonresetting_deadline(self) -> None:
        child = """
import sys
import time
sys.stdin.readline()
time.sleep(0.4)
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
time.sleep(0.4)
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=0.6,
                cwd=Path(temporary),
            )
            self.assertFalse(result["success"])
            self.assertTrue(result["timeout"])
            # process_wall_ms is captured at classification, before
            # _stop_process's cleanup budget (up to about 2 s) runs, so it
            # should sit close to the 0.6 s deadline rather than include
            # cleanup (a wider 800 ms bound here was previously used as a
            # workaround for that cleanup time and was timing-sensitive
            # under load).
            self.assertGreaterEqual(result["process_wall_ms"], 550.0)
            self.assertLess(result["process_wall_ms"], 700.0)

    def test_requested_cpu_outside_current_affinity_is_unavailable(self) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        available_cpus = os.sched_getaffinity(0)
        unavailable_cpu = max(available_cpus, default=-1) + 1
        while unavailable_cpu in available_cpus:
            unavailable_cpu += 1

        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-c", "raise SystemExit(99)"],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=1.0,
                cwd=Path(temporary),
                cpu=unavailable_cpu,
            )

        self.assertFalse(result["available"])
        self.assertFalse(result["success"])
        self.assertIn(f"requested CPU {unavailable_cpu}", result["error"])
        self.assertIn("current process affinity", result["error"])

    def test_marked_process_records_effective_cpu_pin(self) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        if shutil.which("taskset") is None:
            self.skipTest("taskset is unavailable")
        cpu = min(os.sched_getaffinity(0))
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
                cpu=cpu,
            )

        self.assertTrue(result["available"])
        self.assertTrue(result["success"])
        self.assertEqual(result["effective_affinity"], [cpu])
        self.assertTrue(Path(result["launcher_executable"]).is_absolute())
        self.assertRegex(result["launcher_executable_sha256"], r"^[0-9a-f]{64}$")

    def test_marked_process_re_checks_affinity_after_target_marker(self) -> None:
        # The readiness check alone only shows the affinity a backend
        # started with. Here the target changes its own affinity
        # during the measured interval; the post-target re-check must catch
        # that drift and, like the readiness check, fail the sample (it is a
        # protocol failure, not a timeout).
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        if shutil.which("taskset") is None:
            self.skipTest("taskset is unavailable")
        available_cpus = sorted(os.sched_getaffinity(0))
        if len(available_cpus) < 2:
            self.skipTest("at least two available CPUs are required")
        cpu, other_cpu = available_cpus[0], available_cpus[1]
        child = f"""
import os
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
os.sched_setaffinity(0, {{{other_cpu}}})
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
                cpu=cpu,
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("after the target marker", result["error"])
        self.assertIn(f"requested CPU {cpu}", result["error"])
        self.assertEqual(result["effective_affinity"], [cpu])
        self.assertEqual(result["effective_affinity_after_target"], [other_cpu])
        self.assertNotIn("target_wall_ms", result)

    def test_marked_process_fails_when_post_target_affinity_read_errors(self) -> None:
        # The readiness read succeeds, then the post-target sched_getaffinity
        # of the target raises OSError: the sample must fail (not succeed
        # with an unverified affinity) and must not be classified a timeout.
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        if shutil.which("taskset") is None:
            self.skipTest("taskset is unavailable")
        cpu = min(os.sched_getaffinity(0))
        target_reads = [0]

        class FailingOs:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def sched_getaffinity(self, pid: int) -> Any:
                if pid != 0:
                    target_reads[0] += 1
                    if target_reads[0] >= 2:
                        raise OSError("simulated affinity read failure")
                return os.sched_getaffinity(pid)

        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            process_module, "os", FailingOs()
        ):
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=5.0,
                cwd=Path(temporary),
                cpu=cpu,
            )

        self.assertEqual(target_reads[0], 2)
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("after the target marker", result["error"])
        self.assertIn("simulated affinity read failure", result["error"])
        self.assertEqual(result["effective_affinity"], [cpu])
        self.assertIsNone(result["effective_affinity_after_target"])

    def test_marked_process_keeps_an_unchanged_affinity_after_target(self) -> None:
        # Counterpart to the affinity-drift failure above: a target that
        # stays on its requested CPU passes the post-target re-check.
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        if shutil.which("taskset") is None:
            self.skipTest("taskset is unavailable")
        cpu = min(os.sched_getaffinity(0))
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=5.0,
                cwd=Path(temporary),
                cpu=cpu,
            )

        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual(result["effective_affinity_after_target"], [cpu])
        self.assertIsNone(result["failure_origin"])

    def test_cpu_pin_does_not_execute_path_selected_taskset(self) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        cpu = min(os.sched_getaffinity(0))
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "hostile-taskset-ran"
            hostile = root / "taskset"
            hostile.write_text(
                f"#!{sys.executable}\n"
                "import os, pathlib, sys\n"
                "os.sched_setaffinity(0, {int(sys.argv[2])})\n"
                f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
                "sys.stdin.readline()\n"
                "print('READY', flush=True)\n"
                "nonce = sys.stdin.readline().strip()\n"
                "print('TARGET:' + nonce, flush=True)\n"
                "sys.stdin.readline()\n"
            )
            hostile.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = f"{root}{os.pathsep}{environment.get('PATH', '')}"
            with mock.patch.dict(os.environ, environment, clear=True):
                result = run_marked_process(
                    [sys.executable, "-u", "-c", child],
                    ready_input="start\n",
                    target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                    final_input="finish\n",
                    ready_marker="READY",
                    target_marker="TARGET",
                    timeout=2.0,
                    cwd=root,
                    cpu=cpu,
                    env=environment,
                )

            self.assertTrue(result["success"])
            self.assertFalse(marker.exists())
            self.assertNotEqual(Path(result["launcher_executable"]), hostile)
            self.assertEqual(result["effective_affinity"], [cpu])

    def test_marked_process_rejects_target_marker_substrings(self) -> None:
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("prefix-TARGET:" + nonce + "-suffix", flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=1.0,
                cwd=Path(temporary),
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["timeout"])
        self.assertIn("target marker", result["error"])

    def test_marked_process_phase_write_obeys_timeout(self) -> None:
        child = """
import time
time.sleep(2)
"""
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="x" * (512 * 1024),
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=0.05,
                cwd=Path(temporary),
            )
            elapsed = time.monotonic() - started

        self.assertFalse(result["success"])
        self.assertTrue(result["timeout"])
        self.assertLess(elapsed, 0.5)

    def test_marked_process_records_early_exit_before_a_marker(self) -> None:
        # Covers a marked target that exits, with its own exit code, before
        # ever reaching a marker -- as distinct from a deadline timeout. The
        # child reads the ready input (so the
        # write itself succeeds) and then exits without printing READY.
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(3)"],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=5.0,
                cwd=Path(temporary),
            )
        self.assertTrue(result["available"])
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["returncode"], 3)
        self.assertEqual(result["error"], "marked process did not reach ready marker")

    def test_marked_process_reports_broken_pipe_distinctly_from_timeout(self) -> None:
        # A closed input pipe used to be reported with the same "timed out
        # writing ... input" text as an actual deadline expiry, even though
        # timeout was already False. The message must name the real cause
        # instead.
        child = """
import sys
import time
sys.stdin.close()
time.sleep(2)
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                # Larger than a pipe's default kernel buffer (64 KiB): the
                # write must still be in progress, waiting for room, when the
                # child's close() takes effect, so the harness observes the
                # closed pipe instead of finishing the write beforehand.
                ready_input="x" * (512 * 1024),
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=5.0,
                cwd=Path(temporary),
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("pipe closed", result["error"])
        self.assertIn("ready input", result["error"])
        self.assertNotIn("timed out", result["error"])

    def _lagging_harness(self) -> tuple[Any, Any, list[float]]:
        """Stand-ins for the process module's ``os`` and ``time`` that simulate
        harness scheduling lag deterministically.

        From the first EOF the harness reads on a pipe (a captured stream,
        so the target has exited by then), its monotonic clock reads 1000 s
        later, as if the harness had been descheduled past the deadline
        before noticing. EOFs on regular files (read while pinning the
        command) do not count.
        """
        lag = [0.0]

        class LaggingOs:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def read(self, descriptor: int, size: int) -> bytes:
                data = os.read(descriptor, size)
                if not data and stat.S_ISFIFO(os.fstat(descriptor).st_mode):
                    lag[0] = 1000.0
                return data

        class LaggingTime:
            def __getattr__(self, name: str) -> Any:
                return getattr(time, name)

            def monotonic(self) -> float:
                return time.monotonic() + lag[0]

        return LaggingOs(), LaggingTime(), lag

    def test_bounded_communicate_classifies_an_already_exited_process_as_exited(
        self,
    ) -> None:
        # Once the deadline has passed, an exit (and EOF) that had already
        # happened must be classified as an exit, not a timeout. The
        # process has exited (unreaped) before the call, and a
        # zero timeout makes the harness notice only after the deadline.
        process = subprocess.Popen(
            [sys.executable, "-c", "print('done'); raise SystemExit(3)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
            stdout, stderr, error = process_module._bounded_communicate(
                process, None, 0.0
            )
        finally:
            process.stdout.close()
            process.stderr.close()
            process.wait()
        self.assertIsNone(error)
        self.assertEqual(bytes(stdout), b"done\n")
        self.assertEqual(process.returncode, 3)

    def test_bounded_communicate_still_times_out_a_running_process(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _, _, error = process_module._bounded_communicate(process, None, 0.0)
        finally:
            process.kill()
            process.stdout.close()
            process.stderr.close()
            process.wait()
        self.assertEqual(error, "process timed out")

    def test_process_exit_noticed_after_the_deadline_is_not_a_timeout(self) -> None:
        # Same "exit noticed after the deadline" case as above, but through
        # run_process with the real supervisor.
        lagging_os, lagging_time, lag = self._lagging_harness()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            process_module, "os", lagging_os
        ), mock.patch.object(process_module, "time", lagging_time):
            result = run_process(
                [sys.executable, "-c", "print('out'); raise SystemExit(3)"],
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertEqual(lag[0], 1000.0)
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["returncode"], 3)
        self.assertEqual(result["stdout"], "out\n")
        self.assertEqual(result["failure_origin"], "target")

    def test_process_deadline_passing_before_communicate_still_observes_exit(
        self,
    ) -> None:
        # The deadline passes between the supervisor handshake and
        # communicate. The remaining budget is clamped to zero (not
        # negative), so the harness still makes one non-blocking observation
        # and returns a classified row (success if the target had already
        # finished, otherwise a timeout) instead of raising.
        real_popen = process_module._supervised_popen
        jumped = [0.0]

        def popen_then_expire(*args: Any, **kwargs: Any) -> Any:
            returned = real_popen(*args, **kwargs)
            jumped[0] = 1000.0
            return returned

        class JumpingTime:
            def __getattr__(self, name: str) -> Any:
                return getattr(time, name)

            def monotonic(self) -> float:
                return time.monotonic() + jumped[0]

        real_communicate = process_module._bounded_communicate
        budgets: list[float] = []

        def recording_communicate(*args: Any, **kwargs: Any) -> Any:
            budgets.append(args[2])
            return real_communicate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            process_module, "_supervised_popen", popen_then_expire
        ), mock.patch.object(
            process_module, "_bounded_communicate", recording_communicate
        ), mock.patch.object(process_module, "time", JumpingTime()):
            result = run_process(
                [sys.executable, "-c", "print('out')"],
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertEqual(jumped[0], 1000.0)
        self.assertEqual(budgets, [0.0])
        self.assertTrue(result["timeout"] or result["success"], result)
        self.assertIs(result["success"], not result["timeout"])
        if result["timeout"]:
            self.assertEqual(result["error"], "process timed out")
        else:
            self.assertEqual(result["stdout"], "out\n")

    def test_marked_exit_noticed_after_the_deadline_is_not_a_timeout(self) -> None:
        # A marked target that exits early, noticed only after the deadline,
        # is an early exit, not a timeout.
        lagging_os, lagging_time, lag = self._lagging_harness()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            process_module, "os", lagging_os
        ), mock.patch.object(process_module, "time", lagging_time):
            result = run_marked_process(
                [sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(3)"],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertEqual(lag[0], 1000.0)
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["returncode"], 3)
        self.assertEqual(result["error"], "marked process did not reach ready marker")
        self.assertEqual(result["failure_origin"], "target")

    def test_marked_process_rejects_a_leftover_partial_line_read_before_dispatch(
        self,
    ) -> None:
        # A prompt-like unterminated line left before dispatch runs into the
        # target marker. One write puts the ready
        # marker and the partial line in the pipe together, so the harness
        # reads the partial line before dispatch.
        child = """
import os
import sys
sys.stdin.readline()
os.write(1, b"READY\\n? ")
nonce = sys.stdin.readline().strip()
os.write(1, ("TARGET:" + nonce + "\\n").encode())
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(
            result["error"],
            "marked process emitted the target marker after unterminated "
            "output on the same line",
        )

    def test_marked_process_rejects_a_leftover_partial_line_read_after_dispatch(
        self,
    ) -> None:
        # The same partial line, arriving in the same write as the marker
        # (so the harness reads it after dispatch), is the same protocol
        # error, not a timeout.
        child = """
import os
import sys
sys.stdin.readline()
os.write(1, b"READY\\n")
nonce = sys.stdin.readline().strip()
os.write(1, ("? TARGET:" + nonce + "\\n").encode())
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=30.0,
                cwd=Path(temporary),
            )
            elapsed = time.monotonic() - started
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertIn("unterminated output", result["error"])
        self.assertLess(elapsed, 20.0)

    def test_marked_process_accepts_a_partial_line_terminated_before_the_marker(
        self,
    ) -> None:
        # A leftover partial line that the target terminates before printing
        # the marker does not touch the marker line.
        child = """
import os
import sys
sys.stdin.readline()
os.write(1, b"READY\\n? ")
nonce = sys.stdin.readline().strip()
os.write(1, ("\\nTARGET:" + nonce + "\\n").encode())
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertTrue(result["success"], result.get("error"))

    def test_failure_origin_distinguishes_supervisor_and_target_exit_124(
        self,
    ) -> None:
        # Distinguishes exit status 124 from a supervisor stopped after a
        # timeout from a target that itself exits with 124.
        with tempfile.TemporaryDirectory() as temporary:
            stopped = run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=0.5,
                cwd=Path(temporary),
            )
            own = run_process(
                [sys.executable, "-c", "raise SystemExit(124)"],
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertTrue(stopped["timeout"])
        self.assertEqual(stopped["returncode"], 124)
        self.assertEqual(stopped["failure_origin"], "supervisor")
        self.assertFalse(own["timeout"])
        self.assertEqual(own["returncode"], 124)
        self.assertEqual(own["failure_origin"], "target")

    def test_failure_origin_is_target_for_a_target_exit_126(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_process(
                [sys.executable, "-c", "raise SystemExit(126)"],
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 126)
        self.assertEqual(result["failure_origin"], "target")
        self.assertNotIn("error", result)

    def test_post_handshake_supervisor_error_is_reported_with_its_origin(
        self,
    ) -> None:
        # A supervisor that fails after the PID handshake reports on the
        # still-open control channel; the harness records the
        # error and the supervisor origin of exit status 126. A stand-in
        # supervisor writes the whole exchange in one write, so the harness
        # also has to keep the bytes that follow the PID line.
        fake_supervisor = (
            "import os\n"
            f"fd = int(os.environ[{process_module._SUPERVISOR_CONTROL_FD_ENV!r}])\n"
            "os.write(fd, b'PID %d\\nERROR simulated cleanup failure\\n"
            "EXIT supervisor 126\\n' % os.getpid())\n"
            "os._exit(126)\n"
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            process_module, "_SUPERVISOR_SOURCE", fake_supervisor
        ):
            result = run_process(
                [sys.executable, "-c", "pass"],
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertFalse(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["returncode"], 126)
        self.assertEqual(result["failure_origin"], "supervisor")
        self.assertEqual(
            result["error"],
            "process supervisor failed: simulated cleanup failure",
        )

    def test_failure_origin_is_unknown_without_a_matching_exit_report(self) -> None:
        # A supervisor that exits without reporting (or reports a different
        # status than the harness observed) leaves the origin unknown.
        for report in ("", "EXIT target 0\\n"):
            with self.subTest(report=report):
                fake_supervisor = (
                    "import os\n"
                    f"fd = int(os.environ[{process_module._SUPERVISOR_CONTROL_FD_ENV!r}])\n"
                    f"os.write(fd, b'PID %d\\n{report}' % os.getpid())\n"
                    "os._exit(126)\n"
                )
                with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                    process_module, "_SUPERVISOR_SOURCE", fake_supervisor
                ):
                    result = run_process(
                        [sys.executable, "-c", "pass"],
                        timeout=30.0,
                        cwd=Path(temporary),
                    )
                self.assertEqual(result["returncode"], 126)
                self.assertIsNone(result["failure_origin"])
                self.assertNotIn("error", result)

    def test_marked_success_has_no_failure_origin(self) -> None:
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=30.0,
                cwd=Path(temporary),
            )
        self.assertTrue(result["success"], result.get("error"))
        self.assertIsNone(result["failure_origin"])

    def test_timeout_cleanup_does_not_call_unbounded_communicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            subprocess.Popen,
            "communicate",
            side_effect=AssertionError("cleanup must not call communicate"),
        ):
            result = run_process(
                [sys.executable, "-u", "-c", "import time; time.sleep(5)"],
                timeout=0.05,
                cwd=Path(temporary),
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["timeout"])

    @mock.patch("silex_bench.process.shutil.which", return_value=None)
    def test_missing_taskset_is_reported_without_starting_process(
        self, _which: mock.Mock
    ) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        cpu = min(os.sched_getaffinity(0))
        with tempfile.TemporaryDirectory() as temporary:
            result = run_process(
                [sys.executable, "-c", "raise SystemExit(99)"],
                timeout=1.0,
                cwd=Path(temporary),
                cpu=cpu,
            )
            self.assertFalse(result["available"])
            self.assertFalse(result["success"])
            self.assertIn("taskset", result["error"])

    def test_supervisor_runs_off_the_target_cpu_and_its_smt_siblings(self) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        if shutil.which("taskset") is None:
            self.skipTest("taskset is unavailable")
        available_cpus = sorted(os.sched_getaffinity(0))
        if len(available_cpus) < 2:
            self.skipTest("test requires at least two available CPUs")
        cpu = min(available_cpus)
        expected_supervisor_cpus = set(
            process_module._supervisor_housekeeping_cpus(cpu, available_cpus)
        )
        if cpu in expected_supervisor_cpus:
            self.skipTest(
                "no housekeeping CPU is available on this host (single-CPU "
                "affinity, or the target CPU and all its SMT siblings "
                "exhaust the available set); the supervisor necessarily "
                "shares the target CPU in that fallback"
            )

        # The child reads Cpus_allowed_list for itself and for its direct
        # parent (the supervisor) from /proc while both are still alive,
        # since the supervisor is fully reaped by the time run_marked_process
        # returns.
        child = """
import os
import sys


def cpus_allowed(pid):
    with open(f"/proc/{pid}/status") as handle:
        for line in handle:
            if line.startswith("Cpus_allowed_list:"):
                return line.split(":", 1)[1].strip()
    return ""


sys.stdin.readline()
print("SUPERVISOR_CPUS:" + cpus_allowed(os.getppid()), flush=True)
print("TARGET_CPUS:" + cpus_allowed(os.getpid()), flush=True)
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=2.0,
                cwd=Path(temporary),
                cpu=cpu,
            )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["effective_affinity"], [cpu])

        lines = dict(
            line.split(":", 1) for line in result["stdout"].splitlines() if ":" in line
        )
        target_cpus = _parse_cpu_list(lines["TARGET_CPUS"])
        supervisor_cpus = _parse_cpu_list(lines["SUPERVISOR_CPUS"])

        self.assertEqual(target_cpus, {cpu})
        self.assertNotIn(cpu, supervisor_cpus)
        siblings = process_module._read_thread_siblings(cpu) or set()
        self.assertTrue(supervisor_cpus.isdisjoint(siblings))
        self.assertEqual(supervisor_cpus, expected_supervisor_cpus)

    def test_supervisor_stays_asleep_during_the_run(self) -> None:
        # The supervisor once polled every 5 ms (about 200 wakeups/s) instead
        # of blocking; this guards against that regression. Read the
        # supervisor's own
        # voluntary_ctxt_switches count from /proc twice, about 0.3 s apart,
        # while it is blocked in that wait for the whole window (no target
        # marker write or exit happens in between). A tolerant bound (a
        # handful, not the ~60 a 5 ms poller would rack up in that window)
        # catches a reintroduced poll without being flaky under host load.
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("sched_getaffinity is unavailable")
        if shutil.which("taskset") is None:
            self.skipTest("taskset is unavailable")
        cpu = min(os.sched_getaffinity(0))
        child = """
import os
import sys
import time


def voluntary_ctxt_switches(pid):
    with open(f"/proc/{pid}/status") as handle:
        for line in handle:
            if line.startswith("voluntary_ctxt_switches:"):
                return int(line.split(":", 1)[1].strip())
    return None


sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
before = voluntary_ctxt_switches(os.getppid())
time.sleep(0.3)
after = voluntary_ctxt_switches(os.getppid())
print("BEFORE:" + str(before), flush=True)
print("AFTER:" + str(after), flush=True)
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as temporary:
            result = run_marked_process(
                [sys.executable, "-u", "-c", child],
                ready_input="start\n",
                target_input=TARGET_NONCE_PLACEHOLDER + "\n",
                final_input="finish\n",
                ready_marker="READY",
                target_marker="TARGET",
                timeout=3.0,
                cwd=Path(temporary),
                cpu=cpu,
            )

        self.assertTrue(result["success"], result)
        lines = dict(
            line.split(":", 1) for line in result["stdout"].splitlines() if ":" in line
        )
        before = int(lines["BEFORE"])
        after = int(lines["AFTER"])
        self.assertLessEqual(after - before, 8)


class SupervisorHousekeepingCpuTests(unittest.TestCase):
    """Unit coverage for the fallback tiers, independent of host topology."""

    def test_excludes_target_and_smt_siblings_when_available(self) -> None:
        with mock.patch(
            "silex_bench.process._read_thread_siblings", return_value={0, 12}
        ):
            result = process_module._supervisor_housekeeping_cpus(0, list(range(24)))
        self.assertNotIn(0, result)
        self.assertNotIn(12, result)
        self.assertTrue(result)

    def test_falls_back_to_sibling_when_only_the_smt_pair_is_available(self) -> None:
        with mock.patch(
            "silex_bench.process._read_thread_siblings", return_value={0, 12}
        ):
            result = process_module._supervisor_housekeeping_cpus(0, [0, 12])
        self.assertEqual(result, (12,))

    def test_falls_back_to_shared_cpu_when_it_is_the_only_one_available(self) -> None:
        with mock.patch(
            "silex_bench.process._read_thread_siblings", return_value={0, 12}
        ):
            result = process_module._supervisor_housekeeping_cpus(0, [0])
        self.assertEqual(result, (0,))

    def test_treats_unknown_topology_as_no_siblings(self) -> None:
        with mock.patch(
            "silex_bench.process._read_thread_siblings", return_value=None
        ):
            result = process_module._supervisor_housekeeping_cpus(0, [0, 1, 2])
        self.assertEqual(result, (1, 2))


class SupervisorSourceDescendantTests(unittest.TestCase):
    """Mocked coverage of the embedded supervisor's own functions.

    These exec() the supervisor source (see _load_supervisor_namespace) so
    the fail-closed startup check, the multi-thread descendant walk, and the
    stop-path fallback kill can be exercised directly, without depending on
    a specific kernel's /proc support or spawning a real supervisor process.
    """

    def setUp(self) -> None:
        # Snapshot this test process's own signal disposition so tearDown
        # can prove no test in this class leaked a change into it: a test
        # that calls the embedded supervisor's real main() must mock signal
        # installation rather than let it touch this process's handlers or
        # wakeup fd.
        self._signal_snapshot = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGCHLD)
        }
        self._wakeup_fd_snapshot = signal.set_wakeup_fd(-1)
        if self._wakeup_fd_snapshot != -1:
            signal.set_wakeup_fd(self._wakeup_fd_snapshot)

    def tearDown(self) -> None:
        for signum, handler in self._signal_snapshot.items():
            self.assertEqual(
                signal.getsignal(signum),
                handler,
                f"test left signal {signum} handler changed in this process",
            )
        current_wakeup_fd = signal.set_wakeup_fd(-1)
        if current_wakeup_fd != -1:
            signal.set_wakeup_fd(current_wakeup_fd)
        self.assertEqual(
            current_wakeup_fd,
            self._wakeup_fd_snapshot,
            "test left this process's wakeup fd changed",
        )

    def test_missing_own_children_file_fails_closed_at_startup(self) -> None:
        namespace = _load_supervisor_namespace()
        pid = os.getpid()
        missing_path = f"/proc/{pid}/task/{pid}/children"
        real_open = os.open

        def fake_open(path, flags, *args, **kwargs):
            if path == missing_path:
                raise FileNotFoundError(path)
            return real_open(path, flags, *args, **kwargs)

        control_read_fd, control_write_fd = os.pipe()
        # main() opens its own wakeup pipe; give it one under test control
        # (mocked os.pipe below) so it never touches the real signal wakeup
        # fd or leaks an untracked real pipe.
        wakeup_read_fd, wakeup_write_fd = os.pipe()
        signal_calls: list[tuple[int, Any]] = []

        def fake_signal(signum, handler):
            signal_calls.append((signum, handler))
            return signal.SIG_DFL

        try:
            environment = {process_module._SUPERVISOR_CONTROL_FD_ENV: str(control_write_fd)}
            with mock.patch.object(
                os, "open", side_effect=fake_open
            ), mock.patch.object(
                os, "_exit", side_effect=SystemExit
            ) as exit_mock, mock.patch.object(
                os, "pipe", return_value=(wakeup_read_fd, wakeup_write_fd)
            ), mock.patch.object(
                signal, "signal", side_effect=fake_signal
            ), mock.patch.object(
                signal, "set_wakeup_fd", return_value=-1
            ) as set_wakeup_fd, mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaises(SystemExit):
                    namespace["main"]()
            exit_mock.assert_called_once_with(126)
            message = os.read(control_read_fd, 256).decode("ascii")
            self.assertIn("ERROR", message)
            self.assertIn("CONFIG_PROC_CHILDREN", message)
            # main() must route its handler and wakeup-fd installation
            # through the mocks above rather than the real signal module,
            # so this test process's own signal state is left untouched
            # (confirmed independently by tearDown's snapshot check).
            installed_signums = {signum for signum, _ in signal_calls}
            self.assertEqual(
                installed_signums,
                {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGCHLD},
            )
            set_wakeup_fd.assert_called_once_with(
                wakeup_write_fd, warn_on_full_buffer=False
            )
        finally:
            os.close(control_read_fd)
            os.close(control_write_fd)
            os.close(wakeup_read_fd)
            os.close(wakeup_write_fd)

    def test_has_children_support_reflects_the_probe_file(self) -> None:
        namespace = _load_supervisor_namespace()
        self.assertTrue(namespace["has_children_support"]())

        pid = os.getpid()
        missing_path = f"/proc/{pid}/task/{pid}/children"
        real_open = os.open

        def fake_open(path, flags, *args, **kwargs):
            if path == missing_path:
                raise FileNotFoundError(path)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(os, "open", side_effect=fake_open):
            self.assertFalse(namespace["has_children_support"]())

    def test_direct_children_enumerates_every_thread_not_only_the_leader(self) -> None:
        namespace = _load_supervisor_namespace()
        pid = 987654
        task_root = f"/proc/{pid}/task"
        children_by_path = {
            f"/proc/{pid}/task/{pid}/children": b"111 222 \n",
            f"/proc/{pid}/task/222222/children": b"333\n",
        }
        real_listdir = os.listdir
        real_open = os.open
        real_read = os.read
        real_close = os.close
        fd_paths: dict[int, str] = {}
        next_fd = 900_000

        def fake_listdir(path):
            if path == task_root:
                return [str(pid), "222222"]
            return real_listdir(path)

        def fake_open(path, flags, *args, **kwargs):
            nonlocal next_fd
            if path in children_by_path:
                fd = next_fd
                next_fd += 1
                fd_paths[fd] = path
                return fd
            return real_open(path, flags, *args, **kwargs)

        def fake_read(fd, size):
            if fd in fd_paths:
                return children_by_path[fd_paths[fd]]
            return real_read(fd, size)

        def fake_close(fd):
            if fd in fd_paths:
                del fd_paths[fd]
                return
            real_close(fd)

        with mock.patch.object(
            os, "listdir", side_effect=fake_listdir
        ), mock.patch.object(os, "open", side_effect=fake_open), mock.patch.object(
            os, "read", side_effect=fake_read
        ), mock.patch.object(
            os, "close", side_effect=fake_close
        ):
            children = namespace["direct_children"](pid)

        self.assertEqual(sorted(children), [111, 222, 333])

    def test_kill_target_directly_signals_the_pid_and_its_escaped_process_group(
        self,
    ) -> None:
        # A pgid distinct from the supervisor's own group models a target
        # that called setsid, forming its own session/group.
        namespace = _load_supervisor_namespace()
        calls: list[tuple[str, int, int]] = []

        with mock.patch.object(
            os, "getpgid", return_value=555
        ), mock.patch.object(os, "getpgrp", return_value=1), mock.patch.object(
            os, "kill", side_effect=lambda pid, sig: calls.append(("kill", pid, sig))
        ), mock.patch.object(
            os, "killpg", side_effect=lambda pgid, sig: calls.append(("killpg", pgid, sig))
        ):
            namespace["kill_target_directly"](4242)

        self.assertIn(("kill", 4242, signal.SIGKILL), calls)
        self.assertIn(("killpg", 555, signal.SIGKILL), calls)

    def test_kill_target_directly_does_not_killpg_its_own_shared_group(self) -> None:
        # The common case: the target has not called setsid, so its pgid is
        # the supervisor's own. killpg-ing that group would kill the
        # supervisor itself before terminate_descendants() can walk /proc for
        # descendants that escaped into a different session, as the
        # escaped-process-group and reparented-double-fork-descendant tests
        # above cover.
        namespace = _load_supervisor_namespace()
        with mock.patch.object(
            os, "getpgid", return_value=777
        ), mock.patch.object(os, "getpgrp", return_value=777), mock.patch.object(
            os, "kill"
        ) as kill, mock.patch.object(os, "killpg") as killpg:
            namespace["kill_target_directly"](4242)

        kill.assert_called_once_with(4242, signal.SIGKILL)
        killpg.assert_not_called()

    def test_kill_target_directly_tolerates_an_already_reaped_target(self) -> None:
        namespace = _load_supervisor_namespace()
        with mock.patch.object(
            os, "getpgid", side_effect=ProcessLookupError
        ), mock.patch.object(
            os, "kill", side_effect=ProcessLookupError
        ), mock.patch.object(os, "killpg") as killpg:
            namespace["kill_target_directly"](4242)  # must not raise
        killpg.assert_not_called()

    def test_stop_path_kills_target_directly_before_descendant_cleanup(self) -> None:
        namespace = _load_supervisor_namespace()
        order: list[str] = []
        namespace["kill_target_directly"] = lambda pid, pidfd: order.append(
            f"kill:{pid}:{pidfd}"
        )
        namespace["terminate_descendants"] = lambda: order.append("terminate") or True
        target = SimpleNamespace(pid=4242, returncode=None)

        with mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock:
            with self.assertRaises(SystemExit):
                namespace["handle_stop_request"](target, 99)

        self.assertEqual(order, ["kill:4242:99", "terminate"])
        exit_mock.assert_called_once_with(124)

    def test_supervisor_reports_its_exit_origin_on_the_control_channel(
        self,
    ) -> None:
        # After the PID handshake the supervisor reports whether its exit
        # status is the target's own or its own.
        cases = {
            "stop": ("EXIT supervisor 124\n", 124),
            "stop_cleanup_failure": (
                "ERROR cleanup after stop request failed: survived\n"
                "EXIT supervisor 126\n",
                126,
            ),
            "target_exit": ("EXIT target 126\n", 126),
        }
        for case, (expected, status) in cases.items():
            with self.subTest(case=case):
                namespace = _load_supervisor_namespace()
                control_read, control_write = os.pipe()
                namespace["control_channel_fd"] = control_write
                namespace["kill_target_directly"] = lambda pid, pidfd: None

                def failing_terminate() -> None:
                    raise RuntimeError("survived")

                namespace["terminate_descendants"] = (
                    failing_terminate if case == "stop_cleanup_failure" else (lambda: True)
                )
                target = SimpleNamespace(pid=4242, returncode=None)
                try:
                    with mock.patch.object(
                        os, "_exit", side_effect=SystemExit
                    ) as exit_mock:
                        with self.assertRaises(SystemExit):
                            if case == "target_exit":
                                namespace["finish"](126)
                            else:
                                namespace["handle_stop_request"](target, 99)
                    exit_mock.assert_called_once_with(status)
                    os.set_blocking(control_read, False)
                    try:
                        reported = os.read(control_read, 4096).decode("ascii")
                    except BlockingIOError:
                        reported = ""
                    self.assertEqual(reported, expected)
                finally:
                    os.close(control_read)
                    os.close(control_write)

    def test_stop_path_cleanup_failure_exits_once_without_a_second_full_pass(
        self,
    ) -> None:
        namespace = _load_supervisor_namespace()
        terminate_calls: list[int] = []

        def failing_terminate() -> None:
            terminate_calls.append(1)
            raise RuntimeError("process descendants survived bounded cleanup")

        namespace["kill_target_directly"] = lambda pid, pidfd: None
        namespace["terminate_descendants"] = failing_terminate
        target = SimpleNamespace(pid=4242, returncode=None)

        with mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock:
            with self.assertRaises(SystemExit):
                namespace["handle_stop_request"](target, 99)

        # The stop path must not retry the (already exhausted) cleanup budget
        # a second time.
        self.assertEqual(len(terminate_calls), 1)
        exit_mock.assert_called_once_with(126)

    def test_stop_after_reap_signals_neither_the_old_pid_nor_its_group(self) -> None:
        # Once Popen.poll() has reaped the target (target.returncode is
        # set), the kernel is free to recycle its pid for an unrelated
        # process. A stop request that arrives
        # after that point must send no signal at all to that pid, its
        # pgid, or via its pidfd -- only terminate_descendants() (the
        # /proc-walk orphan cleanup) may still run.
        namespace = _load_supervisor_namespace()
        target = SimpleNamespace(pid=4242, returncode=0)
        namespace["terminate_descendants"] = lambda: True

        with mock.patch.object(os, "getpgid") as getpgid, mock.patch.object(
            os, "kill"
        ) as kill, mock.patch.object(os, "killpg") as killpg, mock.patch.object(
            signal, "pidfd_send_signal", create=True
        ) as pidfd_send_signal, mock.patch.object(
            os, "_exit", side_effect=SystemExit
        ) as exit_mock:
            with self.assertRaises(SystemExit):
                namespace["handle_stop_request"](target, 99)

        getpgid.assert_not_called()
        kill.assert_not_called()
        killpg.assert_not_called()
        pidfd_send_signal.assert_not_called()
        exit_mock.assert_called_once_with(124)

    def test_kill_target_directly_prefers_pidfd_send_signal_when_available(
        self,
    ) -> None:
        # When a pidfd is available, it must be preferred over
        # os.kill(pid, ...): it targets the exact process the pidfd was
        # opened for, so it cannot hit a process that has since reused a
        # recycled pid.
        namespace = _load_supervisor_namespace()
        with mock.patch.object(
            os, "getpgid", return_value=777
        ), mock.patch.object(os, "getpgrp", return_value=777), mock.patch.object(
            os, "kill"
        ) as kill, mock.patch.object(
            signal, "pidfd_send_signal", create=True
        ) as pidfd_send_signal:
            namespace["kill_target_directly"](4242, 99)

        pidfd_send_signal.assert_called_once_with(99, signal.SIGKILL)
        kill.assert_not_called()

    def test_kill_target_directly_falls_back_to_kill_when_pidfd_signal_fails(
        self,
    ) -> None:
        namespace = _load_supervisor_namespace()
        with mock.patch.object(
            os, "getpgid", return_value=777
        ), mock.patch.object(os, "getpgrp", return_value=777), mock.patch.object(
            os, "kill"
        ) as kill, mock.patch.object(
            signal, "pidfd_send_signal", create=True, side_effect=OSError
        ):
            namespace["kill_target_directly"](4242, 99)

        kill.assert_called_once_with(4242, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
