from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

from silex_bench.process import (
    MAX_PROTOCOL_INPUT_BYTES,
    TARGET_NONCE_PLACEHOLDER,
    parse_key_values,
    run_marked_process,
    run_process,
)


class ProcessTests(unittest.TestCase):
    def test_process_error_classification_precedes_cleanup(self) -> None:
        now = [0.0]
        supervisor = mock.Mock()

        def cleanup(process):
            now[0] = 11.0
            return -15, b"", b""

        with mock.patch("silex_bench.process.time.monotonic", side_effect=lambda: now[0]), mock.patch(
            "silex_bench.process._supervised_popen", return_value=(supervisor, 12345)
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
            "silex_bench.process._supervised_popen", return_value=(supervisor, 12345)
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
            return_value=(supervisor, 12345),
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
        child = """
import sys
sys.stdin.readline()
print("READY", flush=True)
nonce = sys.stdin.readline().strip()
print("TARGET:" + nonce, flush=True)
sys.stdin.readline()
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
                timeout=2.0,
                cwd=Path(temporary),
            )
            self.assertTrue(result["available"])
            self.assertTrue(result["success"])
            self.assertFalse(result["timeout"])
            self.assertIsNotNone(result["target_wall_ms"])
            self.assertGreaterEqual(result["target_wall_ms"], 0.0)
            self.assertEqual(parse_key_values(result["stdout"])["answer"], "42")

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
            self.assertLess(result["process_wall_ms"], 800.0)

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


if __name__ == "__main__":
    unittest.main()
