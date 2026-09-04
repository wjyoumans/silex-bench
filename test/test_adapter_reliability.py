from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from silex_bench import campaign
from silex_bench.backends.hecke import HeckeBackend
from silex_bench.backends.pari import PariBackend
from silex_bench.backends.silex import SILEX_BENCHMARK_CMAKE_CACHE_REQUIREMENTS
from silex_bench.contracts import InvocationContext
from silex_bench.model import BackendContext
from silex_bench.registry import _sunit_arguments, builtin_registry
from silex_bench.resources import builtin_path
from silex_bench.workloads import sunit_module


def backend_context(root: Path) -> BackendContext:
    return BackendContext(
        workspace=root,
        bench_root=root,
        silex_source=root,
        silex_build_dir=root,
        tools={},
        timeout_seconds=10.0,
        cpu=None,
        primary_clock="cpu",
    )


def process_result(
    *, success: bool, stdout: str = "", stderr: str = ""
) -> dict[str, object]:
    return {
        "available": True,
        "success": success,
        "timeout": False,
        "stdout": stdout,
        "stderr": stderr,
        "process_wall_ms": 1.0,
        "executable_sha256": "a" * 64,
    }


class SilexBuildReliabilityTests(unittest.TestCase):
    @staticmethod
    def _invocation_context(
        root: Path, selected_workloads: tuple[str, ...]
    ) -> InvocationContext:
        return InvocationContext(
            bench_root=root,
            workspace=root,
            silex_source=root,
            silex_build_dir=root,
            tools={},
            timeout_seconds=1.0,
            cpu=None,
            environment={},
            selected_workloads=selected_workloads,
        )

    @staticmethod
    def _write_cache(root: Path) -> None:
        (root / "CMakeCache.txt").write_text(
            "".join(
                f"{key}={value}\n"
                for key, value in SILEX_BENCHMARK_CMAKE_CACHE_REQUIREMENTS
            )
        )

    @staticmethod
    def _write_executable(root: Path, name: str, *, executable: bool = True) -> None:
        path = root / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o700 if executable else 0o600)

    def test_prepare_silex_sets_every_probe_required_option(self) -> None:
        required = {
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSILEX_BUILD_BENCHMARK_ADAPTERS=ON",
            "-DSILEX_BUILD_TESTS=OFF",
            "-DSILEX_BUILD_BENCHMARKS=OFF",
            "-DSILEX_BUILD_EXAMPLES=OFF",
            "-DSILEX_BUILD_DOCS=OFF",
            "-DSILEX_ENABLE_LOGGING=OFF",
            "-DSILEX_ENABLE_DEBUG_CHECKS=OFF",
            "-DSILEX_ENABLE_PROFILING=OFF",
            "-DSILEX_ENABLE_SANITIZERS=OFF",
            "-DSILEX_ENABLE_FRAME_POINTERS=OFF",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "silex"
            source.mkdir()
            build = root / "build"
            context = SimpleNamespace(
                silex_source=source,
                silex_build_dir=build,
            )
            plan = SimpleNamespace(backends=("silex",))
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with mock.patch.object(
                campaign, "invocation_context", return_value=context
            ), mock.patch.object(
                campaign.subprocess, "run", return_value=completed
            ):
                commands = campaign.prepare_silex(plan)

        self.assertEqual(len(commands), 2)
        self.assertTrue(required.issubset(set(commands[0])))

    def test_operation_only_probe_does_not_require_class_unit_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root)
            self._write_executable(root, "silex-operation-instance")

            probe = builtin_registry().backends["silex"].probe(
                self._invocation_context(root, ("maximal_order",))
            )

        self.assertTrue(probe.available)
        self.assertIsNone(
            probe.identity["executable_sha256"]["class_unit_executable"]
        )
        self.assertRegex(
            probe.identity["executable_sha256"]["operation_executable"],
            r"\A[0-9a-f]{64}\Z",
        )

    def test_full_probe_requires_both_silex_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root)
            self._write_executable(root, "silex-class-unit-instance")
            capabilities = builtin_registry().backends["silex"].capabilities

            probe = builtin_registry().backends["silex"].probe(
                self._invocation_context(root, capabilities)
            )

        self.assertFalse(probe.available)
        self.assertIn("silex-operation-instance", probe.error or "")

    def test_sunit_only_probe_uses_class_unit_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root)
            self._write_executable(root, "silex-class-unit-instance")

            probe = builtin_registry().backends["silex"].probe(
                self._invocation_context(root, ("sunit_proven",))
            )

        self.assertTrue(probe.available)
        self.assertRegex(
            probe.identity["executable_sha256"]["class_unit_executable"],
            r"\A[0-9a-f]{64}\Z",
        )
        self.assertIsNone(
            probe.identity["executable_sha256"]["operation_executable"]
        )

    def test_probe_rejects_nonexecutable_required_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root)
            self._write_executable(
                root, "silex-class-unit-instance", executable=False
            )

            probe = builtin_registry().backends["silex"].probe(
                self._invocation_context(root, ("class_unit_proven",))
            )

        self.assertFalse(probe.available)
        self.assertIn("not executable", probe.error or "")


class PariReliabilityTests(unittest.TestCase):
    def test_probe_does_not_adopt_home_directory_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version_file = root / "pari-2.17.3" / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            context = backend_context(root)
            with mock.patch(
                "silex_bench.backends.pari.Path.home", return_value=root
            ), mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value="/usr/bin/gp",
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value=process_result(success=True, stdout="2.17.3\n"),
            ):
                probe = PariBackend().probe(context)

        self.assertTrue(probe["available"])
        self.assertIsNone(probe["engine_identity"]["required_version"])
        self.assertIsNone(probe["engine_identity"]["source"])
        self.assertIsNone(probe["engine_identity"]["source_version"])

    def test_explicit_version_without_source_still_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = backend_context(root)
            context.tools.update({"gp": "gp", "pari_version": "2.17.3"})
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value="/usr/bin/gp",
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value=process_result(success=True, stdout="2.17.4\n"),
            ):
                probe = PariBackend().probe(context)

        self.assertFalse(probe["available"])
        self.assertIn("does not match", probe["error"])


class HeckeReliabilityTests(unittest.TestCase):
    def test_missing_hecke_in_active_environment_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = backend_context(Path(temporary))
            with mock.patch(
                "silex_bench.backends.hecke._resolve_julia",
                return_value=("/usr/bin/julia", "julia"),
            ), mock.patch(
                "silex_bench.backends.hecke.run_process",
                return_value=process_result(
                    success=False,
                    stderr="ArgumentError: Package Hecke not found in current path.",
                ),
            ):
                probe = HeckeBackend().probe(context)

        self.assertFalse(probe["available"])
        self.assertIn("active Julia environment", probe["error"])
        self.assertIn("Pkg.add", probe["error"])

    def test_uninstantiated_configured_project_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "Hecke.jl"
            project.mkdir()
            (project / "Project.toml").write_text("name = \"Hecke\"\n")
            context = backend_context(root)
            context.tools["hecke_project"] = str(project)
            with mock.patch(
                "silex_bench.backends.hecke._resolve_julia",
                return_value=("/usr/bin/julia", "julia"),
            ), mock.patch(
                "silex_bench.backends.hecke.run_process",
                return_value=process_result(
                    success=False,
                    stderr=(
                        "Package Hecke required but does not seem to be installed"
                    ),
                ),
            ):
                probe = HeckeBackend().probe(context)

        self.assertFalse(probe["available"])
        self.assertIn("Pkg.instantiate", probe["error"])
        self.assertIn(str(project), probe["error"])


class SUnitAdapterReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.module = sunit_module()
        cls.row = cls.module.load_field(
            builtin_path("corpora", "sunit_fields"),
            "cubic_x3_minus_2_empty_s",
        )

    def test_pari_path_only_execution_does_not_require_source(self) -> None:
        calls: list[list[str]] = []

        def run_process(command: list[str], **_kwargs: object) -> dict[str, object]:
            calls.append(command)
            if "--version-short" in command:
                return process_result(success=True, stdout="2.17.3\n")
            return process_result(success=False, stderr="fixture stopped")

        args = SimpleNamespace(
            gp="gp",
            pari_source=None,
            pari_version=None,
            timeout=1.0,
            cpu=None,
            environment={},
        )
        with mock.patch.object(
            self.module.shutil, "which", return_value="/usr/bin/gp"
        ), mock.patch.object(self.module, "run_process", side_effect=run_process):
            result = self.module.run_pari(args, self.row)

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["failure_stage"], "external_execution")
        self.assertIsNone(result["engine_identity"]["required_version"])
        self.assertIsNone(result["engine_identity"]["source"])
        self.assertIsNone(result["engine_identity"]["source_version"])

    def test_pari_path_only_identity_is_valid_for_agreement(self) -> None:
        result = {
            "engine": "pari",
            "engine_identity": {
                "executable": "/usr/bin/gp",
                "executable_sha256": "a" * 64,
                "version": "2.17.3",
                "required_version": None,
                "source": None,
                "source_version": None,
            },
        }

        self.assertTrue(
            self.module.standalone_engine_identity_is_valid("pari", result)
        )

    def test_pari_explicit_source_version_mismatch_still_fails(self) -> None:
        args = SimpleNamespace(
            gp="gp",
            pari_source=Path("/tmp/pari-source"),
            pari_version="2.17.3",
            timeout=1.0,
            cpu=None,
            environment={},
        )
        with mock.patch.object(
            self.module.shutil, "which", return_value="/usr/bin/gp"
        ), mock.patch.object(
            self.module, "pari_source_version", return_value="2.17.3"
        ), mock.patch.object(
            self.module,
            "run_process",
            return_value=process_result(success=True, stdout="2.17.4\n"),
        ) as run:
            result = self.module.run_pari(args, self.row)

        self.assertEqual(run.call_count, 1)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "engine_provenance")
        self.assertIn("does not match", result["failure_reason"])

    def test_integrated_arguments_do_not_invent_tool_provenance(self) -> None:
        context = InvocationContext(
            bench_root=self.root,
            workspace=self.root.parent,
            silex_source=self.root.parent / "silex",
            silex_build_dir=self.root.parent / "silex" / "build",
            tools={},
            timeout_seconds=1.0,
            cpu=None,
            environment={},
        )

        args = _sunit_arguments(context)

        self.assertIsNone(args.gp)
        self.assertIsNone(args.pari_version)
        self.assertIsNone(args.pari_source)
        self.assertIsNone(args.julia)
        self.assertIsNone(args.hecke_project)

    def test_hecke_path_only_execution_uses_standard_julia_flags(self) -> None:
        commands: list[list[str]] = []

        def run_process(command: list[str], **_kwargs: object) -> dict[str, object]:
            commands.append(command)
            return process_result(success=False, stderr="fixture stopped")

        with tempfile.TemporaryDirectory() as temporary:
            julia = Path(temporary) / "julia"
            julia.write_text("fixture")
            args = SimpleNamespace(
                julia=None,
                hecke_project=None,
                timeout=1.0,
                cpu=None,
                environment={},
            )
            with mock.patch.object(
                self.module.shutil, "which", return_value=str(julia)
            ), mock.patch.object(
                self.module, "run_process", side_effect=run_process
            ):
                result = self.module.run_hecke(args, self.row)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0], str(julia))
        self.assertIn("--startup-file=no", commands[0])
        self.assertIn("--history-file=no", commands[0])
        self.assertIn("--threads=1", commands[0])
        self.assertFalse(any(item.startswith("--project=") for item in commands[0]))
        self.assertEqual(result["failure_stage"], "external_execution")


if __name__ == "__main__":
    unittest.main()
