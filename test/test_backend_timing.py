from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from silex_bench.backends.hecke import HeckeBackend
from silex_bench.backends.magma import MagmaBackend
from silex_bench.backends.pari import (
    PariBackend,
    _parse_invariants,
    _programs as pari_programs,
    _source_release,
)
from silex_bench.backends.silex import SilexBackend
from silex_bench.model import (
    BackendContext,
    FieldSpec,
    SampleRequest,
    engine_probe_contract_errors,
)
from silex_bench.registry import add_executable_digests


MARKED_PROCESS_AFFINITY = [7]
PROBE_KEYS = {
    "engine",
    "available",
    "success",
    "timeout",
    "status",
    "error",
    "engine_identity",
}


def field(identifier: str, constant: int) -> FieldSpec:
    return FieldSpec(identifier, (constant, 0, 1), 2, "unit-test")


def request(
    *, warm: bool = True, operation: str = "ideal_multiply"
) -> SampleRequest:
    return SampleRequest(
        field=field("target", -5),
        operation=operation,
        sample_kind="warm_algorithm" if warm else "cold_process",
        sample_index=0,
        warmup=field("warmup", 47) if warm else None,
        seed=7,
    )


def context(root: Path) -> BackendContext:
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


def silex_cache_bytes() -> bytes:
    return (
        "\n".join(
            (
                "CMAKE_BUILD_TYPE:STRING=Release",
                "SILEX_BUILD_BENCHMARK_ADAPTERS:BOOL=ON",
                "SILEX_BUILD_TESTS:BOOL=OFF",
                "SILEX_BUILD_BENCHMARKS:BOOL=OFF",
                "SILEX_BUILD_EXAMPLES:BOOL=OFF",
                "SILEX_BUILD_DOCS:BOOL=OFF",
                "SILEX_ENABLE_LOGGING:BOOL=OFF",
                "SILEX_ENABLE_DEBUG_CHECKS:BOOL=OFF",
                "SILEX_ENABLE_PROFILING:BOOL=OFF",
                "SILEX_ENABLE_SANITIZERS:BOOL=OFF",
                "SILEX_ENABLE_FRAME_POINTERS:BOOL=OFF",
            )
        )
        + "\n"
    ).encode("utf-8")


def marked_result(stdout: str) -> dict[str, object]:
    return {
        "available": True,
        "success": True,
        "timeout": False,
        "target_cpu_ms": 11.0,
        "target_wall_ms": 12.0,
        "process_wall_ms": 20.0,
        "effective_affinity": MARKED_PROCESS_AFFINITY,
        "cmd": ["engine"],
        "stdout": stdout,
        "stderr": "",
    }


def class_unit_request() -> SampleRequest:
    return SampleRequest(
        field=field("target", -5),
        operation="class_unit_proven",
        sample_kind="cold_process",
        sample_index=0,
        warmup=None,
        seed=7,
    )


def class_unit_payload() -> dict[str, object]:
    return {
        "success": True,
        "timeout": False,
        "engine_thread_count": 1,
        "equation_order_index": "3",
        "failure_stage": None,
        "failure_reason": None,
        "expectations_passed": True,
        "final_result_published": True,
        "certification_status": "proven",
        "class_group_proof_status": "proven",
        "unit_group_proof_status": "proven",
        "regulator_proof_status": "verified",
        "class_group": {"order": "1", "invariants": []},
        "unit_group": {"free_rank": 1},
        "signature": [2, 0],
        "maximal_order_discriminant": "5",
        "measurement_timing": {
            "target_cpu_ms": 1.0,
            "target_wall_ms": 2.0,
        },
        "component_timing_ms": {"zeta": 0.5},
    }


def operation_payload() -> dict[str, object]:
    return {
        "success": True,
        "engine_thread_count": 1,
        "target_cpu_ms": 1.0,
        "target_wall_ms": 1.0,
        "ideal_norm": "36",
        "timing_scope": "ideal_multiplication_only",
        "timing_clock": {
            "cpu": "std_clock_process_cpu",
            "wall": "steady_clock",
        },
    }


class ProgramTimingTests(unittest.TestCase):
    def test_pari_cyclic_class_invariants_are_normalized(self) -> None:
        self.assertEqual(_parse_invariants("[5]"), ["5"])
        self.assertEqual(_parse_invariants("[]"), [])

    def test_pari_total_clocks_wrap_each_target(self) -> None:
        for operation in (
            "class_unit_proven",
            "maximal_order",
            "ideal_multiply",
            "element_square_root",
        ):
            with self.subTest(operation=operation):
                sample = SampleRequest(
                    field=field("target", -5),
                    operation=operation,
                    sample_kind="cold_process",
                    sample_index=0,
                    warmup=None,
                    seed=7,
                )
                ready, target, final = pari_programs(sample)
                self.assertIn("default(nbthreads, 1);", ready)
                self.assertIn(
                    "benchmark_reported_threads = default(nbthreads);",
                    ready,
                )
                self.assertIn("target_cpu_start_ms = getabstime();", target)
                self.assertIn("target_wall_start_ms = getwalltime();", target)
                self.assertIn(
                    "target_internal_cpu_ms = getabstime() - target_cpu_start_ms;",
                    target,
                )
                self.assertIn(
                    "target_internal_wall_ms = getwalltime() - target_wall_start_ms;",
                    target,
                )
                self.assertLess(
                    target.index("target_internal_wall_ms ="),
                    target.index("__SILEX_BENCH_PARI_TARGET_DONE__"),
                )
                self.assertIn('print("target_internal_cpu_ms="', final)
                self.assertIn('print("target_internal_wall_ms="', final)
                self.assertIn('print("reported_threads="', final)

        _, class_target, _ = pari_programs(
            SampleRequest(
                field=field("target", -5),
                operation="class_unit_proven",
                sample_kind="cold_process",
                sample_index=0,
                warmup=None,
                seed=7,
            )
        )
        self.assertLess(
            class_target.index("P = x^2 - 5;"),
            class_target.index("target_cpu_start_ms ="),
        )


class AdapterTimingTests(unittest.TestCase):
    def assert_persistable_probe(
        self,
        probe: dict[str, object],
        backend: str,
        *,
        selected_operations: tuple[str, ...] = (
            "class_unit_proven",
            "maximal_order",
        ),
    ) -> None:
        enriched = add_executable_digests(
            probe, selected_workloads=selected_operations
        )
        self.assertEqual(set(enriched), PROBE_KEYS)
        self.assertEqual(
            engine_probe_contract_errors(
                enriched,
                expected_backend=backend,
                selected_operations=selected_operations,
            ),
            [],
        )

    def test_live_probe_producers_emit_canonical_persistable_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class_unit = root / "silex-class-unit-instance"
            operation = root / "silex-operation-instance"
            gp = root / "gp"
            julia = root / "julia"
            magma = root / "magma"
            for executable in (class_unit, operation, gp, julia, magma):
                executable.write_bytes(executable.name.encode())
                executable.chmod(0o700)
            (root / "CMakeCache.txt").write_bytes(silex_cache_bytes())
            pari_source = root / "pari-source"
            pari_version = pari_source / "config" / "version"
            pari_version.parent.mkdir(parents=True)
            pari_version.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(gp),
                    "pari_source": str(pari_source),
                    "pari_version": "2.17.3",
                }
            )

            self.assert_persistable_probe(
                SilexBackend().probe(probe_context), "silex"
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(gp),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                self.assert_persistable_probe(
                    PariBackend().probe(probe_context), "pari"
                )
            with mock.patch(
                "silex_bench.backends.hecke._resolve_julia",
                return_value=(str(julia), str(julia)),
            ), mock.patch(
                "silex_bench.backends.hecke._project",
                return_value=(None, None),
            ), mock.patch(
                "silex_bench.backends.hecke.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": (
                        "julia_version=1.12.6\n"
                        "hecke_version=0.39.19\n"
                        "hecke_source=/tmp/Hecke.jl\n"
                    ),
                },
            ):
                self.assert_persistable_probe(
                    HeckeBackend().probe(probe_context), "hecke"
                )
            with mock.patch(
                "silex_bench.backends.magma._resolve_magma",
                return_value=(str(magma), None),
            ), mock.patch(
                "silex_bench.backends.magma.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "magma_version=2.28-16\n",
                },
            ):
                self.assert_persistable_probe(
                    MagmaBackend().probe(probe_context), "magma"
                )

    def test_silex_probe_rejects_symlinked_cmake_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("silex-class-unit-instance", "silex-operation-instance"):
                executable = root / name
                executable.write_bytes(name.encode("utf-8"))
                executable.chmod(0o700)
            target = root / "cache-target"
            target.write_bytes(silex_cache_bytes())
            (root / "CMakeCache.txt").symlink_to(target)

            probe = SilexBackend().probe(context(root))

        self.assertFalse(probe["available"])
        self.assertIn("regular file", probe["error"])

    def test_silex_probe_rejects_oversized_cmake_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("silex-class-unit-instance", "silex-operation-instance"):
                executable = root / name
                executable.write_bytes(name.encode("utf-8"))
                executable.chmod(0o700)
            (root / "CMakeCache.txt").write_bytes(
                silex_cache_bytes() + b"#" * (1 << 20)
            )

            probe = SilexBackend().probe(context(root))

        self.assertFalse(probe["available"])
        self.assertIn("1048576-byte size limit", probe["error"])

    def test_silex_probe_rejects_duplicate_required_cmake_cache_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("silex-class-unit-instance", "silex-operation-instance"):
                executable = root / name
                executable.write_bytes(name.encode("utf-8"))
                executable.chmod(0o700)
            (root / "CMakeCache.txt").write_bytes(
                b"CMAKE_BUILD_TYPE:STRING=Debug\n" + silex_cache_bytes()
            )

            probe = SilexBackend().probe(context(root))

        self.assertFalse(probe["available"])
        self.assertIn("duplicate", probe["error"])
        self.assertIn("CMAKE_BUILD_TYPE:STRING", probe["error"])

    def test_silex_probe_rejects_required_cmake_variable_under_two_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("silex-class-unit-instance", "silex-operation-instance"):
                executable = root / name
                executable.write_bytes(name.encode("utf-8"))
                executable.chmod(0o700)
            (root / "CMakeCache.txt").write_bytes(
                silex_cache_bytes() + b"CMAKE_BUILD_TYPE:INTERNAL=Debug\n"
            )

            probe = SilexBackend().probe(context(root))

        self.assertFalse(probe["available"])
        self.assertIn("duplicate", probe["error"])
        self.assertIn("CMAKE_BUILD_TYPE", probe["error"])

    def test_unavailable_probe_producers_emit_canonical_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            probe_context = context(Path(temporary))
            probes: list[tuple[str, dict[str, object]]] = [
                ("silex", SilexBackend().probe(probe_context)),
            ]
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable", return_value=None
            ):
                probes.append(("pari", PariBackend().probe(probe_context)))
            with mock.patch(
                "silex_bench.backends.hecke._resolve_julia",
                return_value=(None, "missing-julia"),
            ):
                probes.append(("hecke", HeckeBackend().probe(probe_context)))
            with mock.patch(
                "silex_bench.backends.magma._resolve_magma",
                return_value=(None, "missing-magma"),
            ):
                probes.append(("magma", MagmaBackend().probe(probe_context)))

            for backend, probe in probes:
                with self.subTest(backend=backend):
                    self.assert_persistable_probe(probe, backend)

    def test_pari_probe_rejects_executable_source_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(source),
                    "pari_version": "2.17.3",
                }
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.4\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertFalse(probe["available"])
        self.assertIn("does not match", probe["error"])
        self.assertEqual(probe["engine_identity"]["version"], "2.17.4")
        self.assertEqual(
            probe["engine_identity"]["source_version"], "2.17.3"
        )
        self.assertEqual(
            probe["engine_identity"]["required_version"], "2.17.3"
        )

    def test_pari_probe_accepts_exact_executable_source_version_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(source),
                    "pari_version": "2.17.3",
                }
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertTrue(probe["available"])
        self.assertIsNone(probe["error"])
        self.assertEqual(probe["engine_identity"]["version"], "2.17.3")
        self.assertEqual(
            probe["engine_identity"]["source_version"], "2.17.3"
        )
        self.assertEqual(
            len(probe["engine_identity"]["source_version_file_sha256"]), 64
        )

    def test_pari_source_release_rejects_duplicate_version_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='9'\n"
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
            )

            with self.assertRaisesRegex(
                ValueError, "duplicate.*VersionMajor"
            ):
                _source_release(source)

    def test_pari_source_release_rejects_semicolon_hidden_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='9'; VersionMajor='8'\n"
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
            )

            with self.assertRaisesRegex(ValueError, "VersionMajor"):
                _source_release(source)

    def test_pari_source_release_rejects_alternate_assignment_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\n"
                "VersionMajor+='9'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
            )

            with self.assertRaisesRegex(ValueError, "VersionMajor"):
                _source_release(source)

    def test_pari_source_release_rejects_continued_assignment_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMaj\\\n"
                "or='9'\n"
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
            )

            with self.assertRaisesRegex(ValueError, "VersionMajor"):
                _source_release(source)

    def test_pari_source_release_rejects_indirect_required_key_assignment(self) -> None:
        evasions = (
            "version_key=VersionMajor\neval \"${version_key}='9'\"\n",
            "printf -v VersionMajor '%s' 9\n",
            "read -r VersionMajor <<<'9'\n",
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "pari-source"
                version_file = source / "config" / "version"
                version_file.parent.mkdir(parents=True)
                version_file.write_text(
                    evasion
                    + "VersionMajor='2'\n"
                    + "VersionMinor='17'\n"
                    + "patch='3'\n"
                )

                with self.assertRaisesRegex(ValueError, "ambiguous.*VersionMajor"):
                    _source_release(source)

    def test_pari_source_release_rejects_escaped_indirect_assignment(self) -> None:
        escaped_key = f"Version{chr(92)}115inor"
        escaped_eval = f"e{chr(92)}val"
        escaped_printf = f"p{chr(92)}rintf"
        fragmented_eval = "e''val"
        fragmented_printf = 'p""rintf'
        ansi_eval = r"$'\145\166\141\154'"
        ansi_printf = r"$'\160\162\151\156\164\146'"
        ansi_key = r"\126\145\162\163\151\157\156\115\151\156\157\162"
        expanded_commands = (
            "a=e\n"
            "b=val\n"
            "p=pri\n"
            "q=ntf\n"
            f'"$a$b" "$("$p$q" \'{escaped_key}\')=\'8\'"\n'
        )
        dynamic_export = (
            "name=Version\n"
            'name="${name}Minor"\n'
            'export "$name=8"\n'
        )
        evasions = (
            f"eval \"$(printf '{escaped_key}')='8'\"\n",
            f"printf -v \"$(printf '{escaped_key}')\" '%s' 8\n",
            f"read -r \"$(printf '{escaped_key}')\" <<<'8'\n",
            (
                f"{escaped_eval} \"$("
                f"{escaped_printf} '{escaped_key}')='8'\"\n"
            ),
            (
                f"{fragmented_eval} \"$("
                f"{fragmented_printf} '{escaped_key}')='8'\"\n"
            ),
            f'{ansi_eval} "$({ansi_printf} \'{ansi_key}\')=\'8\'"\n',
            expanded_commands,
            dynamic_export,
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "pari-source"
                    version_file = source / "config" / "version"
                    version_file.parent.mkdir(parents=True)
                    version_file.write_text(
                        "VersionMajor='2'\n"
                        "VersionMinor='17'\n"
                        "patch='3'\n"
                        + evasion
                    )

                    with self.assertRaisesRegex(ValueError, "ambiguous"):
                        _source_release(source)

    def test_pari_source_release_rejects_direct_escaped_required_assignment(
        self,
    ) -> None:
        evasions = (
            f"VersionM{chr(92)}inor='8'\n",
            "VersionM''inor='8'\n",
            'VersionM""inor=\'8\'\n',
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "pari-source"
                version_file = source / "config" / "version"
                version_file.parent.mkdir(parents=True)
                version_file.write_text(
                    "VersionMajor='2'\n"
                    "VersionMinor='17'\n"
                    "patch='3'\n"
                    + evasion
                )

                with self.assertRaisesRegex(ValueError, "ambiguous.*VersionMinor"):
                    _source_release(source)

    def test_pari_source_release_rejects_computed_arithmetic_assignment(
        self,
    ) -> None:
        evasions = (
            ': "$(( $name = 8 ))"\n',
            'calculated="$(( $name = 8 ))"\n',
            'echo "$(( $name = 8 ))"\n',
        )
        for evasion in evasions:
            with self.subTest(evasion=evasion), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "pari-source"
                version_file = source / "config" / "version"
                version_file.parent.mkdir(parents=True)
                version_file.write_text(
                    "VersionMajor='2'\n"
                    "VersionMinor='17'\n"
                    "patch='3'\n"
                    "left=VersionM\n"
                    "right=inor\n"
                    'name="$left$right"\n'
                    + evasion
                )

                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    _source_release(source)

    def test_pari_source_release_allows_ordinary_variable_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
                'version="$VersionMajor.${VersionMinor}.$patch"\n'
            )

            self.assertEqual(_source_release(source)[0], "2.17.3")

    def test_pari_source_release_rejects_conditional_required_assignments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "if true; then\n"
                "VersionMajor='2'\n"
                "VersionMinor='17'\n"
                "patch='3'\n"
                "fi\n"
            )

            with self.assertRaisesRegex(ValueError, "noncanonical content"):
                _source_release(source)

    def test_backend_context_preserves_pari_symlink_for_nofollow_rejection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_source = root / "pari-real"
            version_file = real_source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            linked_source = root / "pari-linked"
            linked_source.symlink_to(real_source, target_is_directory=True)
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(linked_source),
                    "pari_version": "2.17.3",
                }
            )
            self.assertEqual(Path(probe_context.tools["pari_source"]), linked_source)
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertFalse(probe["available"])
        self.assertIn("source", probe["error"].lower())

    def test_silex_probe_rejects_configured_symlink_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_source = root / "silex-real"
            real_source.mkdir()
            linked_source = root / "silex-linked"
            linked_source.symlink_to(real_source, target_is_directory=True)
            real_build = root / "build-real"
            real_build.mkdir()
            (real_build / "CMakeCache.txt").write_bytes(silex_cache_bytes())
            operation = real_build / "silex-operation-instance"
            operation.write_text("fixture\n")
            operation.chmod(0o700)
            linked_build = root / "build-linked"
            linked_build.symlink_to(real_build, target_is_directory=True)

            cases = (
                ("source", linked_source, real_build),
                ("build", real_source, linked_build),
            )
            for label, source, build in cases:
                with self.subTest(label=label):
                    probe_context = context(root)
                    probe_context.silex_source = source
                    probe_context.silex_build_dir = build
                    probe_context.selected_operations = ("ideal_multiply",)
                    probe = SilexBackend().probe(probe_context)

                    self.assertFalse(probe["available"])
                    self.assertFalse(probe["success"])
                    self.assertIsInstance(probe["error"], str)
                    self.assertTrue(probe["error"])
                    self.assertIn(label, probe["error"].lower())

    def test_pari_probe_rejects_symlinked_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_source = root / "pari-real"
            version_file = real_source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            linked_source = root / "pari-linked"
            linked_source.symlink_to(real_source, target_is_directory=True)
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(linked_source),
                    "pari_version": "2.17.3",
                }
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertFalse(probe["available"])
        self.assertIn("source", probe["error"].lower())

    def test_pari_probe_binds_source_version_and_digest_to_one_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            original = b"VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            replacement = b"VersionMajor='2'\nVersionMinor='17'\npatch='4'\n"
            version_file.write_bytes(original)
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(source),
                    "pari_version": "2.17.3",
                }
            )

            def replace_after_snapshot(_path: Path, **_kwargs: object) -> bytes:
                version_file.write_bytes(replacement)
                return original

            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ), mock.patch(
                "silex_bench.backends.pari.read_bytes_nofollow",
                side_effect=replace_after_snapshot,
            ):
                probe = PariBackend().probe(probe_context)

        self.assertTrue(probe["available"])
        self.assertEqual(probe["engine_identity"]["source_version"], "2.17.3")
        self.assertEqual(
            probe["engine_identity"]["source_version_file_sha256"],
            hashlib.sha256(original).hexdigest(),
        )

    def test_pari_probe_rejects_symlinked_source_version_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            replacement = root / "replacement-version"
            replacement.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
            )
            version_file.symlink_to(replacement)
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(source),
                    "pari_version": "2.17.3",
                }
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertFalse(probe["available"])
        self.assertIn("regular file", probe["error"])

    def test_pari_probe_rejects_oversized_source_version_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_bytes(
                b"VersionMajor='2'\nVersionMinor='17'\npatch='3'\n"
                + b"#" * (64 << 10)
            )
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(source),
                    "pari_version": "2.17.3",
                }
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertFalse(probe["available"])
        self.assertIn("65536-byte size limit", probe["error"])

    def test_pari_probe_rejects_source_required_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pari-source"
            version_file = source / "config" / "version"
            version_file.parent.mkdir(parents=True)
            version_file.write_text(
                "VersionMajor='2'\nVersionMinor='17'\npatch='4'\n"
            )
            probe_context = context(root)
            probe_context.tools.update(
                {
                    "gp": str(root / "gp"),
                    "pari_source": str(source),
                    "pari_version": "2.17.3",
                }
            )
            with mock.patch(
                "silex_bench.backends.pari.resolve_executable",
                return_value=str(root / "gp"),
            ), mock.patch(
                "silex_bench.backends.pari.run_process",
                return_value={
                    "available": True,
                    "success": True,
                    "timeout": False,
                    "stdout": "2.17.3\n",
                },
            ):
                probe = PariBackend().probe(probe_context)

        self.assertFalse(probe["available"])
        self.assertIn("source version 2.17.4", probe["error"])
        self.assertEqual(probe["engine_identity"]["version"], "2.17.3")
        self.assertEqual(
            probe["engine_identity"]["source_version"], "2.17.4"
        )

    def test_pari_internal_zero_is_authoritative_and_marked_is_audit(self) -> None:
        backend = PariBackend()
        backend._probe = {
            "engine": "pari",
            "available": True,
            "engine_identity": {"executable": "/usr/bin/gp"},
        }
        raw = marked_result(
            "target_internal_cpu_ms=0\n"
            "target_internal_wall_ms=0\n"
            "component_ideal_multiply_ms=0\n"
            "ideal_norm=36\n"
            "reported_threads=1\n"
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.pari.run_marked_process",
            return_value=raw,
        ) as run:
            payload = backend.run(request(), context(Path(temporary)))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["target_cpu_ms"], 0.0)
        self.assertEqual(payload["target_wall_ms"], 0.0)
        self.assertEqual(payload["timing"]["algorithm_clock"], "pari_getabstime_ms")
        self.assertEqual(payload["timing"]["wall_clock"], "pari_getwalltime_ms")
        self.assertEqual(payload["timing"]["marked_target_cpu_ms"], 11.0)
        self.assertEqual(payload["timing"]["marked_target_wall_ms"], 12.0)
        self.assertEqual(
            payload["timing"]["marked_process_affinity"],
            MARKED_PROCESS_AFFINITY,
        )
        self.assertEqual(
            payload["thread_count"],
            {
                "requested": 1,
                "reported": 1,
                "matches_requested": True,
                "source": "pari_default_nbthreads_runtime_query",
            },
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 10.0)

    def test_pari_thread_count_mismatch_or_omission_is_a_hard_failure(self) -> None:
        for reported_line, expected_reported in (
            ("reported_threads=24\n", 24),
            ("reported_threads=01\n", None),
            ("", None),
        ):
            with self.subTest(reported=expected_reported):
                backend = PariBackend()
                backend._probe = {
                    "engine": "pari",
                    "available": True,
                    "engine_identity": {"executable": "/usr/bin/gp"},
                }
                raw = marked_result(
                    "target_internal_cpu_ms=1\n"
                    "target_internal_wall_ms=1\n"
                    "component_ideal_multiply_ms=1\n"
                    "ideal_norm=36\n"
                    + reported_line
                )
                with tempfile.TemporaryDirectory() as temporary, mock.patch(
                    "silex_bench.backends.pari.run_marked_process",
                    return_value=raw,
                ):
                    payload = backend.run(request(), context(Path(temporary)))

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "thread_contract")
                self.assertIn("thread-count contract failed", payload["error"])
                self.assertEqual(
                    payload["thread_count"],
                    {
                        "requested": 1,
                        "reported": expected_reported,
                        "matches_requested": False,
                        "source": "pari_default_nbthreads_runtime_query",
                    },
                )
                self.assertTrue(payload["proof"]["final_result_published"])

    def test_magma_internal_zero_is_authoritative_and_marked_is_audit(self) -> None:
        backend = MagmaBackend()
        backend._probe = {
            "engine": "magma",
            "available": True,
            "engine_identity": {"executable": "/usr/local/bin/magma"},
        }
        raw = marked_result(
            "target_internal_cpu_seconds=0\n"
            "target_internal_wall_seconds=0\n"
            "ideal_norm=36\n"
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.magma.run_marked_process",
            return_value=raw,
        ) as run:
            payload = backend.run(request(), context(Path(temporary)))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["target_cpu_ms"], 0.0)
        self.assertEqual(payload["target_wall_ms"], 0.0)
        self.assertEqual(payload["timing"]["algorithm_clock"], "magma_cputime")
        self.assertEqual(payload["timing"]["wall_clock"], "magma_realtime")
        self.assertEqual(payload["timing"]["marked_target_cpu_ms"], 11.0)
        self.assertEqual(payload["timing"]["marked_target_wall_ms"], 12.0)
        self.assertEqual(
            payload["timing"]["marked_process_affinity"],
            MARKED_PROCESS_AFFINITY,
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 10.0)

    def test_hecke_internal_zero_remains_authoritative(self) -> None:
        backend = HeckeBackend()
        backend._probe = {
            "engine": "hecke",
            "available": True,
            "engine_identity": {
                "executable": "/usr/bin/julia",
                "project": None,
            },
        }
        raw = marked_result(
            "internal_target_cpu_ms=0\n"
            "internal_target_wall_ms=0\n"
            "component_ideal_multiply_ms=0\n"
            "ideal_norm=36\n"
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.hecke.run_marked_process",
            return_value=raw,
        ) as run:
            payload = backend.run(request(), context(Path(temporary)))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["target_cpu_ms"], 0.0)
        self.assertEqual(payload["target_wall_ms"], 0.0)
        self.assertEqual(payload["timing"]["marked_target_cpu_ms"], 11.0)
        self.assertEqual(payload["timing"]["marked_target_wall_ms"], 12.0)
        self.assertEqual(
            payload["timing"]["marked_process_affinity"],
            MARKED_PROCESS_AFFINITY,
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 10.0)

    def test_external_adapters_reject_mistyped_process_success(self) -> None:
        cases = (
            (
                "pari",
                PariBackend,
                {
                    "engine": "pari",
                    "available": True,
                    "engine_identity": {"executable": "/usr/bin/gp"},
                },
                (
                    "target_internal_cpu_ms=1\n"
                    "target_internal_wall_ms=1\n"
                    "component_ideal_multiply_ms=1\n"
                    "ideal_norm=36\n"
                    "reported_threads=1\n"
                ),
                "silex_bench.backends.pari.run_marked_process",
            ),
            (
                "hecke",
                HeckeBackend,
                {
                    "engine": "hecke",
                    "available": True,
                    "engine_identity": {
                        "executable": "/usr/bin/julia",
                        "project": None,
                    },
                },
                (
                    "internal_target_cpu_ms=1\n"
                    "internal_target_wall_ms=1\n"
                    "component_ideal_multiply_ms=1\n"
                    "ideal_norm=36\n"
                ),
                "silex_bench.backends.hecke.run_marked_process",
            ),
            (
                "magma",
                MagmaBackend,
                {
                    "engine": "magma",
                    "available": True,
                    "engine_identity": {"executable": "/usr/local/bin/magma"},
                },
                (
                    "target_internal_cpu_seconds=0.001\n"
                    "target_internal_wall_seconds=0.001\n"
                    "ideal_norm=36\n"
                ),
                "silex_bench.backends.magma.run_marked_process",
            ),
        )
        invalid_process_states = (
            {"success": "false"},
            {"timeout": "true"},
            {"timeout": "false"},
            {"timeout": 1},
        )
        for name, backend_type, probe, stdout, patch_target in cases:
            for invalid_state in invalid_process_states:
                with self.subTest(
                    backend=name, invalid_state=invalid_state
                ), tempfile.TemporaryDirectory() as temporary:
                    backend = backend_type()
                    backend._probe = probe
                    raw = {**marked_result(stdout), **invalid_state}
                    with mock.patch(patch_target, return_value=raw):
                        payload = backend.run(request(), context(Path(temporary)))

                    self.assertFalse(payload["success"])
                    self.assertFalse(payload["proof"]["final_result_published"])

    def test_silex_operation_preserves_native_clock_identity(self) -> None:
        native = {
            "success": True,
            "engine_thread_count": 1,
            "target_cpu_ms": 0.0,
            "target_wall_ms": 0.25,
            "ideal_norm": "36",
            "timing_scope": "ideal_multiplication_only",
            "timing_clock": {
                "cpu": "std_clock_process_cpu",
                "wall": "steady_clock",
            },
            "source": "silex_public_api",
            "warmup": {"used": False},
        }
        process = {
            "available": True,
            "success": True,
            "timeout": False,
            "process_wall_ms": 1.0,
            "effective_affinity": MARKED_PROCESS_AFFINITY,
            "cmd": ["silex"],
            "stdout": json.dumps(native),
            "stderr": "",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=process,
        ) as run:
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertEqual(payload["target_cpu_ms"], 0.0)
        self.assertEqual(
            payload["timing"]["algorithm_clock"], "std_clock_process_cpu"
        )
        self.assertEqual(payload["timing"]["wall_clock"], "steady_clock")
        self.assertEqual(payload["timing"]["target_cpu_ms"], 0.0)
        self.assertEqual(payload["timing"]["target_wall_ms"], 0.25)
        self.assertEqual(
            payload["timing"]["marked_process_affinity"],
            MARKED_PROCESS_AFFINITY,
        )
        self.assertEqual(
            payload["thread_count"],
            {
                "requested": 1,
                "reported": 1,
                "matches_requested": True,
                "source": "silex_flint_get_num_threads",
            },
        )
        self.assertIn("--marked-protocol", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 10.0)

    def test_silex_operation_rejects_duplicate_json_fields(self) -> None:
        native = {
            "success": True,
            "engine_thread_count": 1,
            "target_cpu_ms": 1.0,
            "target_wall_ms": 1.0,
            "ideal_norm": "36",
            "timing_scope": "ideal_multiplication_only",
            "timing_clock": {
                "cpu": "std_clock_process_cpu",
                "wall": "steady_clock",
            },
        }
        contradictory = json.dumps(native).replace(
            "{", '{"success": false, ', 1
        )
        process = marked_result(contradictory)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=process,
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "invalid_output")
        self.assertIn("duplicate key", payload["error"])

    def test_silex_operation_rejects_utf8_replacement_evidence(self) -> None:
        native = operation_payload()
        native["source"] = "\ufffd"
        outputs = (
            json.dumps(native, ensure_ascii=False),
            "\ufffd\n" + json.dumps(operation_payload()),
        )
        for output in outputs:
            with self.subTest(output_prefix=output[:1]), tempfile.TemporaryDirectory() as temporary:
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=marked_result(output),
                ):
                    payload = SilexBackend()._run_operation(
                        request(warm=False), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")
                self.assertIn("UTF-8", payload["error"])

    def test_silex_operation_rejects_invalid_process_availability(self) -> None:
        for invalid_availability in (None, "false", 0):
            with self.subTest(
                invalid_availability=invalid_availability
            ), tempfile.TemporaryDirectory() as temporary:
                process = marked_result(json.dumps(operation_payload()))
                if invalid_availability is None:
                    process.pop("available")
                else:
                    process["available"] = invalid_availability
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=process,
                ):
                    payload = SilexBackend()._run_operation(
                        request(warm=False), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")

    def test_silex_operations_require_semantically_complete_results(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []
        ideal = operation_payload()
        ideal.pop("ideal_norm")
        cases.append(("ideal_multiply", ideal))
        cases.append(("maximal_order", operation_payload()))
        square_root = operation_payload()
        square_root.pop("ideal_norm")
        square_root.update({"root_found": 1, "root_verified": True})
        cases.append(("element_square_root", square_root))

        for operation, native in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=marked_result(json.dumps(native)),
                ):
                    payload = SilexBackend()._run_operation(
                        request(warm=False, operation=operation),
                        context(Path(temporary)),
                        {},
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")
                self.assertIsInstance(payload["error"], str)
                self.assertTrue(payload["error"])

    def test_silex_class_unit_requires_semantically_complete_results(self) -> None:
        cases: dict[str, tuple[str, object]] = {
            "class group": ("class_group", {}),
            "unit group": ("unit_group", {}),
            "signature": ("signature", None),
            "discriminant": ("maximal_order_discriminant", None),
        }
        for label, (key, value) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                native = class_unit_payload()
                native[key] = value
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=marked_result(json.dumps(native)),
                ):
                    payload = SilexBackend()._run_class_unit(
                        class_unit_request(), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")
                self.assertIsInstance(payload["error"], str)
                self.assertTrue(payload["error"])

    def test_silex_operation_failure_has_nonempty_text_diagnostic(self) -> None:
        native = operation_payload()
        native.update({"success": False, "timeout": False, "error": []})
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=marked_result(json.dumps(native)),
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertFalse(payload["success"])
        self.assertIsInstance(payload["error"], str)
        self.assertTrue(payload["error"])

    def test_silex_operation_rejects_mistyped_process_timeout(self) -> None:
        native = operation_payload()
        for invalid_timeout in ("true", "false", 1):
            with self.subTest(
                invalid_timeout=invalid_timeout
            ), tempfile.TemporaryDirectory() as temporary:
                process = marked_result(json.dumps(native))
                process["timeout"] = invalid_timeout
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=process,
                ):
                    payload = SilexBackend()._run_operation(
                        request(warm=False), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")

    def test_silex_operation_rejects_mistyped_native_timeout(self) -> None:
        for invalid_timeout in ("false", 0, None):
            with self.subTest(
                invalid_timeout=invalid_timeout
            ), tempfile.TemporaryDirectory() as temporary:
                native = operation_payload()
                native["timeout"] = invalid_timeout
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=marked_result(json.dumps(native)),
                ):
                    payload = SilexBackend()._run_operation(
                        request(warm=False), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")
                self.assertIn("Boolean", payload["error"])

    def test_silex_process_timeout_overrides_native_timeout_false(self) -> None:
        operation_native = operation_payload()
        operation_native["timeout"] = False
        cases = (
            ("operation", "_run_operation", request(warm=False), operation_native),
            ("class_unit", "_run_class_unit", class_unit_request(), class_unit_payload()),
        )
        for label, method_name, sample, native in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                process = marked_result(json.dumps(native))
                process.update(
                    {
                        "success": False,
                        "timeout": True,
                        "error": "process timed out",
                    }
                )
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=process,
                ):
                    payload = getattr(SilexBackend(), method_name)(
                        sample, context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertTrue(payload["timeout"])
                self.assertEqual(payload["status"], "timeout")
                self.assertIsInstance(payload["error"], str)
                self.assertTrue(payload["error"])

    def test_silex_operation_requires_false_native_timeout_for_success(self) -> None:
        native = operation_payload()
        native["timeout"] = True
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=marked_result(json.dumps(native)),
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertFalse(payload["success"])
        self.assertTrue(payload["timeout"])
        self.assertEqual(payload["status"], "timeout")

    def test_silex_timeout_precedes_result_validation(self) -> None:
        operation = operation_payload()
        operation.pop("ideal_norm")
        operation["timeout"] = True
        class_unit = class_unit_payload()
        class_unit["class_group"] = {}
        class_unit["timeout"] = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "silex_bench.backends.silex.run_marked_process",
                return_value=marked_result(json.dumps(operation)),
            ):
                operation_result = SilexBackend()._run_operation(
                    request(warm=False), context(root), {}
                )
            with mock.patch(
                "silex_bench.backends.silex.run_marked_process",
                return_value=marked_result(json.dumps(class_unit)),
            ):
                class_result = SilexBackend()._run_class_unit(
                    class_unit_request(), context(root), {}
                )

        for result in (operation_result, class_result):
            with self.subTest(status=result["status"]):
                self.assertFalse(result["success"])
                self.assertTrue(result["timeout"])
                self.assertEqual(result["status"], "timeout")
                self.assertIn("timeout", result["error"].lower())

    def test_silex_operation_accepts_false_native_timeout(self) -> None:
        native = operation_payload()
        native["timeout"] = False
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=marked_result(json.dumps(native)),
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertTrue(payload["success"])
        self.assertFalse(payload["timeout"])
        self.assertEqual(payload["status"], "ok")

    def test_silex_operation_accepts_absent_native_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=marked_result(json.dumps(operation_payload())),
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertTrue(payload["success"])
        self.assertFalse(payload["timeout"])
        self.assertEqual(payload["status"], "ok")

    def test_silex_adapters_reject_non_object_nested_payloads(self) -> None:
        for key in (
            "class_group",
            "unit_group",
            "measurement_timing",
            "component_timing_ms",
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                native = class_unit_payload()
                native[key] = []
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=marked_result(json.dumps(native)),
                ):
                    payload = SilexBackend()._run_class_unit(
                        class_unit_request(), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")

        operation = {
            "success": True,
            "engine_thread_count": 1,
            "target_cpu_ms": 1.0,
            "target_wall_ms": 1.0,
            "ideal_norm": "36",
            "timing_scope": "ideal_multiplication_only",
            "timing_clock": [],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=marked_result(json.dumps(operation)),
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "invalid_output")

    def test_silex_class_unit_rejects_mistyped_native_timeout(self) -> None:
        for invalid_timeout in ("false", 0, None):
            with self.subTest(
                invalid_timeout=invalid_timeout
            ), tempfile.TemporaryDirectory() as temporary:
                native = class_unit_payload()
                native["timeout"] = invalid_timeout
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value=marked_result(json.dumps(native)),
                ):
                    payload = SilexBackend()._run_class_unit(
                        class_unit_request(), context(Path(temporary)), {}
                    )

                self.assertFalse(payload["success"])
                self.assertEqual(payload["status"], "invalid_output")

    def test_silex_class_unit_requires_false_native_timeout_for_success(self) -> None:
        native = class_unit_payload()
        native["timeout"] = True
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=marked_result(json.dumps(native)),
        ):
            payload = SilexBackend()._run_class_unit(
                class_unit_request(), context(Path(temporary)), {}
            )

        self.assertFalse(payload["success"])
        self.assertTrue(payload["timeout"])
        self.assertEqual(payload["status"], "timeout")

    def test_silex_class_unit_uses_only_default_backend_and_preserves_audit(self) -> None:
        native = {
            "success": True,
            "expectations_passed": True,
            "engine_thread_count": 1,
            "equation_order_index": "3",
            "failure_stage": None,
            "failure_reason": None,
            "final_result_published": True,
            "certification_status": "proven",
            "class_group_proof_status": "proven",
            "unit_group_proof_status": "proven",
            "regulator_proof_status": "verified",
            "class_group": {"order": "1", "invariants": []},
            "unit_group": {"free_rank": 1},
            "signature": [2, 0],
            "maximal_order_discriminant": "5",
            "measurement_timing": {
                "target_cpu_ms": 1.0,
                "target_wall_ms": 2.0,
            },
            "component_timing_ms": {"zeta": 0.5},
        }
        process = {
            "available": True,
            "success": True,
            "timeout": False,
            "process_wall_ms": 3.0,
            "target_cpu_ms": 1.25,
            "target_wall_ms": 2.25,
            "effective_affinity": MARKED_PROCESS_AFFINITY,
            "cmd": ["silex"],
            "stdout": json.dumps(native),
            "stderr": "",
        }
        class_request = SampleRequest(
            field=field("target", -5),
            operation="class_unit_proven",
            sample_kind="cold_process",
            sample_index=0,
            warmup=None,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=process,
        ):
            payload = SilexBackend()._run_class_unit(
                class_request, context(Path(temporary)), {}
            )

        self.assertTrue(payload["success"])
        self.assertEqual(payload["algorithm"], "default")
        self.assertEqual(payload["equation_order_index"], "3")
        self.assertEqual(
            payload["timing"]["marked_process_affinity"],
            MARKED_PROCESS_AFFINITY,
        )
        self.assertEqual(
            payload["thread_count"],
            {
                "requested": 1,
                "reported": 1,
                "matches_requested": True,
                "source": "silex_flint_get_num_threads",
            },
        )
        incomplete_proof = {**native, "unit_group_proof_status": "unknown"}
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value={**process, "stdout": json.dumps(incomplete_proof)},
        ):
            payload = SilexBackend()._run_class_unit(
                class_request, context(Path(temporary)), {}
            )
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "proof_contract")
        self.assertIn("proven-publication contract failed", payload["error"])

        wrong_regulator = {**native, "regulator_proof_status": "proven"}
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value={**process, "stdout": json.dumps(wrong_regulator)},
        ):
            payload = SilexBackend()._run_class_unit(
                class_request, context(Path(temporary)), {}
            )
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "proof_contract")
        self.assertIn("verified regulator", payload["error"])

        mistyped_booleans = {
            "process success": (
                {**process, "success": "false"},
                native,
            ),
            "payload success": (
                process,
                {**native, "success": "false"},
            ),
            "expectations": (
                process,
                {**native, "expectations_passed": "false"},
            ),
            "publication": (
                process,
                {**native, "final_result_published": "false"},
            ),
        }
        for case, (case_process, case_payload) in mistyped_booleans.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                with mock.patch(
                    "silex_bench.backends.silex.run_marked_process",
                    return_value={
                        **case_process,
                        "stdout": json.dumps(case_payload),
                    },
                ):
                    payload = SilexBackend()._run_class_unit(
                        class_request, context(Path(temporary)), {}
                    )
            self.assertFalse(payload["success"])

    def test_silex_runtime_thread_mismatch_is_a_hard_failure(self) -> None:
        native = {
            "success": True,
            "engine_thread_count": 2,
            "target_cpu_ms": 1.0,
            "target_wall_ms": 1.0,
            "ideal_norm": "36",
            "timing_scope": "ideal_multiplication_only",
            "timing_clock": {
                "cpu": "std_clock_process_cpu",
                "wall": "steady_clock",
            },
        }
        process = marked_result(json.dumps(native))
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "silex_bench.backends.silex.run_marked_process",
            return_value=process,
        ):
            payload = SilexBackend()._run_operation(
                request(warm=False), context(Path(temporary)), {}
            )

        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "thread_contract")
        self.assertIn("thread-count contract failed", payload["error"])
        self.assertEqual(
            payload["thread_count"],
            {
                "requested": 1,
                "reported": 2,
                "matches_requested": False,
                "source": "silex_flint_get_num_threads",
            },
        )


if __name__ == "__main__":
    unittest.main()
