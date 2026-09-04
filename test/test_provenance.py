from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from silex_bench.campaign import (
    _benchmark_identity,
    _benchmark_source_root,
)
from silex_bench.util import git_identity


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkProvenanceTests(unittest.TestCase):
    def test_source_checkout_is_bound_to_the_imported_module_not_cwd(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(_benchmark_source_root(), ROOT)
                identity = _benchmark_identity()
            finally:
                os.chdir(original)

        self.assertEqual(Path(identity["path"]), ROOT)
        self.assertEqual(identity["provenance"], "source_checkout")
        self.assertEqual(len(identity["package_sha256"]), 64)

    def test_installed_distribution_never_inherits_the_callers_git_identity(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "site-packages" / "silex_bench"
            package.mkdir(parents=True)
            module = package / "campaign.py"
            module.write_text("# installed harness\n", encoding="utf-8")
            caller = root / "unrelated-checkout"
            caller.mkdir()
            (caller / ".git").mkdir()
            os.chdir(caller)
            try:
                identity = _benchmark_identity(module)
            finally:
                os.chdir(original)

        self.assertEqual(identity["path"], str(package))
        self.assertEqual(identity["provenance"], "installed_distribution")
        self.assertIsNone(identity["revision"])
        self.assertIsNone(identity["dirty"])
        self.assertEqual(len(identity["package_sha256"]), 64)

    def test_installed_package_content_changes_its_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "site-packages" / "silex_bench"
            package.mkdir(parents=True)
            module = package / "campaign.py"
            module.write_text("first\n", encoding="utf-8")
            first = _benchmark_identity(module)["package_sha256"]
            module.write_text("second\n", encoding="utf-8")
            second = _benchmark_identity(module)["package_sha256"]

        self.assertNotEqual(first, second)

    def test_dirty_worktree_content_changes_even_when_status_text_does_not(self) -> None:
        git = shutil.which("git")
        if git is None:
            self.skipTest("git is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run([git, "init", "-q"], cwd=root, check=True)
            subprocess.run(
                [git, "config", "user.email", "silex-bench@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [git, "config", "user.name", "Silex Bench Tests"],
                cwd=root,
                check=True,
            )
            tracked = root / "tracked.txt"
            tracked.write_text("committed\n", encoding="utf-8")
            subprocess.run([git, "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run([git, "commit", "-qm", "fixture"], cwd=root, check=True)

            tracked.write_text("first dirty content\n", encoding="utf-8")
            first = git_identity(root)
            tracked.write_text("second dirty content\n", encoding="utf-8")
            second = git_identity(root)

        self.assertEqual(first["status"], second["status"])
        self.assertTrue(first["dirty"])
        self.assertNotEqual(first["worktree_sha256"], second["worktree_sha256"])


if __name__ == "__main__":
    unittest.main()
