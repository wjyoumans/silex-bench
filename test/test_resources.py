"""Tests for installed built-in suites, profiles, and corpora."""

from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from silex_bench.configuration import load_profile, load_suite
from silex_bench.resources import builtin_names, builtin_path, resolve_path


ROOT = Path(__file__).resolve().parents[1]


class BuiltinResourceTests(unittest.TestCase):
    def test_wheel_configuration_includes_user_and_license_documents(self) -> None:
        configuration = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(configuration["project"]["readme"], "README.md")
        self.assertEqual(
            configuration["project"]["license-files"],
            ["LICENSE", "NOTICE.md", "THIRD_PARTY_NOTICES.md"],
        )
        for name in ("README.md", "LICENSE", "NOTICE.md", "THIRD_PARTY_NOTICES.md"):
            self.assertTrue((ROOT / name).is_file(), name)
        self.assertIn(
            "PARI/GP source lineage",
            (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
        )

    def test_wheel_configuration_includes_every_resource_collection(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = set(project["tool"]["setuptools"]["package-data"]["silex_bench"])
        self.assertEqual(
            patterns,
            {
                "data/corpora/*.json",
                "data/profiles/*.toml",
                "data/suites/*.toml",
            },
        )

    def test_catalog_exposes_loadable_suite_profile_and_corpora(self) -> None:
        self.assertEqual(builtin_names("suites"), ("number-field",))
        self.assertEqual(
            builtin_names("profiles"),
            ("dev", "publication", "quick", "scale"),
        )
        self.assertEqual(
            builtin_names("corpora"),
            ("class_unit_fields", "sunit_fields"),
        )

        suite = load_suite(builtin_path("suites", "number-field"))
        profile = load_profile(builtin_path("profiles", "quick.toml"))

        self.assertEqual(suite.id, "number-field")
        self.assertEqual(profile.id, "quick")
        self.assertEqual(
            suite.corpora["number_fields"],
            builtin_path("corpora", "class_unit_fields"),
        )
        self.assertEqual(
            suite.corpora["sunit"], builtin_path("corpora", "sunit_fields.json")
        )
        for corpus in suite.corpora.values():
            payload = json.loads(corpus.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)

    def test_resolve_path_prefers_builtins_for_plain_names(self) -> None:
        self.assertEqual(
            resolve_path("suites", "number-field"),
            builtin_path("suites", "number-field"),
        )

    def test_resolve_path_preserves_explicit_custom_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.toml"
            custom.write_text("schema_version = 1\n", encoding="utf-8")
            self.assertEqual(resolve_path("suites", custom), custom.absolute())

        missing = Path("custom") / "missing.toml"
        self.assertEqual(resolve_path("suites", missing), missing.absolute())

    def test_catalog_rejects_unknown_collection_and_unsafe_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "resource collection"):
            builtin_names("unknown")
        with self.assertRaisesRegex(ValueError, "plain resource name"):
            builtin_path("profiles", "../quick")


if __name__ == "__main__":
    unittest.main()
