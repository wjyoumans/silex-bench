from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from silex_bench.resources import builtin_path
from silex_bench.sunit_backend import MAX_SUNIT_MANIFEST_BYTES
from silex_bench.util import MAX_JSON_NESTING_DEPTH
from silex_bench.workloads import SUNIT, _QUICK_SUNIT_FIELDS, load_cases


class IntegratedSUnitCorpusLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = json.loads(
            builtin_path("corpora", "sunit_fields").read_text(encoding="utf-8")
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, payload: object, name: str = "sunit.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _load(path: Path):
        return load_cases({"sunit": path}, (SUNIT,))

    def test_valid_original_manifest_materializes_all_sunit_cases(self) -> None:
        cases = self._load(self._write(self.manifest))

        self.assertEqual(len(cases), len(self.manifest["fields"]))
        self.assertEqual({case.workload for case in cases}, {SUNIT})

    def test_quick_sunit_tag_is_independent_of_manifest_row_order(self) -> None:
        expected_quick_ids = set(_QUICK_SUNIT_FIELDS)
        self.assertTrue(expected_quick_ids)

        original_fields = self.manifest["fields"]
        allow_listed_ids = {row["id"] for row in original_fields} & expected_quick_ids
        self.assertEqual(
            allow_listed_ids,
            expected_quick_ids,
            "the quick allow-list must name rows that exist in the manifest",
        )

        # Move the row that is at index 0 today (in the allow-list) to the
        # end, and put a row that is NOT in the allow-list at index 0
        # instead. The resulting `quick`-tagged set must still be exactly
        # the allow-list, regardless of manifest row position, guarding
        # against a positional fallback re-appearing.
        reordered_fields = original_fields[1:] + original_fields[:1]
        self.assertNotIn(reordered_fields[0]["id"], expected_quick_ids)
        reordered_manifest = copy.deepcopy(self.manifest)
        reordered_manifest["fields"] = reordered_fields

        cases = self._load(self._write(reordered_manifest, "reordered.json"))
        quick_ids = {case.id for case in cases if "quick" in case.tags}
        self.assertEqual(quick_ids, expected_quick_ids)

    def test_original_schema_and_prime_index_convention_are_required(self) -> None:
        mutations = {
            "schema_version": 999,
            "prime_index_convention": "one-based engine-local order",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self._load(self._write(manifest, f"{key}.json"))

    def test_integrated_loader_rejects_duplicate_keys_and_excessive_nesting(self) -> None:
        remainder = copy.deepcopy(self.manifest)
        remainder.pop("schema_version")
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":2,"schema_version":2,'
            + json.dumps(remainder)[1:],
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self._load(duplicate)

        nested: object = 0
        for _ in range(MAX_JSON_NESTING_DEPTH + 2):
            nested = [nested]
        deep_manifest = copy.deepcopy(self.manifest)
        deep_manifest["unexpected_nested_value"] = nested
        with self.assertRaisesRegex(ValueError, "nesting"):
            self._load(self._write(deep_manifest, "deep.json"))

    def test_integrated_loader_is_size_bounded_and_does_not_follow_symlinks(self) -> None:
        target = self._write(self.manifest, "target.json")
        linked = self.root / "linked.json"
        linked.symlink_to(target)
        with self.assertRaises((OSError, ValueError)):
            self._load(linked)

        serialized = json.dumps(self.manifest).encode()
        oversized = self.root / "oversized.json"
        oversized.write_bytes(
            serialized + b" " * (MAX_SUNIT_MANIFEST_BYTES + 1 - len(serialized))
        )
        with self.assertRaisesRegex(ValueError, "size limit"):
            self._load(oversized)


if __name__ == "__main__":
    unittest.main()
