from __future__ import annotations

import contextlib
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from silex_bench.campaign import build_plan, run_campaign
from silex_bench.configuration import (
    ProfileConfig,
    RunOverrides,
    SuiteConfig,
    ToolConfig,
)
from silex_bench.contracts import validate_campaign_manifest
from silex_bench.ledger import RunLedger
from silex_bench.workloads import MAXIMAL_ORDER
from test.test_campaign import fake_registry


class ManifestAndLedgerValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        corpus = self.root / "fields.json"
        corpus.write_text(
            json.dumps(
                {
                    "fields": [
                        {
                            "id": "real_quadratic_5_proven",
                            "coefficients_low_to_high": [-5, 0, 1],
                            "degree": 2,
                            "maximal_order_discriminant": 5,
                            "expected_success": True,
                            "optimization_external_engines": ["pari"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        marker = self.root / "config.toml"
        marker.write_text("schema_version = 1\n", encoding="utf-8")
        suite = SuiteConfig(
            path=marker,
            id="validation",
            title="validation",
            workloads=(MAXIMAL_ORDER,),
            backends=("silex", "pari"),
            required_pairs=(("silex", "pari"),),
            corpora={"number_fields": corpus},
            reports={},
            sha256="suite",
        )
        profile = ProfileConfig(
            path=marker,
            id="validation",
            description="",
            include_tags=("quick",),
            exclude_tags=(),
            repetitions=1,
            warmups=0,
            timeout_seconds=1,
            budget_seconds=None,
            cpu=None,
            threads=1,
            publication=False,
            minimum_repetitions=1,
            require_clean_sources=False,
            metrics={},
            sha256="profile",
        )
        self.plan = build_plan(
            self.root,
            suite,
            profile,
            ToolConfig(None, {}, None),
            RunOverrides(),
            fake_registry(),
            performance=True,
        )
        self.run_dir = self.root / "run"
        run_campaign(self.plan, self.run_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest(self) -> dict[str, object]:
        with RunLedger(self.run_dir) as ledger:
            return ledger.manifest()

    def test_manifest_rejects_missing_nested_structure_and_bad_machine_type(self) -> None:
        missing = copy.deepcopy(self._manifest())
        del missing["plan"]["execution"]
        with self.assertRaisesRegex(ValueError, "plan.execution"):
            validate_campaign_manifest(missing)

        malformed = copy.deepcopy(self._manifest())
        malformed["machine"]["cpu_model"] = {"not": "text"}
        with self.assertRaisesRegex(ValueError, "cpu_model"):
            validate_campaign_manifest(malformed)

    def test_manifest_rejects_self_fingerprint_and_plan_count_drift(self) -> None:
        fingerprint_drift = copy.deepcopy(self._manifest())
        fingerprint_drift["machine"]["cpu_model"] = "another CPU"
        with self.assertRaisesRegex(ValueError, "run_fingerprint"):
            validate_campaign_manifest(fingerprint_drift)

        count_drift = copy.deepcopy(self._manifest())
        count_drift["plan"]["sample_count"] += 1
        with self.assertRaisesRegex(ValueError, "sample_count"):
            validate_campaign_manifest(count_drift)

    def test_existing_ledger_rejects_row_fingerprint_and_state_drift(self) -> None:
        variants = {
            "fingerprint": ("UPDATE run SET fingerprint = ?", ("0" * 64,), "fingerprint"),
            "state": ("UPDATE run SET state = ?", ("invented",), "state"),
        }
        for label, (statement, values, error) in variants.items():
            with self.subTest(label=label):
                run_dir = self.root / f"run-{label}"
                run_campaign(self.plan, run_dir)
                with contextlib.closing(
                    sqlite3.connect(run_dir / "run.sqlite")
                ) as connection:
                    with connection:
                        connection.execute(statement, values)
                with self.assertRaisesRegex(ValueError, error):
                    RunLedger(run_dir)

    def test_terminal_run_requires_all_observations_but_running_may_be_partial(self) -> None:
        with contextlib.closing(
            sqlite3.connect(self.run_dir / "run.sqlite")
        ) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM observations WHERE backend = 'pari' AND repetition = 0"
                )
        with self.assertRaisesRegex(ValueError, "observation count"):
            RunLedger(self.run_dir)

        with contextlib.closing(
            sqlite3.connect(self.run_dir / "run.sqlite")
        ) as connection:
            with connection:
                connection.execute("UPDATE run SET state = 'running'")
        with RunLedger(self.run_dir) as ledger:
            self.assertEqual(ledger.state(), "running")


if __name__ == "__main__":
    unittest.main()
