from __future__ import annotations

import unittest
from types import SimpleNamespace

from silex_bench.campaign import _ensure_timing_samples
from silex_bench.contracts import (
    Observation,
    ObservationStatus,
    ValidationResult,
)


class TargetCpuSelectionTests(unittest.TestCase):
    def test_target_cpu_uses_backend_value_not_marked_diagnostic(self) -> None:
        observation = Observation(
            case_key="k",
            workload="w",
            backend="b",
            repetition=0,
            order_index=0,
            status=ObservationStatus.OK,
            success=True,
            timeout=False,
            result={},
            proof={},
            validation=ValidationResult(True, (), {}),
            target_wall_ns=5_000_000,
            process_wall_ns=9_000_000,
            internal_timing={"target_cpu_ms": 2.0, "marked_target_cpu_ms": 11.0},
            engine_identity={},
            command=(),
            stdout="",
            stderr="",
        )
        case = SimpleNamespace(key="k", workload="w")
        context = SimpleNamespace(timeout_seconds=1.0, timeout_source="t")
        out = _ensure_timing_samples(observation, case, context, "scope")
        self.assertEqual(out.timing_samples[0].target_cpu_ns, 2_000_000)

        only_marked = Observation(
            **{
                **observation.__dict__,
                "internal_timing": {"marked_target_cpu_ms": 11.0},
            }
        )
        out = _ensure_timing_samples(only_marked, case, context, "scope")
        self.assertIsNone(out.timing_samples[0].target_cpu_ns)


if __name__ == "__main__":
    unittest.main()
