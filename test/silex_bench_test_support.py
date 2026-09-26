"""Shared synthetic campaign fixtures for local unittest discovery."""

from __future__ import annotations

from dataclasses import dataclass

from silex_bench.contracts import (
    BackendDescriptor, EngineInfo, Observation, ObservationStatus, Registry,
)
from silex_bench.workloads import MAXIMAL_ORDER, NumberFieldContract


@dataclass(frozen=True)
class FakeImplementation:
    backend: str
    workload: str = MAXIMAL_ORDER
    discriminant: str = "5"

    def run(self, case, *, repetition, order_index, warmup, context, contract):
        result = {"maximal_order_discriminant": self.discriminant}
        validation = contract.validate_observation(case, self.backend, result, {})
        elapsed = 10_000_000 if self.backend == "silex" else 20_000_000
        return Observation(
            case_key=case.key,
            workload=self.workload,
            backend=self.backend,
            repetition=repetition,
            order_index=order_index,
            status=ObservationStatus.OK,
            success=True,
            timeout=False,
            result=result,
            proof={},
            validation=validation,
            target_wall_ns=elapsed,
            process_wall_ns=elapsed + 1_000,
            internal_timing={
                "scope": "maximal_order_only",
                "wall_clock": {
                    "silex": "steady_clock",
                    "pari": "pari_getwalltime_ms",
                    "hecke": "julia_time_ns_monotonic",
                }.get(self.backend, "fixture_monotonic"),
            },
            engine_identity={"engine": self.backend, "version": "test"},
            command=(self.backend,),
            stdout="",
            stderr="",
        )


def fake_registry(
    *, include_unavailable: bool = False, disagree: bool = False
) -> Registry:
    contract = NumberFieldContract(MAXIMAL_ORDER, "Maximal order")
    descriptors = []
    for name in ("silex", "pari"):
        implementation = FakeImplementation(
            name, discriminant="7" if disagree and name == "pari" else "5"
        )

        def probe(context, descriptor, *, backend=name):
            return EngineInfo(
                backend=backend,
                display_name=backend,
                available=True,
                capabilities=(MAXIMAL_ORDER,),
                identity={"engine": backend, "version": "test"},
            )

        descriptors.append(
            BackendDescriptor(
                id=name,
                display_name=name,
                implementations={MAXIMAL_ORDER: implementation},
                probe_callback=probe,
            )
        )
    if include_unavailable:
        implementation = FakeImplementation("hecke")

        def unavailable(context, descriptor):
            return EngineInfo(
                backend="hecke",
                display_name="hecke",
                available=False,
                capabilities=(MAXIMAL_ORDER,),
                identity={},
                error="fixture unavailable",
            )

        descriptors.append(
            BackendDescriptor(
                id="hecke",
                display_name="hecke",
                implementations={MAXIMAL_ORDER: implementation},
                probe_callback=unavailable,
            )
        )
    return Registry([contract], descriptors)
