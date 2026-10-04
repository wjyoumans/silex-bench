
# Comparative benchmarking

This repository owns cross-implementation correctness, proof agreement,
performance, and scaling campaigns. Read [AGENTS.md](../AGENTS.md),
[architecture.md](architecture.md), and [backend-contracts.md](backend-contracts.md)
before changing a campaign or adapter. The separate
[S-unit contract](sunit-contract.md) governs that workload's stronger evidence.

## Measurement ownership

Use Silex's native tests for native mathematical invariants and its native
microbenchmarks for regression tracking, component attribution, and local
optimization. An external match supplements focused native coverage. Native
source fidelity and performance evidence remain governed by the sibling
Silex repository's `docs/development/source_fidelity.rst` and
`docs/development/benchmarking.rst`.

Use Silex Bench for cross-engine comparisons through the in-tree adapters for
Silex, PARI/GP, Hecke/OSCAR, and Magma. Read each repository's agent instructions
before working there. Keep runs, exports, plots, credentials, and machine-local
paths untracked. The ignored `.silex-bench.local.toml` is optional when `PATH`
and conventional sibling-checkout discovery suffice.

## Bounded feature-development loop

1. Identify the authoritative source routine, version, proof mode,
   normalization, and failure behavior before implementing native mathematics.
2. Add the smallest native regression that establishes the intended behavior
   and build the maintained native adapters when the protocol needs them.
3. Preview a bounded Bench plan, then use `check` on the affected cases and
   the most authoritative available pair. Inspect each backend's validation
   and pairwise agreement, not just the process exit code.
4. Stop timing interpretation on any disagreement in results, proof state, or
   normalization. Diagnose the mathematical or adapter difference first.
5. After agreement, use `run` for an exploratory quick or development campaign.
   An unexplained gap should lead to native profiling of setup, conversion,
   allocation, algorithm selection, and repeated work before design changes.
6. Expand to scale cases only after this loop is stable. Predeclare the corpus
   slice, budget, and stop criterion; retain difficult cases and failures.

This feedback can produce a reproducible performance finding, a ranked native
optimization task, native correctness and measurement evidence, independent
review, and a new cross-engine verification. During final CPU-sensitive
measurements no other CPU-intensive work, including other agents' builds and
tests, may run, and the relevant SMT sibling must be idle.

## Commands and selection

For an uninstalled checkout, prefix commands with
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m silex_bench`.
The installed `silex-bench` command is equivalent for ordinary development.
Use `--json` for automation and the default readable output for people.

```sh
silex-bench list
silex-bench plan --suite number-field --profile quick
silex-bench doctor --profile quick --backend silex --backend pari --build-silex
silex-bench check --profile quick --backend silex --backend pari \
  --workload class_unit_proven --tag core --metric-max degree=4
silex-bench run --profile dev --backend silex --backend pari \
  --workload class_unit_proven --tag core --metric-max degree=4 \
  --repetitions 3 --timeout 60 --budget 600
```

The opt-in `class_unit_grh` workload runs each engine's GRH-conditional
route (see the [backend contracts](backend-contracts.md)); select it with
`--workload class_unit_grh`. Its rows are labelled `grh`, stay separate from
`class_unit_proven` rows in ledgers and reports, and are never compared with
proven timings. The report summary carries a `workload` field on every
`backend_status` and `agreement_status` row and splits `status_counts` per
workload, so a grh run and a proven run of the same engine land in separate
rows and counts; the markdown observation and agreement tables show a
Workload column.

`plan` resolves selection, capabilities, required pairs, and execution limits
without starting engines. `doctor` probes availability and provenance; its
`--build-silex` option prepares native adapters. `check` uses one correctness
observation per selected case/backend; `run` collects the configured repeated
timing samples. Both validate results and agreement and retain a ledger and
an exploratory report. A `check` pass does not exempt subsequent `run`
observations from validation. Neither command alone makes a publication claim.

Use repeatable `--backend`, `--workload`, `--case`, and `--tag` filters and
numeric `--metric-min NAME=VALUE` / `--metric-max NAME=VALUE` bounds. Each
`--tag` is required in addition to the profile's default slice, so it only
narrows selection. An explicit `--case` can select outside the default tag
slice while retaining the profile's other filters and execution limits.
Use `--timeout` for the per-observation deadline, `--budget` for the campaign
budget, and `--repetitions` for repeated observations.

The suite's required pair is normally Silex/PARI. When choosing another pair,
set `--require-pair CANDIDATE:BASELINE` explicitly. Use
`--require-all-adapters` when every selected declared capability is a hard
gate. Optional unavailable adapters and unsupported cells remain visible;
Magma currently does not declare S-unit support. Profile policy exclusions are
also explicit; the publication profile excludes Hecke S-unit cells.

## Ledger, timing, and reports

`run.sqlite` is the mutable source of truth; `manifest.json` is a readable
mirror and `exports/HASH/` contains immutable derived artifacts. `resume RUN_DIR`
requires the recorded configuration, engines, sources, and machine fingerprint
to match. Current schema-v2 ledgers can be resumed; schema-v1 ledgers are
report-only. `report RUN_DIR` can render a complete or partial ledger without
mutating it. Retain failures, timeouts, invalid proofs, disagreements,
unavailable engines, unsupported cells, and policy exclusions in the evidence.
Do not remove them to improve a summary.

Reports publish absolute runtime summaries and plots, with separate series by
backend, sample variant, timing scope, and wall clock. Ordinary workloads use
the backend's internal target wall interval, with the backend-specific clock
guarantees described in [backend-contracts.md](backend-contracts.md). For class and unit
groups, field and maximal-order construction precede the interval; class-group
and unit-group calls are inside it; extraction and validation follow it.
Nonce-bound target markers independently audit boundaries and enforce the
observation deadline. Their supervisor clocks remain diagnostics.

Ordinary adapters usually record one `standard` sample. Ordinary Hecke paths
with `jit_repetitions = 1` record `first_call` and `repeat_call` in one Julia
process on independently prepared objects, preventing cross-sample mathematical
cache reuse. Keep these series separate. Integrated S-unit work instead uses
one explicitly labeled supervisor-measured whole-process sample and does not
use this JIT pair policy. That envelope runs from the supervisor's spawn of the
target to its reap of the target on `CLOCK_MONOTONIC`; it excludes harness and
supervisor startup. A target that leaves a descendant alive at its exit has
that descendant killed and the sample fails with
`descendants_outlived_target`; it is never a timing sample. Do not merge
different timing scopes or clocks.

A successful mathematical observation needs valid proof evidence and agreeing
canonical results before its samples can enter a timing summary. Admitted
samples also require successful timing status and a positive target wall
clock. A zero-resolution clock can still supply correctness evidence but is
not timing-eligible. Use matching case and repetition identities for any
subsequent paired analysis; current built-in reports contain absolute times,
not speedup ratios.

## Provenance and publication

The campaign binds exact configuration/corpus bytes, overrides, package
content, source commits and cleanliness, engine probes and executable
identities, machine details, affinity, and timing semantics. Matching engine
version strings alone do not prove binary lineage. State provenance only as
strongly as the recorded evidence permits.

For a publication candidate:

- Predeclare the input regime and metric range before inspecting results.
- Run from the actual clean, committed Bench source checkout with clean,
  committed Silex sources and complete required-engine identities. Ordinary
  installed wheels remain useful for exploratory comparisons.
- Use a publication profile, explicit Linux CPU affinity, one engine thread,
  an idle relevant SMT sibling, and no competing CPU-intensive work.
- Collect at least the profile's required repetitions, never fewer than three.
  Require valid, timing-eligible agreement for every required pair and planned
  repetition. Any optional timing series included in the candidate must cover
  every planned repetition; an entirely unavailable optional adapter is
  non-blocking.
- When PARI participates in a required publication pair, configure matching
  `pari_source` and `pari_version` in the local tools file.
- Run `export RUN_DIR --publication`. Report rejected gates rather than
  bypassing them. State exact inputs, engines/revisions, exclusions, clock and
  scope, statistic, uncertainty, and retained failure/timeout counts.

From a source checkout, with `CPU_ID` replaced by the chosen idle Linux CPU:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m silex_bench run \
  --profile publication --cpu CPU_ID
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m silex_bench export \
  runs/RUN --publication
```

The export gate validates eligibility of an immutable local bundle. External
publication or upload still needs explicit authorization. Gate acceptance
also does not replace source/proof review or external host-idleness evidence.

## Extension ownership

A new backend owns a stable `BackendDescriptor`, its availability/provenance
probe, and one adapter per supported workload. A new `WorkloadContract` owns
case validation, canonical result/proof validation, pairwise equivalence, and
scale axes. Add corpus rows with tags and metrics for an existing workload.
Register through `Registry`; missing capabilities remain unsupported. Keep
backend-specific mathematics out of campaign orchestration and reporting.

Test registry/contract rejection, fake-engine success/failure/timeout paths,
required-pair gates, ledger collision/resume behavior, and reports without
requiring optional or proprietary engines. Live all-adapter checks are opt-in
and need a host with access to the Julia depot and any Magma license.
