# Campaign architecture

The runner has five deliberately narrow layers.

1. `configuration.py` parses strict suite, profile, and machine-local tool
   TOML. Unknown keys fail; resolved paths and exact file hashes enter the plan.
2. `contracts.py` defines typed cases, observations, validation/agreement
   results, workload contracts, backend descriptors, and the in-tree registry.
3. `registry.py` connects those contracts to source-backed Silex, PARI,
   Hecke/OSCAR, and Magma adapters. Backend mechanics do not decide
   mathematical equivalence.
4. `campaign.py` materializes a deterministic plan, probes engines, rotates
   serial paired execution, validates outputs, computes agreements, and
   checkpoints every observation to `run.sqlite`.
5. `reporting.py` reads a ledger snapshot and writes a content-addressed,
   immutable export. Exploratory reports retain publication blockers;
   publication exports fail when any blocker exists.

The SQLite ledger is the only mutable run record. Its primary keys prevent two
different observations from occupying the same `(case, backend, repetition)`
slot. An observation owns correctness, command, and engine identity; its timing
samples own their individual variant, status, deadline, clocks, scope, and
diagnostics. Resume recomputes the campaign fingerprint, rejects drift, skips
existing observations, and fills missing slots. Derived reports never update
the ledger. Schema-v1 ledgers remain readable for reports but cannot be resumed
or mutated.

## Extension contract

A workload owns:

- its stable ID and scale axes;
- input validation;
- validation of one backend's canonical result and proof evidence;
- pairwise equivalence of two validated results.

A backend owns:

- one stable ID and display name;
- a probe producing availability, capabilities, and provenance identity;
- zero or one implementation adapter for each workload.

Register both through `Registry`. The registry checks ID uniqueness,
capability existence, and adapter identity consistency at construction time.
Campaign planning checks suite references against that registry. Missing
backend/workload combinations are modeled as unsupported capabilities, not
special cases in the runner.

The initial API is intentionally in-tree. A new backend normally needs one
adapter, one descriptor entry, contract fixtures, and fake-engine tests. It
does not require edits to the ledger or reporters.

## Eligibility flow

```text
case schema valid
  -> engine observation succeeds
  -> workload validates canonical result and proof
  -> paired workload comparison agrees
  -> timing sample succeeds with a positive target clock
  -> absolute runtime summary is eligible
```

Every failed stage is retained with its status and diagnostics. Reports count
those states and summarize only the final eligible subset. Required pairs are
hard campaign gates; unavailable optional engines do not discard valid
required-pair comparisons. If an optional engine contributes a timing series
to a publication candidate, that series must cover every planned repetition;
partial optional series block publication rather than publishing a biased
subset.

A successful sample may retain a zero reported by a backend clock whose
resolution is coarser than the operation. Such a sample remains correctness
evidence, but it is not timing-eligible and cannot appear on a logarithmic
runtime plot. Publication reports therefore require positive target-clock
values for every admitted timing sample.

## Reproducibility and publication

The run manifest binds suite/profile/tool bytes, selected cases, CLI
overrides, engine probes, the imported benchmark package bytes, Git worktree
contents, machine identity, execution policy, and timing semantics. Successful
observations must match the executable identity captured by their initial
probe. The resulting fingerprint protects resume from drift.

Exploratory reports may be produced from partial or dirty runs. A publication
export additionally requires a complete run, a publication profile, three or
more repetitions, an explicit Linux CPU, one engine thread, clean committed
Silex and silex-bench sources, complete required-engine identities, and
successful timing-eligible agreement for every required pair and repetition.
The exported directory name is derived from the ledger snapshot, report mode,
and renderer capability; installing plotting support can therefore create a
new immutable derived bundle while an existing bundle is never overwritten.
Generated plots contain absolute wall times only and use deterministic SVG
output, with separate degree and logarithmic-discriminant views when data is
available.
