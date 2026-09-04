# Silex Bench Agent Instructions

These instructions govern automated work in this repository.

## Purpose and boundaries

- This repository owns backend-neutral benchmark orchestration, corpora,
  configurations, proof/agreement contracts, reports, and adapters for Silex,
  PARI/GP, Hecke, and Magma.
- The sibling `silex` repository owns native implementation, microbenchmarks,
  replay kernels, and native profiling tools. Do not duplicate them here.
- The sibling `silex-devtools` repository owns reusable skills and context tools.
- Treat the historical source workspace outside this pilot as read-only source
  material. Read exact committed objects; never mutate that workspace.
- Keep `runs/`, plots, caches, virtual environments, bytecode, credentials, and
  machine-local benchmark output untracked.

## First commands

Run the foundation checks from this repository:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s test -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m silex_bench --json plan \
  --suite number-field --profile quick
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m silex_bench --json list
```

Use the quick profile and select Silex/PARI for the bounded foundation gate.
Hecke and Magma remain optional comparison points. Do not describe a timing
run as publication evidence unless `export --publication` accepts it.
Use `doctor` before starting a real engine campaign. The ignored local tools
file is optional; create it only when PATH and conventional workspace
discovery are insufficient. Use `--json` for agent parsing and retain the
default human output for user-facing instructions.

When the host is expected to provide every implementation, run a bounded
`check --require-all-adapters` across all declared capabilities. Unsupported
cells, currently Magma S-units, must remain explicit and are not failures.

## Mathematical and benchmark fidelity

- Follow the user's current direction first, then documented PARI/Hecke/Magma
  behavior, current Silex behavior, and the contracts under `docs/`.
- Do not invent mathematical algorithms, proof labels, normalization,
  certification rules, agreement fields, or performance heuristics.
- Preserve exact integer serialization, backend identity, proof completeness,
  failure semantics, timeout semantics, and sample-pairing rules.
- Successful class/unit and S-unit rows must satisfy their backend-specific
  proof contracts before they can enter agreement or timing summaries.
- Keep subprocess execution shell-free, bounded, process-group aware, and
  explicit about stdout, stderr, timeout, spawn, and descendant cleanup.
- Performance claims require the declared configuration, engine/source identity,
  CPU-affinity policy, machine-readable output, and an idle relevant SMT sibling.
  Do not run final CPU-pinned measurements while other CPU-intensive work runs.
- Use `check` to establish per-engine validity and pairwise agreement before
  interpreting `run` timings. A failed validation or agreement row is never a
  performance sample.
- Treat `run.sqlite` as the run source of truth and `exports/HASH/` as immutable
  derived artifacts. Reject any ledger whose declared campaign format does not
  match the implementation.
- Matching version strings do not prove binary lineage; state provenance claims
  no more strongly than the available evidence supports.

## Host safety

- Treat a Hermes profile, worktree, or process wrapper as workflow isolation,
  not OS process isolation.
- Never run a nonzero signal experiment against PID `-1`, PID `0`, a negative
  process group, a parent/supervisor, the live compositor, `systemd --user`,
  Hermes infrastructure, or other session-wide targets. Never invoke logout,
  shutdown, reboot, or equivalent host/session actions from an agent task.
- The harness manages the lifecycle of trusted configured Silex, PARI, Hecke,
  and Magma executables; it is not an OS sandbox. Deliberately hostile code that
  kills its same-UID cleanup supervisor is out of scope for this harness.
- `PR_SET_CHILD_SUBREAPER`, `PR_SET_PDEATHSIG`, process groups, sessions, and
  Python cleanup logic are lifecycle mechanisms, not security boundaries.
- Keep real supervisor-kill probes out of the default host test suite. Testing
  hostile code requires a disposable VM or microVM, an isolated container, or
  an independently enforced UID and PID boundary; signal 0, mocks, and static
  analysis may document intent but cannot prove kernel-enforced containment.

## Worktrees, staging, and commits

- After the foundation commit, use one writing task, one repository, and one
  Git worktree. Reviewers are read-only unless a correction task is authorized.
- Add a focused failing regression before changing behavior, then run the
  focused test and complete unit suite.
- Stage explicit paths; never use broad staging commands such as `git add .`.
- Keep commits focused and validated.
- Do not create remotes, push, tag, publish, rewrite history, or destructively
  clean without explicit approval.

## Evidence and stop conditions

- AI assistance or generated output does not change the required workflow or
  lower mathematical, provenance, testing, or review standards.
- Report changed paths, exact commands and results, benchmark/proof implications,
  provenance implications, and remaining risks.
- Use concise source traces and machine-readable output for mathematical or
  performance-critical work; ordinary changes need only proportional evidence.
- Stop and ask before changing supported backends, benchmark eligibility, proof
  policy, licensing, publication gates, or release policy.
- Stop on source-workspace drift, unexplained result disagreement, incomplete
  proof metadata, provenance uncertainty, candidate drift, or an unresolved
  review finding.
