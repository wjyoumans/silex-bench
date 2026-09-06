# Silex Bench

`silex-bench` compares Silex with standard mathematical libraries and
implementations for both correctness and performance. Comparisons are provided
through adapters. The built-in adapters support PARI/GP, Hecke/OSCAR, and Magma
when those systems are installed and available; contributors can add further
adapters through the same in-tree framework.

This is separate from Silex's native microbenchmarks. Native microbenchmarks
track regressions and local optimizations. Silex Bench answers
cross-implementation questions: whether results agree, which implementation is
faster, and how that relationship changes as inputs grow.

## AI-assisted development disclosure

Silex and its companion repositories, Silex Bench and Silex Devtools, were
built almost entirely with OpenAI Codex, initially using GPT-5.5 and later
GPT-5.6, under the direction and review of William Youmans.

## Project family

- [Silex](https://github.com/wjyoumans/silex) is the native computational
  algebraic number theory library.
- [Silex Bench](https://github.com/wjyoumans/silex-bench) is this
  correctness-gated cross-implementation benchmark harness. Its comparisons
  supplement rather than replace Silex's native tests.

Bench owns its [comparative benchmarking guide](docs/comparative-benchmarking.md).
Automated work follows [AGENTS.md](AGENTS.md).

## Install

Silex Bench supports Linux and Python 3.11 or newer. Version 0.1.1 is in
release preparation; 0.1.0 is the latest tagged release. The process
supervision and CPU-affinity implementation uses Linux interfaces.

Clone the public repository and install the command and plotting support with:

```sh
git clone https://github.com/wjyoumans/silex-bench.git
cd silex-bench
python3 -m pip install '.[plot]'
```

From an existing checkout, the installation command is:

```sh
python3 -m pip install '.[plot]'
```

You can then invoke `silex-bench` from any directory. If you prefer not to
install it, run from the checkout and replace each invocation with:

```sh
PYTHONPATH=src python3 -m silex_bench ...
```

The `plot` extra is optional. Without it, Markdown, JSON, and CSV artifacts are
still generated, but SVG plots are skipped.

## Run a comparison

Start by seeing the available workloads, profiles, and adapter capabilities:

```sh
silex-bench list
```

Probe the locally available implementations and build the two native Silex
benchmark executables when needed:

```sh
silex-bench doctor --build-silex
```

Run the bounded correctness suite, then an exploratory performance comparison:

```sh
silex-bench check --profile quick --build-silex
silex-bench run --profile quick
```

Both commands print readable status and agreement tables. They also create an
immutable report containing Markdown, CSV, JSON, and any available plots, and
print its location. A failed or interrupted campaign still retains its ledger
and can produce a diagnostic report.

Adapters outside the required comparison are optional by default: unavailable
ones remain visible without discarding a valid required comparison. To verify
every selected adapter and every workload it declares, use the stricter gate:

```sh
silex-bench check --profile quick --build-silex --require-all-adapters
```

Magma currently does not declare S-unit support, so that cell is reported as
unsupported rather than as a failure.

## Choose implementations and inputs

The checked-in suite defines the available comparison, while profiles control
how far it runs:

- `quick`: small inputs and one observation for a fast development check;
- `dev`: broader low-degree inputs and three repetitions;
- `scale`: curated large inputs, five repetitions, and a long budget;
- `publication`: three repetitions over the expanded default corpus slice,
  with strict provenance and execution gates. Hecke S-unit rows are excluded
  by policy from this profile with a default 60-second observation ceiling.
  The [publication policy record](docs/publication-policy.md) details the
  changed selection, overrides, rationale, and open methodology questions.

Narrow a run with repeatable workload, case, backend, or tag options, or use
numeric metric bounds. Each `--tag` is required in addition to the profile's
default tag slice, so it cannot broaden the run. An explicit `--case` may
select a row outside the profile's default tags while retaining that profile's
execution limits:

```sh
silex-bench run --profile dev \
  --backend silex --backend pari \
  --workload class_unit_proven --tag core \
  --metric-min degree=2 --metric-max degree=8 \
  --repetitions 5 --timeout 600 --budget 7200
```

For a comparison pair other than the suite default, make the correctness gate
explicit. The order is candidate then baseline:

```sh
silex-bench run --profile quick \
  --backend silex --backend magma \
  --workload maximal_order \
  --require-pair silex:magma
```

Use `silex-bench plan` with the same options to preview the resolved cases,
adapter cells, repetition count, timeout, budget, and required pairs without
starting an implementation. A budget-limited ledger can be continued with
`silex-bench resume RUN_DIR` when its configuration, engines, sources, and
machine still match.

## Tool discovery

Silex Bench discovers GP, Julia, and Magma from the active environment and
`PATH`. In a conventional workspace it also finds a sibling `silex` checkout
and uses `silex/build/benchmark-adapters` for the native executables.

If your paths differ, create the ignored `.silex-bench.local.toml` in the
directory where you run the command and set only the required overrides. A
checkout includes `.silex-bench.local.toml.example`; `--tools PATH` selects
another tools file. The external programs and any required Magma license are
not bundled.

## Read or regenerate a report

To print the readable report for an existing complete or partial ledger:

```sh
silex-bench report runs/RUN
```

Each run contains a transactional `run.sqlite` source-of-truth ledger, a
readable `manifest.json` mirror, and content-addressed exports under
`exports/HASH/`. Reports retain unavailable
implementations, unsupported cells, failures, timeouts, invalid results, and
disagreements. Timing tables include only validated, agreeing pairs. Raw timing
samples and correctness records remain available in JSON and CSV. When plotting
support is installed, each workload gets absolute-runtime SVG plots by degree
and by `log10(abs(D_K))` whenever those metrics are present.

Automation can request one JSON document instead of terminal tables:

```sh
silex-bench --json check --profile quick
```

## Add benchmarks or adapters

- To add inputs for an existing workload, add corpus rows with the appropriate
  tags and scale metrics.
- To add a new mathematical workload, implement a `WorkloadContract` defining
  case validation, canonical result/proof validation, pairwise comparison, and
  scale axes, then add adapters for the implementations that support it.
- To add another comparison implementation, add a `BackendDescriptor`, its
  availability/provenance probe, and one implementation adapter per supported
  workload. Unsupported workload cells remain explicit.

Registry construction rejects duplicate IDs, unknown capabilities, and
adapter/descriptor mismatches. See [the architecture](docs/architecture.md)
and [backend contracts](docs/backend-contracts.md) for the contributor-facing
details.

## Correctness, timing, and publication

Each observation is validated against its workload contract before any
pairwise comparison. Canonical mathematical results, not formatted output, are
compared. Failed validation or disagreement is never admitted to timing
summaries. Every measured observation starts a fresh backend process. Ordinary
adapters normally record one `standard` timing sample. When a profile sets
`jit_repetitions = 1`, ordinary Hecke workloads—the only currently JIT-compiled
adapter path—record separate `first_call` and `repeat_call` samples in one Julia
process. The two calls use independently prepared objects and prevent
cross-sample mathematical-cache reuse, so the repeat timing is not made
artificially cheap by the first call. Reports keep these as distinct series.

For class and unit groups, field and maximal-order construction happens before
the timer. The measured interval contains only the requested class-group and
unit-group computation; result extraction and correctness comparison happen
afterward. Other ordinary workloads likewise use their declared internal target
interval, with the marked protocol auditing the boundary and enforcing the
deadline. Integrated S-unit work retains its separately labeled, single-sample
whole-process scope; the ordinary Hecke JIT-pair policy does not apply there.

Publication bundles additionally require a clean source checkout, explicit
CPU affinity, one engine thread, complete identities, at least three paired
repetitions, and agreement for every configured required pair. An optional
timing series that appears in a publication candidate must cover every planned
repetition; an entirely unavailable optional adapter remains non-blocking:

```sh
PYTHONPATH=src python3 -m silex_bench run --profile publication --cpu 2
PYTHONPATH=src python3 -m silex_bench export runs/RUN --publication
```

Installed wheels support ordinary comparisons and exploratory reports;
publication campaigns must run from the actual `silex-bench` source checkout
so its commit and worktree can be identified. The explicit `PYTHONPATH` form
above or an editable install (`python3 -m pip install -e '.[plot]'`) preserves
that source identity.

When PARI is part of a required publication pair, configure a matching
`pari_source` and `pari_version` in the local tools file.

Exploratory output must not be presented as publication evidence unless this
gate accepts it.

Silex Bench is GPL-3.0-or-later; see `LICENSE` and
`THIRD_PARTY_NOTICES.md`. It invokes external engines through their public
command interfaces and does not bundle them.

## Contributing

AI-assisted and AI-generated contributions are welcome under the same
mathematical, provenance, licensing, correctness, proof, performance, and
review requirements as other contributions. The human submitter must
understand, review, and be able to explain the change, and remains accountable
for it. A tool/model declaration and prompt transcripts are not required.

Follow [AGENTS.md](AGENTS.md) for repository-specific workflow and validation
rules. The [Silex contributor guide](https://github.com/wjyoumans/silex/blob/main/CONTRIBUTING.md)
contains the family-wide contribution policy.
