# Silex Bench 0.1.1 release notes

**Status:** Unreleased; 0.1.1 release preparation. The latest tagged release is
0.1.0 (2026-09-03).

Silex Bench 0.1.1 updates the Linux benchmark harness for correctness-gated
comparisons between Silex and other mathematical systems.

## AI-assisted development disclosure

Silex and its companion repositories, Silex Bench and Silex Devtools, were
built almost entirely with OpenAI Codex, initially using GPT-5.5 and later
GPT-5.6, under the direction and review of William Youmans.

## Changes in 0.1.1

- Schema-v2 ledgers store each timing variant independently while retaining
  correctness observations and agreement records. Schema-v1 ledgers remain
  immutable and reportable, but cannot be resumed.
- Ordinary workloads now use backend-internal timing around the intended
  operation. Class/unit adapters prepare maximal orders before timing and keep
  result extraction and validation outside the interval.
- Profiles can request Hecke `first_call` and `repeat_call` samples. They run in
  one Julia process on independently prepared objects with cross-sample
  mathematical-cache reuse disabled; other adapters retain one `standard`
  sample.
- Timeout handling uses one effective per-observation deadline across adapters.
  Interrupts preserve committed rows for resume, and terminal progress reports
  each completed observation with its recorded samples.
- Square-root adapters construct a known square before timing and verify the
  recovered root afterward.
- Reports contain absolute timings rather than speedup ratios and generate one
  SVG format, with views by degree and by `log10(abs(D_K))`. Exact signed
  maximal-order discriminants are retained in case metadata.
- The publication profile uses three repetitions across the expanded
  default field slice and a 60-second observation ceiling. Hecke S-unit
  cells are explicitly excluded from this profile by policy. Default
  square-root selection is limited to degree 9; explicit case and timeout
  overrides remain supported. See the [policy record](docs/publication-policy.md)
  for the retained rationale and remaining methodology questions.

The command-line interface is the supported 0.1.1 interface. In-tree Python
extension APIs remain subject to change during the 0.x series. Magma does not
currently declare S-unit support. Comparative checks supplement rather than
replace Silex's native tests. This preparation publishes no benchmark results.

## Project family and licensing

- [Silex](https://github.com/wjyoumans/silex) provides the native library.
- Silex Bench owns its [comparative guidance](docs/comparative-benchmarking.md).
  Migrated Devtools guidance retains its
  [Apache-2.0 attribution](LICENSES/Silex-Devtools-NOTICE.md).

Silex Bench is distributed under [GPL-3.0-or-later](LICENSE). Project notices
are in [NOTICE.md](NOTICE.md), and upstream attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
