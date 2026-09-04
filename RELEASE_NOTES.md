# Silex Bench 0.1.0 release notes

**Released:** 2026-09-03

Silex Bench 0.1.0 is the first public release of the Linux benchmark harness
for correctness-gated comparisons between Silex and other mathematical
systems.

## AI-assisted development disclosure

Silex and its companion repositories, Silex Bench and Silex Devtools, were
built almost entirely with OpenAI Codex, initially using GPT-5.5 and later
GPT-5.6, under the direction and review of William Youmans.

## Included surface

- The `silex-bench` command plans, checks, runs, resumes, reports, and exports
  cross-implementation campaigns.
- Built-in adapters cover Silex, PARI/GP, Hecke/OSCAR, and Magma when those
  separately installed systems are available.
- Workload contracts validate canonical results and proof evidence before a
  comparison can enter a timing summary.
- Publication exports require the documented clean-source, provenance,
  agreement, repetition, thread, and CPU-affinity gates.

The command-line interface is the supported 0.1.0 interface. In-tree Python
extension APIs remain subject to change during the 0.x series. Magma does not
currently declare S-unit support. Comparative checks supplement rather than
replace Silex's native tests, and this release publishes no benchmark results.

## Project family and licensing

- [Silex](https://github.com/wjyoumans/silex) provides the native library.
- [Silex Devtools](https://github.com/wjyoumans/silex-devtools) provides the
  maintained Codex workflows.

Silex Bench is distributed under [GPL-3.0-or-later](LICENSE). Project notices
are in [NOTICE.md](NOTICE.md), and upstream attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
