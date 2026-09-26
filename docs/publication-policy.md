# Publication policy in the 0.1.1 preparation

This records the material methodology changes already present in the working
tree reviewed after `8bed228`. They are preserved as part of the timing
redesign, separately identified here from its mechanical implementation.
They do not establish that 0.1.1 is released or that a benchmark result is
publication evidence.

## Default policy

| Setting | 0.1.0 / pre-redesign | Prepared policy |
| --- | --- | --- |
| Repetitions and publication minimum | 9 | 3 |
| Observation timeout ceiling | 7,200 seconds | 60 seconds |
| Campaign budget | 259,200 seconds | 259,200 seconds |
| Ordinary field publication tags | Core, scale, diversity, holdout roles | All ordinary fields, subject to workload selection |
| Default square-root slice | Publication-tagged fields | Degree at most 9 |
| Hecke S-units | Declared capability, included | Explicit profile exclusion |
| Magma S-units | Unsupported | Unsupported |

The default plan has 193 cases: 19 class/unit, 61 maximal-order, 61 ideal
multiplication, 46 square-root, and six S-unit cases. It contains 746
case/backend cells and 2,238 repeated observation slots, including explicit
unsupported Magma S-unit cells. All 61 ordinary corpus fields are retained;
the obsolete separate warmup corpus is removed. Existing case/backend
restrictions and expected-success selection still apply.

The profile sets `jit_repetitions = 1`: ordinary Hecke observations contain
`first_call` and `repeat_call` samples on independently prepared objects in
one process. These share the observation deadline and remain distinct timing
series. Other ordinary adapters and integrated S-units use one `standard`
sample. This replaces the former alternate-field warmup policy.

The 60-second value is a default ceiling, not an unconditional limit on every
publication invocation. Explicit `--timeout` overrides remain supported;
case hints can shorten the resolved ceiling. Explicit `--case` selection
bypasses default profile tags, including the square-root degree slice, while
retaining other filters and execution limits. Overrides and exact selection
enter the campaign fingerprint. The materialized plan is the authoritative
description of a particular campaign.

## Rationale and evidence limits

The profile's recorded Hecke S-unit exclusion reason is exactly:
“Policy exclusion from the 60-second publication profile.” The existing
release draft describes an expanded field slice with a bounded observation
ceiling. Neither text supplies a statistical justification for three
repetitions, the 60-second cutoff, or the degree-9 square-root slice.

Local historical evidence inspected during review included the ignored
`runs/20260903-221008-number-field-publication` ledger and report. It planned
145 cases with nine repetitions and a 7,200-second ceiling, retained
alternate-warmup and high-degree square-root failures, and was unpinned and
unfinished. That evidence supports investigating the old warmup behavior;
it does not establish statistical adequacy, justify excluding a backend, or
prove that the new slice was predeclared before results were inspected.
Machine-local runs remain untracked and are not publication evidence.

Before using this policy for a scientific claim, the campaign owner still
needs to justify repetition count and uncertainty estimates, timeout
censoring, backend coverage, and the square-root slice for the intended input
regime. Record that rationale before collecting the candidate. Changing the
policy further requires an explicit methodology decision.

## Gates and verification

The prepared export gate requires at least three repetitions and the
profile's declared minimum, clean committed sources, complete required-engine
identities, explicit CPU affinity, one engine thread, and successful
timing-eligible agreement for every required pair and repetition. An optional
timing series included in the report must cover every planned repetition;
an entirely unavailable optional adapter remains non-blocking. Failures,
timeouts, zero-resolution clocks, disagreements, and exclusions stay visible.
The required suite pair remains Silex/PARI. No new backend or proof mode is
introduced by this policy record.

Inspect the exact default selection without starting engines:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m silex_bench --json plan \
  --suite number-field --profile publication
```

`test/test_resources.py` checks the default profile and corpus counts.
`test/test_reporting.py` checks repetition, optional-series, and timing
admission blockers. These establish implemented behavior, not methodological
adequacy. Only an accepted `export --publication` establishes the harness's
eligibility gates; host idleness and the scientific rationale require their
own evidence.
