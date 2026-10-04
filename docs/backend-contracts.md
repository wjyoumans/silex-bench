# Backend contract source trace

The adapter layer supplies orchestration and instrumentation; mathematical
calls and result normalization remain source-backed. Every engine receives the
same monic defining polynomial. Exact integers are serialized as canonical
decimal strings.

## Proven class and unit groups

- **Silex** invokes the maintained `silex-class-unit-instance` adapter in
  proven mode. A successful observation must publish the final result and
  completed certification, class-group, unit-group, and regulator proof states.
- **PARI/GP** prepares `nf = nfinit(P)` before timing, then times
  `bnfinit(nf, 1)` followed by a checked `bnfcertify`. The
  implementation lives in PARI's `src/basemath/buch2.c` and `buch3.c`. The
  adapter reads `no`, `cyc`, `disc`, `r1`, and `r2`, and reports the unit
  count as `#b.fu`, the number of fundamental units stored by
  `bnfinit(nf, 1)`. All proof labels come from the `bnfcertify` result. The
  adapter records the selected executable and observed version, and validates
  optional version/source pins when configured. Publication export requires
  the stronger source identity.
- **Hecke/OSCAR** constructs uncached fields and LLL-reduced maximal orders
  before timing and calls
  `class_group(...; GRH=false, redo=true, do_lll=false)`. The adapter passes
  `do_lll=false` explicitly (in the generated Julia program built in
  `src/silex_bench/backends/hecke.py`) because the order is already
  LLL-reduced before timing. The adapter reads `do_lll` as a tuning option
  and not a proof option, so the proof labels still come only from the `GRH`
  flags described below. This repository carries no pinned Hecke source, so
  that reading of Hecke's interface, including that `do_lll` leaves the
  non-GRH proof route unchanged, is not independently verified. The following `unit_group` lookup reads the
  shared class/unit result created within that same sample; fresh independently
  prepared orders prevent reuse across timing samples. It uses the active Julia
  environment unless a project override is configured. Because Julia is JIT
  compiled, profiles with `jit_repetitions = 1` record a `first_call` and
  `repeat_call` in the same process. The calls use independently prepared
  objects and prevent cross-sample mathematical-cache reuse; reports keep both
  timings as distinct series. Other profiles record one `standard` sample.
  The unit count is the length of the fundamental-unit list in the order's
  `UnitGrpCtx`, which is the list the returned unit-group map evaluates. The
  abstract group returned by `unit_group` is not used for the count because
  Hecke builds its free rank from the signature. This readback is a guard,
  not independent evidence of completeness: Hecke's own `add_unit!` only
  appends independent units until the list reaches the full unit rank, and
  `_class_unit_group` asserts that the stored `UnitGrpCtx` is full rank
  before returning, so the count can fail only on an engine-internal
  inconsistency or an adapter/attribute regression. The evidence for
  completeness is the `GRH` flag below. Proof labels come from Hecke's own
  `GRH` flags: the class-group status is proven when `ClassGrpCtx.GRH` is
  false, and the unit-group status is proven when `UnitGrpCtx.GRH` is false.
  Hecke clears those flags only after its unconditional proofs in
  `Clgp/Proof.jl`. The one exception is unit rank zero, where
  `_class_unit_group` skips the unit proof because the unit group is only
  torsion, so rank zero also counts as proven. Certification, the regulator,
  the class-group status, and `proof_complete` all require both `GRH` flags
  and a matching unit count. This is not merely a conjunction the adapter
  imposes: Hecke's class-group proof consumes the same `UnitGrpCtx` the unit
  count is read from. `_class_unit_group` asserts that `UnitGrpCtx` is full
  rank before calling `_class_group_proof` (`NfOrd/Clgp.jl`), and
  `_class_group_proof` saturates the class-group relation lattice together
  with that unit group (`saturate!`/`simplify` in `Clgp/Saturate.jl` fold
  every `U.units` entry into the lattice being saturated). A returned count
  below `r1 + r2 - 1` means the `UnitGrpCtx` the class proof used was not the
  full-rank group the assertion required; a count above `r1 + r2 - 1` means
  the list the adapter read is longer than the group the proof asserted and
  saturated against, so it is equally not that group. Either way, a count
  mismatch drops every label to unknown, class-group included. A JIT pair
  requires the flags and
  unit counts of both calls.
  The adapter relays Hecke's own proof flags as reported and adds no
  additional guard on them: `_unit_group_proof` (`Clgp/Proof.jl`) computes a
  regulator-index bound from `Nemo.unique_integer` and does not check that
  helper's own success flag, so in principle `U.GRH` could clear without a
  completed saturation step. This is an upstream Hecke soundness question,
  not a normalization or proof-rule choice for this adapter to second-guess,
  so the labels are reported exactly as Hecke sets them.
  Hecke's field and attribute names (`ClassGrpCtx`, `UnitGrpCtx`, `.GRH`,
  `.units`, `unit_group_rank`) are internal to the pinned Hecke version, not
  a published interface; an upstream Hecke upgrade can rename or remove them
  without notice, and the adapter has no independent way to detect that
  short of the resulting `get_attribute` lookup or field access failing.
- **Magma** calls `ClassGroup(O : Proof := "Full")` and
  `UnitGroup(O : GRH := false)`, matching the installed handbook contracts.
  The unit count is the number of infinite-order generators of the returned
  abstract unit group; each generator is also evaluated under the unit map,
  but the mapped values are discarded and do not affect the count. Magma is
  closed source, so whether that abstract group's free rank could ever
  differ from the field signature is not established; the count is reported
  as what it is, a structural
  property of the returned group, and not claimed as independent evidence.
  The V2.28 handbook documents no intrinsic that reports a computed group's
  proof state. Magma's proof labels therefore follow the call contract: with
  `Proof := "Full"` the class group is guaranteed, and with `GRH := false`
  `UnitGroup` runs its rigorous proof phase. A call that returns a complete,
  count-checked result is labeled proven; a unit-count mismatch labels the
  result unknown.

## GRH-conditional class and unit groups

`class_unit_grh` is an opt-in workload, separate from `class_unit_proven`. It
is selected with `--workload class_unit_grh`, is not part of the default
`number-field` suite or the publication profile, and only corpus rows that
carry a `grh` object (`expected_success`, `source`, optional
`timeout_seconds`) produce a case. Its case keys, ledger rows, agreements,
timing rows and report section are scoped by the workload id, so a grh
observation is never pooled with a proven one. The shared expected values on a
corpus row come from proven routes and remain the oracle for both workloads; a
grh result that differs from them fails validation and is never a timing
sample. The report summary keeps the populations apart as well: each
`backend_status` and `agreement_status` row has a `workload` field, and
`status_counts` maps each workload id to its own status counts (report renderer
version 3). Each engine's conditional route:

- **Silex** runs `silex-class-unit-instance --mode grh`. The echoed `mode`
  must be `grh`, which rejects binaries that predate grh mode. The final result
  must be published with class, unit and overall labels exactly `grh`. The
  regulator state is recorded and never required.
- **PARI/GP** times `bnfinit(nf, 1)` alone, the same call and flag as the first
  component of the proven route, with no `bnfcertify`. PARI documents the
  results as conditional on the GRH. Without `bnfcertify` the class number,
  structure, generators, regulator and units may be wrong, independently of
  each other. PARI's completeness check is computed in double precision with
  fitted constants; Bench labels PARI rows `grh` on PARI's documented contract
  and does not verify the rigor of that check.
- **Hecke/OSCAR** calls `class_group(...; GRH=true, redo=true, do_lll=false)`
  and `unit_group(...; GRH=true)`. Both `GRH` flags stay set. The class label is
  `grh` while `ClassGrpCtx.GRH` holds; the unit label is `grh` while
  `UnitGrpCtx.GRH` holds and the rank is positive, and `proven` at rank zero
  (a torsion-only unit group), the same flag rule as the proven route.
- **Magma** calls `ClassGroup(O : Proof := "GRH")` and
  `UnitGroup(O : GRH := true)`. The V2.28 handbook says the class group is
  correct under the GRH and that the unit computation has the same level of
  rigour. Magma reports no proof state, so the labels follow the call contract.
  Magma was not run to confirm this route; smoke-check an imaginary quadratic
  field, a real quadratic field and `x^3+x+200` against PARI before relying on
  it.

External adapters report `proof_complete: false` (no unconditional proof ran)
and `conditional_result_complete: true` when the unit count was read back and
every field is present. The overall `certification_status` of a successful grh
observation is `grh`, even when a component such as a rank-zero Hecke unit
group is proven. Validation requires the overall label `grh`, class and unit
labels `grh` or `proven`, and a published result; `unknown` and `heuristic`
fail. Agreement compares the same canonical fields as the proven workload
(class order, invariants, unit rank, signature, maximal-order discriminant),
not labels, and is consistency evidence among engines that all assume the GRH,
not proof. Reports render grh timings in their own "GRH-conditional" section
and never rank them against proven timings.

For every external engine, the unit-count readback runs after the target
marker and the internal clocks have stopped. It is outside the timed region,
but it counts toward the process wall time and the observation timeout. An
adapter marks a class/unit observation as a failed computation, with no
published final result, when the unit count is missing or differs from
`r1 + r2 - 1`. All three adapters also agree on proof labelling in that
case: a unit-count mismatch never leaves a result labeled proven, even for
the fields (PARI's `certified`, Hecke's class-group `GRH` flag) that a
successful earlier stage already set. PARI's `bnfcertify` proves the whole
bnf structure together, so a mismatch clears certification and all three
proof-status fields; Hecke's class-group proof consumes the same
`UnitGrpCtx` its unit-group proof does, so a mismatch clears every label,
class-group included; Magma has no independent per-stage flag, so a
mismatch clears every label. All three adapters also agree that
`certification_status` on a mismatch reads `unknown`, not `failed`: the
mismatch is a readback disagreement between the adapter and the engine's
returned unit group, not evidence that the engine's own certification step
failed, so none of the three claims a failure it cannot show.

Validation requires a positive class order, normalized invariant factors whose
product is the order, a signature of the field degree, the Dirichlet unit-rank
relation between the returned unit count and the signature, an exact
maximal-order discriminant, and completed proof metadata.
Pairwise agreement checks all of those canonical result fields.

## Other operations

- Maximal orders use PARI `nfinit`/`nfbasis`, Hecke maximal-order routines,
  Magma `MaximalOrder`/`Discriminant`, and the public Silex order API.
- Integral ideal multiplication compares `(2)(3)` and independently requires
  the exact norm `6^degree`.
- Element square-root recovery uses PARI `nfeltissquare`, Hecke
  `is_square_with_sqrt`, Magma `IsSquare`, and the public Silex element API.
  Each adapter verifies the recovered root by squaring it.

The S-class/S-unit route retains its stronger standalone proof, membership,
prime-witness, and valuation-lattice contract; see
[sunit-contract.md](sunit-contract.md).

## Execution and provenance boundary

Every ordinary observation starts a fresh process through the bounded process
helper. Stdout and stderr are drained independently with caps; timeouts and
protocol failures terminate and reap the process tree. Configured engines are
trusted local dependencies. Sealed executables, process groups, subreaping,
and cleanup improve lifecycle integrity but do not isolate malicious same-UID
programs.

At startup the supervisor requires `/proc/<pid>/task/<pid>/children`
(`CONFIG_PROC_CHILDREN`); without it, it exits with an error (code 126)
instead of silently treating every process as childless for its whole
lifetime. Live descendant enumeration reads every thread's children file
under `/proc/<pid>/task/*/children`, not only the thread-group leader's, so a
child forked by a non-leader thread is not missed. On a stop request
(timeout or interruption), the supervisor first sends `SIGKILL` straight to
the known target PID, preferring a pidfd-scoped signal (immune to PID reuse)
over a plain `kill(pid, ...)` where the pidfd is available, and this direct
kill is skipped entirely once the target has already been reaped: signalling
a PID (or a process group derived from it) after reap risks hitting an
unrelated same-UID process that the kernel has since given that recycled
PID. The supervisor also `SIGKILL`s the target's own process group, but only
when that group differs from the supervisor's own; when the target has not
called `setsid`, it shares the supervisor's group, and killing that group
would kill the supervisor itself before it can finish enumerating the wider
tree. Descendants that share the supervisor's own group are instead reached
by the `/proc` walk below. Independent of the direct kill, the supervisor
still runs one bounded enumeration-and-kill pass (up to 32 rounds of a 2 ms
freeze-and-confirm loop, then a single 0.75 s hard-kill loop) to confirm and
reap the wider descendant tree; that pass is not retried a second time on
failure. The 2 ms/0.75 s figures are a nominal sleep budget, not a wall-clock
bound: each round also pays for a full `/proc` walk, and the loop can
overshoot its deadline by up to one iteration, so actual worst-case latency
runs somewhat higher, especially for a large descendant tree or under load.
The harness's own stop path budgets 1.5 s of margin over that nominal budget
before forcing the supervisor's process group closed itself; giving the
supervisor a way to report when its own cleanup pass has actually finished,
instead of relying on a fixed wait, remains open as a follow-up.

The supervisor's control pipe stays open after it reports the target PID.
Later supervisor errors are reported on it, and just before exiting the
supervisor reports whether its exit status is the target's own or one it
produced itself: 124 after a stop request, or 126 after an internal failure
(such as a descendant-cleanup failure). Each process result records this as
`failure_origin`: `"target"` or `"supervisor"` for a failed run with an exit
status, and `null` for a successful run, a run without an exit status, or a
supervisor that exited without reporting the status the harness observed. A
target that itself exits with 124 or 126 is therefore distinguishable from the
supervisor. When the supervisor is the origin, its reported error text is
included in the result's `error`. The field is a process-helper result field;
backend adapters do not copy process exit statuses into observation payloads.

The whole-process envelope reported as `process_wall_ms` is measured by the
supervisor on `CLOCK_MONOTONIC`, from immediately before it spawns the target
to the moment it reaps the target, and is sent over the control pipe as a
`WALL <nanoseconds>` line. It excludes the harness-side launch, the
supervisor's interpreter startup, the handshake, and the exit-poll quantum.
The `taskset` launcher's exec is still inside the envelope, as it is part of
the spawned target. When no such line arrives (timeouts, stop requests,
supervisor failures) the harness-measured elapsed time at classification is
reported instead.

At the moment the target is reaped the supervisor enumerates its remaining
descendants. If any are alive it reports `OUTLIVED`, kills them immediately
(there is no polling drain), and the result is a failure with
`success = false`, `descendants_outlived_target = true`, and
`descendants_outlived_target` in `error` (appended to an existing error text).
The target's own `returncode` is preserved. A leader that exits and leaves a
live detached child is therefore never a success, and the child's lifetime is
never part of the envelope. `failure_origin` describes only who produced the
exit status; the outlived reason is carried by `error` and
`descendants_outlived_target`.

Timeout classification uses what the harness can already observe, not when it
happens to look. When the observation deadline has passed, the helper first
drains whatever output is already readable without waiting; if every captured
stream is already at end of file (the supervisor keeps both pipes open until
it exits) or the supervisor has already exited, the run is classified by its
exit, not as a timeout, even though the harness noticed it only after the
deadline.

When a CPU is pinned, only the target is placed on the requested CPU (through
`taskset`); the process supervisor itself runs on the remaining CPUs, chosen
by reading the target CPU's SMT sibling set from
`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`, so its own
scheduling and control-channel activity does not compete with the target for
that CPU's execution resources. The supervisor also waits for the target
event-driven (on the target's exit and on its own termination signals)
instead of polling. Where the available CPU set is too small to exclude the
full sibling set, the supervisor falls back in stages: first to any CPU
outside the target alone (still sharing SMT execution resources with it),
and only when the target CPU is the sole available CPU does the supervisor
share it, as it always did before this separation existed.

The primary ordinary comparison clock is each backend's internal wall
interval around the exact target operation. For class and unit groups, every
adapter constructs the field and maximal order before starting that interval;
only the class-group and unit-group calls are timed. Result extraction,
serialization, and correctness comparison happen afterward. The supervisor's
nonce-bound ready/target markers independently audit those boundaries and
enforce the observation deadline; its marked wall and CPU values remain in the
sample diagnostics: the marked CPU value
(`marked_target_cpu_ms`) is the whole thread group's user+system time from
`/proc/<pid>/stat` (including exited threads, clock-tick resolution) and is
diagnostic only; `target_cpu_ns` comes from the backend's own CPU value. With
a requested CPU, the affinity check covers every thread of the target. The marked wall interval (`target_wall_ms`) runs from
writing the target's input to observing its target marker line, so it also
includes writing that input (up to 1 MiB) and the harness's own wake latency
in reading the marker off the target's stdout; it is not purely the target's
own execution time. Integrated S-unit comparisons use their supervisor-measured
whole-process wall envelope and label that different scope and clock explicitly.
The supervisor also reports `effective_affinity` (read once the target is
ready) and, when a CPU was requested, `effective_affinity_after_target` (read
again after the target marker). When a CPU was requested, either read failing
or differing from that singleton CPU fails the sample as a protocol failure.
That failure does not set the timeout flag itself, but it is classified like
any other failure at the moment it is raised: if the observation deadline has
already passed and the target is still running, the row is recorded as a
timeout. Both reads inspect the target's thread-group leader only.

Timeout enforcement is best effort at the margins. A deadline that passes
while the harness is between steps (for example between the supervisor
handshake and the first read) still gets one non-blocking observation of the
process, so a successful sample can report a process wall time slightly above
the configured cutoff. Such a sample is kept as a success and is not
reclassified.

The target marker must be a complete line of its own. The harness scans the
target segment from the start of the stdout line in progress at dispatch, so a
leftover unterminated pre-dispatch line (for example a prompt) is joined to the
bytes that complete it whether it was read before or after dispatch. A line
that ends with the exact nonce-bound target marker but has other bytes before
it fails the observation as a protocol error ("emitted the target marker after
unterminated output on the same line"), not as a timeout. Because the nonce is
generated after readiness, ordinary output cannot end with it by accident. A
leftover partial line that the target terminates before printing its marker
does not affect the marker.

Clock identities remain backend-specific: Silex uses `steady_clock`, Hecke
uses Julia `time_ns`, PARI uses `getwalltime`, and Magma uses `Realtime`.
PARI 2.17.3 implements its wall timer in `src/language/init.c` using
`CLOCK_REALTIME` or `gettimeofday` where available; it is not a monotonic-clock
guarantee. No monotonic guarantee is claimed for Magma `Realtime`. The
supervisor's observation deadline uses a separate monotonic clock regardless
of the selected backend timer.

CPU-pinned Linux runs resolve `taskset` from the system default path, reject
group/world-writable path components, execute a sealed byte snapshot, and
record its path and SHA-256. UID 0 ownership is not treated as a security
boundary. Publication provenance instead relies on recorded content identity,
engine probes, source revisions and cleanliness, the materialized campaign,
and the requested singleton affinity.

Successful observations must carry exact Boolean lifecycle states,
object-shaped result/proof/timing payloads, nonnegative selected clocks, and a
complete backend-specific engine identity. A zero from a coarse backend clock
is retained but is not timing-eligible. Workload validation occurs before
pairwise comparison. Pairwise agreement occurs before timing eligibility. No
failure, malformed proof, disagreement, missing clock, or nonpositive clock can
enter an absolute-runtime summary.
