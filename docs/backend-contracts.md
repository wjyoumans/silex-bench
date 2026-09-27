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
  adapter reads `no`, `cyc`, `fu`, `disc`, and `r1`, records the selected
  executable and observed version, and validates optional version/source pins
  when configured. Publication export requires the stronger source identity.
- **Hecke/OSCAR** constructs uncached fields and LLL-reduced maximal orders
  before timing and calls the non-GRH
  `class_group(...; GRH=false, redo=true)` route in Hecke's
  `src/NumFieldOrd/NfOrd/Clgp.jl`. The following `unit_group` lookup reads the
  shared class/unit result created within that same sample; fresh independently
  prepared orders prevent reuse across timing samples. It uses the active Julia
  environment unless a project override is configured. Because Julia is JIT
  compiled, profiles with `jit_repetitions = 1` record a `first_call` and
  `repeat_call` in the same process. The calls use independently prepared
  objects and prevent cross-sample mathematical-cache reuse; reports keep both
  timings as distinct series. Other profiles record one `standard` sample.
- **Magma** calls `ClassGroup(O : Proof := "Full")` and
  `UnitGroup(O : GRH := false)`, matching the installed handbook contracts.

Validation requires a positive class order, normalized invariant factors whose
product is the order, a signature of the field degree, the Dirichlet unit-rank
relation, an exact maximal-order discriminant, and completed proof metadata.
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
sample diagnostics. The marked wall interval (`target_wall_ms`) runs from
writing the target's input to observing its target marker line, so it also
includes writing that input (up to 1 MiB) and the harness's own wake latency
in reading the marker off the target's stdout; it is not purely the target's
own execution time. Integrated S-unit comparisons use their supervisor-measured
whole-process wall envelope and label that different scope and clock explicitly.
The supervisor also reports `effective_affinity` (read once the target is
ready) and, when a CPU was requested, `effective_affinity_after_target` (read
again after the target marker). When a CPU was requested, either read failing
or differing from that singleton CPU fails the sample as a protocol failure,
not a timeout. Both reads inspect the target's thread-group leader only.

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
