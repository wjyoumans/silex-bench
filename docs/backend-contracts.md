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
sample diagnostics. Integrated S-unit comparisons use their supervisor-measured
whole-process wall envelope and label that different scope and clock explicitly.

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
