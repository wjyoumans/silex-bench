# Backend contract source trace

The adapter layer supplies orchestration and instrumentation; mathematical
calls and result normalization remain source-backed. Every engine receives the
same monic defining polynomial. Exact integers are serialized as canonical
decimal strings.

## Proven class and unit groups

- **Silex** invokes the maintained `silex-class-unit-instance` adapter in
  proven mode. A successful observation must publish the final result and
  completed certification, class-group, unit-group, and regulator proof states.
- **PARI/GP** uses `bnfinit(P, 1)` followed by a checked `bnfcertify`. The
  implementation lives in PARI's `src/basemath/buch2.c` and `buch3.c`. The
  adapter reads `no`, `cyc`, `fu`, `disc`, and `r1`, records the selected
  executable and observed version, and validates optional version/source pins
  when configured. Publication export requires the stronger source identity.
- **Hecke/OSCAR** constructs an uncached field and calls the non-GRH
  `class_group(...; GRH=false, redo=true)` and `unit_group(...; GRH=false)`
  routes in Hecke's `src/NumFieldOrd/NfOrd/Clgp.jl`. It uses the active Julia
  environment unless a project override is configured.
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

The external comparison clock is the supervisor's marked target wall interval:
it begins when the target command is dispatched after optional preparation and
ends at the target-done marker. It includes phase handoff, command parsing,
marker I/O, and small post-operation audit work; it excludes process startup,
warmup, and final publication. Internal CPU and wall clocks are retained as
diagnostic evidence, not silently substituted into cross-engine ratios.
Integrated S-unit comparisons use their whole-process wall envelope and label
that different scope explicitly.

CPU-pinned Linux runs resolve `taskset` from the system default path, reject
group/world-writable path components, execute a sealed byte snapshot, and
record its path and SHA-256. UID 0 ownership is not treated as a security
boundary. Publication provenance instead relies on recorded content identity,
engine probes, source revisions and cleanliness, the materialized campaign,
and the requested singleton affinity.

Successful observations must carry exact Boolean lifecycle states,
object-shaped result/proof/timing payloads, positive selected clocks, and a
complete backend-specific engine identity. Workload validation occurs before
pairwise comparison. Pairwise agreement occurs before timing eligibility. No
failure, malformed proof, disagreement, or missing clock can produce a
performance ratio.
