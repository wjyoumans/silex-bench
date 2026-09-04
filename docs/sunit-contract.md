# S-class and S-unit comparison contract

`silex-bench` owns the cross-engine S-class/S-unit comparison contract and its
six-row corpus. It exposes the contract as the registered `sunit_proven`
workload for correctness checks and explicitly scoped whole-process timing
comparisons.

Run a focused comparison from this repository with:

```sh
PYTHONPATH=src python3 -m silex_bench check \
  --suite number-field --profile quick \
  --case cubic_x3_minus_2_empty_s \
  --backend silex --backend pari
```

The S-unit corpus schema is version 2. The loader reads at most 1 MiB through
a stable, regular-file, no-follow boundary, rejects duplicate JSON keys and excessive
nesting, and validates the complete manifest before it selects a field:
`fields` is a nonempty list with unique nonempty IDs, each
defining polynomial is a monic nonconstant integer coefficient list, and each
selected-prime entry has a prime signed-64-bit rational-prime value and either
a nonnegative index or `"all"`. Every row also declares the exact maximal-order
discriminant and a complete `prime_ideal_witnesses` list for every rational
prime named by a selector. A witness consists of `p`, a contiguous zero-based
`canonical_index`, positive `e` and `f`, and `beta_power_basis`: canonical
rational coefficients for an element beta in the defining polynomial's power
basis. It names the basis-independent two-generator ideal `(p, beta)`.
The manifest order of these exact witnesses, not an engine-local matrix sort,
defines each canonical index. Every row must provide exactly these typed and
canonical expectations:

Primality is checked deterministically in that bounded domain with the
seven-base Miller-Rabin set recorded by Jim Sinclair at
<https://miller-rabin.appspot.com/>. Values outside the domain fail schema
validation; this check validates benchmark input and does not replace a backend
proof operation.

- `s_class_order`, `torsion_order`, and `valuation_lattice_index` as canonical
  positive decimal strings;
- `s_class_invariants` as a nondecreasing list of canonical decimal strings at
  least two; and
- `ordinary_free_rank`, `nonunit_rank`, and `free_rank` as nonnegative integers,
  with `free_rank = ordinary_free_rank + nonunit_rank`.

A successful selected backend must publish all seven exact fields with those
same types. Its backend slot must match its reported `engine` identity, and the
identity must bind the exact immutable executable SHA-256 used for the
computation. PARI additionally requires its version probe and computation to
use the same executable digest. For each
manifest witness, the backend constructs `(p, beta)` and requires it to match
exactly one member of that backend's complete prime decomposition; every member
must be matched exactly once. This check is performed by native ideal equality,
not by relabeling the engine's local decomposition order. An explicit selector
publishes its named witnesses, while an `"all"` selector publishes every
witness for that rational prime.

Each selected-prime descriptor includes `p`, `canonical_index`, `e`, `f`, the
exact `beta_power_basis`, and the backend-local HNF and basis label. Each backend
also publishes `canonical_prime_decompositions`, containing the complete
manifest-ordered evidence for every rational prime named by the row. The basis
label is fixed by engine: `silex_maximal_order_basis_rows`,
`pari_bnf_integral_basis_rows`, or `hecke_maximal_order_basis_rows`. The HNF is
a degree-by-degree matrix of canonical integer strings in the declared local
basis. Its determinant has absolute value `p^f`, but matrix bytes are not a
cross-engine identity.

The runner independently validates the local evidence. Each backend publishes
its maximal-order basis in the common polynomial power basis and the basis
discriminant must equal the manifest value. Exact rational arithmetic verifies
that the basis contains one, is closed under multiplication, and is integral.
For each descriptor, the runner verifies the local HNF lattice equals the
two-generator ideal `(p, beta)`: the lattice has index `p^f`, contains `pO` and
`beta O`, and has the same image as those generators modulo `p`. Thus a
correctly shaped, correctly normed nonideal HNF cannot authenticate itself.

Descriptor rational-prime and canonical-index coverage must match the row's
selectors, and the selected descriptor count must equal the expected
`nonunit_rank`. Descriptor objects must be unique; repeating a local HNF cannot
satisfy distinct witnesses. Malformed, empty, partial, duplicate, wrong-prime,
wrong-index, wrong-witness, wrong-basis, or wrong-lattice descriptor sets fail
closed. Agreement also requires every selected backend to succeed, satisfy its
proof and membership contracts, match the corpus expectations, and agree on
the normalized exact fields and complete manifest witness identities
`(p, canonical_index, e, f, beta_power_basis)`. Local HNFs remain
basis-specific and are compared only with that backend's complete evidence.

The native Silex process and payload `success` states, timeout state,
membership checks, and final-publication state are exact booleans; truthy JSON
values are not accepted as substitutes, and success requires an exact false
timeout. Native payload, `sunit`, class-group, unit-group, membership,
selected-prime, descriptor, and HNF containers retain their exact object/array
shapes. Before normalization, native output uses the repository's bounded JSON
parser: duplicate keys, non-finite values, invalid UTF-8 decode evidence,
oversized input, and excessive nesting fail closed. Malformed native JSON and
external descriptor integers produce structured failed results rather than
coercion or tracebacks. A reported
`regulator_midpoint` is
optional, but every present value must be a finite JSON number before regulator
agreement can succeed.  `NaN` and either infinity fail agreement even when all
selected backends report the same non-finite value.

For ordinary class/unit proof states, Silex must report `"proven"` for
certification, class group, and unit group, and exactly `"verified"` for its
native regulator proof. PARI reports `"proven"` for all four fields. Hecke must
report the row's declared external certification state for all four fields.

The Silex adapter invokes the native `silex-class-unit-instance` executable
directly through the benchmark process helper's sealed executable snapshot; it
does not delegate correctness-sensitive execution to the sibling repository's
Python instance driver. Its source and build roots remain explicit command-line
inputs. PARI uses `bnfinit(P, 1)`, checked `bnfcertify`, and `bnfsunit` through
the GP executable selected from configuration or PATH; its observed version
and executable digest are recorded. Hecke uses the active Julia environment
and its explicit non-GRH class/unit and factored S-unit operations unless a
corpus row declares a GRH comparison. The optional
`external_hecke_certification` field accepts
only `"proven"` or `"grh"`; omission means `"proven"`, and every other value is
rejected. Exact upstream source traces remain in the private agent references
and the general [backend contracts](backend-contracts.md).

The campaign admits S-unit timings only when each individual backend satisfies
this contract and the selected pair agrees. They remain labeled
`whole_process`, so reports must not pool them with ordinary marked-target
timings. Magma currently declares this workload unsupported.
