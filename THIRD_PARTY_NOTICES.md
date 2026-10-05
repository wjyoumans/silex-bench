# Third-Party Notices

## Project license and fresh-history notice

Silex Bench is distributed under the GNU General Public License, version 3 or
(at your option) any later version (`GPL-3.0-or-later`). The license text is in
`LICENSE`.

This repository began with a curated fresh-history import from Silex's
pre-public benchmark tooling. Fresh history changes repository organization;
it does not erase authorship, source provenance, or applicable license
obligations.

The S-unit comparison runner and corpus were first developed in the historical
native Silex tree and moved to Silex Bench before its first public release.
The current foundation preserves that lineage rather than claiming an
independent or clean-room origin.

## PARI/GP source lineage

The S-regulator computation in
`src/silex_bench/sunit_backend.py:2314-2320` translates the identity
implemented by PARI/GP 2.17.3 in `src/basemath/bnfunits.c:203-234`, especially
lines 213-228, and documented in `doc/usersch3.tex:20056-20071`: the ordinary
regulator is multiplied by the S-class number and the logarithms of the norms
of the selected prime ideals. This material was translated and adapted for
Silex and Silex Bench in 2026.

The S-unit corpus also retains fixture and test provenance from PARI/GP 2.17.3,
including `src/test/in/bnfsunit:2-34` and `src/test/32/bnfsunit:2-30` for the
`x^3 - 2`, empty-S, `x^2 + 23`, and `x^2 - 210` cases.

PARI/GP is copyright (C) 2000-2023 The PARI Group, Bordeaux, with additional
authors and component notices recorded by its distribution. PARI/GP 2.17.3 is
licensed under the GNU General Public License, version 2 or (at your option) any
later version (`GPL-2.0-or-later`). Silex Bench's `GPL-3.0-or-later`
distribution terms preserve the applicable copyleft obligations for retained
translated material. Complete PARI/GP notices remain available from the
corresponding PARI/GP source distribution.

## External engines and libraries

Silex Bench can invoke these separately installed systems:

- Silex;
- PARI/GP;
- Hecke.jl and Julia;
- Magma.

It may optionally use Matplotlib for plots. These external programs and
libraries are not bundled in this repository and remain under the copyright
and license terms supplied by their respective distributors. Magma is
proprietary and optional; this repository includes no Magma program source or
documentation.

The mathematical calls, proof expectations, and source versions used for
comparison are documented in `docs/backend-contracts.md` and
`docs/sunit-contract.md`. Those references identify provenance and behavior;
they do not change the licensing of an external system.
