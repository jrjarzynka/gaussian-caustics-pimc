# Double-well convention provenance audit — Ref. [1]

## Question

Does the one-dimensional three-stage gate use the intended double-well
Hamiltonian of Sadhasivam, Althorpe, and Kapil, or should the published
quartic term `g x^4 / 4` be replaced by `g x^4` because of the additive
constant `1/(16 g)`?

## Published convention

The published/preprint text states Eq. (44), with `m = omega = 1`, as

`V(x) = -x^2/2 + g x^4/4 + 1/(16 g)`

for `g = 0.1` and `g = 0.5`, and labels the former a deep well and the
latter a shallow well. The public Cambridge record DOI
`10.17863/CAM.129713` identifies its `FIG3` dataset as the double-well
benchmark supporting the same paper.

## Algebra of the printed Eq. (44)

For the printed potential, the nonzero minima satisfy `x_min^2 = 1/g`,
so `V_min = -3/(16 g)`, while `V(0) = 1/(16 g)`. The additive constant
therefore does **not** set the minima to zero. The barrier height relative
to the minima is `Delta V = 1/(4 g)`.

| g | V_min | V(0) | Delta V |
|---:|---:|---:|---:|
| 0.1 | -1.875 | 0.625 | 2.5 |
| 0.5 | -0.375 | 0.125 | 0.5 |

The curvature is `V''(x) = -1 + 3 g x^2`, so `min V'' = -1` at `x=0`,
and the first scalar Gaussian caustic remains `beta_c = pi`.

## Independent spectral consistency check

A converged finite-difference diagonalization gives the number of
eigenvalues below the central barrier `V(0)`:

- printed Eq. (44), `g=0.1`: **4**;
- printed Eq. (44), `g=0.5`: **1**.

This matches the paper's qualitative description: `g=0.1` is the deeper
case with a comparatively large density of intrawell states, while in the
`g=0.5` shallow case the number of states in the wells is less than or equal
to one.

For the alternative zero-minimum convention
`V_alt=-x^2/2+g x^4+1/(16g)`, the corresponding below-barrier counts are
1 (`g=0.1`) and 0 (`g=0.5`). In particular, its `g=0.1` case does not match
the paper's stated deep-well / large-intrawell-state characterization.

## Figure-level consistency

The published Fig. 3 uses the same `g=0.1` / `g=0.5` deep-versus-shallow
classification and captions the plotted potential as Eq. (44), including
the `g x^4/4` term. Thus the printed equation and its physical
interpretation support retaining the `/4` convention in the gate.

## Decision

**PASS — retain `g x^4/4`.**

The v6.3 numerical gate already implements the literal published Eq. (44)
and does not require recomputation. The earlier explanatory sentence that
the "chosen zero" makes the shallow-well energy small was misleading and
has been removed. The Supplement now states explicitly that the published
offset does not set `V_min=0`, gives the correct barrier heights 2.5 and
0.5, and uses normalized-density L1 as the primary metric.

## Remaining provenance limitation

The Apollo landing page and published paper were accessible during this
audit. The binary `manu_data_eff.zip` endpoint returned a cache miss in the
present tool environment, so this audit does not claim a byte-level
comparison against the raw FIG3 files. No numerical claim in the gate
depends on such a comparison: the implemented Hamiltonian is the literal
published Eq. (44), and the independent spectral check matches the paper's
stated well-depth/state-count interpretation.
