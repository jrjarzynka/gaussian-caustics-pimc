# Point 7 — Figure 2 exact-BZ anchors and Gamma-guide removal

Date: 2026-09-08
Status: predeclared before inspecting any new Point-7 exact-BZ anchor values.

## Purpose
Replace the dense Gamma-sector visual guide in Figure 2 by a denser set of genuine Brillouin-zone-averaged exact anchors, while retaining the already validated triangular benchmark and the central beta=2.5 pre-caustic result.

## Frozen beta grid

Use exactly 13 anchors:

`0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.10, 2.25, 2.35, 2.42, 2.48, 2.50, 2.54`

The first four legacy anchors 0.50, 1.00, 1.90, 2.50 must reproduce the archived LHA-12 exact-BZ values. No anchor may be added, removed, or shifted based on the new numerical curve.

All anchors must satisfy beta < beta_c = pi/sqrt(3/2) = 2.565099660... .

## Scientific sources of truth

Continuum exact BZ density/energy:
`validation_tests/lha12a1_exact_bz_density_reference.py`

Finite-P=64 primitive exact energy:
`validation_tests/lha12c0a_exact_P64_multibeta_converged.py`

SPLH reconstruction and registered 16x16 density integration:
`validation_tests/lha12c2a_harmonized_density_comparison.py`

Existing Figure 2 layout/source:
`validation_tests/p2c_make_figure2_before_caustic.py`

## Numerical controls

Continuum BZ production remains exactly K2=81, Nk=16x16, real-space density grid 64x64, with the source script's K2=64/Nk=12 basis control and Nk=12->16 k-grid control.

Primitive P=64 production remains exactly the converged source-script controls (K2=81, Nk=12 production with K2=64/Nk=8 basis control and Nk=8->12 k-grid control).

No Gamma-only result may be used as a plotted guide or interpolant.

## Figure quantities

Panel (a): signed SPLH energy error against continuum exact BZ,
`100*(V_SPLH - V_exact_BZ)/V_exact_BZ`.

Panel (b): registered 16x16 L1 distance between SPLH density and continuum exact-BZ density, using the same Fourier exact-bin integration and high-resolution SPLH bin integration as LHA-12C2a.

Panel (c): absolute SPLH energy error and absolute finite-P=64 primitive Trotter bias at the same 13 exact-BZ anchors.

The beta=2.5 annotation must use SPLH nomenclature and round the energy error to +17.40%.

## Acceptance gates

1. Legacy exact-BZ anchors at beta=0.5,1.0,1.9,2.5 reproduce archived values to <=1e-10 in continuum <V> and <=1e-10 L1 for the registered SPLH density comparison.
2. Continuum exact-BZ basis/k-grid controls remain numerically negligible relative to plotted effects; record maxima across all anchors.
3. P=64 basis/k-grid controls remain numerically negligible relative to plotted effects; record maxima across all anchors.
4. SPLH 16x16 bin integration 256->512 sensitivity remains <1e-4 L1 at every anchor.
5. Exact Fourier-bin imaginary residual remains <1e-12.
6. All 13 zeta_max < 1.
7. Figure 2 contains no Gamma-sector curve/guide and no legend entry referring to Gamma-sector data.
8. Do not change the underlying benchmark, formulas, exact-reference method, caustic location, or central beta=2.5 interpretation.

## Required outputs

- `point7_exact_bz_anchors.csv`
- `point7_exact_bz_anchors.npz`
- `point7_splh_bz_comparison.csv`
- `point7_fig2_anchor_audit.md`
- regenerated Figure 2 as PDF and PNG
- provenance hashes for source and new Point-7 scripts/results.
