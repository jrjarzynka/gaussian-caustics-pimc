# Point 7 — Figure 2 exact-BZ anchors / Gamma-guide removal — forensic audit

Date: 2026-09-08

## Verdict

**PASS / Point 7 can be closed scientifically.**

Figure 2 has been rebuilt from 13 predeclared Brillouin-zone-averaged exact anchors. The former dense Gamma-sector guide is not used in the final plotted data and no Gamma-sector curve or legend entry appears in the regenerated figure.

## Frozen anchor set

`beta = 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.10, 2.25, 2.35, 2.42, 2.48, 2.50, 2.54`

The anchor grid was fixed before inspection of the new exact-BZ values. No point was added, removed, or shifted after seeing the curve.

All anchors are globally pre-caustic; the largest is

`zeta_max(beta=2.54) = 0.990214937567 < 1`.

## Scientific source identity

The Point-7 calculations preserve the established methods in:

- `lha12a1_exact_bz_density_reference.py` — continuum exact BZ density and energy;
- `lha12c0a_exact_P64_multibeta_converged.py` — exact primitive P=64 energy;
- `lha12c2a_harmonized_density_comparison.py` — SPLH reconstruction and registered 16x16 bin integration;
- `p2c_make_figure2_before_caustic.py` — legacy Figure-2 layout.

The only scientific extension is evaluation at the frozen 13-beta grid. Finite-P anchors were executed in independent parallel one-beta batches solely to fit execution wall-time limits; every batch retained the identical K2/Nk controls and formulas. No result-dependent tuning was performed.

The Gamma-only density calculation remains present inside the canonical exact-reference source as an internal diagnostic, but it is not used to form any final Point-7 plotted quantity.

## Legacy-anchor reconstruction

The four previous BZ anchors beta=0.5, 1.0, 1.9, 2.5 reproduce the archived values essentially to floating-point roundoff:

- max continuum-energy difference: `2.44249e-15`;
- max P=64-energy difference: `1.33227e-15`;
- max SPLH-energy difference: `0`;
- max registered SPLH-density-L1 difference: `6.66134e-16`.

## Convergence gates across all 13 anchors

- max continuum BZ basis-density L1, K2=64 -> 81 at Nk=12: `4.5349172e-7` — PASS;
- max continuum BZ k-grid density L1, Nk=12 -> 16 at K2=81: `2.7927697e-11` — PASS;
- max P=64 basis energy delta, K2=64 -> 81: `6.0305815e-7` — PASS;
- max P=64 k-grid energy delta, Nk=8 -> 12 at K2=81: `9.5602193e-11` — PASS;
- max SPLH 16x16 bin integration sensitivity, 256 -> 512: `6.3569774e-5` — PASS;
- max exact Fourier-bin imaginary residual: `2.1094237e-15` — PASS;
- all zeta_max < 1 — PASS.

## Central beta=2.5 anchor

- `zeta_max = 0.974621001542`;
- `V_exact_BZ = 0.712951818447`;
- `V_SPLH = 0.836975604153`;
- SPLH energy error = `+17.3958158878%`, reported as **+17.40%**;
- registered 16x16 SPLH density L1 = `0.115041336100`;
- exact primitive P=64 Trotter bias = `-0.020376866640%`;
- absolute error-scale separation = `853.704x`, reported as about **854x**.

## New near-caustic exact anchor

At beta=2.54, still before the caustic:

- `zeta_max = 0.990214937567`;
- SPLH energy error = `+21.251763%`;
- registered SPLH density L1 = `0.134162249`.

This point strengthens the resolved trend but does not replace beta=2.5 as the manuscript's central benchmark.

## Figure audit

Final Figure 2:

- contains 13 genuine BZ-exact markers;
- contains no dense Gamma-sector guide;
- uses SPLH nomenclature;
- annotates beta=2.5 as `+17.40%`;
- retains the analytic caustic line at `beta_c = pi/sqrt(3/2)`;
- shows exact P=64 primitive bias only at the same frozen BZ anchors;
- uses markers without connecting interpolation curves, avoiding any implication that uncomputed intermediate points are exact data.

No manuscript, Supplement, or other publication figure was modified during this Point-7 calculation.
