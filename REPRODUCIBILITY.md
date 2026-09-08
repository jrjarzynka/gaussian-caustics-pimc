# Reproducibility guide

This guide distinguishes three levels: deterministic reconstruction, replay of
archived Monte Carlo outputs, and full stochastic reruns. The manuscript does
not treat beads as independent top-level observations; independent chains are
the statistical units where uncertainty is reported.

## 1. Tensorial harmonic validation and first caustic (Fig. 1)

Foundational calculations:

```bash
python validation_tests/lha10d1_anisotropic_harmonic_2d.py
python validation_tests/lha10d2_negative_curvature_caustic.py
```

Publication rendering:

```bash
python figure_scripts/p2f_make_figure1_tensorial_theory_massweighted_v63.py
```

Authoritative compact inputs are in `figure_data/`.

## 2. Symmetric pre-caustic accuracy and classical baseline (Fig. 2)

The final exact-BZ anchors and their frozen provenance are in
`point7_exact_bz/point7_work/`. Rebuild the publication plot with:

```bash
python figure_scripts/p2c_make_figure2_before_caustic_v63.py
```

The central state point is beta=2.5 with zeta_max<1; the figure compares exact-BZ,
SPLH, classical, and primitive finite-P/Trotter scales.

## 3. Asymmetric off-stationary caustic (Fig. 3)

Foundational scripts:

```bash
python validation_tests/lha13a0_asymmetric_moire_geometry.py
python validation_tests/lha13a1_exact_bz_reference.py
python validation_tests/lha13a2_tensor_rlh_offstationary_caustic.py
python validation_tests/lha13a3_probability_weighted_offstationary_caustic.py
python validation_tests/lha13a3b_caustic_grid_convergence.py
```

The `tensor_rlh` filename is historical; see `LEGACY_NAMING.md`.
Publication rendering:

```bash
python figure_scripts/p2d_make_figure3_offstationary_caustic_v63.py
```

## 4. Finite-P / PI-QMC density validation and noise floor (Fig. 4)

The publication PI-QMC chain outputs and deterministic finite-P reference are
under `validation_tests/`. The final chain-level residual bootstrap is:

```bash
python paper2_secVII_l1_noise_floor/analyze_point6_l1_noise_floor_variance_corrected.py
```

It multiplies centered whole-chain residuals by `sqrt(16/15)` before the fixed
200000-replicate bootstrap. Final values are in
`paper2_secVII_l1_noise_floor/results/` and mirrored under `validation/`.
Render Fig. 4 with:

```bash
python figure_scripts/make_fig4_point6_v63.py
```

## 5. Long-statistics energy check

The complete 16-chain campaign (seeds 132001--132016), per-chain NPZ outputs,
analysis, and provenance are in `paper2_secVII_energy_longcheck/`.
To inspect the archived campaign, use the saved result files. To rerun it, follow
that directory's `RUN_COMMAND.txt` after installing the package.

## 6. Stage-independence and scalar three-stage control

Correct first-boundary regression:

```bash
python validation/stage_independence_corrected.py
```

Three-stage double-well density benchmark:

```bash
python validation/three_stage_doublewell_gate_v63.py
python validation/three_stage_doublewell_energy_origin_sensitivity.py
```

The final mapped density is tested under both the literal published additive
constant and a minimum-zero shift. Exact, bare, and renormalized normalized
densities are invariant to that constant shift; the final mapped density is not.

## 7. Interacting unequal-mass benchmark (Figs. 5 and 6)

The canonical finite-P aggregate and per-seed summaries are under `secVIII/results/`.
The production runner and configuration are preserved under `secVIII/`.
The publication finite-P summary figure is rebuilt by:

```bash
python figure_scripts/p2f_make_figure6_interacting_caustic_v63.py
```

Mode-attribution inputs are:

```text
paper2_reviewer_hardening_audit/mode_attribution_P256.csv
paper2_reviewer_hardening_audit/mode_attribution_distributions_P256.csv
paper2_reviewer_hardening_audit/counterfactual_stability.csv
```

The final vector Fig. 5 is preserved in `figures/Fig5_mode_attribution.pdf`.

## 8. P=256 long-chain stationarity (Fig. S1)

The four long raw trajectories are intentionally omitted from ordinary Git due
to size. Their SHA-256 hashes and the four derived observable-cache hashes are
listed in `provenance/OMITTED_LARGE_DATA.csv`. The figure and all aggregate/block
statistics are present in Git. A fully archived replay uses the Zenodo raw-data
package once its DOI is minted.

## 9. Manuscript

Compile the exact v6.3.3 source from the repository root using
`BUILD_INSTRUCTIONS.md`. Before journal submission, fill only the release
metadata placeholders documented there; do not change scientific values merely
as part of the repository release.
