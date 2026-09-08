# Gaussian caustics in multidimensional and interacting systems

Reproducibility package for the manuscript

> **Local Stability Does Not Guarantee Quantum Accuracy: Gaussian Caustics in Multidimensional and Interacting Systems**

Author: **Jaroslaw R. Jarzynka**  
Target journal: *Physical Review E*  
Repository state: **v1.0.0-rc1 reproducibility release candidate, aligned to manuscript v6.3.3**

This repository contains the numerical library, deterministic quantum references,
PI-QMC validation outputs, caustic/stability analyses, publication figures, and
compact provenance needed to support the manuscript. Manuscript and Supplemental
Material source files and compiled article PDFs are intentionally maintained outside
this Git release candidate; `provenance/MANUSCRIPT_REFERENCE_FREEZE_v6_3_3.md`
records the manuscript snapshot to which this reproducibility package is aligned.

## Scientific scope

The tensorial SPLH object benchmarked in the multidimensional sections is the
**renormalized pre-harmonic-mapping** positional distribution of Sadhasivam,
Althorpe, and Kapil and its mass-weighted tensorial extension. The scalar
three-stage control separately tests the bare, renormalized pre-mapping, and
final mapped forms. The first scalar Gaussian-existence boundary is shared by
all three stages at `zeta = 1`; quantitative accuracy is assessed independently.

Historical filenames and variable names containing `RLH` are preserved where
needed for provenance. See `LEGACY_NAMING.md`; these labels must not be read as
meaning the bare/unrenormalized stage.

## Repository map

- `numerics/tmd_pimc/` — PI-QMC numerical library used by the validation code.
- `validation_tests/` — foundational harmonic, periodic, exact-reference,
  finite-P, PI-QMC, asymmetric-caustic, and interacting validation scripts.
- `validation/` — **current v6.3.3 hardening checks**, including the corrected
  stage-independence, three-stage double-well, additive-energy-origin, BLK, and
  Point-6 variance-correction audits.
- `paper2_secVII_energy_longcheck/` — full 16-chain long-statistics energy check.
- `paper2_secVII_l1_noise_floor/` — 8x8/16x16/32x32 chain-level density-noise-floor
  analysis; `results/` contains the final `sqrt(16/15)` variance-corrected result.
- `point7_exact_bz/` — exact-BZ anchor protocol, run scripts, source snapshots,
  anchors, audits, and figure reconstruction assets.
- `secVIII/` — interacting-pair runner, configuration, stability field, and
  canonical aggregate results.
- `paper2_P256_long_stationarity_spotcheck/` — long-chain stationarity summaries,
  source provenance, and Fig. S1 source PNG. Raw trajectories are archived
  separately for Zenodo.
- `paper2_reviewer_hardening_audit/` — retained numerical Hessian/mode-attribution
  artifacts used by the interacting analysis; superseded manuscript-patch files
  have been removed.
- `figure_data/`, `figure_scripts/`, `figures/` — authoritative v6.3.3 figure
  inputs, generators, and publication assets.
- `provenance/` — source-bundle lineage and public release manifests.

## Environment

The audited environment was:

```text
Python      3.11.15
NumPy       2.4.6
SciPy       1.17.1
Numba       0.66.0
pandas      3.0.5
matplotlib  3.11.1
pytest      9.1.1
```

For a close reproduction:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e .
```

The looser compatibility requirements are in `requirements.txt`.

## Fast deterministic checks

From the repository root:

```bash
python validation/stage_independence_corrected.py
python validation/three_stage_doublewell_gate_v63.py
python validation/three_stage_doublewell_energy_origin_sensitivity.py
python paper2_secVII_l1_noise_floor/analyze_point6_l1_noise_floor_variance_corrected.py
```

The first three require no Monte Carlo. Point 6 replays the already archived
whole-chain histograms and uses a fixed bootstrap seed; it does **not** treat
beads or sweeps as independent observations.

## Publication figures

Figures 1, 2, 3, 4, and 6 can be rebuilt from compact authoritative inputs:

```bash
python figure_scripts/p2f_make_figure1_tensorial_theory_massweighted_v63.py
python figure_scripts/p2c_make_figure2_before_caustic_v63.py
python figure_scripts/p2d_make_figure3_offstationary_caustic_v63.py
python figure_scripts/make_fig4_point6_v63.py
python figure_scripts/p2f_make_figure6_interacting_caustic_v63.py
```

Figure 5 is preserved as the final vector PDF together with the mode-attribution
CSV data and generator under `paper2_reviewer_hardening_audit/`. Fig. S1 is the
author-supplied long-chain stationarity image, preserved as PNG and as the PDF
wrapper used by the Supplement.

## Longer / stochastic calculations

The full long-energy campaign and interacting finite-P campaign are deliberately
separate from the fast checks. Their frozen commands, seeds, inputs, and outputs
are preserved in their respective directories. Running them again can be
expensive and is not required merely to inspect the published results.

## Reproduction matrix and data archive

See:

- `REPRODUCIBILITY.md` — claim-by-claim reproduction guide;
- `REPRODUCTION_MATRIX.csv` — machine-readable map from claims/figures to code/data;
- `DATA_AVAILABILITY.md` — what is in Git and what is reserved for Zenodo;
- `provenance/OMITTED_LARGE_DATA.csv` — exact hashes of the raw P=256 long chains
  and derived caches omitted from ordinary Git;
- `SHA256SUMS` — hashes of the complete public-candidate repository tree.

## Relationship to the manuscript

This repository is the reproducibility/code-and-data release, not the working
manuscript repository. The RC1 scientific contents are aligned to the v6.3.3
submission-candidate reference freeze recorded under `provenance/`. The final
public `v1.0.0` release will point to the corresponding article and immutable
Zenodo archive once their identifiers exist.

## License and citation

The software is released under the MIT License; see `LICENSE` and
`LICENSE_SCOPE.md`. Citation metadata are provided in `CITATION.cff`. The final
repository URL and Zenodo DOI will be added at release freeze.
