# v1.0.1 — PRE referee-revision reproducibility release

This release updates the reproducibility package for the hardened PRE
referee revision.

Main additions:

- `R_wall = 30 nm` equilibrium sensitivity campaign at `P=256`, `T=20 K`;
- four independent 240000-step chains with compact and dissociated
  initializations;
- chain-level Student-t uncertainty, burn-in and block diagnostics;
- Hessian no-wall reclassification on the R30 ensemble;
- compact R30 result summaries and source provenance.

The original canonical `R_wall = 15 nm` production calculations remain
unchanged.

Large raw trajectories remain outside ordinary Git and are archived in
Zenodo version 1.0.1:

https://doi.org/10.5281/zenodo.22695962
