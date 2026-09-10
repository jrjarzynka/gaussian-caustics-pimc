# Paper 2 — Sec. VIII auxiliary-wall equilibrium sensitivity protocol

## Scientific question

Does the central interacting conclusion — that the finite-P equilibrium bead marginal is predominantly Gaussian-invalid — survive a deliberately large change in the auxiliary radial confinement?

This protocol changes the **sampled Hamiltonian**. It is therefore distinct from the existing Hessian-only `full-no-wall` reclassification.

## Predeclared Hamiltonian change

Canonical benchmark:

`V_wall(rho) = 0.08 eV * (rho / 15 nm)^8`

Sensitivity benchmark:

`V_wall(rho) = 0.08 eV * (rho / 30 nm)^8`

Everything else in the physical Hamiltonian is unchanged.

For a purely wall-controlled large-separation radial integral,

`Integral 2*pi*rho*exp[-beta*A*(rho/R)^8] d rho ∝ R^2`.

Thus doubling R from 15 to 30 nm increases that asymptotic configurational-volume scale by a factor of 4. This is a stress test, not a small perturbation.

## Fixed Monte Carlo protocol

- `T = 20 K`
- `P = 256`
- 4 independent chains
- seeds: `930300, 930301, 930302, 930303`
- `n_steps = 240000`
- raw trajectories stored from `burn_in = 15000`
- primary analysis burn-in: `60000`
- predeclared burn-in audit: `15000, 30000, 60000`
- `sample_every = 20`
- Brownian-bridge staging exactly as in archived Sec. VIII runtime
- local step `0.15 nm`
- global step `12 nm`
- global-move probability `0.20`
- same-slice electron-hole pairing
- independent chains, not beads, are the top-level statistical units

## Deliberately hostile initial conditions

Seeds `930300` and `930301` start compact: both carriers at the canonical true moire minimum.

Seeds `930302` and `930303` start dissociated: electron at the canonical minimum and hole displaced by `(0, L)` with `L=20 nm`. This is an exact Bravais translation of the analytic moire potential, so the two carriers start in equivalent one-body minima while their pair separation is 20 nm, beyond the isolated-BLK outer `zeta=1` crossing near 18.7 nm.

The two start classes are a mixing diagnostic, not separate statistical populations to be pooled selectively.

## Primary observable and predeclared gate

Primary observable: `p_inv` from the full wall-inclusive 4D mass-weighted Hessian evaluated on the newly equilibrated `Rwall=30 nm` trajectories.

Primary qualitative gate:

`lower endpoint of the 95% Student-t CI across the four independent chain means > 0.5`.

No requirement is imposed that the R=30 result remain numerically near 0.925. A substantial change is scientifically allowed; the test asks whether the **predominantly invalid** conclusion survives.

## Secondary diagnostics

Record per chain:

- `p_inv`
- no-wall-Hessian reclassification of the same R30 samples
- `zeta_50`
- `P(zeta >= 5)`
- median/mean/p95/p99/max pair separation
- `P(rho > 15 nm)`
- `P(rho > 18.7 nm)`
- `P(rho > 20 nm)`
- `P(rho > 30 nm)`
- first-half vs second-half `p_inv` and mean-rho drift
- 8-block diagnostics
- comparison of compact-start and dissociated-start chain means

The no-wall-Hessian quantity remains a **classification diagnostic**, not the equilibrium probability of a wall-free Hamiltonian.

## Source-of-truth freeze

The runner aborts unless the following archived Sec. VIII hashes match:

- `secVIII/run_tb03_pinv_ladder.py` — `63de5e2ba07f577a00627f22b28686ba1c9f57b66c5a5deee01411d99db8697b`
- `secVIII/lha15a0_pair_stability_field_TB03.py` — `f423834e37c640bf855e799e75080a34cc0fa2dd21a90dd0390390efb719d926`
- `secVIII/configs/two_body/landscape_scan_config_v2_P256_matched.json` — `9d7779ad7da3554fda7118b645547d7b251141286e738afc5f5254cfc2163719`
- `numerics/tmd_pimc/two_body_sampler_periodic_jit.py` — `34b33ae11fefe2c25ec9e6dd438fd9b6d7c18ad8c3c16510b71ce94faed82d2c`

Do not edit the archived production files to make the sensitivity run work. The new runner imports them and overrides the wall radius only in memory.
