# Point 6 — Sec. VII L1 density noise floor: forensic audit

## Verdict

**PASS. Point 6 can be closed scientifically.**

The original publication-density PI-QMC discrepancy is statistically consistent with the positive chain-level sampling floor at 8x8, 16x16, and 32x32 registered bins. The deterministic SPLH discrepancy remains far larger than that floor at every resolution.

## Source requirement and predeclaration

The master handover required a chain-aware L1 sampling/noise-floor calculation and binning sensitivity at 8x8, 16x16, and 32x32, with chains rather than beads as the top-level statistical units. The Point-6 protocol was frozen before any new Point-6 L1 result was inspected.

The first attempted recovery of Point-5 long-run density was rejected under its strict predeclared bitwise energy-array gate: sampler acceptance and means matched, but postprocessed energy arrays differed by ~2.9e-15. The gate was not relaxed after inspection.

A separate archival 32-bin recovery protocol was then frozen before any 8/16/32 Point-6 L1 result was inspected.

## Archival trajectory recovery

The publication ensemble seeds 131001..131016 was replayed using the frozen publication controls:

- beta*=2.70;
- P=64;
- 40000 sweeps per chain;
- 8000 burn-in;
- sample_every=5;
- same initialization `default_rng(seed+991)`;
- same Hamiltonian, physical embedding, 512x512 potential grid, proposal scales, periodic wrapping, and `run_pimc_core_jit_periodic_cell` kernel;
- no staging and no tuning.

Only post-sampler 32x32 histogram accumulation was added.

**Replay gate: PASS 16/16.** For every chain, the replayed 32x32 full, first-half, and second-half physical-bead count arrays, after exact adjacent 2x2 aggregation, are integer-array-equal to the corresponding archived 16x16 publication histograms. Thus the 32x32 arrays are a lossless finer-bin reconstruction of the same archived publication trajectories, not a new independent Monte Carlo ensemble.

Each chain contributes 409600 physical-bead entries; pooled count is 6553600. No counts are lost under 32->16->8 aggregation.

## Deterministic target audit

The exact finite-P production density matrix was recomputed once at K2=100, Nk=16, then analytically bin-integrated at 8x8, 16x16, and 32x32 using the same Fourier/bin-integration formula as the historical deterministic reference.

At 16x16 the recomputed continuum density differs from the archived reference by at most 2.7e-15; the recomputed finite-P density differs by at most 2.1e-13. The P=64 energy is 0.6567185985810481, consistent with the archived 0.6567185985810541 at floating-point roundoff scale.

The exact multibin targets are internally nested: averaging 32x32 bin densities into 16x16 or 8x8 reproduces the directly integrated coarser targets to <=2.7e-15.

The SPLH density was recomputed on the historical N=1024 midpoint grid at all three binnings. At 16x16 it reproduces the archived SPLH/RLH density to max absolute difference 7.3e-15. SPLH 32->16 and 32->8 nested-bin consistency is <=5.4e-15.

## Statistical method

For each resolution, the 16 normalized whole-chain density vectors are the independent observations. Let `pbar` be their mean. Centered whole-chain residuals `r_i = p_i - pbar` were resampled with replacement under the exact-finite-P null. For each of 200000 fixed bootstrap replicates (seed 20260908), the null statistic is the L1 norm of the mean resampled residual. This estimates the positive L1 floor without treating beads or sweeps as IID.

The same bootstrap index matrix was reused for 8x8, 16x16, and 32x32. No linear noise-floor subtraction was performed.

## Results

| bins | PI-QMC vs exact P64 | null median | null 95% interval | null 99% interval | p_null | SPLH vs exact P64 |
|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 0.017726009 | 0.016839600 | [0.009600510, 0.028330383] | [0.007980038, 0.033074344] | 0.427278 | 0.090985682 |
| 16x16 | 0.020058018 | 0.020359802 | [0.012504868, 0.032640999] | [0.010639038, 0.037684698] | 0.523592 | 0.093910187 |
| 32x32 | 0.021999645 | 0.022659302 | [0.014384460, 0.035281372] | [0.012326656, 0.040430309] | 0.551557 | 0.095227361 |

The historical 16x16 pooled PI-QMC vs archived exact-P64 value is reconstructed as `0.020058017556`, i.e. the published rounded `0.02006`.

The observed PI-QMC discrepancy is near the median of the null sampling floor at every resolution and has no small null-tail probability. There is no evidence here for a resolved PI-QMC density bias relative to the exact finite-P target at the available resolution/statistics.

SPLH differs from exact P64 by about 0.091--0.095 across the same binnings, while the 99% upper null-noise bounds are only about 0.033--0.040. Thus the SPLH density discrepancy is several times larger than the chain-level PI-QMC sampling floor and is robust to binning.

## Additional internal diagnostics

| bins | first 8 vs last 8 chains | odd vs even chains | first vs second time half |
|---:|---:|---:|---:|
| 8x8 | 0.040255737 | 0.039443359 | 0.045064087 |
| 16x16 | 0.050521851 | 0.044873047 | 0.053944092 |
| 32x32 | 0.055347900 | 0.049064331 | 0.057661743 |

These are sampling/stationarity diagnostics for smaller half-ensembles, not IID error bars.

## Reviewer-facing conclusion

A defensible manuscript statement is:

> Across registered 8x8, 16x16, and 32x32 partitions, the PI-QMC-to-exact finite-P L1 discrepancy is statistically consistent with the positive sampling floor estimated by whole-chain residual resampling. At the original 16x16 resolution, the observed value 0.02006 is essentially equal to the null-floor median 0.02036 (null-tail probability 0.52). In contrast, the deterministic SPLH discrepancy remains 0.091--0.095 across these resolutions, well above the 99% PI-QMC sampling floor.

This supports retaining density as the primary PI-QMC validation diagnostic while avoiding the incorrect interpretation of the old positive percentile-bootstrap interval as evidence for a nonzero true density discrepancy.
