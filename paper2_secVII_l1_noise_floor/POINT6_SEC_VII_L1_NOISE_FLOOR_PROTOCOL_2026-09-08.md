# Point 6 — Sec. VII L1 density noise floor and binning dependence

**Status:** predeclared before any Point-6 replay or 8/16/32-bin result is inspected.

## Fixed scientific question
Determine whether the observed PI-QMC density discrepancy from the deterministic exact finite-P P=64 bead marginal is resolved beyond chain-level sampling noise, and whether the conclusion is stable under 8x8, 16x16, and 32x32 registered reduced-coordinate bins.

## Fixed reference and units
- beta*=2.70, P=64.
- Exact null target: deterministic finite-P P=64 bead marginal from the validated `lha13b3a_exact_finiteP_registered_density.py` construction.
- Primary binning: 16x16 (the existing publication comparison).
- Sensitivity binnings: 8x8 and 32x32.
- L1 convention: mean absolute difference of densities normalized to mean 1 on the registered equal-area reduced-coordinate bins. Equivalently, for bin probabilities, the sum of absolute probability differences.
- Independent top-level statistical units: complete independent PI-QMC chains. Beads and saved sweeps are not treated as IID top-level observations.

## Data hierarchy
Primary preference is to use the completed Point-5 long-statistics ensemble, seeds 132001..132016, 160000 sweeps, 32000 burn-in, sample_every=5, if its original trajectories can be deterministically reconstructed.

Because Point 5 archived energy time series but not bead coordinates/histograms, a replay may be used ONLY as trajectory reconstruction. For each seed, the replay must use exactly the frozen Point-5 sampler, seed, initialization, Hamiltonian, grid, proposals, wrapping, kernel, and controls. No extra RNG draw may occur before or during the sampler.

**Replay acceptance gate:** the replayed `Vstar_time_series` must be exactly array-equal to the archived Point-5 `Vstar_time_series` for every used seed. If any seed fails exact equality, reconstructed density from that replay is not used as Point-5 data. No parameter adjustment is allowed to force agreement.

If the long-run replay gate fails, fall back to the archived publication ensemble for 8x8 and 16x16 only; do not fabricate 32x32 from 16x16 histograms. A new independent density campaign would then require a separate protocol.

## Histogram construction
For each accepted chain, pool physical beads only for the chain density point estimate. Transform physical coordinates to periodic reduced coordinates with the same `Ainv`, wrap modulo 1, and use `histogram2d(t,s)` with common edges. Save per-chain count arrays at 32x32; obtain 16x16 and 8x8 by exact adjacent-bin aggregation of those 32x32 arrays. Verify count conservation at every resolution.

Deterministic exact finite-P and SPLH densities must be integrated/averaged onto exactly the same registered bins at each resolution. No interpolation between mismatched bin conventions is allowed.

## Observed discrepancy
For resolution r in {8,16,32}, with equal-count chains,

`pbar_r = mean_i p_{i,r}`

and

`D_obs(r) = L1(pbar_r, p_exact,r)`.

Also report `D_SPLH(r) = L1(p_SPLH,r, p_exact,r)` and deterministic finite-P vs continuum density discrepancy where available.

## Chain-aware null sampling floor
Use a centered whole-chain residual bootstrap under the exact finite-P null.

For each resolution:
1. Form normalized per-chain density vectors `p_i`.
2. Compute `pbar = mean_i p_i`.
3. Form centered chain residuals `r_i = p_i - pbar`; these sum to zero across chains.
4. For each bootstrap replicate, sample 16 residual vectors with replacement and form `delta_b = mean_j r_{I_j}`.
5. Define the null-noise statistic `D_null_b = L1(delta_b, 0)`; this is the positive sampling floor expected for the estimator if the true density equals the exact target, while preserving chain-level bin covariance and within-chain autocorrelation through the chain aggregate.

Fixed bootstrap controls:
- 200000 replicates per binning.
- RNG seed 20260908.
- Same resampling index matrix may be reused across binnings to make binning comparisons paired.

Report median, 68%, 95%, and 99% quantiles of `D_null`.

## Null-tail probability / decision
For each resolution report

`p_null = (1 + number(D_null_b >= D_obs)) / (B + 1)`.

Interpretation is descriptive, not a hard publication gate:
- If D_obs lies comfortably inside the 95% null-noise range and p_null is not small, PI-QMC is statistically consistent with the exact finite-P density at that resolution.
- If D_obs exceeds the 99% null-noise quantile or p_null < 0.01 reproducibly across resolutions, treat a residual density discrepancy as resolved and investigate before strengthening validation language.
- Borderline/mixed binning results are reported without selecting a preferred grid after inspection.

Do not subtract the noise-floor median linearly from D_obs and call the remainder a corrected L1 distance.

## Additional diagnostics
For each binning report:
- pooled observed L1 PI-QMC vs exact P64;
- SPLH vs exact P64;
- null floor quantiles and p_null;
- first 8 chains vs last 8 chains L1;
- odd vs even chain L1;
- per-chain L1 distribution vs exact (diagnostic only);
- exact count conservation and normalization checks.

For the long-run reconstructed ensemble also report first-half vs second-half density L1 if half histograms are retained during replay. This is a stationarity diagnostic, not an IID uncertainty estimate.

## No adaptivity
Do not change binnings, bootstrap seed/reps, chain set, burn-in, or inclusion rules after inspecting Point-6 outcomes. No seed rejection. No tuning. No manuscript editing until the numerical audit is complete.
