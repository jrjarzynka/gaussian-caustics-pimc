# Point 6 archival 32x32 recovery protocol

**Status:** predeclared after the long-run replay gate failed for a purely floating-point postprocessing reason, and before any new Point-6 L1 result at 8x8, 16x16, or 32x32 is inspected.

The Point-5 long-run replay produced exactly matching sampler acceptance and energy means for the first audited seed, but the postprocessed energy arrays differed at ~2.9e-15 and therefore failed the previously declared strict `array_equal` gate. That long-run replay is not used for Point-6 density inference.

To recover 32x32 information for the density ensemble actually used in the publication, replay the frozen publication chains seeds 131001..131016 with the canonical publication controls (40000 sweeps, 8000 burn-in, sample_every=5, P=64, beta*=2.70, same initialization, Hamiltonian, grid, proposals, wrapping and sampler/kernel). The only added post-sampler operation is accumulation of a 32x32 physical-bead histogram on the same registered reduced-coordinate cell.

For every chain, aggregate the replayed 32x32 count array by summing adjacent 2x2 bins to 16x16. The replay is accepted for Point-6 archival recovery only if this reconstructed 16x16 count array is exactly integer-array-equal to the archived `lhap0a_seed_<seed>_hist.npz::H_full_ts` for that same seed. Also require the replayed first-half and second-half 32x32 histograms, after 2x2 aggregation, to equal the archived first-half and second-half 16x16 histograms exactly.

All 16 seeds must pass. If any seed fails, no replayed 32x32 result is used and Point 6 is reported only at 8x8 and 16x16 unless a separate new sampling campaign is later approved.

If all 16 pass, the replayed 32x32 histograms are treated as a lossless finer-bin reconstruction of the exact archived publication trajectories, not as a new independent Monte Carlo ensemble. The 16x16 Point-6 result must reproduce the historical pooled value before any 32x32 inference is accepted.

No parameters may be changed after replay inspection. No seed may be rejected. No L1 result may be used to decide replay acceptance.
