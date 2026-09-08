# Section VII density-noise-floor validation

The **authoritative publication result** is the variance-corrected whole-chain
residual bootstrap in `results/`, produced by
`analyze_point6_l1_noise_floor_variance_corrected.py`.

The residual multiplier is `sqrt(16/15)`. The archived pre-correction analysis
is retained only for provenance under `results_pre_variance_correction/` and
`analyze_point6_l1_noise_floor_PRE_VARIANCE_CORRECTION.py`; it is not the value
quoted in manuscript v6.3.3.

Run from the repository root:

```bash
python paper2_secVII_l1_noise_floor/analyze_point6_l1_noise_floor_variance_corrected.py
```

The 16 replayed whole-chain histograms, exact finite-P references, and SPLH
multibin references required for this deterministic bootstrap are included.
No beads or individual sweeps are treated as independent top-level observations.

The original Point-6 archive checksum manifests are preserved under
`provenance/original_archive/`. They intentionally record historical build
paths and pre-variance-correction filenames; they are provenance records, not
current run instructions. Repository-wide current hashes are frozen in the
top-level `SHA256SUMS`.
