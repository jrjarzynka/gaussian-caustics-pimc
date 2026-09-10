# Data availability and archive split

## Included directly in Git

The Git repository contains all compact deterministic references, exact-BZ
anchors, publication PI-QMC density summaries, the full 16-chain Section VII
energy-check outputs, Point-6 replay histograms and corrected bootstrap outputs,
interacting finite-P aggregate/per-chain summaries, publication figures, and the
scripts required to reproduce the deterministic/derived analyses.

## Reserved for the immutable Zenodo archive

The only intentionally omitted large scientific files are the four independent
P=256 long-stationarity raw trajectories and four derived observable caches
(about 410 MB in the author's working tree). Their exact SHA-256 hashes are in
`provenance/OMITTED_LARGE_DATA.csv`.

The raw chains are the primary archival objects; the caches are convenience
artifacts that can be reconstructed from them. They are omitted from ordinary
Git to keep the repository lightweight, not because they are scientifically
irrelevant.

## Candidate manuscript Data Availability Statement

> The code, compact numerical data, validation workflows, and figure-generation
> scripts supporting this work are available in the versioned public repository
> https://github.com/jrjarzynka/gaussian-caustics-pimc. The corresponding
> immutable software/data release, including the large raw P=256 long-chain
> trajectories omitted from ordinary Git, is archived
> at 10.5281/zenodo.22695962. SHA-256 manifests are provided in both releases.

The DOI above was reserved for the corresponding archival Zenodo record before
the final `v1.0.1` repository freeze.

## v1.0.1 referee-revision addition

Version `v1.0.1` adds the `R_wall = 30 nm` equilibrium sensitivity
campaign reported in the PRE referee revision. The runner, frozen protocol,
run command, compact chain-level results, diagnostics, and source provenance
are included in Git under `paper2_wall_radius_sensitivity_R30/`.

The four new raw `P=256` R30 trajectories are intentionally omitted from
ordinary Git and are archived in the immutable Zenodo version `1.0.1` at
DOI `10.5281/zenodo.22695962`. The corresponding derived observable caches are
reconstructable convenience artifacts.
