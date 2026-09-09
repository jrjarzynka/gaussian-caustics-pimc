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
> at 10.5281/zenodo.22668090. SHA-256 manifests are provided in both releases.

The DOI above was reserved for the corresponding archival Zenodo record before
the final `v1.0.0` repository freeze.
