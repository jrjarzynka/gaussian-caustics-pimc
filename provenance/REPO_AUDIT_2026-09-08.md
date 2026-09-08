# Public-repository audit - 2026-09-08

## Source bundle

Audited input: `paper2_caustic_repo_bundle_2026-09-08.zip` produced by Codex.
The source bundle contained 234 files / 81 Python scripts and its 233-entry
`SHA256SUMS` verified with zero failures. All 81 Python sources passed syntax
compilation. No credential/token/private-key pattern was detected and no
symlinks were present.

## Material issues found before public release

1. The source bundle predated manuscript v6.3.3 and still exposed
   `docs/manuscript_v4.tex` plus publication assets using obsolete RLH naming.
2. `validation_tests/paper2_caustic_stage_analysis/` contained the rejected
   exploratory bare-stage formula with an extra `sqrt(Xi)` prefactor. Its common
   root was useful, but its bare phase/continuation analysis was invalid. The
   entire exploratory directory was removed from the public candidate and
   replaced by `validation/stage_independence_corrected.py` and its audit.
3. Fig. 6 and the real Fig. S1 were absent from the old publication-figure layer.
   All final v6.3.3 Figs. 1--6 and Fig. S1 now come from the authoritative
   submission-candidate bundle.
4. The source bundle did not contain the final finite-n-corrected Point-6
   residual bootstrap. The full archived Point-6 inputs were added and the
   authoritative analysis now rescales whole-chain residuals by `sqrt(16/15)`.
5. `pyproject.toml` described the unrelated centre-of-mass switching project and
   pointed at a stale repository URL. Project metadata were rewritten for this
   Gaussian-caustics paper; no GitHub URL is invented before repository creation.
6. The long-stationarity `RUN_COMMAND.txt` contained an author-local Python path.
   It is now repository-relative. Historical absolute paths remain only in
   frozen provenance files, where they are documentary rather than executable.
7. The original bundle lacked `CITATION.cff`, a claim-level reproduction matrix,
   and an explicit Git-vs-Zenodo data split. These were added.

## Deliberate exclusions

- superseded manuscript v4 and proposed patch files;
- obsolete RLH publication figures;
- the rejected exploratory stage-analysis directory;
- four raw P=256 long-stationarity trajectories and four derived caches (~410 MB),
  which are reserved for the immutable Zenodo archive and frozen by SHA-256.

## Deliberate retention

Historical validation filenames/variables containing `RLH` remain when they are
part of frozen source/output provenance. `LEGACY_NAMING.md` defines their modern
interpretation and prevents confusion with the bare stage.
