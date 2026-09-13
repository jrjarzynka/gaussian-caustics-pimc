# Source collection lineage

The initial curated source bundle was assembled on 2026-09-08 from the author's
working checkout. Its internal Git snapshot was branch
`codex/paper2-pre-hardening-freeze-2026-09-01`, head
`72c5cf17cc207f73a6d0ba8bac55cb587ab07e75`.

That bundle passed its own 233-entry SHA-256 manifest and Python syntax audit.
This public-candidate repository then replaced superseded manuscript/figure
assets with the authoritative v6.3.3 submission-candidate files and removed the
rejected exploratory bare-stage implementation. The lineage of that initial
bundle is preserved in the repository-root `PROVENANCE_MAP.csv`: the 183 files
inherited from it are marked `source_group = codex_curated_bundle` and carry
their originating `source_path` together with source and repository SHA-256
digests. Recorded source paths are relative to their respective bundle roots,
are documentary only, and are not runtime requirements.
