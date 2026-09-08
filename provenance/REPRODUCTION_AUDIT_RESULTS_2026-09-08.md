# Public repository reproduction audit — 2026-09-08

## Candidate

- repository version: `v1.0.0-rc1`
- manuscript freeze: `Paper2_manuscript_v6_3_3.tex`
- Supplement freeze: `Paper2_Supplementary_v2_3_3.tex`
- source curated bundle SHA-256: `89adedc2554fa6f27f766c5a90e48fec40f3804a352ebfddbfb69dc02dd33010`
- v6.3.3 submission-candidate ZIP SHA-256: `28848f1e0f70f66316eb36bd18e260f71db781fea0b43cb2fdbcb6d48b8540a2`
- Point-5 archive SHA-256: `86a80caad406a5337c0dfb6c8d87c7e181f18b3bd484c5fe6418e6bfcd2422db`
- Point-6 archive SHA-256: `270b1e6e7bff6437dc4a397d7cdee273c6dea2c9448dc1cbf04793aa19447631`
- Point-7 archive SHA-256: `e251996555519a3f6f3ac1f7df802909d3cc2ca2677050c485225a1602709698`

## Source-bundle integrity

The original Codex-curated bundle contained 233 entries in its `SHA256SUMS`.
`sha256sum -c SHA256SUMS` returned **233 OK / 0 FAILED** before hardening.

The rejected exploratory `validation_tests/paper2_caustic_stage_analysis/`
package was removed from the public candidate because its bare-stage formula
contained the already-audited extra `sqrt(Xi)` factor. The corrected
stage-independence calculation is under `validation/`.

## Python / portability / secret scan

Final public-candidate checks:

- Python source files parsed with `ast.parse`: **111 PASS / 0 FAIL**.
- executable/public paths containing `/home/jaro` or `/mnt/data` outside
  provenance-only records: **0**.
- common GitHub/OpenAI/AWS token and private-key patterns: **0**.
- `__pycache__` directories: **0**.
- `.pyc` files: **0**.

Historical paths are retained only inside files explicitly placed under
`provenance/`; they document source lineage and are not run instructions.

## Deterministic scientific replay

### Stage-independent first caustic

`python validation/stage_independence_corrected.py`

- reference `beta_c = 2.7116988514311098`
- `cos(eta)=0` root: `2.7116988514311102`
- `sin(2 eta)=0`, Dirichlet zero mode, and `1/Xi=0`: `2.7116988514311098`
- maximum root spread: `4.441e-16`
- result: **PASS**.

### Three-stage double-well gate

`python validation/three_stage_doublewell_gate_v63.py`

The regenerated CSV has SHA-256
`993f81713a1295c71cf106b6547c6d89eb96a393e1f78c8837f8746e06f3c6ec`,
identical to the v6.3.3 freeze. At `g=0.5`, `beta=3.10`,
`zeta_max=0.9867606472`, the normalized-density errors are
`0.648272` (bare), `0.238121` (renormalized pre-map), and `0.262335`
(mapped with the published additive constant).

### Additive-energy-origin audit

`python validation/three_stage_doublewell_energy_origin_sensitivity.py`

The regenerated CSV has SHA-256
`865127dc7210bc3749efdb5acc23c2c1b241a847770ecfdd88d46305a398e678`,
identical to the v6.3.3 freeze. Maximum normalized-density changes under the
constant shift are `3.61e-16` (bare) and `3.89e-16` (renormalized pre-map),
while the final mapped distribution changes substantially.

### Point 6 whole-chain L1 noise floor

`python paper2_secVII_l1_noise_floor/analyze_point6_l1_noise_floor_variance_corrected.py`

The final residual multiplier is `sqrt(16/15) = 1.032795558989`.
The regenerated outputs match the v6.3.3 freeze exactly:

- summary CSV SHA-256: `01cf81aa142ed92607dda76e3468a89b0ae840882558645e1d61bf72577a768a`
- bootstrap NPZ SHA-256: `177164e67e8ea8347fa6573f3cfa2efb709e15cef0062315f5686eed79f3385c`

For the primary 16x16 partition: observed `D_L1=0.020058017556`, null median
`0.021027513342`, central 95% interval `[0.012914971675, 0.033711478643]`,
and `p_null=0.575807120964`.

### Point 7 exact-BZ anchors

`python point7_exact_bz/point7_work/build_point7_anchor_table_and_audit.py`

All acceptance gates pass. The beta=2.5 anchor reproduces:

- `zeta_max = 0.974621001542`
- `V_exact_BZ = 0.712951818447`
- `V_SPLH = 0.836975604153`
- SPLH energy error `+17.3958158878%` -> `+17.40%`
- SPLH density `L1 = 0.115041336100`
- primitive P64 bias `-0.02037686664%`
- error-scale ratio `853.704x`.

## Publication assets

All authoritative figure PDFs in `figures/` are byte-identical to the v6.3.3
submission-candidate freeze, including the author-supplied real Fig. S1:

- Fig. 1: PASS
- Fig. 2: PASS
- Fig. 3: PASS
- Fig. 4: PASS
- Fig. 5: PASS
- Fig. 6: PASS
- Fig. S1: PASS

Figure generators for Figs. 1, 2, 3, 4, and 6 were also replayed during repo
hardening; rendered output was visually/pixel consistent with the authoritative
freeze. The authoritative frozen figure files were restored afterward.

## LaTeX build

A fresh build from the repository source and `figures/` completed successfully:

- main paper: **22 pages**, 0 unresolved references/citations;
- Supplemental Material: **8 pages**, 0 unresolved references/citations.

PDF binary hashes may differ across fresh LaTeX runs because of PDF metadata;
the authoritative v6.3.3 compiled PDFs are retained under `manuscript_compiled/`.

## Data intentionally outside ordinary Git

Four raw P=256 long-chain trajectories and four derived observable caches,
approximately 410 MB total, are intentionally omitted from this Git candidate.
Their exact SHA-256 hashes and intended Zenodo names are frozen in
`provenance/OMITTED_LARGE_DATA.csv`. Raw chains are the primary archival objects;
the caches are reconstructable.

## Release blockers intentionally not guessed

Before changing the release candidate to `v1.0.0` and journal submission:

1. replace `[GITHUB URL]` with the final public repository URL;
2. replace `[ZENODO DOI]` with the deposited archive DOI;
3. insert exact historical AI/Codex model-version strings rather than guessing;
4. choose/document the license for non-code manuscript, figure, and data assets.

The APS publisher-inserted Supplemental Material URL placeholder is intentional.
