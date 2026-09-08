> **Repository note.** This file records the v6.3.3 manuscript/Supplement reference freeze used to align the reproducibility package. The manuscript source and compiled article PDFs are intentionally not distributed in this Git release candidate.

# Paper 2 v6.3.3 - full submission-candidate audit

Date: 2026-09-08

## Status

**SUBMISSION CANDIDATE / TECHNICAL PASS.**

The scientific manuscript, Supplemental Material, figures, compact figure data,
figure-generation scripts, and validation/audit artifacts are assembled in one bundle.
The main manuscript and Supplemental Material compile successfully with no unresolved
citations or cross-references.

This is not labelled `submission-final` because the public-release metadata still contains
three author-supplied placeholders: the final GitHub URL, Zenodo DOI, and exact historical
AI/Codex model-version strings. The APS publisher-inserted Supplemental Material URL
placeholder is intentionally retained.

## v6.3.3 hardening relative to v6.3.2

1. The Abstract now states that a separate one-dimensional benchmark of all three scalar
   stages shows that the existence-versus-accuracy separation can persist through the final
   harmonic mapping.
2. The Introduction and Discussion no longer say that the final mapped stage is "not
   benchmarked" without qualification. They now distinguish the multidimensional tensorial
   benchmarks (pre-mapping SPLH) from the separate unambiguous scalar three-stage control.
3. Supplemental Sec. S9/Table S12 now explicitly closes the cherry-picking interpretation:
   for `g=0.1`, the renormalized pre-mapping stage is the most accurate stage throughout the
   near-caustic range `beta >= 2.5`, but its L1 error still reaches `0.15355` at
   `zeta_max = 0.9868 < 1`.
4. The nonmonotonic renormalized error (`0.03663 -> 0.02637 -> 0.09365 -> 0.15355`) is
   explicitly described as a fortuitous local minimum rather than a controlled accuracy
   window, paralleling the practical interpretation of the near-zero signed-energy error in
   the triangular benchmark.
5. The final mapped double-well control retains both additive-energy-origin conventions.
   Exact, bare, and renormalized normalized densities are invariant under the constant shift;
   the mapped result is not, and the substantial pre-caustic error survives both tested
   conventions.
6. The real long-chain stationarity figure supplied by the author replaced the prior
   syntax-compile placeholder. The supplied PNG is preserved verbatim as
   `figures/FigS1_P256_stationarity.png`; `FigS1_P256_stationarity.pdf` embeds the same image
   for the existing LaTeX source path.

## Compile verification

Main:

- source: `Paper2_manuscript_v6_3_3.tex`
- compiled PDF: `Paper2_manuscript_v6_3_3_COMPILED.pdf`
- pages: 22
- build: `pdflatex -> bibtex.original -> pdflatex -> pdflatex -> pdflatex`
- unresolved citations/references: 0

Supplement:

- source: `Paper2_Supplementary_v2_3_3.tex`
- compiled PDF: `Paper2_Supplementary_v2_3_3_COMPILED.pdf`
- pages: 8
- build: `pdflatex -> pdflatex`
- unresolved citations/references: 0

PDF preflight reports both PDFs openable, unencrypted, and text-based (not scanned).
Selected rendered pages were visually inspected: main pp. 1, 20-21 and Supplement pp. 7-8.
No clipping, overlap, broken glyphs, or placeholder Fig. S1 remains.

## Fig. S1 provenance

Author-supplied source image:

- dimensions: 2016 x 1458 px
- content: four-panel P=256 long-chain stationarity spot-check for seeds 912560-912563
- burn-in: 15000 steps
- panels: cumulative p_inv, eight temporal blocks + fits, cumulative zeta_50,
  cumulative median/mean pair separation

The PDF wrapper was rendered back at 300 dpi to exactly 2016 x 1458 px as a visual
embedding check. The scientific plot content was not redrawn or recomputed.

## Deliberately unresolved release metadata

Before actual journal submission, replace:

- `[GITHUB URL]`
- `[ZENODO DOI]`
- `[INSERT MODEL/VERSION(S) FROM PROJECT LOG; GPT-5.6 Sol WAS USED FOR THE FINAL REVISION]`
- `[INSERT CODEX MODEL/VERSION FROM PROJECT LOG]`

GPT-5.6 Sol is explicitly known for the final revision. The earlier Codex model/version is
not available in the recovered project context and is therefore not guessed.

The bibliography entry for Supplemental Material retains `[URL will be inserted by publisher]`
intentionally, consistent with the APS Supplemental Material workflow.

## Scientific freeze

No new MCMC, FK benchmark, temperature scan, wall-radius ensemble scan, or 4D exact
calculation was introduced in v6.3.3. The changes are interpretive/consistency hardening plus
replacement of the missing Fig. S1 asset.
