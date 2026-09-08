# Suggested first Git freeze

Run only after reviewing `git status` and the release placeholders.

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Initial reproducibility release candidate for Gaussian caustics paper"
git tag v1.0.0-rc1
```

Before the public `v1.0.0` tag, insert the final GitHub URL, Zenodo DOI, exact
historical AI/Codex version strings, and the selected non-code/data license.
Do not add the ~410 MB raw P=256 trajectories to ordinary Git; archive them in
Zenodo as listed in `ZENODO_UPLOAD_MANIFEST.md`.
