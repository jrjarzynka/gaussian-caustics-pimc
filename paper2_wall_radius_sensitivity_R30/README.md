# Rwall=30 nm sensitivity bundle

This directory is intended to be copied into the root of the released Gaussian-caustics repository.

Run `--stage dry-run` first. It verifies the four frozen Sec. VIII source hashes and performs the sampler/stability model-match check without launching a Markov chain.

The scientific production command is in `RUN_COMMAND.txt`. The production script never edits the archived Sec. VIII code or config; it changes `wall_radius_nm` from 15 to 30 only in an in-memory copy of the config.

Results are written under `paper2_wall_radius_sensitivity_R30/results/`; raw chains and observable caches remain outside the scientific source files in `raw_chains/` and `observable_cache/`.
