# Legacy internal naming

The publication terminology is **SPLH** (starting-point local harmonic), defined
in the manuscript as the renormalized pre-harmonic-mapping positional
distribution of Ref. [1] and its mass-weighted tensorial extension.

Some frozen historical validation filenames, variable names, CSV column names,
and comments contain `RLH`, `tensor_rlh`, or similar labels. They are retained
only because changing historical source/output names would break provenance and
cross-checks. In this repository those legacy labels refer to the same
renormalized pre-mapping object now called SPLH; they must not be interpreted as
the bare/unrenormalized stage.

The earlier exploratory `paper2_caustic_stage_analysis` directory is deliberately
excluded because it contained a rejected bare-stage prefactor definition. The
authoritative stage-boundary check is `validation/stage_independence_corrected.py`.
