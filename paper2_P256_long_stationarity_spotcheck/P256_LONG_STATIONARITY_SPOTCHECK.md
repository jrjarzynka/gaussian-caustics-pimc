# Paper 2 — P=256 long-chain stationarity spot-check

## Verdict

**B. PASS WITH SMALL RESIDUAL DRIFT**

The four predeclared long chains give `p_inv = 0.925221615 ± 0.000398` (SEM across independent chains; burn-in 15,000). The change from the canonical `p_inv = 0.924852431` is `0.00036918`, or `0.659σ` using the combined old/new chain-level uncertainty; it is compatible.

The mean new half-difference is `-0.000546701 ± 0.000145` with 95% t interval `[-0.0010075, -8.59071e-05]` and signs 4 negative / 0 positive. Yes, in a limited numerical sense: the new independent chains reproduce a weak early-time negative half-drift. This is evidence of residual slow relaxation after the 15,000-step burn-in, but not evidence that the canonical Sec. VIII estimate is materially biased.

## Exact reproduction configuration

This audit uses the same production assembly and the exact JIT sampler runtime that passed the prior Sec. VIII reproduction gate. It is orchestration around that path, not a replacement sampler. The phrase “identical electron/hole masses” in the task is interpreted as *identical to production*: `m_e=0.58 m0`, `m_h=0.36 m0`; setting the two masses equal would change the Hamiltonian.

| Item | Value |
| --- | --- |
| Temperature / beads | 20 K / P=256 |
| Masses | electron 0.58 m0; hole 0.36 m0 |
| Bilayer Keldysh | D=0.60 nm; r0,1=4.479911124019045 nm; r0,2=3.4934510307918494 nm; kappa=4.945 |
| Moire landscapes | amplitude 0.045 eV for each carrier; period 20 nm; origins (0,0) |
| Radial wall | R=15 nm; H=0.08 eV; power=8 |
| Initial center | (10,0) nm for both carriers; initial spread is the sampler default 0.1 nm |
| Moves | local sweep on; local step 0.15 nm; 2 staging moves/step; lengths 4,8,16,32,64,128 (256 is filtered because L<P); joint global probability 0.20 and scale 12 nm |
| Periodic lookup | triangular moire cell, grid 200; interaction table rmax 80 nm, 20,000 points |
| Trajectory | 240,000 steps; stored from step 15,000 every 20 steps; 11,250 frames/chain |
| Seeds | 912560, 912561, 912562, 912563; none occurs in the original 30-chain set |
| Observable | wall-inclusive mass-weighted zeta; `p_inv=P(zeta>=1)`; production `w_zeta_gt5=P(zeta>=5)`; exact same-slice paired flattening |

The JSON config itself contains 120,000/30,000 trajectory defaults inherited from an earlier scan. As in the reproduced Sec. VIII production runner, trajectory controls are command-line/orchestration values and override those JSON fields. No physical config value is overridden.

### Code and provenance

| Component | Path / revision |
| --- | --- |
| Production checkout | `historical production checkout` @ `775f3911a8a9acd6805800dfd17bd9b3bc2bee4e` |
| Production runner | `secVIII/run_tb03_pinv_ladder.py`; SHA-256 `63de5e2ba07f577a00627f22b28686ba1c9f57b66c5a5deee01411d99db8697b` |
| Stability field | `secVIII/lha15a0_pair_stability_field_TB03.py`; SHA-256 `f423834e37c640bf855e799e75080a34cc0fa2dd21a90dd0390390efb719d926` |
| Config | `secVIII/configs/two_body/landscape_scan_config_v2_P256_matched.json`; SHA-256 `9d7779ad7da3554fda7118b645547d7b251141286e738afc5f5254cfc2163719` |
| Reproducing sampler runtime | `numerics/tmd_pimc/two_body_sampler_periodic_jit.py` @ repo `93450b81e787d4f656b31cc475bf13d3e69bfae5`; SHA-256 `34b33ae11fefe2c25ec9e6dd438fd9b6d7c18ad8c3c16510b71ce94faed82d2c` |
| This audit runner | `paper2_P256_long_stationarity_spotcheck/run_P256_long_stationarity.py`; SHA-256 before execution `d52377e1db2cb72871781cb8666fc57af3515e043eca43d4b9b52a42ab0a7ff4` |

Exact command:

```bash
python paper2_P256_long_stationarity_spotcheck/run_P256_long_stationarity.py --stage all
```

Raw trajectories and cached same-slice observables remain under `raw_chains/` and `observable_cache/`; they are not merged into the canonical production set. Per-file hashes are in `source_provenance.json`.

## Primary result: p_inv

Block SEM is `SD(eight block means)/sqrt(8)`. Linear fits use only the eight temporal block estimates; slopes are change in p_inv per 100,000 MC steps.

| seed | full | H1 | H2 | H2-H1 | block SEM | slope / 100k | slope/SE | zeta50 | w>=5 | rho50 | rho mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 912560 | 0.9248201 | 0.9250632 | 0.9245771 | -0.00048611 | 0.0003791 | -0.0002939 ± 0.000624 | -0.471 | 17.3464 | 0.9087083 | 1.63396 | 1.82371 |
| 912561 | 0.9246747 | 0.9251556 | 0.9241938 | -0.00096181 | 0.0002115 | -0.0006813 ± 0.00022 | -3.1 | 17.3506 | 0.9082219 | 1.63359 | 1.82368 |
| 912562 | 0.9249934 | 0.9251382 | 0.9248486 | -0.00028958 | 0.0003161 | -0.0004633 ± 0.000495 | -0.936 | 17.3073 | 0.9083208 | 1.6374 | 1.82939 |
| 912563 | 0.9263983 | 0.9266229 | 0.9261736 | -0.00044931 | 0.0004534 | -0.0003815 ± 0.000744 | -0.513 | 17.2648 | 0.909316 | 1.64657 | 1.83992 |

### Burn-in sensitivity and cross-chain test

Independent chains are the top-level units. Intervals below are Student-t intervals with 3 degrees of freedom, not bead-level intervals.

| burn-in | stored | mean p ± SEM | mean H2-H1 ± SEM | 95% t CI | signs -/+ | zeta50 | w>=5 | rho50 | rho mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15000 | 11250 | 0.9252216 ± 0.000398 | -0.0005467 ± 0.000145 | [-0.0010075, -8.5907e-05] | 4/0 | 17.3173 | 0.9086418 | 1.63788 | 1.82917 |
| 30000 | 10500 | 0.9251645 ± 0.000424 | -0.00053106 ± 0.000282 | [-0.001429, 0.00036684] | 3/1 | 17.3171 | 0.9086658 | 1.63787 | 1.82912 |
| 60000 | 9000 | 0.9251429 ± 0.000362 | -0.00021853 ± 0.000397 | [-0.0014835, 0.0010464] | 3/1 | 17.3142 | 0.9086452 | 1.63815 | 1.82964 |

Maximum aggregate p_inv shift induced by moving burn-in from 15,000 to 30,000 or 60,000 is `7.8711e-05`.

### Quarter estimates

| seed | burn-in | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- | --- |
| 912560 | 15000 | 0.924519 | 0.9255979 | 0.9250103 | 0.9241532 |
| 912560 | 30000 | 0.9247143 | 0.9248229 | 0.9253259 | 0.9243438 |
| 912560 | 60000 | 0.925276 | 0.9249826 | 0.9246406 | 0.9246389 |
| 912561 | 15000 | 0.9250966 | 0.9252119 | 0.9244991 | 0.9238906 |
| 912561 | 30000 | 0.9249777 | 0.9249509 | 0.9239077 | 0.9243274 |
| 912561 | 60000 | 0.9253542 | 0.9248854 | 0.923941 | 0.9245712 |
| 912562 | 15000 | 0.9256327 | 0.9246426 | 0.9249464 | 0.9247519 |
| 912562 | 30000 | 0.925369 | 0.924622 | 0.9246384 | 0.9249673 |
| 912562 | 60000 | 0.9250903 | 0.9235191 | 0.9258056 | 0.9246875 |
| 912563 | 15000 | 0.9270019 | 0.9262451 | 0.9257076 | 0.9266383 |
| 912563 | 30000 | 0.9266339 | 0.9273497 | 0.9250432 | 0.9266384 |
| 912563 | 60000 | 0.9259948 | 0.9269149 | 0.9264097 | 0.9255747 |

### Eight-block estimates (burn-in 15,000)

All three burn-in variants are available in `P256_long_chain_blocks.csv`.

| seed | block | MC steps | p_inv | zeta50 | w>=5 | rho50 | rho mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 912560 | 1 | 15000–43120 | 0.9250316 | 17.3995 | 0.9088042 | 1.62882 | 1.81674 |
| 912560 | 2 | 43140–71260 | 0.92401 | 17.4182 | 0.9075299 | 1.6208 | 1.81153 |
| 912560 | 3 | 71280–99380 | 0.9267564 | 17.2981 | 0.9099118 | 1.64637 | 1.83867 |
| 912560 | 4 | 99400–127500 | 0.9244366 | 17.3535 | 0.9080809 | 1.63275 | 1.82246 |
| 912560 | 5 | 127520–155620 | 0.9241921 | 17.3327 | 0.9083865 | 1.63614 | 1.82892 |
| 912560 | 6 | 155640–183740 | 0.9258285 | 17.244 | 0.9103896 | 1.64897 | 1.83809 |
| 912560 | 7 | 183760–211860 | 0.9233947 | 17.3869 | 0.9078086 | 1.62577 | 1.81027 |
| 912560 | 8 | 211880–239980 | 0.9249117 | 17.3396 | 0.908756 | 1.63322 | 1.82304 |
| 912561 | 1 | 15000–43120 | 0.9250233 | 17.3557 | 0.9078269 | 1.63358 | 1.82711 |
| 912561 | 2 | 43140–71260 | 0.9251538 | 17.3594 | 0.9090374 | 1.63571 | 1.82132 |
| 912561 | 3 | 71280–99380 | 0.9251145 | 17.3499 | 0.9095478 | 1.6367 | 1.82589 |
| 912561 | 4 | 99400–127500 | 0.9253256 | 17.2645 | 0.908881 | 1.64291 | 1.83207 |
| 912561 | 5 | 127520–155620 | 0.924156 | 17.3577 | 0.9074308 | 1.63243 | 1.82417 |
| 912561 | 6 | 155640–183740 | 0.9248422 | 17.3086 | 0.9076447 | 1.63459 | 1.82709 |
| 912561 | 7 | 183760–211860 | 0.9238226 | 17.4068 | 0.9079531 | 1.62847 | 1.81773 |
| 912561 | 8 | 211880–239980 | 0.9239587 | 17.4007 | 0.907453 | 1.62524 | 1.81402 |
| 912562 | 1 | 15000–43120 | 0.9264837 | 17.1823 | 0.9088764 | 1.65188 | 1.84577 |
| 912562 | 2 | 43140–71260 | 0.924754 | 17.401 | 0.9075521 | 1.62869 | 1.82028 |
| 912562 | 3 | 71280–99380 | 0.9251172 | 17.2759 | 0.9086365 | 1.64392 | 1.83496 |
| 912562 | 4 | 99400–127500 | 0.9241949 | 17.2681 | 0.9073752 | 1.63898 | 1.83115 |
| 912562 | 5 | 127520–155620 | 0.9239476 | 17.3525 | 0.9076086 | 1.62856 | 1.81858 |
| 912562 | 6 | 155640–183740 | 0.9259452 | 17.3292 | 0.9097201 | 1.63874 | 1.82964 |
| 912562 | 7 | 183760–211860 | 0.9252756 | 17.2199 | 0.909195 | 1.64743 | 1.84157 |
| 912562 | 8 | 211880–239980 | 0.9242282 | 17.4266 | 0.907603 | 1.62191 | 1.81313 |
| 912563 | 1 | 15000–43120 | 0.9270667 | 17.2937 | 0.9097703 | 1.64402 | 1.83578 |
| 912563 | 2 | 43140–71260 | 0.9269528 | 17.247 | 0.9099175 | 1.65018 | 1.84091 |
| 912563 | 3 | 71280–99380 | 0.9249978 | 17.2538 | 0.9091533 | 1.64649 | 1.84127 |
| 912563 | 4 | 99400–127500 | 0.927476 | 17.1898 | 0.9096951 | 1.6574 | 1.8532 |
| 912563 | 5 | 127520–155620 | 0.9264591 | 17.2776 | 0.9093172 | 1.64626 | 1.84019 |
| 912563 | 6 | 155640–183740 | 0.9249561 | 17.3126 | 0.9080864 | 1.63443 | 1.83134 |
| 912563 | 7 | 183760–211860 | 0.9282873 | 17.2343 | 0.9101924 | 1.65413 | 1.84509 |
| 912563 | 8 | 211880–239980 | 0.9249894 | 17.3051 | 0.9083948 | 1.63861 | 1.83154 |

## Secondary temporal checks

Mean chainwise H2-H1 changes at burn-in 15,000:

| observable | mean delta | SEM | 95% t CI |
| --- | --- | --- | --- |
| zeta50 | 0.0205503 | 0.02065 | [-0.0451811, 0.0862816] |
| w(zeta>=5) | -0.00029184 | 0.0003808 | [-0.00150377, 0.000920087] |
| rho median (nm) | -0.00403759 | 0.002533 | [-0.0120975, 0.0040223] |
| rho mean (nm) | -0.00402638 | 0.002261 | [-0.0112215, 0.00316872] |

The optional radial-mode overlap O_rho was not recomputed: evaluating full eigenvectors for roughly 11.5 million paired beads is not inexpensive, while the prior attribution audit already established that result. This spot-check leaves mode attribution untouched and focuses on stationarity of p_inv, zeta depth, and the sampled compact-pair distribution.

### Conservative several-times-longer projection

Because the classification is B, the requested drift-size estimate is made explicitly. As a conservative diagnostic, the mean H2-H1 contrast after the largest burn-in (60,000) is linearly extended over another 480,000 steps, i.e. from a 240,000-step chain to a hypothetical 720,000-step chain. The retained-half centers are 90,000 steps apart, so the multiplier is 5.333. This is not treated as a physical forecast: the drift weakens with increasing burn-in and every 60k-burn half contrast is compatible with zero.

| observable | projected shift over +480k | projected SEM | projected level |
| --- | --- | --- | --- |
| p_inv | -0.00116551 | 0.00212 | 0.9239774 |
| zeta50 | 0.15421 | 0.1047 | 17.46837 |
| w(zeta>=5) | -0.00156481 | 0.002793 | 0.9070804 |

Even this deliberately conservative extrapolation leaves p_inv near 92.4%, zeta50 near 17.5, and w(zeta>=5) near 90.7%. It does not alter any qualitative conclusion. The spot-check supports quoting p_inv at the existing manuscript scale (approximately 92.5%, not additional decimal precision); no canonical number should be replaced by this noise-dominated extrapolation.

## Compatibility with canonical P=256 production

The new estimate is the unweighted mean of four independent chain estimates. For zeta50 this follows the production convention (mean of chain medians). Old secondary SEMs are reconstructed across the ten reproduced production chains.

| observable | old | new | new-old | combined SEM | difference / combined SEM |
| --- | --- | --- | --- | --- | --- |
| p_inv | 0.92485243 | 0.92522161 | 0.00036918 | 0.0005598 | 0.659 |
| zeta50 | 17.318187 | 17.317287 | -0.00089912 | 0.02629 | -0.0342 |
| w(zeta>=5) | 0.90877066 | 0.90864175 | -0.00012891 | 0.0003322 | -0.388 |
| rho median (nm) | 1.6374945 | 1.6378803 | 0.00038579 | 0.003852 | 0.1 |
| rho mean (nm) | 1.8270548 | 1.8291727 | 0.0021179 | 0.004981 | 0.425 |

The old whole-chain bootstrap interval for p_inv is `[0.924004861, 0.925468924]`.

## Time-series diagnostics

`P256_stationarity_diagnostics.pdf` contains cumulative p_inv, eight-block p_inv with block-level regressions, cumulative zeta50, and cumulative median/mean rho for every chain. Cumulative medians are evaluated at 24 predeclared time points; no bead-level trend regression is used.

## Reviewer-facing answers

> Does the previously observed H2-H1 = -7.51e-4 represent evidence that the P=256 Sec. VIII result was not equilibrated?

Yes, in a limited numerical sense: the new independent chains reproduce a weak early-time negative half-drift. This is evidence of residual slow relaxation after the 15,000-step burn-in, but not evidence that the canonical Sec. VIII estimate is materially biased. The decisive test is the independent-chain mean and its t interval above, not the sign of one aggregate split.

> Would the result change the manuscript conclusions?

No. None of the listed manuscript conclusions changes.

| Manuscript conclusion | Spot-check implication |
| --- | --- |
| p_inv approximately 92.5% | Retained; the long-chain estimate is compatible at publication scale. |
| Most probability lies deep beyond the caustic | Retained; w(zeta>=5) remains near 90.9%. |
| zeta50 approximately 17.3 | Retained. |
| w(zeta>5) approximately 90.9% | Retained. |
| BLK dominates the deep unstable mode | Unchanged by this stationarity test; the Hamiltonian and attribution result were not modified. |
| Deep mode is almost purely radial relative motion | Unchanged by this stationarity test; established in the prior mode-attribution audit. |
| Wall contribution is negligible | Unchanged by this stationarity test; established in the prior counterfactual Hessian audit. |

## Classification rule and interpretation

The rule was fixed in the audit script: `C` requires both a >2σ incompatibility and an absolute p_inv displacement above 0.005; `B` captures a resolved cross-chain half drift, any individual >=3σ block slope, a burn-in displacement above 0.002, or >2σ old/new tension below the material threshold; otherwise the result is `A`. Statistical significance alone cannot trigger `C`.

**Final classification: B. PASS WITH SMALL RESIDUAL DRIFT.**

A brief qualification should report the measured residual drift and its negligible publication-scale effect; no physical conclusion needs to change.

## Reproducibility files

- `P256_long_chain_summary.csv`: one row per chain and burn-in.
- `P256_long_chain_blocks.csv`: 96 block rows (4 chains x 3 burn-ins x 8 blocks).
- `P256_long_chain_aggregate.csv`: chain-level means, SEMs, and t intervals.
- `P256_stationarity_diagnostics.pdf`: requested compact diagnostic.
- `source_provenance.json`: paths, hashes, exact command, configuration, raw-output hashes.
