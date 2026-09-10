# Rwall=30 nm equilibrium sensitivity result

Primary analysis: P=256, T=20 K, 4 independent chains, 240000 steps/chain, primary burn-in 60000, sample every 20.

- p_inv = 0.92548600 +/- 0.00041865 (SEM across chains)
- 95% Student-t CI = [0.92415368, 0.92681832]
- lower-CI > 0.5 gate: PASS
- no-wall-Hessian reclassification on the same R30 samples = 0.92548600
- zeta50 chain mean = 17.276225
- P(zeta>=5) chain mean = 0.90930707
- rho median chain mean = 1.642740 nm
- rho mean chain mean = 1.834463 nm
- P(rho>18.7 nm) chain mean = 0.00000000
- max rho across retained samples = 11.884666 nm
- compact-start p_inv mean = 0.92583290
- dissociated-start p_inv mean = 0.92513911
- dissociated minus compact start-class difference = -0.00069379
- canonical R15 p_inv reference = 0.92485243
- R30 minus R15 p_inv = +0.00063357

Independent chains are the top-level statistical units. Beads enter only as within-chain estimators of same-slice observables.
