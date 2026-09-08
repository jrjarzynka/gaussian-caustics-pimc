# Point 7 Figure 2 exact-BZ anchor audit

Overall: **PASS**

## Frozen anchors

0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.10, 2.25, 2.35, 2.42, 2.48, 2.50, 2.54

## Legacy-anchor reconstruction

| beta | dV continuum | dV P64 | dV SPLH | dL1 SPLH |
|---:|---:|---:|---:|---:|
| 0.50 | 8.882e-16 | 4.441e-16 | 0.000e+00 | 2.238e-16 |
| 1.00 | 2.442e-15 | 0.000e+00 | 0.000e+00 | 3.261e-16 |
| 1.90 | 4.441e-16 | 1.332e-15 | 0.000e+00 | 1.613e-16 |
| 2.50 | 1.554e-15 | 0.000e+00 | 0.000e+00 | 6.661e-16 |

## Maximum convergence diagnostics

- legacy_dV_cont: `2.44249065418e-15`
- legacy_dV_P64: `1.33226762955e-15`
- legacy_dL1_SPLH: `6.66133814775e-16`
- legacy_dV_SPLH: `0`
- bz_basis_l1: `4.53491720222e-07`
- bz_kgrid_l1: `2.79276970473e-11`
- p64_basis: `6.03058152748e-07`
- p64_kgrid: `9.56021928289e-11`
- splh_binconv: `6.35697744663e-05`
- exact_imag: `2.10942374679e-15`
- zeta_max: `0.990214937567`

## Acceptance gates

- legacy_continuum: **PASS**
- legacy_P64: **PASS**
- legacy_SPLH_L1: **PASS**
- legacy_SPLH_V: **PASS**
- BZ_basis: **PASS**
- BZ_kgrid: **PASS**
- P64_basis: **PASS**
- P64_kgrid: **PASS**
- SPLH_bin: **PASS**
- exact_imag: **PASS**
- all_precaustic: **PASS**

## Central beta=2.5 anchor

- zeta_max = `0.974621001542`
- V_exact_BZ = `0.712951818447`
- V_SPLH = `0.836975604153`
- SPLH energy error = `+17.395815887756%` -> `+17.40%`
- SPLH density L1 (16x16) = `0.115041336100`
- P64 Trotter bias = `-0.020376866640%`
- error-scale separation = `853.704163x` -> `854x`

