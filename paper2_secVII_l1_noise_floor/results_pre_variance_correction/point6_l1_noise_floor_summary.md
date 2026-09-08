# Point 6 — L1 density noise floor and binning dependence

Archival replay gate: **PASS 16/16**.

Historical 16x16 pooled PI-QMC vs archived exact P64 reconstruction: `0.020058017556`.

| bins | PI-QMC vs exact P64 | null median | null 95% | null 99% | p_null | SPLH vs exact P64 | seed 8/8 | odd/even | time halves |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 0.017726009 | 0.016839600 | [0.009600510, 0.028330383] | [0.007980038, 0.033074344] | 0.427278 | 0.090985682 | 0.040255737 | 0.039443359 | 0.045064087 |
| 16x16 | 0.020058018 | 0.020359802 | [0.012504868, 0.032640999] | [0.010639038, 0.037684698] | 0.523592 | 0.093910187 | 0.050521851 | 0.044873047 | 0.053944092 |
| 32x32 | 0.021999645 | 0.022659302 | [0.014384460, 0.035281372] | [0.012326656, 0.040430309] | 0.551557 | 0.095227361 | 0.055347900 | 0.049064331 | 0.057661743 |

Interpretation rule was predeclared. No linear subtraction of the noise floor from observed L1 is performed.
