# TB-04A finite-D BLK radial-curvature diagnostic

## Parameters

- `D = 0.600000 nm`
- `r0e = 4.479911 nm`
- `r0h = 3.493451 nm`
- `kappa = 4.945000`
- `me = 0.580000 m0`
- `mh = 0.360000 m0`
- `mu = 0.22212766 m0`
- `T = 20.000 K`

## Deterministic results

- `V_BLK(0) = -0.252624794610 eV`
- `U''(0) = 0.377842197092 eV/nm^2`
- radial-curvature sign change: `rho = 0.528086702624 nm`
- zeta=1 inner crossing: `rho = 0.528282918499 nm`
- zeta=1 outer crossing: `rho = 18.690987326792 nm`
- zeta=1 curvature threshold: `U'' = -8.545849163462e-05 eV/nm^2`
- `zeta_BLK(rho=1.64 nm) = 20.64093281`
- `zeta_BLK(rho=1.83 nm) = 18.84924348`

## Interpretation

The finite interlayer separation regularizes the BLK interaction at
rho=0. The radial curvature is finite and positive at contact, so the
deep Gaussian-invalid regime is not caused by a Coulombic
rho->0 curvature divergence. Negative radial curvature emerges only
after the finite-separation sign change near 0.528 nm.

At 20 K, the BLK-only radial-relative zeta exceeds unity over a broad
finite-separation interval from approximately 0.528 to
18.69 nm. At the publication-scale sampled median pair
separation 1.64 nm, the simple BLK-only radial estimate is
zeta ~= 20.6, consistent in scale with the full 4D
PI-QMC/mode-attribution result zeta_50 ~= 17.3.

The 1D radial analysis predicts the physical mechanism; it does not
determine the equilibrium occupation probability p_inv. The latter
requires the full temperature-dependent quantum equilibrium measure.
