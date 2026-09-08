# Stage-independence audit - corrected bare stage

## Status

**PASS for the location of the first scalar Gaussian-existence boundary.**

The review package correctly identified the common root but its phase analysis defined the bare density with an extra `sqrt(Xi)` factor. The correct bare stage uses the Dirichlet determinant factor

`D = [sinh(2 xi)/(2 xi)]^(-1/2)`

in place of the renormalized `sqrt(Xi)` prefactor. Therefore the claimed phase cancellation and real bare continuation beyond `zeta=1` are rejected.

## Common first boundary

For negative curvature, `xi=i eta`:

- renormalized pre-mapping and mapped stages: `Xi=tan(eta)/eta`, first pole at `eta=pi/2`;
- bare stage: `D=[sin(2 eta)/(2 eta)]^(-1/2)`, first branch singularity at `eta=pi/2`;
- fixed-endpoint fluctuation operator: `lambda_1=(pi/beta)^2+kappa`, first zero at the same point.

Hence the first scalar real-domain boundary is common:

`eta_c=pi/2 <=> zeta_c=1`.

At the asymmetric off-stationary benchmark, all algebraically equivalent root conditions return the same double-precision beta. This is a regression check, not four independent physical measurements.

## Scope

The shared boundary strengthens the caustic-field and occupation results, because diagnostics defined solely by the `zeta>=1` set are not artifacts of selecting the renormalized pre-mapping stage. This stage-independence audit concerns the existence boundary only. The later one-dimensional three-stage double-well gate, preserved separately in this bundle, provides the distinct scalar accuracy test of the final mapped stage; no unique multidimensional mapped continuation is claimed.

## BLK side audit

The independent analytic BLK derivative script in the review package reproduces the finite-D contact curvature and inner/outer `zeta=1` crossings. Its very high-precision outer crossing differs from the earlier tabulation only at physically irrelevant digits, so v6.2 rounds the main-text value to `~18.7 nm`. The BLK-only radial invalid interval persists geometrically up to roughly 538 K, but this is **not** an equilibrium temperature scan of `p_inv(T)` and is not promoted to a main-paper result.
