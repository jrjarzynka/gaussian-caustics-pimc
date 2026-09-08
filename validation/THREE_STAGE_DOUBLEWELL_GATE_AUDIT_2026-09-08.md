# 1D three-stage double-well pre-caustic gate

Potential and convention follow Sadhasivam, Althorpe, and Kapil (Phys. Rev. E 114, 024126 (2026)):

`V(x) = -x^2/2 + g x^4/4 + 1/(16 g)`, with `M = hbar = omega = 1` and `g = 0.1, 0.5`.

Since `min_x V''(x) = -1` at `x=0`, the scalar first caustic is exactly `beta_c = pi`; hence `zeta_max = beta/pi`.

The exact thermal position density was obtained by converged finite-difference diagonalization of the 1D Hamiltonian. The three local-harmonic stages were implemented directly from the published scalar forms: bare determinant prefactor, renormalized pre-mapping form, and final harmonic-mapped form. The regular combination `F_beta(kappa)=(Xi-1)/kappa` was used through `kappa=0`.

## Primary near-caustic gate

At `beta=3.10`, `zeta_max=0.98676065 < 1`:

| g | L1 bare | L1 renormalized pre-map | L1 final mapped |
|---:|---:|---:|---:|
| 0.1 | 0.467417 | 0.153554 | 0.602413 |
| 0.5 | 0.648272 | 0.238121 | 0.262335 |

Thus the final mapped scalar stage also develops a substantial density error before the shared first caustic. This gate supports extending the existence-versus-accuracy distinction to the final scalar mapped stage; it does not define or benchmark a unique multidimensional harmonic mapping.

## Convergence

For `beta=3.10`, repeating the exact diagonalization on coarser boxes/grids changed the exact potential energy by approximately `1.1e-6` (`g=0.1`) and `2.1e-7` (`g=0.5`), and changed each reported L1 distance by less than `2e-6`.

Status: **PASS**.

## Convention provenance clarification (2026-09-08)

A separate audit (`DOUBLEWELL_CONVENTION_PROVENANCE_AUDIT_2026-09-08.md`)
checked the apparent mismatch between the additive constant `1/(16g)` and
the quartic coefficient. The gate intentionally uses the literal published
Eq. (44), `-x^2/2 + g x^4/4 + 1/(16g)`. Its minima are negative,
`V_min=-3/(16g)`; the offset is not a zero-minimum convention. The barrier
heights are `1/(4g)`, i.e. 2.5 and 0.5 for `g=0.1` and `0.5`. A converged
spectral count gives 4 and 1 below-barrier eigenvalues, respectively,
consistent with Ref. [1]'s deep-versus-shallow characterization. Numerical
L1 values in this audit are retained unchanged.
