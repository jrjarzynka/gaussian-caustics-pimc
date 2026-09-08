# Final mapped-stage additive-energy-origin audit

## Scope

This audit checks the structural dependence of the scalar final harmonic-mapped distribution of Ref. [1],

\[
\widetilde P_{\rm mapped}(x)\propto \sqrt{\Xi(x)}\,e^{-\beta V(x)\Xi(x)},
\]

on an additive constant in the potential.  The one-dimensional double-well convention printed in Ref. [1] is

\[
V_{\rm pub}(x)=-\frac{x^2}{2}+\frac{g x^4}{4}+\frac{1}{16g},
\]

with `M=hbar=omega=1`.  Its minima satisfy `x_min^2=1/g` and `V_min=-3/(16g)`.  A minimum-zero energy origin is therefore

\[
V_0(x)=V_{\rm pub}(x)+\frac{3}{16g}
      =-\frac{x^2}{2}+\frac{g x^4}{4}+\frac{1}{4g}.
\]

The shift changes neither derivatives nor the caustic field.

## Analytic invariance check

For the exact quantum density, the classical Boltzmann density, the bare local-harmonic density, and the renormalized pre-mapping density, `V -> V+c` contributes only a coordinate-independent factor `exp(-beta c)` before normalization.  Their normalized positional densities are therefore invariant.

For the final mapped stage,

\[
\widetilde P_{\rm mapped}^{(c)}(x)
=\widetilde P_{\rm mapped}^{(0)}(x)e^{-\beta c\Xi(x)}.
\]

For an anharmonic potential, `Xi(x)` is generally coordinate dependent, so this factor cannot be absorbed into the global normalization.  Additive-energy-origin invariance is recovered only when `Xi` is coordinate independent, including the harmonic limit.

## Numerical check

The deterministic script `three_stage_doublewell_energy_origin_sensitivity.py` repeats the converged one-dimensional thermal-density calculation and compares the literal published offset against the minimum-zero shift.  Across the complete beta grid, the largest pointwise difference between the normalized densities obtained under the two energy origins is

- bare stage: < 4e-16;
- renormalized pre-map stage: < 4e-16.

The final mapped density changes substantially.

Representative `L1` errors relative to the same exact normalized quantum density are:

| g | beta | zeta_max | bare | renorm | mapped, published offset | mapped, V_min=0 |
|---:|---:|---:|---:|---:|---:|---:|
|0.1|1.00|0.3183|0.01260|0.03663|0.06718|0.07470|
|0.1|2.50|0.7958|0.18960|0.02637|0.44253|0.18866|
|0.1|3.00|0.9549|0.37060|0.09365|0.57532|0.20464|
|0.1|3.10|0.9868|0.46742|0.15355|0.60241|0.20728|
|0.5|1.00|0.3183|0.02767|0.05371|0.03982|0.08487|
|0.5|2.50|0.7958|0.25686|0.08720|0.09529|0.39962|
|0.5|3.00|0.9549|0.49393|0.08663|0.21538|0.55579|
|0.5|3.10|0.9868|0.64827|0.23812|0.26234|0.57842|

At `beta=3.10`, the minimum-zero mapped `L1` values are independently stable under the pre-existing coarser-grid convergence check to about `1e-6`.

## Interpretation

PASS.  The quantitative output of the final scalar harmonic mapping is structurally sensitive to the additive energy origin for anharmonic potentials.  The magnitude of the mapped error in the double-well benchmark changes strongly under the minimum-zero shift, but the central pre-caustic conclusion survives both tested conventions: at `zeta_max=0.9868<1`, the final mapped density has a substantial error for both `g=0.1` and `g=0.5` under either energy origin.

This audit does not assume that Ref. [1] intended the minimum-zero convention; it reports the literal published convention and the minimum-zero shift as an explicit sensitivity test.
