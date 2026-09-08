#!/usr/bin/env python3
"""
TB-04A: finite-D bilayer-Keldysh radial-curvature diagnostic.

Purpose
-------
Reviewer-facing deterministic check for Sec. VIII of
"Local Stability Does Not Guarantee Quantum Accuracy:
 Gaussian Caustics in Multidimensional and Interacting Systems".

The script demonstrates that the finite interlayer separation D regularizes
the BLK interaction at rho=0, producing a finite *positive* radial curvature
there. The negative radial curvature that drives the deep Gaussian-invalid
regime emerges only at finite in-plane separation.

It also evaluates the BLK-only radial-relative instability parameter

    zeta_BLK(rho) = (beta*hbar/pi) * sqrt(max(0, -U''(rho)/mu))

using the manuscript parameters.

No Monte Carlo data are generated or modified.

Outputs
-------
- tb04a_blk_radial_curvature.csv
- tb04a_blk_radial_curvature_summary.md

Dependencies
------------
numpy, scipy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import constants, optimize, special


# ------------------------- manuscript parameters -------------------------

D_NM = 0.60
R0E_NM = 4.479911
R0H_NM = 3.493451
KAPPA = 4.945

ME_M0 = 0.58
MH_M0 = 0.36
MU_M0 = ME_M0 * MH_M0 / (ME_M0 + MH_M0)

T_K = 20.0

# Publication-scale pair separations reported in Sec. VIII.
RHO_MEDIAN_NM = 1.64
RHO_MEAN_NM = 1.83


# ------------------------- BLK implementation ----------------------------

# Match the production BLK source constant exactly.
C_EVN_NM = 1.43996448
R0SUM_NM = R0E_NM + R0H_NM
A_Ev = np.pi * C_EVN_NM / (2.0 * R0SUM_NM)
ALPHA_INV_NM = KAPPA / R0SUM_NM


def blk_value_first_second(rho_nm: np.ndarray | float):
    """
    Return V_BLK, dV/drho, d2V/drho2 in eV, eV/nm, eV/nm^2.

    V_BLK(rho) = -A [H_0(x)-Y_0(x)]
    x = alpha * sqrt(rho^2 + D^2)

    Exact derivative identities:
      d H_0 / dx = H_{-1}
      d Y_0 / dx = -Y_1
      d2 H_0 / dx2 = H_{-2} + H_{-1}/x
      d2 Y_0 / dx2 = -Y_0 + Y_1/x
    """
    rho = np.asarray(rho_nm, dtype=float)
    s = np.sqrt(rho * rho + D_NM * D_NM)
    x = ALPHA_INV_NM * s

    f = special.struve(0, x) - special.yv(0, x)
    fp = special.struve(-1, x) + special.yv(1, x)
    fpp = (
        special.struve(-2, x)
        + special.struve(-1, x) / x
        + special.yv(0, x)
        - special.yv(1, x) / x
    )

    xp = ALPHA_INV_NM * rho / s
    xpp = ALPHA_INV_NM * D_NM * D_NM / s**3

    V = -A_Ev * f
    Vp = -A_Ev * fp * xp
    Vpp = -A_Ev * (fpp * xp * xp + fp * xpp)
    return V, Vp, Vpp


def tangential_curvature(rho_nm: np.ndarray | float):
    """
    U'(rho)/rho in eV/nm^2, with the rho->0 limit set to U''(0).
    """
    rho = np.asarray(rho_nm, dtype=float)
    _, Vp, Vpp = blk_value_first_second(rho)
    out = np.empty_like(rho, dtype=float)

    nz = rho != 0.0
    out[nz] = Vp[nz] / rho[nz]
    out[~nz] = Vpp[~nz]

    if out.ndim == 0:
        return float(out)
    return out


def zeta_from_radial_curvature(Vpp_eV_nm2: np.ndarray | float):
    """
    BLK-only radial-relative zeta from Lambda_rho = U''/mu.
    """
    Vpp = np.asarray(Vpp_eV_nm2, dtype=float)

    # eV/nm^2 -> J/m^2
    Vpp_SI = Vpp * constants.electron_volt / (1.0e-9**2)
    mu_kg = MU_M0 * constants.m_e
    Lambda_s2 = Vpp_SI / mu_kg

    beta_J_inv = 1.0 / (constants.k * T_K)
    zeta = (
        beta_J_inv
        * constants.hbar
        / np.pi
        * np.sqrt(np.maximum(0.0, -Lambda_s2))
    )
    return zeta


def radial_curvature_scalar(rho_nm: float) -> float:
    return float(blk_value_first_second(float(rho_nm))[2])


def zeta_scalar(rho_nm: float) -> float:
    return float(zeta_from_radial_curvature(radial_curvature_scalar(rho_nm)))


def find_roots():
    # U'' sign change: positive core -> negative radial-curvature region.
    rho_sign = optimize.brentq(radial_curvature_scalar, 0.1, 1.0)

    # Solve the curvature threshold corresponding to zeta = 1.
    beta_J_inv = 1.0 / (constants.k * T_K)
    mu_kg = MU_M0 * constants.m_e
    lambda_thr_s2 = -(np.pi / (beta_J_inv * constants.hbar)) ** 2
    curvature_thr_eV_nm2 = (
        lambda_thr_s2
        * mu_kg
        * (1.0e-9**2)
        / constants.electron_volt
    )

    def g(rho):
        return radial_curvature_scalar(rho) - curvature_thr_eV_nm2

    rho_zeta1_inner = optimize.brentq(g, rho_sign, 0.60)
    rho_zeta1_outer = optimize.brentq(g, 10.0, 30.0)
    return rho_sign, rho_zeta1_inner, rho_zeta1_outer, curvature_thr_eV_nm2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default=".",
        help="Output directory (default: current directory).",
    )
    ap.add_argument(
        "--rmax-nm",
        type=float,
        default=25.0,
        help="Maximum rho for diagnostic table (default: 25 nm).",
    )
    ap.add_argument(
        "--n-grid",
        type=int,
        default=2501,
        help="Number of rho grid points (default: 2501).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rho_sign, rho_z1_in, rho_z1_out, curv_thr = find_roots()

    V0, Vp0, Vpp0 = blk_value_first_second(0.0)
    zeta_med = zeta_scalar(RHO_MEDIAN_NM)
    zeta_mean = zeta_scalar(RHO_MEAN_NM)

    # Dense deterministic table.
    rho = np.linspace(0.0, args.rmax_nm, args.n_grid)
    V, Vp, Vpp = blk_value_first_second(rho)
    tang = tangential_curvature(rho)
    zeta = zeta_from_radial_curvature(Vpp)

    csv_path = outdir / "tb04a_blk_radial_curvature.csv"
    header = (
        "rho_nm,V_BLK_eV,dV_drho_eV_per_nm,"
        "d2V_drho2_eV_per_nm2,"
        "tangential_curvature_eV_per_nm2,zeta_BLK_radial"
    )
    np.savetxt(
        csv_path,
        np.column_stack([rho, V, Vp, Vpp, tang, zeta]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e",
    )

    summary_path = outdir / "tb04a_blk_radial_curvature_summary.md"
    summary = f"""# TB-04A finite-D BLK radial-curvature diagnostic

## Parameters

- `D = {D_NM:.6f} nm`
- `r0e = {R0E_NM:.6f} nm`
- `r0h = {R0H_NM:.6f} nm`
- `kappa = {KAPPA:.6f}`
- `me = {ME_M0:.6f} m0`
- `mh = {MH_M0:.6f} m0`
- `mu = {MU_M0:.8f} m0`
- `T = {T_K:.3f} K`

## Deterministic results

- `V_BLK(0) = {float(V0):.12f} eV`
- `U''(0) = {float(Vpp0):.12f} eV/nm^2`
- radial-curvature sign change: `rho = {rho_sign:.12f} nm`
- zeta=1 inner crossing: `rho = {rho_z1_in:.12f} nm`
- zeta=1 outer crossing: `rho = {rho_z1_out:.12f} nm`
- zeta=1 curvature threshold: `U'' = {curv_thr:.12e} eV/nm^2`
- `zeta_BLK(rho=1.64 nm) = {zeta_med:.8f}`
- `zeta_BLK(rho=1.83 nm) = {zeta_mean:.8f}`

## Interpretation

The finite interlayer separation regularizes the BLK interaction at
rho=0. The radial curvature is finite and positive at contact, so the
deep Gaussian-invalid regime is not caused by a Coulombic
rho->0 curvature divergence. Negative radial curvature emerges only
after the finite-separation sign change near {rho_sign:.3f} nm.

At 20 K, the BLK-only radial-relative zeta exceeds unity over a broad
finite-separation interval from approximately {rho_z1_in:.3f} to
{rho_z1_out:.2f} nm. At the publication-scale sampled median pair
separation 1.64 nm, the simple BLK-only radial estimate is
zeta ~= {zeta_med:.1f}, consistent in scale with the full 4D
PI-QMC/mode-attribution result zeta_50 ~= 17.3.

The 1D radial analysis predicts the physical mechanism; it does not
determine the equilibrium occupation probability p_inv. The latter
requires the full temperature-dependent quantum equilibrium measure.
"""
    summary_path.write_text(summary, encoding="utf-8")

    print("=== TB-04A FINITE-D BLK RADIAL-CURVATURE CHECK ===")
    print(f"mu/m0                    = {MU_M0:.8f}")
    print(f"U''(0)                   = {float(Vpp0):+.12f} eV/nm^2")
    print(f"radial sign change       = {rho_sign:.12f} nm")
    print(f"zeta=1 inner crossing    = {rho_z1_in:.12f} nm")
    print(f"zeta=1 outer crossing    = {rho_z1_out:.12f} nm")
    print(f"zeta_BLK(1.64 nm)        = {zeta_med:.8f}")
    print(f"zeta_BLK(1.83 nm)        = {zeta_mean:.8f}")
    print()
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
