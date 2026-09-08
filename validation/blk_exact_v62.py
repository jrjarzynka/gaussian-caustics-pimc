"""
Analytic V''_BLK(rho) using Struve/Bessel recurrences, so that the outer
caustic radius (where |V''| ~ 1e-4 eV nm^-2) is not limited by cancellation
in a finite difference.

V_BLK(rho) = -A [ H0(x) - Y0(x) ],  A = pi C / (2 r0s),  x = kappa sqrt(rho^2+D^2)/r0s

  H0'(x) = 2/pi - H1(x)          Y0'(x) = -Y1(x)
  H1'(x) = H0(x) - H1(x)/x       Y1'(x) = Y0(x) - Y1(x)/x
"""
import numpy as np
from scipy.special import struve, y0, y1
from scipy.optimize import brentq, minimize_scalar

HBAR_C, M0C2 = 197.3269804, 510998.95
HBAR2_M0 = HBAR_C**2 / M0C2
KB = 8.617333262e-5
C_COUL = 1.439964548

r0e, r0h = 4.479911, 3.493451
r0s = r0e + r0h
Dsep, kscr, mu = 0.60, 4.945, 0.22213
A_PREF = np.pi * C_COUL / (2.0 * r0s)


def _F(x):    return struve(0, x) - y0(x)
def _Fp(x):   return 2.0 / np.pi - struve(1, x) + y1(x)
def _Fpp(x):  return -_F(x) + (struve(1, x) - y1(x)) / x


def V_blk(rho):
    u = np.sqrt(rho**2 + Dsep**2)
    return -A_PREF * _F(kscr * u / r0s)


def d2V_blk(rho):
    u = np.sqrt(rho**2 + Dsep**2)
    x = kscr * u / r0s
    xp = kscr * rho / (r0s * u)
    xpp = kscr * Dsep**2 / (r0s * u**3)
    return -A_PREF * (_Fpp(x) * xp**2 + _Fp(x) * xpp)


if __name__ == "__main__":
    print("=" * 76)
    print("STEP 7'' analytic re-derivation of Eq. (101)")
    print("=" * 76)
    print(f"  V_BLK(0)   = {V_blk(0.0)*1e3:.4f} meV")
    print(f"  V''_BLK(0) = {d2V_blk(1e-9):.7f} eV nm^-2   "
          f"(manuscript: +0.3778422)")

    rho0 = brentq(d2V_blk, 0.05, 3.0, xtol=1e-15, rtol=8.9e-16)
    print(f"  rho_0      = {rho0:.7f} nm              "
          f"(manuscript: 0.5280867)")

    thresh = mu * (np.pi * KB * 20.0) ** 2 / HBAR2_M0
    f = lambda r: -d2V_blk(r) - thresh
    rho_in = brentq(f, rho0 + 1e-12, rho0 + 1.0, xtol=1e-15, rtol=8.9e-16)
    rho_out = brentq(f, 3.0, 100.0, xtol=1e-13, rtol=8.9e-16)
    print()
    print(f"  threshold |V''| = {thresh:.7e} eV nm^-2")
    print(f"  {'':16s} {'analytic':>16s} {'manuscript':>16s} {'rel. diff':>12s}")
    for nm, a, m in [("rho_in  (nm)", rho_in, 0.5282829),
                     ("rho_out (nm)", rho_out, 18.6909873)]:
        print(f"  {nm:16s} {a:16.7f} {m:16.7f} {abs(a-m)/m:12.2e}")

    # deepest negative curvature and the temperature at which the annulus closes
    res = minimize_scalar(lambda r: d2V_blk(r), bounds=(rho0, 20.0),
                          method="bounded", options={"xatol": 1e-12})
    print()
    print(f"  most negative V''_BLK = {res.fun:.7f} eV nm^-2 at rho = {res.x:.5f} nm")
    T_close = np.sqrt(-res.fun * HBAR2_M0 / mu) / (np.pi * KB)
    print(f"  -> BLK radial annulus closes entirely above T = {T_close:.0f} K")

    print()
    print("  Temperature dependence of the BLK-only radial invalid annulus:")
    print(f"  {'T (K)':>7s} {'rho_in (nm)':>13s} {'rho_out (nm)':>13s} {'width (nm)':>12s}")
    for Tk in [10, 20, 50, 100, 200, 300, 500, 1000]:
        th = mu * (np.pi * KB * Tk) ** 2 / HBAR2_M0
        g = lambda r: -d2V_blk(r) - th
        if -res.fun < th:
            print(f"  {Tk:7.0f} {'--':>13s} {'--':>13s} {'no annulus':>12s}")
            continue
        ri = brentq(g, rho0 + 1e-12, res.x, xtol=1e-15, rtol=8.9e-16)
        ro = brentq(g, res.x, 400.0, xtol=1e-13, rtol=8.9e-16)
        print(f"  {Tk:7.0f} {ri:13.6f} {ro:13.4f} {ro-ri:12.4f}")

    np.save("blk_exact_v62_results.npy", np.array([rho0, rho_in, rho_out, thresh, res.x, res.fun]))
