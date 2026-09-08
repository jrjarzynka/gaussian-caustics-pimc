"""
Local anharmonicity diagnostic for the tensorial RLH construction.

Companion code for "Before the Caustic: Why Local Stability Does Not
Guarantee Quantum Accuracy".

Motivation
----------
The stability parameter zeta certifies that the local quadratic Gaussian
exists, but says nothing about whether the quadratic truncation is a good
model of the potential over the width that the construction itself smears.
This module supplies a second, equally cheap field that does address that
question.

The construction assigns to each local normal mode a a smearing variance

    sigma_a^2(r) = -F_beta(kappa_a(r)) / beta                        (D1)

where F_beta is the same spectral function that already appears in the
gradient term of the RLH density.  Equation (D1) is not an extra postulate:
at zero curvature it reduces to the free-particle width

    sigma^2(0) = beta hbar^2 / (12 M),

which is exactly the content of F_beta(0) = -beta^2 hbar^2 / (12 M).

The quadratic truncation neglects the cubic term of the local expansion.
Measuring that term over the smearing width, in units of k_B T, gives the
dimensionless diagnostic

    epsilon_a(r) = (beta / 6) |T_aaa(r)| sigma_a(r)^3,               (D2)
    epsilon(r)   = max_a epsilon_a(r),

with T_aaa the third derivative of V along local normal mode a.

Two properties make Eq. (D2) suitable as a companion to zeta rather than a
rival to it:

  1. It is computable from V, its derivatives, and the RLH construction
     alone -- no exact quantum reference is required.
  2. sigma_a diverges at the caustic, so epsilon -> infinity as zeta -> 1.
     The diagnostic therefore contains the existence criterion as a limiting
     case and degrades continuously before it, which is precisely the
     behaviour the caustic itself lacks.

Units follow the manuscript: M = hbar = |G_j| = 1.

Author: J. R. Jarzynka
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "FourierPotential2D",
    "triangular_benchmark",
    "asymmetric_benchmark",
    "spectral_Xi",
    "spectral_F",
    "zeta_field",
    "epsilon_field",
    "rlh_density",
    "diagnostic_scan",
]


# ---------------------------------------------------------------------
# Potential
# ---------------------------------------------------------------------


@dataclass
class FourierPotential2D:
    """
    First-shell Fourier potential on a triangular reciprocal lattice,

        V(r) = V0 - sum_j A_j cos(G_j . r + phi_j),

    with G_1 + G_2 + G_3 = 0 and |G_j| = 1.

    Derivatives are analytic:

        grad V      = + sum_j A_j sin(u_j) G_j
        Hess V      = + sum_j A_j cos(u_j) G_j (x) G_j
        third V     = - sum_j A_j sin(u_j) G_j (x) G_j (x) G_j

    where u_j = G_j . r + phi_j.

    Reduced coordinates
    -------------------
    The real-space lattice is chosen dual to (G_1, G_2), so that in reduced
    coordinates (s, t) the phases are exactly

        u_1 = 2 pi s + phi_1
        u_2 = 2 pi t + phi_2
        u_3 = -2 pi (s + t) + phi_3

    which is how the benchmark figures of the manuscript are parameterised.
    """

    V0: float = 3.0
    amplitudes: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    phases: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    hbar: float = 1.0

    G: np.ndarray = field(init=False, repr=False)
    lattice: np.ndarray = field(init=False, repr=False)
    cell_area: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g1 = np.array([1.0, 0.0])
        g2 = np.array([-0.5, np.sqrt(3.0) / 2.0])
        g3 = -(g1 + g2)
        self.G = np.vstack([g1, g2, g3])                      # (3, 2)

        B = np.vstack([g1, g2])                               # rows G1, G2
        self.lattice = 2.0 * np.pi * np.linalg.inv(B).T       # rows a1, a2
        self.cell_area = float(abs(np.linalg.det(self.lattice)))

    # -- geometry -----------------------------------------------------

    def reduced_to_cartesian(self, st: np.ndarray) -> np.ndarray:
        """(..., 2) reduced coordinates -> (..., 2) Cartesian."""
        return st @ self.lattice

    def phase(self, r: np.ndarray) -> np.ndarray:
        """u_j = G_j . r + phi_j, shape (..., 3)."""
        return r @ self.G.T + np.asarray(self.phases)

    # -- potential and derivatives ------------------------------------

    def value(self, r: np.ndarray) -> np.ndarray:
        A = np.asarray(self.amplitudes)
        return self.V0 - np.sum(A * np.cos(self.phase(r)), axis=-1)

    def gradient(self, r: np.ndarray) -> np.ndarray:
        A = np.asarray(self.amplitudes)
        w = A * np.sin(self.phase(r))                          # (..., 3)
        return np.einsum("...j,jk->...k", w, self.G)

    def hessian(self, r: np.ndarray) -> np.ndarray:
        A = np.asarray(self.amplitudes)
        w = A * np.cos(self.phase(r))                          # (..., 3)
        GG = np.einsum("jk,jl->jkl", self.G, self.G)           # (3, 2, 2)
        return np.einsum("...j,jkl->...kl", w, GG)

    def third_along(self, r: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Directional third derivative T_aaa = e_a . grad^3 V . e_a e_a.

        Parameters
        ----------
        r : (..., 2) positions
        e : (..., d, 2) unit normal-mode vectors, one row per mode

        Returns
        -------
        (..., d) third derivative along each mode direction
        """
        A = np.asarray(self.amplitudes)
        w = -A * np.sin(self.phase(r))                         # (..., 3)
        proj = np.einsum("...ak,jk->...aj", e, self.G)         # (..., d, 3)
        return np.einsum("...j,...aj->...a", w, proj ** 3)


def triangular_benchmark() -> FourierPotential2D:
    """Symmetric benchmark, Eq. (40) of the manuscript."""
    return FourierPotential2D(
        V0=3.0,
        amplitudes=(1.0, 1.0, 1.0),
        phases=(0.0, 0.0, 0.0),
    )


def asymmetric_benchmark(V0: Optional[float] = None,
                         n: int = 2048) -> FourierPotential2D:
    """
    Asymmetric benchmark, Eqs. (48)-(49) of the manuscript.

    V0 is fixed by the convention V_min = 0, i.e. V0 = max_r sum_j A_j
    cos(G_j.r + phi_j).  Passing V0 explicitly overrides this.  Getting the
    offset wrong shifts <V> by a constant and silently corrupts every
    relative energy error, so it is determined here rather than hard-coded.
    """
    pot = FourierPotential2D(
        V0=0.0,
        amplitudes=(1.0, 0.78, 0.62),
        phases=(0.0, 0.23, 0.71),
    )
    if V0 is None:
        S, T = _grid(n)
        r = pot.reduced_to_cartesian(np.stack([S, T], axis=-1))
        V0 = float(-np.min(pot.value(r)))
    pot.V0 = V0
    return pot


# ---------------------------------------------------------------------
# Spectral response functions
# ---------------------------------------------------------------------


def spectral_Xi(kappa: np.ndarray,
                beta: float,
                mass: float = 1.0,
                hbar: float = 1.0) -> np.ndarray:
    """
    Mode response Xi_a, Eqs. (6) and (15).

        kappa > 0 :  tanh(xi) / xi ,   xi  = beta hbar omega / 2
        kappa < 0 :  tan(eta) / eta ,  eta = beta hbar nu    / 2
        kappa = 0 :  1

    Returns +inf beyond the caustic (eta >= pi/2) rather than the spurious
    finite values produced by naive continuation of tan past its pole.
    """
    kappa = np.asarray(kappa, dtype=float)
    out = np.ones_like(kappa)

    pos = kappa > 0.0
    if np.any(pos):
        xi = 0.5 * beta * hbar * np.sqrt(kappa[pos] / mass)
        out[pos] = np.tanh(xi) / xi

    neg = kappa < 0.0
    if np.any(neg):
        eta = 0.5 * beta * hbar * np.sqrt(-kappa[neg] / mass)
        val = np.full_like(eta, np.inf)
        ok = eta < 0.5 * np.pi
        val[ok] = np.tan(eta[ok]) / eta[ok]
        out[neg] = val

    return out


def spectral_F(kappa: np.ndarray,
               beta: float,
               mass: float = 1.0,
               hbar: float = 1.0,
               small: float = 1e-10) -> np.ndarray:
    """
    F_beta(kappa) = (Xi - 1) / kappa, Eq. (8), with the finite limit

        F_beta(0) = -beta^2 hbar^2 / (12 M)                    Eq. (10)

    applied near kappa = 0 to avoid catastrophic cancellation.
    """
    kappa = np.asarray(kappa, dtype=float)
    Xi = spectral_Xi(kappa, beta, mass, hbar)

    out = np.empty_like(kappa)
    tiny = np.abs(kappa) < small
    out[tiny] = -(beta ** 2) * (hbar ** 2) / (12.0 * mass)

    big = ~tiny
    out[big] = (Xi[big] - 1.0) / kappa[big]
    return out


def smearing_variance(kappa: np.ndarray,
                      beta: float,
                      mass: float = 1.0,
                      hbar: float = 1.0) -> np.ndarray:
    """
    sigma_a^2 = -F_beta(kappa_a) / beta, Eq. (D1).

    Positive for both signs of curvature; diverges at the caustic.
    """
    return -spectral_F(kappa, beta, mass, hbar) / beta


# ---------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------


def _grid(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centred (s, t) sampling of the primitive cell."""
    u = (np.arange(n) + 0.5) / n
    S, T = np.meshgrid(u, u, indexing="ij")
    return S, T


def _eigh_fields(pot: FourierPotential2D,
                 r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hessian eigenvalues (ascending) and eigenvectors as row directions."""
    H = pot.hessian(r)
    kappa, R = np.linalg.eigh(H)               # R columns are eigenvectors
    e = np.swapaxes(R, -1, -2)                 # rows are eigenvectors
    return kappa, e


def zeta_field(pot: FourierPotential2D,
               beta: float,
               n: int = 256) -> np.ndarray:
    """
    zeta(r) = max_a (beta hbar / pi) sqrt(max(0, -kappa_a) / M), Eq. (19).
    """
    S, T = _grid(n)
    r = pot.reduced_to_cartesian(np.stack([S, T], axis=-1))
    kappa, _ = _eigh_fields(pot, r)
    neg = np.maximum(0.0, -kappa)
    z = (beta * pot.hbar / np.pi) * np.sqrt(neg / pot.mass)
    return np.max(z, axis=-1)


def epsilon_field(pot: FourierPotential2D,
                  beta: float,
                  n: int = 256) -> np.ndarray:
    """
    epsilon(r) = max_a (beta / 6) |T_aaa| sigma_a^3, Eq. (D2).
    """
    S, T = _grid(n)
    r = pot.reduced_to_cartesian(np.stack([S, T], axis=-1))
    kappa, e = _eigh_fields(pot, r)

    var = smearing_variance(kappa, beta, pot.mass, pot.hbar)
    sigma = np.sqrt(np.clip(var, 0.0, np.inf))
    T3 = pot.third_along(r, e)

    eps = (beta / 6.0) * np.abs(T3) * sigma ** 3
    return np.max(eps, axis=-1)


def rlh_density(pot: FourierPotential2D,
                beta: float,
                n: int = 256) -> np.ndarray:
    """
    Normalised tensorial RLH probability on the (s, t) grid, Eq. (11).

    Returns zeros in Gaussian-invalid cells (zeta >= 1), where the
    construction is undefined.
    """
    S, T = _grid(n)
    r = pot.reduced_to_cartesian(np.stack([S, T], axis=-1))

    kappa, e = _eigh_fields(pot, r)
    Xi = spectral_Xi(kappa, beta, pot.mass, pot.hbar)
    Fb = spectral_F(kappa, beta, pot.mass, pot.hbar)

    g = pot.gradient(r)
    g_mode = np.einsum("...ak,...k->...a", e, g)
    quad = np.sum(Fb * g_mode ** 2, axis=-1)

    V = pot.value(r)
    logw = -beta * (V + 0.5 * quad) + 0.5 * np.sum(np.log(Xi), axis=-1)

    bad = ~np.isfinite(logw)
    logw = np.where(bad, -np.inf, logw)

    w = np.exp(logw - np.max(logw[np.isfinite(logw)]))
    w = np.where(np.isfinite(w), w, 0.0)
    total = w.sum()
    return w / total if total > 0 else w


# ---------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------


def diagnostic_scan(pot: FourierPotential2D,
                    betas: np.ndarray,
                    n: int = 256) -> dict:
    """
    Both diagnostics over a temperature ladder.

    The RLH-weighted mean of epsilon is the quantity to correlate against
    the measured RLH error: like zeta, it needs no exact reference, but
    unlike zeta it responds to anharmonicity.

    Returns a dict of 1-D arrays suitable for pandas.DataFrame(...).
    """
    rows = {k: [] for k in
            ("beta", "zeta_max", "eps_max", "eps_mean_rlh", "eps_p95_rlh")}

    for beta in np.atleast_1d(betas):
        z = zeta_field(pot, beta, n)
        eps = epsilon_field(pot, beta, n)
        p = rlh_density(pot, beta, n)

        finite = np.isfinite(eps)
        w = np.where(finite, p, 0.0)
        w = w / w.sum() if w.sum() > 0 else w

        flat_e = eps[finite]
        flat_w = w[finite]
        order = np.argsort(flat_e)
        cw = np.cumsum(flat_w[order])
        p95 = flat_e[order][np.searchsorted(cw, 0.95 * cw[-1])] \
            if cw[-1] > 0 else np.nan

        rows["beta"].append(float(beta))
        rows["zeta_max"].append(float(np.max(z)))
        rows["eps_max"].append(float(np.max(eps[finite])) if finite.any()
                               else np.inf)
        rows["eps_mean_rlh"].append(float(np.sum(flat_w * flat_e)))
        rows["eps_p95_rlh"].append(float(p95))

    return {k: np.asarray(v) for k, v in rows.items()}


# ---------------------------------------------------------------------
# Self-test against manuscript values
# ---------------------------------------------------------------------


def _self_test() -> None:
    from scipy.optimize import minimize

    print("=" * 66)
    print("Consistency checks against manuscript values")
    print("=" * 66)

    tri = triangular_benchmark()

    H_min = tri.hessian(np.zeros(2))
    print(f"triangular, Hessian eigenvalues at minimum : "
          f"{np.linalg.eigvalsh(H_min)}   (paper: 1.5, 1.5)")

    r_sad = tri.reduced_to_cartesian(np.array([0.5, 0.0]))
    print(f"triangular, Hessian eigenvalues at saddle  : "
          f"{np.linalg.eigvalsh(tri.hessian(r_sad))}   (paper: -1.5, 0.5)")

    print(f"triangular, V range                        : "
          f"{tri.value(np.zeros(2)):.4f} .. "
          f"{tri.value(tri.reduced_to_cartesian(np.array([1/3, 1/3]))):.4f}"
          f"   (paper: 0 .. 4.5)")

    beta_c_tri = np.pi / np.sqrt(1.5)
    print(f"triangular, beta_c                         : "
          f"{beta_c_tri:.9f}   (paper: 2.565099660)")

    S, T = _grid(512)
    r = tri.reduced_to_cartesian(np.stack([S, T], axis=-1))
    kap, _ = _eigh_fields(tri, r)
    frac_neg = float(np.mean(kap[..., 0] < 0.0))
    print(f"triangular, fraction with a negative kappa : "
          f"{100 * frac_neg:.1f}%   (paper: 76.6%)")

    print("-" * 66)
    asym = asymmetric_benchmark()

    def kmin(st):
        rr = asym.reduced_to_cartesian(np.asarray(st))
        return float(np.linalg.eigvalsh(asym.hessian(rr))[0])

    S, T = _grid(400)
    grid = np.stack([S, T], axis=-1)
    kap, _ = _eigh_fields(asym, asym.reduced_to_cartesian(grid))
    idx = np.unravel_index(np.argmin(kap[..., 0]), kap[..., 0].shape)
    seed = grid[idx]

    res = minimize(kmin, seed, method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    st_star = np.mod(res.x, 1.0)

    print(f"asymmetric, global kappa_min               : "
          f"{res.fun:.9f}   (paper: -1.342192423)")
    print(f"asymmetric, location (s, t)                : "
          f"({st_star[0]:.3f}, {st_star[1]:.3f})   (paper: 0.500, 0.463)")
    print(f"asymmetric, beta_c global                  : "
          f"{np.pi / np.sqrt(-res.fun):.9f}   (paper: 2.711705246)")

    r_max = asym.reduced_to_cartesian(np.array([0.4157, 0.3466]))
    res_max = minimize(lambda st: -asym.value(
        asym.reduced_to_cartesian(np.asarray(st))),
        np.array([0.4157, 0.3466]), method="Nelder-Mead",
        options=dict(xatol=1e-12, fatol=1e-14))
    r_max = asym.reduced_to_cartesian(res_max.x)
    kmx = np.linalg.eigvalsh(asym.hessian(r_max))
    print(f"asymmetric, Hessian at maximum             : "
          f"{kmx}   (paper: -1.12373, -0.68538)")
    print(f"asymmetric, beta_c at maximum              : "
          f"{np.pi / np.sqrt(-kmx[0]):.5f}   (paper: 2.96360)")

    print("=" * 66)
    print("Diagnostic scan, asymmetric benchmark")
    print("=" * 66)
    betas = np.array([0.5, 1.0, 1.5, 1.9, 2.3, 2.5, 2.7])
    out = diagnostic_scan(asym, betas, n=256)
    print(f"{'beta':>6} {'zeta_max':>10} {'eps_mean':>10} {'eps_p95':>10}")
    for i in range(len(betas)):
        print(f"{out['beta'][i]:6.2f} {out['zeta_max'][i]:10.4f} "
              f"{out['eps_mean_rlh'][i]:10.4f} {out['eps_p95_rlh'][i]:10.4f}")


if __name__ == "__main__":
    _self_test()
