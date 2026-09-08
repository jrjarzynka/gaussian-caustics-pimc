"""
Mass-weighted 4D local Gaussian stability field for the interacting pair.

Companion module for the two-body extension of "Before the Caustic".

What it computes
----------------
For an electron at r_e and a hole at r_h the total potential is

    V(r_e, r_h) = V_e(r_e) + V_h(r_h) + V_eh(|r_e - r_h|)

with a 4x4 Hessian in block form

    H = [[ H_e + H_rel ,   -H_rel      ],
         [   -H_rel     ,  H_h + H_rel ]]

where H_rel = grad grad V_eh evaluated at rho = r_e - r_h.  Because the
carriers have unequal effective masses, the quantum response follows the
eigenmodes of the MASS-WEIGHTED Hessian

    W = M^{-1/2} H M^{-1/2},     M = diag(m_e, m_e, m_h, m_h),

whose eigenvectors are in general NOT those of H.  This is the point at
which the isotropic-mass construction of the single-particle paper has to
be generalised: diagonalising H itself is correct only for a scalar mass.

The stability parameter is then, per mode,

    zeta_a = (beta / pi) * sqrt( max(0, -lambda_a) * hbar^2 / m_0 )

with lambda_a the eigenvalues of W in eV nm^-2 m_0^-1.

Interaction Hessian
-------------------
For a central V_eh(rho) in two dimensions,

    H_rel = V''(rho) * (rho^ x rho^) + (V'(rho)/rho) * (I - rho^ x rho^),

i.e. radial curvature V'' and transverse curvature V'/rho.  For the
bilayer Keldysh potential V depends on sqrt(rho^2 + D^2), so V has a
minimum at rho = 0: the radial curvature is positive at small separation
and negative beyond a crossover set by the interlayer distance D.

The full density
----------------
The stability criterion needs only the spectrum of W, but the RLH density
itself also requires the linear term to be transformed.  Writing the local
fluctuation in mass-weighted coordinates w = M^{1/2} u turns the gradient
into g~ = M^{-1/2} g, so

    P_RLH(r_e, r_h) ~ sqrt(det Xi_beta(W))
                      * exp{ -beta [ V + (1/2) g^T M^{-1/2} F_beta(W)
                                            M^{-1/2} g ] }.

Using Xi_beta(W) without also conjugating F_beta by M^{-1/2} gives the
right caustic but the wrong density.  With M = M I this reduces exactly to
the isotropic single-particle construction; ``_self_test`` checks the
identity numerically.

Validation
----------
Setting the landscape amplitudes to zero must reproduce the isolated-pair
result: a Gaussian-invalid annulus whose inner edge is temperature
independent.  ``python lha15a0_pair_stability_field.py`` checks this.

Units: nm, eV, masses in m_0.

Author: J. R. Jarzynka
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.special import struve, yv

__all__ = [
    "PairStabilityField",
    "production_field",
]

COULOMB_CONSTANT_EV_NM = 1.43996448
HBAR2_OVER_2M0 = 0.0380998                 # eV nm^2 (times m_0)
HBAR2_OVER_M0 = 2.0 * HBAR2_OVER_2M0
KB_EV_PER_K = 8.617333e-5


# ---------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------


@dataclass
class BilayerKeldysh:
    """
    Bilayer Keldysh interaction, Kamban and Pedersen, Sci. Rep. 10, 5537 (2020).

        V(rho) = -pi C / (2 r0_sum) [H_0(x) - Y_0(x)],
        x      = kappa sqrt(rho^2 + D^2) / r0_sum,
        r0_sum = r0_1 + r0_2.

    Mirrors the convention of ``tmd_pimc.bilayer_keldysh_potential``.  The
    finite separation D makes V(0) finite, which is what makes the
    invalid region an annulus rather than a disc.
    """

    separation_nm: float
    screening_length_layer1_nm: float
    screening_length_layer2_nm: float
    kappa_environment: float
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM

    _r0_sum: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("separation_nm", "screening_length_layer1_nm",
                     "screening_length_layer2_nm", "kappa_environment"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self._r0_sum = (self.screening_length_layer1_nm
                        + self.screening_length_layer2_nm)

    def value(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        s = np.sqrt(rho ** 2 + self.separation_nm ** 2)
        x = self.kappa_environment * s / self._r0_sum
        return (-np.pi * self.coulomb_constant_eV_nm
                / (2.0 * self._r0_sum) * (struve(0, x) - yv(0, x)))

    def radial_derivatives(self, rho: np.ndarray,
                           rel: float = 3e-3,
                           floor: float = 1e-4
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (V'(rho), V''(rho)) by scaled central differences.

        The step is scaled with rho because V'' spans ten orders of
        magnitude across the relevant range; a fixed step is dominated by
        roundoff beyond a few tens of nm.
        """
        rho = np.asarray(rho, dtype=float)
        h = np.maximum(floor, np.abs(rho) * rel)
        vp = self.value(rho + h)
        vm = self.value(rho - h)
        v0 = self.value(rho)
        return (vp - vm) / (2.0 * h), (vp - 2.0 * v0 + vm) / h ** 2

    def gradient(self, d: np.ndarray) -> np.ndarray:
        """grad_{r_e} V_eh at relative displacement d = r_e - r_h."""
        d = np.asarray(d, dtype=float)
        rho = np.linalg.norm(d, axis=-1)
        safe = np.maximum(rho, 1e-12)
        v1, _ = self.radial_derivatives(rho)
        return (v1 / safe)[..., None] * d

    def hessian(self, d: np.ndarray) -> np.ndarray:
        """
        Cartesian Hessian of V_eh at relative displacement d, shape (..., 2, 2).

        Uses the radial/transverse decomposition rather than a Cartesian
        finite difference, which keeps the transverse mode exact near
        rho = 0 where V'/rho is a 0/0 limit.
        """
        d = np.asarray(d, dtype=float)
        rho = np.linalg.norm(d, axis=-1)
        safe = np.maximum(rho, 1e-12)
        n = d / safe[..., None]

        v1, v2 = self.radial_derivatives(rho)
        transverse = np.where(rho > 1e-9, v1 / safe, v2)

        nn = np.einsum("...i,...j->...ij", n, n)
        eye = np.broadcast_to(np.eye(2), nn.shape)
        return (v2[..., None, None] * nn
                + transverse[..., None, None] * (eye - nn))


# ---------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------


@dataclass
class MoireLandscape:
    """
    Hexagonal first-shell moire potential matching ``tmd_pimc.potentials``:

        V(r) = A [cos(G1.r) + cos(G2.r) + cos(G3.r)],
        |G| = 4 pi / (sqrt(3) P).

    ``origin_nm`` shifts the landscape seen by one carrier relative to the
    other, i.e. the registry offset.
    """

    amplitude_eV: float
    period_nm: float = 20.0
    origin_nm: Tuple[float, float] = (0.0, 0.0)

    G: np.ndarray = field(init=False, repr=False)
    lattice: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.period_nm <= 0.0:
            raise ValueError("period_nm must be positive")
        G = 4.0 * np.pi / (np.sqrt(3.0) * self.period_nm)
        self.G = np.array([
            [G, 0.0],
            [-0.5 * G, np.sqrt(3.0) / 2.0 * G],
            [-0.5 * G, -np.sqrt(3.0) / 2.0 * G],
        ])
        # Real-space lattice DUAL to (G1, G2).  Deriving it rather than
        # writing (P, 0), (P/2, sqrt3 P/2) matters: those two vectors are
        # not dual to this reciprocal basis and are not lattice vectors of
        # V, so any cell built from them is not a period of the potential.
        self.lattice = 2.0 * np.pi * np.linalg.inv(self.G[:2]).T

    def _phase(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=float)
                - np.asarray(self.origin_nm)) @ self.G.T

    def value(self, r: np.ndarray) -> np.ndarray:
        return self.amplitude_eV * np.sum(np.cos(self._phase(r)), axis=-1)

    def gradient(self, r: np.ndarray) -> np.ndarray:
        """grad V = -A sum_j sin(G_j.r) G_j, shape (..., 2)."""
        w = -self.amplitude_eV * np.sin(self._phase(r))
        return np.einsum("...j,jk->...k", w, self.G)

    def hessian(self, r: np.ndarray) -> np.ndarray:
        """grad grad V = -A sum_j cos(G_j.r) G_j (x) G_j, shape (..., 2, 2)."""
        w = -self.amplitude_eV * np.cos(self._phase(r))
        GG = np.einsum("jk,jl->jkl", self.G, self.G)
        return np.einsum("...j,jkl->...kl", w, GG)


# ---------------------------------------------------------------------
# Spectral response on the mass-weighted spectrum
# ---------------------------------------------------------------------


def spectral_Xi(lam: np.ndarray, beta: float) -> np.ndarray:
    """
    Mass-weighted curvature eigenvalues lambda_a.
    With carrier masses expressed in units of m_0,

    (hbar * omega_a)^2 = (hbar^2 / m_0) * lambda_a

    for lambda_a > 0; for lambda_a < 0 the same relation defines the
    imaginary frequency nu through

    (hbar * nu_a)^2 = (hbar^2 / m_0) * (-lambda_a).

    The spectral response is

    lambda > 0 : tanh(xi)/xi,  xi  = beta hbar omega / 2
    lambda < 0 : tan(eta)/eta, eta = beta hbar nu    / 2

    Returns +inf beyond the caustic rather than continuing tan past its pole.
    """
    lam = np.asarray(lam, dtype=float)
    out = np.ones_like(lam)

    pos = lam > 0.0
    if np.any(pos):
        xi = 0.5 * beta * np.sqrt(HBAR2_OVER_M0 * lam[pos])
        out[pos] = np.tanh(xi) / xi

    neg = lam < 0.0
    if np.any(neg):
        eta = 0.5 * beta * np.sqrt(-HBAR2_OVER_M0 * lam[neg])
        val = np.full_like(eta, np.inf)
        ok = eta < 0.5 * np.pi
        val[ok] = np.tan(eta[ok]) / eta[ok]
        out[neg] = val
    return out


def spectral_F(lam: np.ndarray, beta: float, small: float = 1e-10
               ) -> np.ndarray:
    """
    F_beta(lambda) = (Xi - 1)/lambda, with the finite limit

        F_beta(0) = -beta^2 hbar^2 / 12

    (the mass is already carried by lambda, so no explicit M appears).
    """
    lam = np.asarray(lam, dtype=float)
    Xi = spectral_Xi(lam, beta)
    out = np.empty_like(lam)
    tiny = np.abs(lam) < small
    out[tiny] = -(beta ** 2) * HBAR2_OVER_M0 / 12.0
    big = ~tiny
    out[big] = (Xi[big] - 1.0) / lam[big]
    return out


# ---------------------------------------------------------------------
# Stability field
# ---------------------------------------------------------------------


@dataclass
class PairStabilityField:
    """Mass-weighted 4D local Gaussian stability for an interacting pair."""

    interaction: BilayerKeldysh
    landscape_e: MoireLandscape
    landscape_h: MoireLandscape
    mass_e_m0: float
    mass_h_m0: float

    _Minv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        inv = np.array([1.0 / np.sqrt(self.mass_e_m0)] * 2
                       + [1.0 / np.sqrt(self.mass_h_m0)] * 2)
        self._Minv = inv

    @property
    def reduced_mass_m0(self) -> float:
        return (self.mass_e_m0 * self.mass_h_m0
                / (self.mass_e_m0 + self.mass_h_m0))

    def hessian_4d(self, r_e: np.ndarray, r_h: np.ndarray) -> np.ndarray:
        """Full 4x4 Hessian, shape (..., 4, 4)."""
        r_e = np.asarray(r_e, dtype=float)
        r_h = np.asarray(r_h, dtype=float)
        H_rel = self.interaction.hessian(r_e - r_h)
        H_e = self.landscape_e.hessian(r_e) + H_rel
        H_h = self.landscape_h.hessian(r_h) + H_rel

        shape = np.broadcast_shapes(H_e.shape[:-2], H_h.shape[:-2])
        out = np.zeros(shape + (4, 4))
        out[..., :2, :2] = H_e
        out[..., 2:, 2:] = H_h
        out[..., :2, 2:] = -H_rel
        out[..., 2:, :2] = -H_rel
        return out

    def mass_weighted_eigenvalues(self, r_e, r_h) -> np.ndarray:
        """
        Eigenvalues of M^{-1/2} H M^{-1/2}, ascending, shape (..., 4).

        These are the squared local normal-mode frequencies in
        eV nm^-2 m_0^-1.  Their eigenvectors differ from those of H
        whenever m_e != m_h.
        """
        H = self.hessian_4d(r_e, r_h)
        W = H * self._Minv[:, None] * self._Minv[None, :]
        return np.linalg.eigvalsh(W)

    def value(self, r_e, r_h) -> np.ndarray:
        """Total potential V = V_e + V_h + V_eh."""
        r_e = np.asarray(r_e, dtype=float)
        r_h = np.asarray(r_h, dtype=float)
        return (self.landscape_e.value(r_e)
                + self.landscape_h.value(r_h)
                + self.interaction.value(
                    np.linalg.norm(r_e - r_h, axis=-1)))

    def gradient_4d(self, r_e, r_h) -> np.ndarray:
        """Gradient of the total potential, shape (..., 4)."""
        r_e = np.asarray(r_e, dtype=float)
        r_h = np.asarray(r_h, dtype=float)
        g_rel = self.interaction.gradient(r_e - r_h)
        g_e = self.landscape_e.gradient(r_e) + g_rel
        g_h = self.landscape_h.gradient(r_h) - g_rel
        return np.concatenate([g_e, g_h], axis=-1)

    def rlh_log_density(self, r_e, r_h, temperature_K: float) -> np.ndarray:
        """
        Unnormalised log of the mass-weighted tensorial RLH density.

            log P = (1/2) sum_a log Xi_a
                    - beta [ V + (1/2) g~^T R diag(F_a) R^T g~ ],
            g~    = M^{-1/2} g.

        Returns -inf wherever the local Gaussian does not exist, so the
        caller can see the invalid set directly.
        """
        beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
        H = self.hessian_4d(r_e, r_h)
        W = H * self._Minv[:, None] * self._Minv[None, :]
        lam, R = np.linalg.eigh(W)

        Xi = spectral_Xi(lam, beta)
        Fb = spectral_F(lam, beta)

        g_tilde = self.gradient_4d(r_e, r_h) * self._Minv
        g_mode = np.einsum("...ka,...k->...a", R, g_tilde)
        quad = np.sum(Fb * g_mode ** 2, axis=-1)

        V = self.value(r_e, r_h)
        with np.errstate(invalid="ignore", divide="ignore"):
            logP = (0.5 * np.sum(np.log(Xi), axis=-1)
                    - beta * (V + 0.5 * quad))
        return np.where(np.isfinite(logP), logP, -np.inf)

    def zeta(self, r_e, r_h, temperature_K: float) -> np.ndarray:
        """zeta = max_a (beta/pi) sqrt(max(0,-lambda_a) hbar^2/m_0)."""
        beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
        lam = self.mass_weighted_eigenvalues(r_e, r_h)
        z = (beta / np.pi) * np.sqrt(
            np.maximum(0.0, -lam) * HBAR2_OVER_M0)
        return np.max(z, axis=-1)

    # -- what the paper actually needs ---------------------------------

    def occupation_from_samples(self, r_e, r_h,
                                temperature_K: float,
                                chain_id=None,
                                n_bootstrap: int = 2000,
                                rng_seed: int = 0) -> dict:
        """
        Probability weight of the Gaussian-invalid region from PI-QMC.

            p_inv = < Theta[ zeta(r_e, r_h) - 1 ] >_{PI-QMC}

        Pass PAIRED coordinates from the same imaginary-time slice:
        r_e[i] and r_h[i] must be the electron and hole of one bead index
        j of one configuration.  Mixing slices is meaningless, since the
        pair Hessian is a property of a single configuration.

        The beads of a ring polymer are correlated, so uncertainty is
        estimated by resampling whole chains, not individual samples.
        Pass ``chain_id`` (one integer per sample) to get a whole-chain
        bootstrap interval; without it only the point estimate is
        returned.

        Note that this is p_inv relative to the finite-P ensemble that the
        sampler draws from, not to the continuum density.  Because
        threshold statistics need not converge in P at the same rate as
        smooth observables, check it on a P ladder before quoting it.
        """
        r_e = np.asarray(r_e, dtype=float)
        r_h = np.asarray(r_h, dtype=float)
        if r_e.shape != r_h.shape:
            raise ValueError(
                f"r_e and r_h must be paired sample-for-sample; got "
                f"{r_e.shape} and {r_h.shape}")
        if r_e.shape[-1] != 2:
            raise ValueError("coordinates must have trailing dimension 2")
        if chain_id is not None and np.shape(chain_id)[0] != r_e.shape[0]:
            raise ValueError(
                f"chain_id has length {np.shape(chain_id)[0]} but there are "
                f"{r_e.shape[0]} samples")

        z = self.zeta(r_e, r_h, temperature_K)
        invalid = z >= 1.0

        out = {
            "temperature_K": float(temperature_K),
            "n_samples": int(z.size),
            "p_invalid": float(np.mean(invalid)),
            "zeta_median": float(np.median(z)),
            "zeta_p95": float(np.percentile(z, 95)),
            "zeta_max": float(np.max(z)),
        }

        if chain_id is None:
            return out

        chain_id = np.asarray(chain_id)
        chains = np.unique(chain_id)
        per_chain = np.array([float(np.mean(invalid[chain_id == c]))
                              for c in chains])
        rng = np.random.default_rng(rng_seed)
        draws = rng.integers(0, chains.size,
                             size=(n_bootstrap, chains.size))
        boot = per_chain[draws].mean(axis=1)

        out.update({
            "n_chains": int(chains.size),
            "p_invalid_chain_mean": float(per_chain.mean()),
            "p_invalid_chain_sem": float(per_chain.std(ddof=1)
                                         / np.sqrt(chains.size)),
            "p_invalid_ci95": (float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))),
        })
        return out

    def invalid_fraction_at_separation(self, rho_nm, temperature_K: float,
                                       n_cell: int = 24,
                                       n_angle: int = 24) -> np.ndarray:
        """
        Conditional Gaussian-invalid fraction at fixed pair separation.

        Why not an unconditional f_inv
        ------------------------------
        The interacting potential V_e(r_e) + V_h(r_h) + V_eh(|r_e - r_h|)
        is invariant only under the DIAGONAL lattice translation
        (r_e, r_h) -> (r_e + R, r_h + R); translating one carrier alone
        changes the energy.  The production sampler confirms this: it
        evaluates the landscapes periodically but takes the ordinary
        unwrapped Euclidean pair separation, with no minimum image.

        The fundamental domain is therefore A x R^2 -- one cell for the
        pair position, and an UNBOUNDED relative coordinate.  A uniform
        geometric fraction over an unbounded direction does not exist
        without an arbitrary cutoff, so there is no cutoff-free scalar
        f_inv for this system.  An earlier product-cell version of this
        method assumed A_e x A_h, which is not a fundamental domain, and
        has been removed.

        What is well defined
        --------------------
        At fixed |rho| the remaining domain is compact: pair position over
        one cell, and the direction of rho over the circle.  This method
        returns

            f_inv(rho) = < Theta[ zeta - 1 ] >_{cell, angle}

        which is a genuine cutoff-free curve.  Report it against the
        PI-QMC separation distribution rather than collapsing it to a
        single number.

        Returns an array with the shape of rho_nm.
        """
        rho_nm = np.atleast_1d(np.asarray(rho_nm, dtype=float))
        lat = self.landscape_e.lattice

        u = (np.arange(n_cell) + 0.5) / n_cell
        S, T = np.meshgrid(u, u, indexing="ij")
        cell = (S[..., None] * lat[0]
                + T[..., None] * lat[1]).reshape(-1, 2)

        ang = (np.arange(n_angle) + 0.5) / n_angle * 2.0 * np.pi
        direction = np.stack([np.cos(ang), np.sin(ang)], axis=-1)

        out = np.empty(rho_nm.shape)
        for i, rho in enumerate(rho_nm):
            d = rho * direction                      # (n_angle, 2)
            r_e = cell[:, None, :] + 0.5 * d[None, :, :]
            r_h = cell[:, None, :] - 0.5 * d[None, :, :]
            z = self.zeta(r_e, r_h, temperature_K)
            out[i] = float(np.mean(z >= 1.0))
        return out


# ---------------------------------------------------------------------
# Production configuration
# ---------------------------------------------------------------------


def production_field(amplitude_e_eV: float = 0.045,
                     amplitude_h_eV: Optional[float] = None,
                     origin_h_nm: Tuple[float, float] = (0.0, 0.0),
                     period_nm: float = 20.0) -> PairStabilityField:
    """
    Values from configs/two_body/adaptive_field_config_production_shift0000.

    The v1.1 refactor allows per-carrier amplitudes; pass amplitude_h_eV
    explicitly if the electron and hole landscapes are decoupled in the
    run being analysed.  The defaults reproduce the shared-amplitude case.
    """
    if amplitude_h_eV is None:
        amplitude_h_eV = amplitude_e_eV
    return PairStabilityField(
        interaction=BilayerKeldysh(
            separation_nm=0.60,
            screening_length_layer1_nm=4.479911124019045,
            screening_length_layer2_nm=3.4934510307918494,
            kappa_environment=4.945,
        ),
        landscape_e=MoireLandscape(amplitude_e_eV, period_nm, (0.0, 0.0)),
        landscape_h=MoireLandscape(amplitude_h_eV, period_nm, origin_h_nm),
        mass_e_m0=0.58,
        mass_h_m0=0.36,
    )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


def _self_test() -> None:
    print("=" * 70)
    print("A. Landscape off: reproduces the isolated-pair annulus")
    print("=" * 70)
    bare = production_field(amplitude_e_eV=0.0)
    print(f"reduced mass mu = {bare.reduced_mass_m0:.5f} m0")
    rho = np.logspace(-3, np.log10(40.0), 4000)
    rel = np.stack([rho, np.zeros_like(rho)], axis=-1)
    print(f"\n{'T[K]':>6} {'rho_in':>8} {'rho_out':>9} {'zeta_max':>9}")
    for T in (5, 10, 20, 43, 100, 300):
        z = bare.zeta(0.5 * rel, -0.5 * rel, T)
        inv = rho[z >= 1.0]
        lo = inv.min() if inv.size else np.nan
        hi = inv.max() if inv.size else np.nan
        print(f"{T:6.0f} {lo:8.3f} {hi:9.3f} {z.max():9.2f}")
    print("  expected at 20 K: 0.53 - 18.7 nm, zeta_max ~ 27")

    print()
    print("=" * 70)
    print("B. TB-01: mass-weighted density reduces to the isotropic case")
    print("=" * 70)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 2)); H = A + A.T
    g = rng.normal(size=2); m = 0.37; beta = 3.1

    lam, R = np.linalg.eigh(H / m)
    gt = g / np.sqrt(m)
    gm = R.T @ gt
    quad_mw = float(np.sum(spectral_F(lam, beta) * gm ** 2))

    lh, Rh = np.linalg.eigh(H)
    Xi_h = spectral_Xi(lh / m, beta)
    F_h = np.where(np.abs(lh) < 1e-10,
                   -beta ** 2 * HBAR2_OVER_M0 / (12.0 * m),
                   (Xi_h - 1) / lh)
    gh = Rh.T @ g
    quad_sp = float(np.sum(F_h * gh ** 2))

    print(f"  mass-weighted   g~^T F(W) g~             = {quad_mw:.12f}")
    print(f"  isotropic       g^T F_beta(H) g            = {quad_sp:.12f}")
    print(f"  identical: {np.isclose(quad_mw, quad_sp)}")
    print("  => using Xi(W) without conjugating F by M^-1/2 would be wrong.")

    print()
    print("=" * 70)
    print("C. Mass weighting changes the spectrum")
    print("=" * 70)
    f = production_field()
    r_e = np.array([3.0, 1.0]); r_h = np.array([-2.0, 0.5])
    lam_H = np.linalg.eigvalsh(f.hessian_4d(r_e, r_h))
    lam_W = f.mass_weighted_eigenvalues(r_e, r_h)
    print(f"  eig(H)                : {lam_H}")
    print(f"  eig(M^-1/2 H M^-1/2)  : {lam_W}")
    print(f"  eig(H)/mu             : {lam_H / f.reduced_mass_m0}")
    print("  three different spectra: neither H nor a scalar mass suffices.")

    print()
    print("=" * 70)
    print("D. TB-02: lattice duality and the conditional invalid fraction")
    print("=" * 70)
    lat = f.landscape_e.lattice
    rng2 = np.random.default_rng(1)
    rr = rng2.normal(size=(5, 2)) * 10.0
    for i, a in enumerate(lat):
        d = np.max(np.abs(f.landscape_e.value(rr + a)
                          - f.landscape_e.value(rr)))
        print(f"  |V(r+a{i+1})-V(r)| = {d:.2e}   a{i+1} = {a}")

    R = lat[0]
    r_e2 = rng2.normal(size=(6, 2)) * 8.0
    r_h2 = rng2.normal(size=(6, 2)) * 8.0
    both = np.max(np.abs(f.value(r_e2 + R, r_h2 + R) - f.value(r_e2, r_h2)))
    one = np.max(np.abs(f.value(r_e2 + R, r_h2) - f.value(r_e2, r_h2)))
    print(f"\n  diagonal translation (r_e+R, r_h+R) : {both:.2e}  invariant")
    print(f"  electron only        (r_e+R, r_h)   : {one:.2e}  NOT invariant")
    print("  => fundamental domain is A x R^2; no cutoff-free scalar f_inv.")

    print(f"\n  conditional f_inv(rho) at 20 K:")
    rho = np.array([0.3, 0.6, 1.0, 2.0, 5.0, 11.55, 18.0, 25.0])
    fi = f.invalid_fraction_at_separation(rho, 20.0, n_cell=12, n_angle=12)
    print(f"  {'rho[nm]':>9} {'f_inv(rho)':>11}")
    for r_, v_ in zip(rho, fi):
        tag = "  <- median separation L/sqrt(3)" if abs(r_ - 11.55) < 0.1 else ""
        print(f"  {r_:9.2f} {v_:11.4f}{tag}")

    print()
    print("=" * 70)
    print("E. RLH density is finite where the Gaussian exists")
    print("=" * 70)
    for T in (20, 300):
        lp = f.rlh_log_density(r_e, r_h, T)
        z = f.zeta(r_e, r_h, T)
        print(f"  T={T:4.0f} K   zeta={float(z):6.3f}   log P = {float(lp)}")


if __name__ == "__main__":
    _self_test()
