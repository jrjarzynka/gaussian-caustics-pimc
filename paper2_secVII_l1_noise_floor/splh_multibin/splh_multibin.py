import csv
import numpy as np


# ============================================================
# LHA-13B3C
#
# Harmonized 16x16 comparison:
#
#   tensorial RLH
#   exact continuum QM
#   exact primitive P=64
#   pooled PI-QMC physical-bead density
#
# beta = 2.70
#
# No new Monte Carlo.
# ============================================================


BETA = 2.70

NBIN = 16


N_CONTROL = 512
N_PROD = 1024


AMPLITUDES = np.array([
    1.00,
    0.78,
    0.62,
])


PHASES = np.array([
    0.00,
    0.23,
    0.71,
])


sqrt3 = np.sqrt(3.0)


G1 = np.array([
    1.0,
    0.0,
])


G2 = np.array([
    -0.5,
    sqrt3/2.0,
])


G3 = -(G1+G2)


GS = np.array([
    G1,
    G2,
    G3,
])


# ============================================================
# Energy zero from A0
# ============================================================


with open(
    "lha13a0_stationary_points.csv",
    newline="",
) as f:

    stationary = list(
        csv.DictReader(f)
    )


minimum = [
    r
    for r in stationary
    if r["type"] == "minimum"
][0]


s0 = float(minimum["s"])
t0 = float(minimum["t"])


theta_min = np.array([
    2*np.pi*s0 + PHASES[0],
    2*np.pi*t0 + PHASES[1],
    -2*np.pi*(s0+t0) + PHASES[2],
])


Umin = -float(
    np.sum(
        AMPLITUDES
        * np.cos(theta_min)
    )
)


# ============================================================
# Tensorial spectral functions
# ============================================================


def spectral_Xi_F(
    kappa,
    beta,
):

    kappa = np.asarray(
        kappa,
        dtype=float,
    )


    Xi = np.empty_like(
        kappa
    )


    F = np.empty_like(
        kappa
    )


    eps = 1e-10


    pos = (
        kappa > eps
    )


    neg = (
        kappa < -eps
    )


    zer = ~(
        pos | neg
    )


    if np.any(pos):

        xi = (
            0.5
            * beta
            * np.sqrt(
                kappa[pos]
            )
        )


        Xi[pos] = (
            np.tanh(xi)
            / xi
        )


        F[pos] = (
            Xi[pos]-1.0
        ) / kappa[pos]


    if np.any(neg):

        eta = (
            0.5
            * beta
            * np.sqrt(
                -kappa[neg]
            )
        )


        if np.any(
            eta >= np.pi/2
        ):

            raise RuntimeError(
                "RLH crossed the local Gaussian caustic."
            )


        Xi[neg] = (
            np.tan(eta)
            / eta
        )


        F[neg] = (
            Xi[neg]-1.0
        ) / kappa[neg]


    if np.any(zer):

        k = kappa[zer]


        Xi[zer] = (
            1.0
            - beta**2*k/12.0
            + beta**4*k*k/120.0
        )


        F[zer] = (
            -beta**2/12.0
            + beta**4*k/120.0
        )


    return (
        Xi,
        F,
    )


# ============================================================
# RLH on a dense endpoint-centered grid.
#
# Array convention:
#
# axis 0 = t
# axis 1 = s
#
# matching exact and PIMC bin arrays.
# ============================================================


def rlh_density(
    N,
):

    if (
        N % NBIN
        != 0
    ):

        raise RuntimeError(
            "Dense grid must divide exactly into histogram bins."
        )


    # Midpoint grid avoids placing quadrature points exactly
    # on coarse-bin boundaries.

    axis = (
        (
            np.arange(
                N,
                dtype=float,
            )
            + 0.5
        )
        / N
    )


    S, T = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )


    theta = np.stack([

        2*np.pi*S
        + PHASES[0],

        2*np.pi*T
        + PHASES[1],

        -2*np.pi*(S+T)
        + PHASES[2],

    ], axis=-1)


    c = np.cos(theta)
    sn = np.sin(theta)


    U = -np.sum(
        AMPLITUDES*c,
        axis=-1,
    )


    V = (
        U-Umin
    )


    grad = np.einsum(
        "...a,ai->...i",
        AMPLITUDES*sn,
        GS,
    )


    H = np.einsum(
        "...a,ai,aj->...ij",
        AMPLITUDES*c,
        GS,
        GS,
    )


    kappa, R = np.linalg.eigh(
        H
    )


    zeta = (
        BETA
        / np.pi
        * np.sqrt(
            np.maximum(
                0.0,
                -kappa
            )
        )
    )


    zeta_max = float(
        np.max(zeta)
    )


    if zeta_max >= 1.0:

        raise RuntimeError(
            f"Unexpected super-caustic grid point: "
            f"zeta={zeta_max}"
        )


    Xi, F = spectral_Xi_F(
        kappa,
        BETA,
    )


    # R[..., i, a] = Cartesian component i
    # of eigenvector a.
    gmodal = np.einsum(
        "...ia,...i->...a",
        R,
        grad,
    )


    Phi = (
        V
        + 0.5
        * np.sum(
            F*gmodal*gmodal,
            axis=-1,
        )
    )


    logu = (
        0.5
        * np.sum(
            np.log(Xi),
            axis=-1,
        )
        - BETA*Phi
    )


    logu -= np.max(
        logu
    )


    P = np.exp(
        logu
    )


    P /= np.mean(
        P
    )


    Vmean = float(
        np.mean(
            P*V
        )
    )


    # Exact coarse-bin average.
    #
    # Shape before reduction:
    #
    #   (t-bin, fine-t,
    #    s-bin, fine-s)
    #
    block = (
        N//NBIN
    )


    Pbin = (
        P.reshape(
            NBIN,
            block,
            NBIN,
            block,
        )
        .mean(
            axis=(
                1,
                3,
            )
        )
    )


    return (
        Pbin,
        Vmean,
        zeta_max,
    )

for nb in (8,16,32):
    NBIN=nb
    P512,V512,z512=rlh_density(N_CONTROL)
    P1024,V1024,z1024=rlh_density(N_PROD)
    grid_l1=float(np.mean(np.abs(P1024-P512)))
    np.savez_compressed(f'splh_multibin_{nb}.npz',P_SPLH_bin_ts=P1024,nbin=nb,V_SPLH=V1024,zeta_max=z1024,grid_L1=grid_l1,V_grid_delta=abs(V1024-V512))
    print(nb,'V',V1024,'zeta',z1024,'gridL1',grid_l1,'norm',float(np.mean(P1024)),flush=True)
