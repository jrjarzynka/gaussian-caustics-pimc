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


# ============================================================
# RLH quadrature convergence
# ============================================================


P_rlh_512, V_rlh_512, zeta_512 = (
    rlh_density(
        N_CONTROL
    )
)


P_rlh_1024, V_rlh_1024, zeta_1024 = (
    rlh_density(
        N_PROD
    )
)


rlh_grid_L1 = float(
    np.mean(
        np.abs(
            P_rlh_1024
            - P_rlh_512
        )
    )
)


rlh_V_grid_delta = abs(
    V_rlh_1024
    - V_rlh_512
)


P_RLH = (
    P_rlh_1024
)


V_RLH = (
    V_rlh_1024
)


ZETA = (
    zeta_1024
)


# ============================================================
# Existing harmonized exact + PIMC densities
# ============================================================


b3a = np.load(
    "lha13b3a_exact_finiteP_registered_density.npz"
)


b3b = np.load(
    "lha13b3b_multiseed_density.npz"
)


P_exact_P64 = np.asarray(
    b3a[
        "P_finiteP_bin_ts"
    ],
    dtype=float,
)


P_exact_cont = np.asarray(
    b3a[
        "P_continuum_bin_ts"
    ],
    dtype=float,
)


P_pimc = np.asarray(
    b3b[
        "P_pimc_bin_ts"
    ],
    dtype=float,
)


P_time_1 = np.asarray(
    b3b[
        "P_time_first_bin_ts"
    ],
    dtype=float,
)


P_time_2 = np.asarray(
    b3b[
        "P_time_second_bin_ts"
    ],
    dtype=float,
)


P_seed_1 = np.asarray(
    b3b[
        "P_seed_first_bin_ts"
    ],
    dtype=float,
)


P_seed_2 = np.asarray(
    b3b[
        "P_seed_second_bin_ts"
    ],
    dtype=float,
)


# ============================================================
# L1 metrics
# ============================================================


def L1(
    A,
    B,
):

    return float(
        np.mean(
            np.abs(
                A-B
            )
        )
    )


L1_RLH_P64 = L1(
    P_RLH,
    P_exact_P64,
)


L1_RLH_CONT = L1(
    P_RLH,
    P_exact_cont,
)


L1_PIMC_P64 = L1(
    P_pimc,
    P_exact_P64,
)


L1_PIMC_CONT = L1(
    P_pimc,
    P_exact_cont,
)


L1_P64_CONT = L1(
    P_exact_P64,
    P_exact_cont,
)


L1_time = L1(
    P_time_1,
    P_time_2,
)


L1_seed = L1(
    P_seed_1,
    P_seed_2,
)


RLH_over_PIMC = (
    L1_RLH_P64
    / L1_PIMC_P64
)


RLH_over_time = (
    L1_RLH_P64
    / L1_time
)


RLH_over_seed = (
    L1_RLH_P64
    / L1_seed
)


PIMC_over_time = (
    L1_PIMC_P64
    / L1_time
)


PIMC_over_seed = (
    L1_PIMC_P64
    / L1_seed
)


# ============================================================
# Cross-check RLH energy against A2
# ============================================================


with open(
    "lha13a2_tensor_rlh_offstationary_caustic.csv",
    newline="",
) as f:

    a2rows = list(
        csv.DictReader(f)
    )


a2 = min(
    a2rows,
    key=lambda r:
        abs(
            float(r["beta"])
            - BETA
        ),
)


V_RLH_A2 = float(
    a2["V_RLH"]
)


V_RLH_A2_delta = abs(
    V_RLH
    - V_RLH_A2
)


V_exact = float(
    a2[
        "V_exact_BZ"
    ]
)


V_error_pct = (
    100
    * (
        V_RLH-V_exact
    )
    / V_exact
)


# ============================================================
# Report
# ============================================================


print(
    "=== LHA-13B3C "
    "HARMONIZED RLH / EXACT / PI-QMC ==="
)

print()


print(
    f"beta                    = "
    f"{BETA}"
)

print(
    f"zeta_max                = "
    f"{ZETA:.9f}"
)

print()


print(
    "=== RLH QUADRATURE CONVERGENCE ==="
)

print(
    f"L1 512 -> 1024          = "
    f"{rlh_grid_L1:.9e}"
)

print(
    f"|dV| 512 -> 1024        = "
    f"{rlh_V_grid_delta:.9e}"
)

print()


print(
    "=== ENERGY ==="
)

print(
    f"V exact continuum       = "
    f"{V_exact:.12f}"
)

print(
    f"V RLH                   = "
    f"{V_RLH:.12f}"
)

print(
    f"RLH V error             = "
    f"{V_error_pct:+.6f}%"
)

print(
    f"RLH energy delta vs A2  = "
    f"{V_RLH_A2_delta:.9e}"
)

print()


print(
    "=== HARMONIZED 16x16 DENSITY ==="
)

print(
    f"L1 exact P64-continuum  = "
    f"{L1_P64_CONT:.9f}"
)

print()

print(
    f"L1 PIMC vs exact P64    = "
    f"{L1_PIMC_P64:.9f}"
)

print(
    f"L1 RLH  vs exact P64    = "
    f"{L1_RLH_P64:.9f}"
)

print()

print(
    f"L1 PIMC vs continuum    = "
    f"{L1_PIMC_CONT:.9f}"
)

print(
    f"L1 RLH  vs continuum    = "
    f"{L1_RLH_CONT:.9f}"
)

print()


print(
    "=== INTERNAL PI-QMC SAMPLING SCALE ==="
)

print(
    f"time-half L1            = "
    f"{L1_time:.9f}"
)

print(
    f"seed-half L1            = "
    f"{L1_seed:.9f}"
)

print()


print(
    "=== RATIOS ==="
)

print(
    f"RLH / PIMC exact-L1     = "
    f"{RLH_over_PIMC:.6f}"
)

print(
    f"RLH / time-half scale   = "
    f"{RLH_over_time:.6f}"
)

print(
    f"RLH / seed-half scale   = "
    f"{RLH_over_seed:.6f}"
)

print()

print(
    f"PIMC / time-half scale  = "
    f"{PIMC_over_time:.6f}"
)

print(
    f"PIMC / seed-half scale  = "
    f"{PIMC_over_seed:.6f}"
)


# ============================================================
# Structural audit
# ============================================================


checks = {

    "SUBCAUSTIC RLH":
        ZETA < 1.0,

    "RLH BIN CONVERGENCE":
        rlh_grid_L1
        < 1e-4,

    "RLH ENERGY CONVERGENCE":
        rlh_V_grid_delta
        < 1e-8,

    "RLH REPRODUCES A2 ENERGY":
        V_RLH_A2_delta
        < 1e-8,

    "RLH DENSITY NORMALIZED":
        abs(
            np.mean(P_RLH)
            - 1.0
        )
        < 1e-12,

    "PIMC DENSITY NORMALIZED":
        abs(
            np.mean(P_pimc)
            - 1.0
        )
        < 1e-12,

    "PIMC WITHIN TIME-HALF SCALE":
        L1_PIMC_P64
        < L1_time,

    "PIMC WITHIN SEED-HALF SCALE":
        L1_PIMC_P64
        < L1_seed,
}


print()

print(
    "=== STRUCTURAL AUDIT ==="
)


for name, passed in checks.items():

    print(
        f"{name:34s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


overall = all(
    checks.values()
)


print()

print(
    "STRUCTURAL OVERALL:",
    "PASS"
    if overall
    else "FAIL"
)


# ============================================================
# Save
# ============================================================


with open(
    "lha13b3c_harmonized_density_summary.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)


    w.writerow([
        "beta",
        "zeta_max",

        "V_exact",
        "V_RLH",
        "V_RLH_error_pct",

        "RLH_grid_L1_512_to_1024",
        "RLH_V_grid_delta",
        "RLH_A2_energy_delta",

        "exact_P64_vs_continuum_L1",

        "PIMC_vs_exact_P64_L1",
        "RLH_vs_exact_P64_L1",

        "PIMC_vs_continuum_L1",
        "RLH_vs_continuum_L1",

        "time_half_L1",
        "seed_half_L1",

        "RLH_over_PIMC_L1",
        "RLH_over_time_half",
        "RLH_over_seed_half",

        "PIMC_over_time_half",
        "PIMC_over_seed_half",

        "structural_overall",
    ])


    w.writerow([
        BETA,
        ZETA,

        V_exact,
        V_RLH,
        V_error_pct,

        rlh_grid_L1,
        rlh_V_grid_delta,
        V_RLH_A2_delta,

        L1_P64_CONT,

        L1_PIMC_P64,
        L1_RLH_P64,

        L1_PIMC_CONT,
        L1_RLH_CONT,

        L1_time,
        L1_seed,

        RLH_over_PIMC,
        RLH_over_time,
        RLH_over_seed,

        PIMC_over_time,
        PIMC_over_seed,

        "PASS"
        if overall
        else "FAIL",
    ])


np.savez_compressed(

    "lha13b3c_harmonized_density.npz",

    P_RLH_bin_ts=
        P_RLH,

    P_PIMC_bin_ts=
        P_pimc,

    P_exact_P64_bin_ts=
        P_exact_P64,

    P_exact_continuum_bin_ts=
        P_exact_cont,
)


print()

print(
    "saved: "
    "lha13b3c_harmonized_density_summary.csv"
)

print(
    "saved: "
    "lha13b3c_harmonized_density.npz"
)
