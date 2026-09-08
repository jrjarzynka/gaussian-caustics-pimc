import csv
import numpy as np


# ============================================================
# LHA-12B0
#
# Exact finite-P reference for the SAME primitive-action
# ring polymer sampled by PI-QMC.
#
# Model:
#
# H* = -1/2 nabla^2 + V
#
# V = 3
#     - cos(2pi s)
#     - cos(2pi t)
#     - cos(2pi(s+t))
#
# beta* = 1.9
#
# Primitive transfer step:
#
# B_tau
# =
# exp(-tau V/2)
# exp(-tau T)
# exp(-tau V/2)
#
# The finite-P partition function is Tr(B_tau^P).
#
# We compute exact primitive finite-P <V> with BZ averaging.
# ============================================================


BETA = 1.9

P_LIST = [
    8,
    16,
    32,
    64,
    128,
    256,
]


sqrt3 = np.sqrt(3.0)


G1 = np.array([
    1.0,
    0.0,
])

G2 = np.array([
    -0.5,
    sqrt3/2.0,
])


# ============================================================
# Numerical controls
# ============================================================


NFFT = 256


# Production
K2_PROD = 49
NK_PROD = 12


# One-point convergence audit at P=64
K2_CONTROL = 36
NK_CONTROL = 8


# ============================================================
# Basis
# ============================================================


def K2_of(
    n1,
    n2,
):

    return (
        n1*n1
        + n2*n2
        - n1*n2
    )


def build_basis(
    K2_max,
):

    R = (
        int(
            np.ceil(
                2.0*np.sqrt(
                    K2_max
                )
            )
        )
        + 2
    )


    basis = []


    for n1 in range(
        -R,
        R+1,
    ):

        for n2 in range(
            -R,
            R+1,
        ):

            K2 = K2_of(
                n1,
                n2,
            )


            if K2 <= K2_max:

                basis.append(
                    (
                        n1,
                        n2,
                        float(K2),
                    )
                )


    basis.sort(
        key=lambda z: (
            z[2],
            z[0],
            z[1],
        )
    )


    return basis


# ============================================================
# Potential matrix
# ============================================================


def build_V_matrix(
    basis,
):

    nvec = np.array([
        [
            n1,
            n2,
        ]

        for (
            n1,
            n2,
            _
        ) in basis
    ], dtype=int)


    d = (
        nvec[:, None, :]
        -
        nvec[None, :, :]
    )


    d1 = d[..., 0]
    d2 = d[..., 1]


    V = np.zeros(
        (
            len(basis),
            len(basis),
        ),
        dtype=float,
    )


    V[
        (d1 == 0)
        &
        (d2 == 0)
    ] = 3.0


    for a, b in [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
    ]:

        V[
            (d1 == a)
            &
            (d2 == b)
        ] = -0.5


    return V


# ============================================================
# Fourier multiplication matrix
#
# Matrix representation of f(s,t):
#
# <n|f|m> = f_(n-m)
# ============================================================


def multiplication_matrix(
    basis,
    fgrid,
):

    N = fgrid.shape[0]


    F = (
        np.fft.fft2(
            fgrid
        )
        / (
            N*N
        )
    )


    nvec = np.array([
        [
            n1,
            n2,
        ]

        for (
            n1,
            n2,
            _
        ) in basis
    ], dtype=int)


    d = (
        nvec[:, None, :]
        -
        nvec[None, :, :]
    )


    M = F[
        d[..., 1] % N,
        d[..., 0] % N,
    ]


    # numerical Hermitization
    M = (
        M
        + M.conj().T
    ) / 2.0


    return M


# ============================================================
# Exact finite-P <V>
# ============================================================


def finiteP_V(
    beta,
    P,
    K2_max,
    nk,
):

    tau = (
        beta
        / P
    )


    basis = build_basis(
        K2_max
    )


    nbasis = len(
        basis
    )


    nvec = np.array([
        [
            n1,
            n2,
        ]

        for (
            n1,
            n2,
            _
        ) in basis
    ], dtype=float)


    Gcart = (
        nvec[:, 0, None]*G1[None, :]
        +
        nvec[:, 1, None]*G2[None, :]
    )


    Vmat = build_V_matrix(
        basis
    )


    # --------------------------------------------------------
    # exp(-tau V/2) in plane-wave basis
    # --------------------------------------------------------


    axis = (
        np.arange(
            NFFT,
            dtype=float,
        )
        / NFFT
    )


    S, T = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )


    Vgrid = (
        3.0
        - np.cos(
            2.0*np.pi*S
        )
        - np.cos(
            2.0*np.pi*T
        )
        - np.cos(
            2.0*np.pi*(S+T)
        )
    )


    W = multiplication_matrix(
        basis,
        np.exp(
            -0.5
            * tau
            * Vgrid
        ),
    )


    # --------------------------------------------------------
    # Midpoint BZ grid
    # --------------------------------------------------------


    kvals = (
        (
            np.arange(
                nk,
                dtype=float,
            )
            + 0.5
        )
        / nk
        - 0.5
    )


    all_logw = []
    all_Vstate = []


    for ku in kvals:

        for kv in kvals:

            kcart = (
                ku*G1
                + kv*G2
            )


            KG = (
                Gcart
                + kcart[None, :]
            )


            Tdiag = (
                0.5
                * np.sum(
                    KG*KG,
                    axis=1,
                )
            )


            D = np.exp(
                -tau
                * Tdiag
            )


            # B = W D W

            B = (
                (
                    W
                    * D[None, :]
                )
                @ W
            )


            B = (
                B
                + B.conj().T
            ) / 2.0


            q, C = np.linalg.eigh(
                B
            )


            q = np.clip(
                q.real,
                1.0e-300,
                None,
            )


            logw = (
                P
                * np.log(
                    q
                )
            )


            # <eigenstate|V|eigenstate>

            VC = (
                Vmat
                @ C
            )


            Vstate = np.real(
                np.sum(
                    C.conj()
                    * VC,
                    axis=0,
                )
            )


            all_logw.append(
                logw
            )


            all_Vstate.append(
                Vstate
            )


    logw = np.concatenate(
        all_logw
    )


    Vstate = np.concatenate(
        all_Vstate
    )


    shift = float(
        np.max(
            logw
        )
    )


    w = np.exp(
        logw-shift
    )


    Vmean = float(
        np.sum(
            w*Vstate
        )
        /
        np.sum(
            w
        )
    )


    return (
        Vmean,
        nbasis,
    )



# ============================================================
# LHA-12C0
#
# Exact primitive finite-P=64 targets at all four beta anchors.
#
# Continuum exact reference:
# LHA-12A1.
#
# For each beta:
#   - K2=36,Nk=8 control
#   - K2=49,Nk=8 basis control
#   - K2=49,Nk=12 production
#
# ============================================================

BETAS_C = [2.48]

P_PROD = 64


exact = np.load(
    "point7_exact_bz_continuum_13anchors.npz"
)

betas_exact = np.asarray(
    exact["betas"],
    dtype=float,
)

densities_exact = np.asarray(
    exact["densities_BZ"],
    dtype=float,
)

Vgrid_exact = np.asarray(
    exact["V"],
    dtype=float,
)


rows = []


print(
    "=== LHA-12C0a "
    "EXACT P=64 MULTIBETA TARGETS ==="
)

print()

print(
    " beta      V_continuum       V_P64"
    "          bias(%)       basis-delta"
    "     kgrid-delta"
)

print(
    "------------------------------------------------"
    "--------------------------------"
)


all_basis_pass = True
all_kgrid_pass = True


for beta in BETAS_C:

    ibeta = int(
        np.argmin(
            np.abs(
                betas_exact-beta
            )
        )
    )

    if abs(
        betas_exact[ibeta]-beta
    ) > 1.0e-12:
        raise RuntimeError(
            f"beta={beta} absent "
            "from exact continuum reference"
        )


    V_cont = float(
        np.mean(
            densities_exact[ibeta]
            * Vgrid_exact
        )
    )


    V64_8, _ = finiteP_V(
        beta,
        P_PROD,
        64,
        8,
    )


    V81_8, _ = finiteP_V(
        beta,
        P_PROD,
        81,
        8,
    )


    V81_12, _ = finiteP_V(
        beta,
        P_PROD,
        81,
        12,
    )


    basis_delta = abs(
        V81_8
        - V64_8
    )


    kgrid_delta = abs(
        V81_12
        - V81_8
    )


    bias_pct = (
        100.0
        * (
            V81_12-V_cont
        )
        / V_cont
    )


    basis_pass = (
        basis_delta < 1.0e-6
    )


    kgrid_pass = (
        kgrid_delta < 1.0e-6
    )


    all_basis_pass &= basis_pass
    all_kgrid_pass &= kgrid_pass


    rows.append([
        beta,
        P_PROD,
        V_cont,
        V81_12,
        bias_pct,
        basis_delta,
        kgrid_delta,
    ])


    print(
        f"{beta:5.2f}   "
        f"{V_cont:14.10f}   "
        f"{V81_12:14.10f}   "
        f"{bias_pct:+10.6f}   "
        f"{basis_delta:12.4e}   "
        f"{kgrid_delta:12.4e}"
    )


print()

print(
    "BASIS CONVERGENCE ALL BETAS:",
    "PASS"
    if all_basis_pass
    else "FAIL"
)

print(
    "k-GRID CONVERGENCE ALL BETAS:",
    "PASS"
    if all_kgrid_pass
    else "FAIL"
)


overall = (
    all_basis_pass
    and all_kgrid_pass
)


print()

print(
    "STRUCTURAL OVERALL:",
    "PASS"
    if overall
    else "FAIL"
)


with open(
    "p7_P64_beta_2p48.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "beta",
        "P",
        "V_continuum",
        "V_finiteP64",
        "finiteP_bias_pct",
        "basis_delta_K64_to_K81",
        "kgrid_delta_N8_to_N12_at_K81",
    ])

    writer.writerows(rows)


np.savez_compressed(
    "p7_P64_beta_2p48.npz",

    beta=np.asarray(
        [r[0] for r in rows],
        dtype=float,
    ),

    P=P_PROD,

    V_continuum=np.asarray(
        [r[2] for r in rows],
        dtype=float,
    ),

    V_finiteP64=np.asarray(
        [r[3] for r in rows],
        dtype=float,
    ),

    finiteP_bias_pct=np.asarray(
        [r[4] for r in rows],
        dtype=float,
    ),
)


print()

print(
    "saved: "
    "p7_P64_beta_2p48.csv"
)

print(
    "saved: "
    "p7_P64_beta_2p48.npz"
)
