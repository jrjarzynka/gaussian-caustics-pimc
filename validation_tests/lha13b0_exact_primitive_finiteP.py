import csv
import numpy as np
from scipy.linalg import expm


# ============================================================
# LHA-13B0
#
# Exact primitive finite-P transfer reference for the
# asymmetric periodic benchmark.
#
# B_tau(k) =
#   exp(-tau V / 2)
#   exp(-tau T_k)
#   exp(-tau V / 2)
#
# Finite-P partition operator:
#
#   B_tau(k)^P
#
# This gives the exact target sampled by the primitive
# ring-polymer discretization at the same P.
#
# NO PI-QMC.
# NO RLH.
# ============================================================


BETAS = np.array([
    0.50,
    1.00,
    1.90,
    2.50,
    2.65,
    2.70,
    2.72,
    2.80,
    3.00,
])


P_VALUES = [
    16,
    32,
    64,
    128,
]


K2_CONTROL = 81
K2_PROD = 100

NK_CONTROL = 12
NK_PROD = 16


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


# ============================================================
# Energy-zero shift from A0
# ============================================================


with open(
    "lha13a0_stationary_points.csv",
    newline="",
) as f:

    stationary = list(
        csv.DictReader(f)
    )


minimum = [
    r for r in stationary
    if r["type"] == "minimum"
][0]


s0 = float(minimum["s"])
t0 = float(minimum["t"])


theta_min = np.array([
    2.0*np.pi*s0 + PHASES[0],
    2.0*np.pi*t0 + PHASES[1],
    -2.0*np.pi*(s0+t0) + PHASES[2],
])


Umin = -float(
    np.sum(
        AMPLITUDES*np.cos(theta_min)
    )
)


V0 = -Umin


V_FOURIER = {

    (0, 0):
        V0,

    (1, 0):
        -0.5
        * AMPLITUDES[0]
        * np.exp(1j*PHASES[0]),

    (-1, 0):
        -0.5
        * AMPLITUDES[0]
        * np.exp(-1j*PHASES[0]),

    (0, 1):
        -0.5
        * AMPLITUDES[1]
        * np.exp(1j*PHASES[1]),

    (0, -1):
        -0.5
        * AMPLITUDES[1]
        * np.exp(-1j*PHASES[1]),

    (-1, -1):
        -0.5
        * AMPLITUDES[2]
        * np.exp(1j*PHASES[2]),

    (1, 1):
        -0.5
        * AMPLITUDES[2]
        * np.exp(-1j*PHASES[2]),
}


# ============================================================
# Basis
# ============================================================


def K2_of(n1, n2):

    return (
        n1*n1
        + n2*n2
        - n1*n2
    )


def build_basis(K2max):

    R = (
        int(
            np.ceil(
                2*np.sqrt(K2max)
            )
        )
        + 2
    )


    basis = []


    for n1 in range(-R, R+1):

        for n2 in range(-R, R+1):

            k2 = K2_of(
                n1,
                n2,
            )


            if k2 <= K2max:

                basis.append(
                    (
                        n1,
                        n2,
                        k2,
                    )
                )


    basis.sort(
        key=lambda x: (
            x[2],
            x[0],
            x[1],
        )
    )


    return basis


def prepare(K2max):

    basis = build_basis(
        K2max
    )


    nvec = np.array([
        [n1, n2]
        for n1, n2, _ in basis
    ], dtype=int)


    index = {
        (n1, n2): i
        for i, (n1, n2, _) in enumerate(basis)
    }


    Gcart = (
        nvec[:, 0, None]*G1[None, :]
        +
        nvec[:, 1, None]*G2[None, :]
    )


    Vmat = np.zeros(
        (
            len(basis),
            len(basis),
        ),
        dtype=complex,
    )


    for i, (n1, n2, _) in enumerate(basis):

        for (dn1, dn2), coeff in V_FOURIER.items():

            j = index.get(
                (
                    n1-dn1,
                    n2-dn2,
                )
            )


            if j is not None:

                Vmat[i, j] = coeff


    return (
        basis,
        Gcart,
        Vmat,
    )


# ============================================================
# Exact continuum BZ targets from A1
# ============================================================


with open(
    "lha13a1_exact_bz_reference.csv",
    newline="",
) as f:

    continuum_rows = list(
        csv.DictReader(f)
    )


continuum = {
    float(r["beta"]):
        float(r["V_BZ"])

    for r in continuum_rows
}


# ============================================================
# Stable trace of B^P and V B^P
#
# B is positive Hermitian:
#
#   B = D Vhalf D
#
# after diagonalizing V once.
#
# We diagonalize B directly.  If
#
#   B = U diag(lambda) U^\dagger,
#
# then
#
#   Tr(B^P) = sum lambda^P
#
# and
#
#   Tr(V B^P)
#   = sum_i lambda_i^P <i|V|i>.
# ============================================================


def finiteP_reference(
    beta,
    P,
    K2max,
    NK,
):

    basis, Gcart, Vmat = prepare(
        K2max
    )


    # V exponential is k-independent.

    vE, vU = np.linalg.eigh(
        Vmat
    )


    tau = beta/P


    expVhalf = (
        (
            vU
            * np.exp(
                -0.5*tau*vE
            )[None, :]
        )
        @ vU.conj().T
    )


    kvals = (
        (
            np.arange(
                NK,
                dtype=float,
            )
            + 0.5
        )
        / NK
        - 0.5
    )


    # First pass: find the largest log eigenvalue among
    # all k points for safe scaling.

    solved = []

    max_log_lambda = -np.inf


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


            kinetic = (
                0.5
                * np.sum(
                    KG*KG,
                    axis=1,
                )
            )


            expT = np.exp(
                -tau*kinetic
            )


            Btau = (
                expVhalf
                @ (
                    expT[:, None]
                    * expVhalf
                )
            )


            Btau = (
                0.5
                * (
                    Btau
                    + Btau.conj().T
                )
            )


            lam, U = np.linalg.eigh(
                Btau
            )


            # Tiny negative roundoff is not physical.
            if np.min(lam) < -1.0e-12:

                raise RuntimeError(
                    "Transfer matrix has significantly "
                    "negative eigenvalue."
                )


            lam = np.maximum(
                lam,
                np.finfo(float).tiny,
            )


            loglam = np.log(
                lam
            )


            max_log_lambda = max(
                max_log_lambda,
                float(
                    np.max(loglam)
                ),
            )


            VU = (
                Vmat @ U
            )


            Vdiag = np.real(
                np.sum(
                    np.conj(U)*VU,
                    axis=0,
                )
            )


            solved.append(
                (
                    loglam,
                    Vdiag,
                )
            )


    Zscaled = 0.0
    Vscaled = 0.0


    for loglam, Vdiag in solved:

        w = np.exp(
            P
            * (
                loglam
                - max_log_lambda
            )
        )


        Zscaled += np.sum(
            w
        )


        Vscaled += np.sum(
            w*Vdiag
        )


    Vmean = (
        Vscaled/Zscaled
    )


    return (
        float(Vmean),
        len(basis),
    )


# ============================================================
# Production ladder
# ============================================================


print(
    "=== LHA-13B0 "
    "EXACT PRIMITIVE FINITE-P REFERENCE ==="
)

print()


rows = []


for beta in BETAS:

    print(
        f"beta = {beta:.3f}"
    )


    for P in P_VALUES:

        Vp, nbasis = finiteP_reference(
            beta,
            P,
            K2_PROD,
            NK_PROD,
        )


        Vinf = continuum[
            beta
        ]


        bias_pct = (
            100.0
            * (
                Vp-Vinf
            )
            / Vinf
        )


        rows.append([
            beta,
            P,
            Vinf,
            Vp,
            bias_pct,
        ])


        print(
            f"  P={P:3d}  "
            f"V_inf={Vinf:.10f}  "
            f"V_P={Vp:.10f}  "
            f"bias={bias_pct:+.6f}%"
        )


    print()


# ============================================================
# P=64 basis and k-grid convergence
# ============================================================


P_AUDIT = 64


audit_rows = []


max_basis_delta = 0.0
max_kgrid_delta = 0.0


for beta in BETAS:

    V_control_basis, n81 = finiteP_reference(
        beta,
        P_AUDIT,
        K2_CONTROL,
        NK_CONTROL,
    )


    V_prod_basis_same_k, n100 = finiteP_reference(
        beta,
        P_AUDIT,
        K2_PROD,
        NK_CONTROL,
    )


    V_prod, _ = finiteP_reference(
        beta,
        P_AUDIT,
        K2_PROD,
        NK_PROD,
    )


    basis_delta = abs(
        V_prod_basis_same_k
        - V_control_basis
    )


    kgrid_delta = abs(
        V_prod
        - V_prod_basis_same_k
    )


    max_basis_delta = max(
        max_basis_delta,
        basis_delta,
    )


    max_kgrid_delta = max(
        max_kgrid_delta,
        kgrid_delta,
    )


    audit_rows.append([
        beta,
        basis_delta,
        kgrid_delta,
    ])


print(
    "=== P=64 NUMERICAL CONVERGENCE ==="
)

print(
    f"K2 {K2_CONTROL}->{K2_PROD}: "
    f"max delta = "
    f"{max_basis_delta:.6e}"
)

print(
    f"Nk {NK_CONTROL}->{NK_PROD}: "
    f"max delta = "
    f"{max_kgrid_delta:.6e}"
)

print()


# ============================================================
# Primitive scaling at beta=2.70
# ============================================================


BETA_SCALING = 2.70


sub = [
    r for r in rows
    if abs(
        r[0]-BETA_SCALING
    ) < 1e-12
]


P_arr = np.array([
    r[1]
    for r in sub
], dtype=float)


bias_abs = np.abs(
    np.array([
        r[3]-r[2]
        for r in sub
    ])
)


mask = (
    bias_abs > 0.0
)


coef = np.polyfit(
    np.log(
        P_arr[mask]
    ),
    np.log(
        bias_abs[mask]
    ),
    1,
)


scaling_exponent = float(
    -coef[0]
)


print(
    "=== PRIMITIVE SCALING AT beta=2.70 ==="
)

print(
    f"|V_P - V_inf| ~ P^(-z)"
)

print(
    f"z = {scaling_exponent:.8f}"
)

print()


# ============================================================
# Structural audit
# ============================================================


checks = {

    "P64 BASIS CONVERGENCE":
        max_basis_delta < 1.0e-6,

    "P64 k-GRID CONVERGENCE":
        max_kgrid_delta < 1.0e-8,

    "PRIMITIVE SECOND-ORDER SCALING":
        abs(
            scaling_exponent-2.0
        ) < 0.05,

    "FINITE-P VALUES POSITIVE":
        all(
            r[3] > 0.0
            for r in rows
        ),
}


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
    "lha13b0_exact_primitive_finiteP.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "P",
        "V_continuum_BZ",
        "V_exact_finiteP",
        "finiteP_bias_pct",
    ])

    w.writerows(
        rows
    )


with open(
    "lha13b0_exact_primitive_finiteP_convergence.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "basis_delta_K2_81_to_100",
        "kgrid_delta_Nk_12_to_16",
    ])

    w.writerows(
        audit_rows
    )


print()

print(
    "saved: "
    "lha13b0_exact_primitive_finiteP.csv"
)

print(
    "saved: "
    "lha13b0_exact_primitive_finiteP_convergence.csv"
)
