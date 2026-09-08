import csv
import numpy as np


# ============================================================
# LHA-13A3
#
# Exact BZ density and probability-weighted caustic for the
# asymmetric moire-like benchmark.
#
# Questions:
#
#   geometric invalid fraction:
#       f_invalid
#
#   exact quantum probability in invalid region:
#       p_invalid
#
#   depletion/enrichment ratio:
#       R_invalid = p_invalid / f_invalid
#
# No RLH regularization.
# No PI-QMC.
# ============================================================


BETAS = np.array([
    2.65,
    2.70,
    2.711,
    2.7117,
    2.712,
    2.72,
    2.75,
    2.80,
    3.00,
    3.30,
])


NGRID = 128

K2 = 100
NK = 16


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
# Stationary minimum -> energy shift
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
    2*np.pi*s0 + PHASES[0],
    2*np.pi*t0 + PHASES[1],
    -2*np.pi*(s0+t0) + PHASES[2],
])


Umin = -float(
    np.sum(
        AMPLITUDES*np.cos(theta_min)
    )
)


V0 = -Umin


# ============================================================
# Fourier potential
# ============================================================


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
        int(np.ceil(2*np.sqrt(K2max)))
        + 2
    )

    basis = []

    for n1 in range(-R, R+1):

        for n2 in range(-R, R+1):

            k2 = K2_of(n1, n2)

            if k2 <= K2max:

                basis.append(
                    (n1, n2, k2)
                )

    basis.sort(
        key=lambda x: (
            x[2],
            x[0],
            x[1],
        )
    )

    return basis


basis = build_basis(K2)


nvec = np.array([
    [n1, n2]
    for n1, n2, _ in basis
], dtype=int)


Gcart = (
    nvec[:, 0, None]*G1[None, :]
    +
    nvec[:, 1, None]*G2[None, :]
)


index = {
    (n1, n2): i
    for i, (n1, n2, _) in enumerate(basis)
}


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


hermiticity = float(
    np.max(
        np.abs(
            Vmat
            - Vmat.conj().T
        )
    )
)


# ============================================================
# Real-space grid
# ============================================================


axis = (
    np.arange(
        NGRID,
        dtype=float,
    )
    / NGRID
)


S, T = np.meshgrid(
    axis,
    axis,
    indexing="xy",
)


# Plane-wave phases:
#
# exp[2pi i(n1 s+n2 t)]
#
# Shape:
#   (NGRID*NGRID, NBASIS)
# ============================================================


points_s = S.ravel()
points_t = T.ravel()


phase_matrix = np.exp(
    2j*np.pi
    * (
        points_s[:, None]
        * nvec[None, :, 0]
        +
        points_t[:, None]
        * nvec[None, :, 1]
    )
)


# ============================================================
# Hessian / zeta field
# ============================================================


theta = np.stack([
    2*np.pi*S + PHASES[0],
    2*np.pi*T + PHASES[1],
    -2*np.pi*(S+T) + PHASES[2],
], axis=-1)


ct = np.cos(theta)


Hgrid = np.einsum(
    "...a,ai,aj->...ij",
    AMPLITUDES*ct,
    GS,
    GS,
)


eig = np.linalg.eigvalsh(
    Hgrid
)


kappa_min = eig[..., 0]


global_kappa_min = float(
    np.min(kappa_min)
)


beta_c = (
    np.pi
    / np.sqrt(
        -global_kappa_min
    )
)


negative_mask = (
    kappa_min < 0.0
)


negative_area = float(
    np.mean(
        negative_mask
    )
)


# ============================================================
# Solve Bloch states once
# ============================================================


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


solved = []


global_E0 = np.inf


print(
    "=== LHA-13A3 "
    "PROBABILITY-WEIGHTED OFF-STATIONARY CAUSTIC ==="
)

print()

print(
    f"NBASIS           = {len(basis)}"
)

print(
    f"Nk               = {NK} x {NK}"
)

print(
    f"real-space grid  = {NGRID} x {NGRID}"
)

print(
    f"Hermiticity      = {hermiticity:.6e}"
)

print(
    f"global kappa_min = {global_kappa_min:+.12f}"
)

print(
    f"grid beta_c      = {beta_c:.12f}"
)

print(
    f"negative area    = "
    f"{100*negative_area:.6f}%"
)

print()


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


        H = Vmat.copy()


        H[
            np.diag_indices_from(H)
        ] += (
            0.5
            * np.sum(
                KG*KG,
                axis=1,
            )
        )


        E, C = np.linalg.eigh(H)


        global_E0 = min(
            global_E0,
            float(E[0]),
        )


        solved.append(
            (
                E,
                C,
            )
        )


# ============================================================
# Density
# ============================================================


def exact_density(beta):

    rho = np.zeros(
        (
            len(basis),
            len(basis),
        ),
        dtype=complex,
    )


    Z = 0.0


    for E, C in solved:

        w = np.exp(
            -beta*(E-global_E0)
        )


        Z += np.sum(w)


        rho += (
            (C*w)
            @ C.conj().T
        )


    rho /= Z


    # diagonal real-space density:
    #
    # P(r) = phi(r)^† rho phi(r)
    #
    # evaluated in chunks to control memory.

    out = np.empty(
        NGRID*NGRID,
        dtype=float,
    )


    CHUNK = 1024


    for start in range(
        0,
        len(out),
        CHUNK,
    ):

        stop = min(
            start+CHUNK,
            len(out),
        )


        ph = phase_matrix[
            start:stop
        ]


        tmp = (
            ph @ rho
        )


        out[
            start:stop
        ] = np.real(
            np.sum(
                tmp
                * np.conj(ph),
                axis=1,
            )
        )


    P = out.reshape(
        NGRID,
        NGRID,
    )


    P /= np.mean(P)


    return P


# ============================================================
# Main scan
# ============================================================


rows = []


print(
    " beta      zeta_max    "
    "invalid area     p_invalid"
    "       R=p/f       p_negative"
)

print(
    "------------------------------------------------"
    "----------------------"
)


for beta in BETAS:

    zeta = (
        beta
        / np.pi
        * np.sqrt(
            np.maximum(
                0.0,
                -kappa_min
            )
        )
    )


    invalid = (
        zeta >= 1.0
    )


    f_invalid = float(
        np.mean(invalid)
    )


    P = exact_density(beta)


    p_invalid = float(
        np.mean(
            P*invalid
        )
    )


    p_negative = float(
        np.mean(
            P*negative_mask
        )
    )


    if f_invalid > 0.0:

        R = (
            p_invalid
            / f_invalid
        )

    else:

        R = np.nan


    zeta_max = float(
        np.max(zeta)
    )


    rows.append([
        beta,
        zeta_max,

        f_invalid,
        p_invalid,
        R,

        negative_area,
        p_negative,

        float(
            np.min(P)
        ),

        float(
            np.max(P)
        ),

        abs(
            float(
                np.mean(P)
            )
            - 1.0
        ),
    ])


    print(
        f"{beta:6.4f}   "
        f"{zeta_max:10.6f}   "
        f"{100*f_invalid:11.6f}%   "
        f"{100*p_invalid:11.7f}%   "
        f"{R:10.6f}   "
        f"{100*p_negative:11.6f}%"
    )


# ============================================================
# Structural audit
# ============================================================


A = np.asarray(
    rows,
    dtype=float,
)


norm_max = float(
    np.max(
        A[:, 9]
    )
)


pre = A[
    A[:, 0] < beta_c
]


post = A[
    A[:, 0] > beta_c
]


checks = {

    "HAMILTONIAN HERMITICITY":
        hermiticity < 1e-12,

    "EXACT DENSITY NORMALIZATION":
        norm_max < 1e-12,

    "PRECAUSTIC INVALID AREA ZERO":
        bool(
            np.all(
                pre[:, 2] == 0.0
            )
        ),

    "POSTCAUSTIC INVALID AREA POSITIVE":
        bool(
            np.all(
                post[:, 2] > 0.0
            )
        ),

    "EXACT DENSITY POSITIVE":
        bool(
            np.all(
                A[:, 7] > 0.0
            )
        ),
}


print()

print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    f"max normalization error = "
    f"{norm_max:.6e}"
)

print()


for name, passed in checks.items():

    print(
        f"{name:36s}: "
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
    "lha13a3_probability_weighted_offstationary_caustic.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "zeta_max",

        "invalid_area_fraction",
        "exact_invalid_probability",
        "invalid_probability_over_area",

        "negative_curvature_area_fraction",
        "exact_negative_curvature_probability",

        "density_min",
        "density_max",
        "density_norm_error",
    ])

    w.writerows(rows)


print()

print(
    "saved: "
    "lha13a3_probability_weighted_offstationary_caustic.csv"
)
