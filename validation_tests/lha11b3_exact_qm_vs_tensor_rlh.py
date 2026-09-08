import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LHA-11B3
#
# First direct comparison:
#
# exact periodic quantum density
#             vs
# tensorial multidimensional RLH
#
# Triangular periodic potential:
#
# V(r) =
# 3
# - cos(G1.r)
# - cos(G2.r)
# - cos(G3.r)
#
# Units:
#   hbar = 1
#   M    = 1
#   V0   = 1
#
# Only globally sub-caustic temperatures are used here.
# ============================================================


INPUT = (
    "lha11b1_exact_periodic_reference_K2_81.npz"
)


data = np.load(
    INPUT
)


betas = data[
    "betas"
]

exact_densities = data[
    "densities"
]

axis = data[
    "axis"
]

S = data[
    "S"
]

T = data[
    "T"
]

V_saved = data[
    "V_real"
]


N = len(
    axis
)


sqrt3 = np.sqrt(3.0)


# ============================================================
# Reciprocal vectors
# ============================================================

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
# Analytic periodic geometry
# ============================================================

phase = np.stack([
    2.0*np.pi*S,
    2.0*np.pi*T,
    -2.0*np.pi*(S+T),
], axis=0)


cos_phase = np.cos(
    phase
)

sin_phase = np.sin(
    phase
)


V = (
    3.0
    - np.sum(
        cos_phase,
        axis=0,
    )
)


potential_reconstruction_error = float(
    np.max(
        np.abs(
            V-V_saved
        )
    )
)


# ------------------------------------------------------------
# Gradient in physical Cartesian coordinates
# ------------------------------------------------------------

gx = sum(
    sin_phase[i]
    * GS[i, 0]
    for i in range(3)
)

gy = sum(
    sin_phase[i]
    * GS[i, 1]
    for i in range(3)
)


gradient = np.stack([
    gx,
    gy,
], axis=-1)


# ------------------------------------------------------------
# Hessian in Cartesian coordinates
# ------------------------------------------------------------

Hxx = sum(
    cos_phase[i]
    * GS[i, 0]**2
    for i in range(3)
)

Hyy = sum(
    cos_phase[i]
    * GS[i, 1]**2
    for i in range(3)
)

Hxy = sum(
    cos_phase[i]
    * GS[i, 0]
    * GS[i, 1]
    for i in range(3)
)


H = np.empty(
    (
        N,
        N,
        2,
        2,
    )
)

H[..., 0, 0] = Hxx
H[..., 1, 1] = Hyy
H[..., 0, 1] = Hxy
H[..., 1, 0] = Hxy


kappa, R = np.linalg.eigh(
    H
)


# ============================================================
# Spectral functions
# ============================================================

def spectral_Xi_F(
    kappa,
    beta,
):

    Xi = np.empty_like(
        kappa
    )

    F = np.empty_like(
        kappa
    )

    valid = np.ones(
        kappa.shape,
        dtype=bool,
    )


    # --------------------------------------------------------
    # Near zero curvature
    # --------------------------------------------------------

    small = (
        np.abs(kappa)
        < 1.0e-10
    )

    Xi[small] = (
        1.0
        - beta**2
        * kappa[small]
        / 12.0
        + beta**4
        * kappa[small]**2
        / 120.0
    )

    F[small] = (
        -beta**2/12.0
        + beta**4
        * kappa[small]
        / 120.0
    )


    # --------------------------------------------------------
    # Positive curvature
    # --------------------------------------------------------

    pos = (
        kappa > 1.0e-10
    )

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


    # --------------------------------------------------------
    # Negative curvature:
    #
    # Xi = tan(eta)/eta
    #
    # valid only for eta < pi/2.
    # --------------------------------------------------------

    neg = (
        kappa < -1.0e-10
    )

    neg_flat = np.flatnonzero(
        neg
    )

    eta = (
        0.5
        * beta
        * np.sqrt(
            -kappa[neg]
        )
    )

    stable = (
        eta < np.pi/2.0
    )

    good = neg_flat[
        stable
    ]

    bad = neg_flat[
        ~stable
    ]


    Xi_flat = Xi.ravel()
    F_flat = F.ravel()
    valid_flat = valid.ravel()
    kappa_flat = kappa.ravel()


    eta_good = eta[
        stable
    ]

    Xi_good = (
        np.tan(
            eta_good
        )
        / eta_good
    )


    Xi_flat[
        good
    ] = Xi_good

    F_flat[
        good
    ] = (
        Xi_good-1.0
    ) / kappa_flat[
        good
    ]


    Xi_flat[
        bad
    ] = np.nan

    F_flat[
        bad
    ] = np.nan

    valid_flat[
        bad
    ] = False


    return (
        Xi,
        F,
        valid,
    )


# ============================================================
# RLH
# ============================================================

def tensor_rlh(
    beta,
):

    Xi, F, valid = (
        spectral_Xi_F(
            kappa,
            beta,
        )
    )


    # --------------------------------------------------------
    # Transform gradient into local Hessian eigenbasis:
    #
    # g_mode = R^T g
    # --------------------------------------------------------

    g_mode = np.einsum(
        "...ji,...j->...i",
        R,
        gradient,
    )


    contraction = np.sum(
        F
        * g_mode*g_mode,
        axis=-1,
    )


    Phi = (
        V
        + 0.5*contraction
    )


    detXi = np.prod(
        Xi,
        axis=-1,
    )


    log_density = (
        0.5*np.log(
            detXi
        )
        - beta*Phi
    )


    if not np.all(
        valid
    ):

        return (
            None,
            valid,
            Xi,
            Phi,
        )


    log_density -= np.max(
        log_density
    )


    density = np.exp(
        log_density
    )


    # Reduced-coordinate density normalization:
    #
    # integral_0^1 ds dt P(s,t) = 1
    #
    # On the uniform periodic grid:
    #
    # mean(P) = 1.
    #

    density /= np.mean(
        density
    )


    return (
        density,
        valid,
        Xi,
        Phi,
    )


# ============================================================
# Symmetry machinery
# ============================================================

s_idx = np.broadcast_to(
    np.arange(
        N,
        dtype=int,
    )[None, :],
    (
        N,
        N,
    ),
)

t_idx = np.broadcast_to(
    np.arange(
        N,
        dtype=int,
    )[:, None],
    (
        N,
        N,
    ),
)


transformations = {
    "swap_s_t": (
        t_idx,
        s_idx,
    ),

    "inversion": (
        -s_idx,
        -t_idx,
    ),

    "rotation_60": (
        -t_idx,
        s_idx+t_idx,
    ),
}


def transformed(
    field,
    s_new,
    t_new,
):

    return field[
        np.mod(
            t_new,
            N,
        ),
        np.mod(
            s_new,
            N,
        ),
    ]


def max_symmetry_L1(
    density,
):

    errors = []

    for (
        s_new,
        t_new
    ) in transformations.values():

        test = transformed(
            density,
            s_new,
            t_new,
        )

        errors.append(
            np.mean(
                np.abs(
                    density-test
                )
            )
        )

    return float(
        max(errors)
    )


# ============================================================
# Metrics
# ============================================================

def L1(
    p,
    q,
):

    return float(
        np.mean(
            np.abs(
                p-q
            )
        )
    )


def expectation(
    observable,
    density,
):

    return float(
        np.mean(
            observable
            * density
        )
    )


half = N//2


# Three exact saddle grid points.

saddle_indices = [
    (0, half),
    (half, 0),
    (half, half),
]


# ============================================================
# Main comparison
# ============================================================

rows = []

rlh_stack = []

difference_stack = []


print(
    "=== LHA-11B3 "
    "EXACT PERIODIC QM VS "
    "TENSORIAL RLH ==="
)

print()

print(
    f"input = {INPUT}"
)

print(
    f"N     = {N}"
)

print(
    "potential reconstruction error = "
    f"{potential_reconstruction_error:.3e}"
)

print()


global_norm_error = 0.0
global_negative_violation = 0.0
global_symmetry_error = 0.0

all_valid = True


for ibeta, beta in enumerate(
    betas
):

    exact = exact_densities[
        ibeta
    ]


    (
        rlh,
        valid,
        Xi,
        Phi,
    ) = tensor_rlh(
        beta
    )


    zeta = (
        beta/np.pi
        * np.sqrt(
            np.maximum(
                0.0,
                -kappa[..., 0],
            )
        )
    )


    zeta_max = float(
        np.max(
            zeta
        )
    )


    valid_fraction = float(
        np.mean(
            valid
        )
    )


    print(
        "========================================"
    )

    print(
        f"beta = {beta:.6f}"
    )

    print(
        "========================================"
    )

    print(
        f"zeta_max       = "
        f"{zeta_max:.9f}"
    )

    print(
        f"valid fraction = "
        f"{100*valid_fraction:.6f}%"
    )


    if rlh is None:

        print(
            "RLH STATUS: INVALID"
        )

        all_valid = False

        continue


    rlh_norm = float(
        np.mean(
            rlh
        )
    )

    norm_error = abs(
        rlh_norm-1.0
    )

    min_rlh = float(
        np.min(
            rlh
        )
    )

    negative_violation = max(
        0.0,
        -min_rlh,
    )

    symmetry_error = (
        max_symmetry_L1(
            rlh
        )
    )


    global_norm_error = max(
        global_norm_error,
        norm_error,
    )

    global_negative_violation = max(
        global_negative_violation,
        negative_violation,
    )

    global_symmetry_error = max(
        global_symmetry_error,
        symmetry_error,
    )


    # --------------------------------------------------------
    # Density metrics
    # --------------------------------------------------------

    l1 = L1(
        rlh,
        exact,
    )

    total_variation = (
        0.5*l1
    )


    V_exact = expectation(
        V,
        exact,
    )

    V_rlh = expectation(
        V,
        rlh,
    )


    V_error_pct = (
        100.0
        * (
            V_rlh-V_exact
        )
        / V_exact
    )


    # --------------------------------------------------------
    # Density at potential minimum
    # --------------------------------------------------------

    Pmin_exact = float(
        exact[
            0,
            0,
        ]
    )

    Pmin_rlh = float(
        rlh[
            0,
            0,
        ]
    )

    Pmin_ratio = (
        Pmin_rlh
        / Pmin_exact
    )


    # --------------------------------------------------------
    # Density at equivalent saddles
    # --------------------------------------------------------

    Psaddle_exact = float(
        np.mean([
            exact[idx]
            for idx
            in saddle_indices
        ])
    )

    Psaddle_rlh = float(
        np.mean([
            rlh[idx]
            for idx
            in saddle_indices
        ])
    )

    Psaddle_ratio = (
        Psaddle_rlh
        / Psaddle_exact
    )


    # --------------------------------------------------------
    # Grid point nearest a maximum
    # --------------------------------------------------------

    maxV_index = np.unravel_index(
        np.argmax(
            V
        ),
        V.shape,
    )

    Pmax_exact = float(
        exact[
            maxV_index
        ]
    )

    Pmax_rlh = float(
        rlh[
            maxV_index
        ]
    )

    Pmax_ratio = (
        Pmax_rlh
        / Pmax_exact
    )


    print(
        f"L1 density      = "
        f"{l1:.9f}"
    )

    print(
        f"total variation = "
        f"{total_variation:.9f}"
    )

    print()

    print(
        f"<V> exact       = "
        f"{V_exact:.10f}"
    )

    print(
        f"<V> RLH         = "
        f"{V_rlh:.10f}"
    )

    print(
        f"<V> error       = "
        f"{V_error_pct:+.6f}%"
    )

    print()

    print(
        f"P(min) ratio    = "
        f"{Pmin_ratio:.8f}"
    )

    print(
        f"P(saddle) ratio = "
        f"{Psaddle_ratio:.8f}"
    )

    print(
        f"P(max) ratio    = "
        f"{Pmax_ratio:.8f}"
    )

    print()

    print(
        f"RLH norm error  = "
        f"{norm_error:.3e}"
    )

    print(
        f"RLH symmetry L1 = "
        f"{symmetry_error:.3e}"
    )

    print()


    rows.append([
        beta,
        zeta_max,
        valid_fraction,
        l1,
        total_variation,
        V_exact,
        V_rlh,
        V_error_pct,
        Pmin_exact,
        Pmin_rlh,
        Pmin_ratio,
        Psaddle_exact,
        Psaddle_rlh,
        Psaddle_ratio,
        Pmax_exact,
        Pmax_rlh,
        Pmax_ratio,
        rlh_norm,
        norm_error,
        min_rlh,
        symmetry_error,
    ])


    rlh_stack.append(
        rlh.copy()
    )

    difference_stack.append(
        (
            rlh-exact
        ).copy()
    )


# ============================================================
# Structural PASS / FAIL
# ============================================================

subcaustic_pass = (
    all(
        row[1] < 1.0
        for row in rows
    )
    and all_valid
)

normalization_pass = (
    global_norm_error
    < 1.0e-12
)

positivity_pass = (
    global_negative_violation
    < 1.0e-14
)

symmetry_pass = (
    global_symmetry_error
    < 1.0e-12
)

potential_pass = (
    potential_reconstruction_error
    < 1.0e-12
)


print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    "max RLH norm error     = "
    f"{global_norm_error:.6e}"
)

print(
    "max negative violation = "
    f"{global_negative_violation:.6e}"
)

print(
    "max RLH symmetry L1    = "
    f"{global_symmetry_error:.6e}"
)

print()


checks = {
    "POTENTIAL RECONSTRUCTION":
        potential_pass,

    "GLOBAL SUB-CAUSTIC DOMAIN":
        subcaustic_pass,

    "RLH NORMALIZATION":
        normalization_pass,

    "RLH POSITIVITY":
        positivity_pass,

    "RLH LATTICE SYMMETRY":
        symmetry_pass,
}


for name, passed in checks.items():

    print(
        f"{name:30s}: "
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

print()

print(
    "NOTE:"
)

print(
    "No arbitrary physical-accuracy "
    "threshold is applied here."
)

print(
    "L1 and observable errors must be "
    "interpreted separately."
)


# ============================================================
# CSV
# ============================================================

with open(
    "lha11b3_exact_qm_vs_tensor_rlh.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "beta",
        "zeta_max",
        "valid_fraction",
        "density_L1",
        "total_variation",
        "V_exact",
        "V_RLH",
        "V_error_pct",
        "Pmin_exact",
        "Pmin_RLH",
        "Pmin_ratio",
        "Psaddle_exact",
        "Psaddle_RLH",
        "Psaddle_ratio",
        "Pmax_exact",
        "Pmax_RLH",
        "Pmax_ratio",
        "RLH_norm",
        "RLH_norm_error",
        "RLH_min_density",
        "RLH_symmetry_L1",
    ])

    writer.writerows(
        rows
    )


# ============================================================
# NPZ
# ============================================================

np.savez_compressed(
    "lha11b3_exact_qm_vs_tensor_rlh.npz",
    betas=betas,
    axis=axis,
    S=S,
    T=T,
    V=V,
    exact_densities=exact_densities,
    rlh_densities=np.stack(
        rlh_stack,
        axis=0,
    ),
    density_differences=np.stack(
        difference_stack,
        axis=0,
    ),
    kappa=kappa,
)


# ============================================================
# Figure — L1 and <V> error vs beta
# ============================================================

rows_array = np.array(
    rows,
    dtype=float,
)


fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

ax.plot(
    rows_array[:, 0],
    rows_array[:, 3],
    marker="o",
)

ax.set_xlabel(
    r"$\beta$"
)

ax.set_ylabel(
    r"$L^1(P_{\rm RLH},P_{\rm QM})$"
)

ax.set_title(
    "Exact periodic quantum density vs tensorial RLH"
)

fig.tight_layout()

fig.savefig(
    "lha11b3_density_L1_vs_beta.png",
    dpi=300,
)

fig.savefig(
    "lha11b3_density_L1_vs_beta.pdf",
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

ax.plot(
    rows_array[:, 0],
    rows_array[:, 7],
    marker="o",
)

ax.axhline(
    0.0,
    linestyle="--",
)

ax.set_xlabel(
    r"$\beta$"
)

ax.set_ylabel(
    r"RLH error in $\langle V\rangle$ (%)"
)

ax.set_title(
    "Observable-level error on the periodic landscape"
)

fig.tight_layout()

fig.savefig(
    "lha11b3_V_error_vs_beta.png",
    dpi=300,
)

fig.savefig(
    "lha11b3_V_error_vs_beta.pdf",
)

plt.close(fig)


print()

print(
    "saved: "
    "lha11b3_exact_qm_vs_tensor_rlh.csv"
)

print(
    "saved: "
    "lha11b3_exact_qm_vs_tensor_rlh.npz"
)

print(
    "saved: "
    "lha11b3_density_L1_vs_beta.png/pdf"
)

print(
    "saved: "
    "lha11b3_V_error_vs_beta.png/pdf"
)
