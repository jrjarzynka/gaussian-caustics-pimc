import csv
import numpy as np
from scipy.optimize import root


# ============================================================
# LHA-13A0
#
# Geometry / symmetry / Hessian / caustic audit for an
# asymmetric first-star moire-like periodic landscape.
#
# No exact diagonalization.
# No PI-QMC.
#
# Dimensionless units:
#   M = hbar = 1
#   |G_j| = 1
#
# Raw potential:
#
# U(r) = - sum_j A_j cos(G_j.r + phi_j)
#
# Final potential:
#
# V(r) = U(r) - U_min
#
# so that V_min = 0.
# ============================================================


NGRID = 512


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


BETA_SCAN = np.array([
    0.50,
    1.00,
    1.90,
    2.50,
    2.70,
    2.72,
    2.80,
    3.00,
])


sqrt3 = np.sqrt(3.0)


# ============================================================
# Reciprocal lattice
# ============================================================


G1 = np.array([
    1.0,
    0.0,
])


G2 = np.array([
    -0.5,
    sqrt3/2.0,
])


G3 = -(
    G1
    + G2
)


GS = np.array([
    G1,
    G2,
    G3,
])


B = np.column_stack([
    G1,
    G2,
])


# B^T A = 2 pi I

A_DIRECT = (
    2.0*np.pi
    * np.linalg.inv(
        B.T
    )
)


duality_error = float(
    np.max(
        np.abs(
            B.T @ A_DIRECT
            - 2.0*np.pi*np.eye(2)
        )
    )
)


# ============================================================
# Gauge-invariant phase combination
#
# Since G1 + G2 + G3 = 0,
#
# Phi = phi1 + phi2 + phi3
#
# cannot be changed by a translation of the origin.
# ============================================================


PHI_INVARIANT = float(
    np.mod(
        np.sum(PHASES),
        2.0*np.pi,
    )
)


distance_to_trivial_phase = float(
    min(
        abs(PHI_INVARIANT),
        abs(PHI_INVARIANT-np.pi),
        abs(PHI_INVARIANT-2.0*np.pi),
    )
)


# ============================================================
# Potential and derivatives in Cartesian coordinates
# ============================================================


def raw_cart(r):

    r = np.asarray(
        r,
        dtype=float,
    )


    theta = (
        GS @ r
        + PHASES
    )


    c = np.cos(theta)
    s = np.sin(theta)


    U = -float(
        np.sum(
            AMPLITUDES*c
        )
    )


    grad = np.sum(
        (
            AMPLITUDES*s
        )[:, None]
        * GS,
        axis=0,
    )


    H = np.einsum(
        "a,ai,aj->ij",
        AMPLITUDES*c,
        GS,
        GS,
    )


    return (
        U,
        grad,
        H,
    )


def raw_uv(uv):

    uv = np.asarray(
        uv,
        dtype=float,
    )


    r = (
        A_DIRECT
        @ uv
    )


    return raw_cart(
        r
    )


# ============================================================
# Analytic derivative audit
# ============================================================


rng = np.random.default_rng(
    130001
)


EPS = 1.0e-5


max_grad_fd_error = 0.0
max_hess_fd_error = 0.0


for _ in range(30):

    uv = rng.random(2)

    r = (
        A_DIRECT
        @ uv
    )


    U, grad, H = raw_cart(
        r
    )


    grad_fd = np.zeros(2)

    H_fd = np.zeros(
        (
            2,
            2,
        )
    )


    for a in range(2):

        e = np.zeros(2)
        e[a] = EPS


        Up = raw_cart(
            r+e
        )[0]

        Um = raw_cart(
            r-e
        )[0]


        grad_fd[a] = (
            Up-Um
        ) / (
            2.0*EPS
        )


        gp = raw_cart(
            r+e
        )[1]

        gm = raw_cart(
            r-e
        )[1]


        H_fd[:, a] = (
            gp-gm
        ) / (
            2.0*EPS
        )


    max_grad_fd_error = max(
        max_grad_fd_error,
        float(
            np.max(
                np.abs(
                    grad-grad_fd
                )
            )
        ),
    )


    max_hess_fd_error = max(
        max_hess_fd_error,
        float(
            np.max(
                np.abs(
                    H-H_fd
                )
            )
        ),
    )


# ============================================================
# Stationary-point search on the torus
# ============================================================


def torus_distance(
    u,
    v,
):

    d = (
        u-v+0.5
    ) % 1.0 - 0.5


    return float(
        np.linalg.norm(
            d
        )
    )


stationary = []


SEED_N = 18


for s0 in np.linspace(
    0.0,
    1.0,
    SEED_N,
    endpoint=False,
):

    for t0 in np.linspace(
        0.0,
        1.0,
        SEED_N,
        endpoint=False,
    ):

        x0 = np.array([
            s0,
            t0,
        ])


        sol = root(

            fun=lambda uv:
                raw_uv(uv)[1],

            x0=x0,

            jac=lambda uv:
                raw_uv(uv)[2]
                @ A_DIRECT,

            tol=1.0e-12,
        )


        if not sol.success:
            continue


        if np.linalg.norm(
            raw_uv(
                sol.x
            )[1]
        ) > 1.0e-10:

            continue


        uv = np.mod(
            sol.x,
            1.0,
        )


        if any(
            torus_distance(
                uv,
                old,
            ) < 1.0e-7

            for old in stationary
        ):

            continue


        stationary.append(
            uv
        )


stationary.sort(
    key=lambda uv:
        raw_uv(uv)[0]
)


if not stationary:
    raise RuntimeError(
        "No stationary points found."
    )


Umin = min(
    raw_uv(uv)[0]
    for uv in stationary
)


# ============================================================
# Classify stationary points
# ============================================================


stationary_rows = []


n_min = 0
n_saddle = 0
n_max = 0


for i, uv in enumerate(
    stationary
):

    U, grad, H = raw_uv(
        uv
    )


    eig = np.linalg.eigvalsh(
        H
    )


    V = (
        U-Umin
    )


    if np.all(
        eig > 1.0e-8
    ):

        kind = "minimum"
        n_min += 1

    elif np.all(
        eig < -1.0e-8
    ):

        kind = "maximum"
        n_max += 1

    elif (
        eig[0] < -1.0e-8
        and eig[1] > 1.0e-8
    ):

        kind = "saddle"
        n_saddle += 1

    else:

        kind = "degenerate"


    if eig[0] < 0.0:

        beta_c_local = (
            np.pi
            / np.sqrt(
                -eig[0]
            )
        )

    else:

        beta_c_local = np.inf


    stationary_rows.append([
        i,
        kind,
        uv[0],
        uv[1],
        V,
        eig[0],
        eig[1],
        float(
            np.linalg.norm(
                grad
            )
        ),
        beta_c_local,
    ])


morse_index = (
    n_min
    - n_saddle
    + n_max
)


# ============================================================
# Dense-cell geometry
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


theta = np.stack([

    2.0*np.pi*S
    + PHASES[0],

    2.0*np.pi*T
    + PHASES[1],

    -2.0*np.pi*(
        S+T
    )
    + PHASES[2],

], axis=-1)


cos_theta = np.cos(
    theta
)


U_grid = -np.sum(
    AMPLITUDES
    * cos_theta,
    axis=-1,
)


V_grid = (
    U_grid
    - Umin
)


H_grid = np.einsum(
    "...a,ai,aj->...ij",
    AMPLITUDES*cos_theta,
    GS,
    GS,
)


eigvals, eigvecs = np.linalg.eigh(
    H_grid
)


kappa_min = eigvals[..., 0]
kappa_max = eigvals[..., 1]


V_grid_min = float(
    np.min(
        V_grid
    )
)


V_grid_max = float(
    np.max(
        V_grid
    )
)


stationary_Vmax = max(
    row[4]
    for row in stationary_rows
)


global_kappa_min = float(
    np.min(
        kappa_min
    )
)


global_kappa_max = float(
    np.max(
        kappa_max
    )
)


negative_curvature_fraction = float(
    np.mean(
        kappa_min < 0.0
    )
)


max_abs_Hxy = float(
    np.max(
        np.abs(
            H_grid[..., 0, 1]
        )
    )
)


# ============================================================
# Rotating Hessian eigenframe
#
# Eigenvector orientation is an unoriented axis:
#
# theta and theta + pi are equivalent.
#
# Therefore use exp(2 i theta).
#
# R2 = 1 -> almost fixed principal axis
# R2 << 1 -> principal direction explores many orientations.
# ============================================================


v_soft = eigvecs[..., :, 0]


orientation = np.arctan2(
    v_soft[..., 1],
    v_soft[..., 0],
)


eig_separation = (
    eigvals[..., 1]
    - eigvals[..., 0]
)


orientation_mask = (
    eig_separation > 1.0e-6
)


orientation_R2 = float(
    np.abs(
        np.mean(
            np.exp(
                2.0j
                * orientation[
                    orientation_mask
                ]
            )
        )
    )
)


# ============================================================
# Symmetry breaking diagnostics
# ============================================================


s_idx = np.broadcast_to(
    np.arange(
        NGRID,
        dtype=int,
    )[None, :],
    (
        NGRID,
        NGRID,
    ),
)


t_idx = np.broadcast_to(
    np.arange(
        NGRID,
        dtype=int,
    )[:, None],
    (
        NGRID,
        NGRID,
    ),
)


TRANSFORMS = {

    "swap": (
        t_idx,
        s_idx,
    ),

    "inversion": (
        -s_idx,
        -t_idx,
    ),

    "rotation60": (
        -t_idx,
        s_idx+t_idx,
    ),
}


symmetry_rows = []


for name, (
    sn,
    tn,
) in TRANSFORMS.items():

    transformed = V_grid[
        np.mod(
            tn,
            NGRID,
        ),
        np.mod(
            sn,
            NGRID,
        ),
    ]


    diff = np.abs(
        V_grid
        - transformed
    )


    symmetry_rows.append([
        name,
        float(
            np.mean(
                diff
            )
        ),
        float(
            np.max(
                diff
            )
        ),
    ])


symmetry_L1 = {
    row[0]: row[1]
    for row in symmetry_rows
}


# ============================================================
# Global caustic structure
# ============================================================


if global_kappa_min >= 0.0:

    beta_c_global = np.inf

else:

    beta_c_global = (
        np.pi
        / np.sqrt(
            -global_kappa_min
        )
    )


caustic_rows = []


for beta in BETA_SCAN:

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


    zeta_max = float(
        np.max(
            zeta
        )
    )


    invalid_fraction = float(
        np.mean(
            zeta >= 1.0
        )
    )


    caustic_rows.append([
        beta,
        zeta_max,
        invalid_fraction,
    ])


zeta_by_beta = {
    row[0]: row[1]
    for row in caustic_rows
}


# ============================================================
# Saddle inequivalence
# ============================================================


saddle_rows = [
    row
    for row in stationary_rows
    if row[1] == "saddle"
]


if len(
    saddle_rows
) >= 2:

    saddle_energy_spread = float(
        max(
            row[4]
            for row in saddle_rows
        )
        -
        min(
            row[4]
            for row in saddle_rows
        )
    )

else:

    saddle_energy_spread = 0.0


minimum_rows = [
    row
    for row in stationary_rows
    if row[1] == "minimum"
]


if minimum_rows:

    min_curvature_anisotropy = float(
        abs(
            minimum_rows[0][6]
            - minimum_rows[0][5]
        )
    )

else:

    min_curvature_anisotropy = 0.0


# ============================================================
# Report
# ============================================================


print(
    "=== LHA-13A0 "
    "ASYMMETRIC MOIRE-LIKE GEOMETRY ==="
)

print()


print(
    "amplitudes =",
    AMPLITUDES,
)

print(
    "phases     =",
    PHASES,
)

print(
    f"gauge-invariant phase sum = "
    f"{PHI_INVARIANT:.12f}"
)

print(
    f"distance from trivial 0/pi = "
    f"{distance_to_trivial_phase:.12f}"
)

print()


print(
    "=== LATTICE / DERIVATIVE AUDIT ==="
)

print(
    f"max lattice duality error = "
    f"{duality_error:.6e}"
)

print(
    f"max gradient FD error     = "
    f"{max_grad_fd_error:.6e}"
)

print(
    f"max Hessian FD error      = "
    f"{max_hess_fd_error:.6e}"
)

print()


print(
    "=== STATIONARY POINTS ==="
)

print(
    " id   type       s           t"
    "           V          kappa1"
    "       kappa2       beta_c(local)"
)

print(
    "------------------------------------------------"
    "------------------------------------------"
)


for row in stationary_rows:

    (
        i,
        kind,
        s,
        t,
        V,
        k1,
        k2,
        gnorm,
        bc,
    ) = row


    bc_text = (
        f"{bc:.8f}"
        if np.isfinite(bc)
        else "inf"
    )


    print(
        f"{i:3d}  "
        f"{kind:8s}  "
        f"{s:10.7f}  "
        f"{t:10.7f}  "
        f"{V:10.7f}  "
        f"{k1:+10.7f}  "
        f"{k2:+10.7f}  "
        f"{bc_text}"
    )


print()

print(
    f"counts min/saddle/max = "
    f"{n_min}/{n_saddle}/{n_max}"
)

print(
    f"Morse index "
    f"Nmin-Nsaddle+Nmax = "
    f"{morse_index}"
)

print(
    f"saddle energy spread = "
    f"{saddle_energy_spread:.8f}"
)

print(
    f"minimum curvature anisotropy = "
    f"{min_curvature_anisotropy:.8f}"
)

print()


print(
    "=== DENSE-CELL GEOMETRY ==="
)

print(
    f"grid                 = "
    f"{NGRID} x {NGRID}"
)

print(
    f"V grid min/max       = "
    f"{V_grid_min:.10f} / "
    f"{V_grid_max:.10f}"
)

print(
    f"stationary Vmax      = "
    f"{stationary_Vmax:.10f}"
)

print(
    f"global kappa_min     = "
    f"{global_kappa_min:+.10f}"
)

print(
    f"global kappa_max     = "
    f"{global_kappa_max:+.10f}"
)

print(
    f"negative-curvature area = "
    f"{100.0*negative_curvature_fraction:.4f}%"
)

print(
    f"max |Hxy|            = "
    f"{max_abs_Hxy:.10f}"
)

print(
    f"principal-axis R2    = "
    f"{orientation_R2:.10f}"
)

print()


print(
    "=== BROKEN-SYMMETRY AUDIT ==="
)

for row in symmetry_rows:

    print(
        f"{row[0]:10s}: "
        f"L1={row[1]:.8f}, "
        f"Linf={row[2]:.8f}"
    )


print()


print(
    "=== CAUSTIC AUDIT ==="
)

print(
    f"global beta_c = "
    f"{beta_c_global:.10f}"
)

print()

print(
    " beta      zeta_max      "
    "invalid area"
)

print(
    "--------------------------------"
)


for beta, zmax, finv in caustic_rows:

    print(
        f"{beta:5.2f}   "
        f"{zmax:11.7f}   "
        f"{100.0*finv:10.6f}%"
    )


# ============================================================
# Structural checks
# ============================================================


checks = {

    "LATTICE DUALITY":
        duality_error
        < 1.0e-12,

    "ANALYTIC GRADIENT":
        max_grad_fd_error
        < 1.0e-8,

    "ANALYTIC HESSIAN":
        max_hess_fd_error
        < 1.0e-8,

    "FOUR STATIONARY POINTS":
        len(stationary_rows)
        == 4,

    "MORSE COUNTS 1/2/1":
        (
            n_min == 1
            and n_saddle == 2
            and n_max == 1
        ),

    "TORUS MORSE INDEX":
        morse_index
        == 0,

    "INEQUIVALENT SADDLES":
        saddle_energy_spread
        > 1.0e-2,

    "ANISOTROPIC MINIMUM":
        min_curvature_anisotropy
        > 1.0e-1,

    "NONZERO MIXED HESSIAN":
        max_abs_Hxy
        > 1.0e-1,

    "ROTATING HESSIAN FRAME":
        orientation_R2
        < 0.5,

    "NONTRIVIAL PHASE":
        distance_to_trivial_phase
        > 0.1,

    "BROKEN INVERSION":
        symmetry_L1[
            "inversion"
        ]
        > 0.05,

    "BROKEN SWAP":
        symmetry_L1[
            "swap"
        ]
        > 0.05,

    "GRID MINIMUM RESOLVED":
        V_grid_min
        < 1.0e-4,

    "GRID MAXIMUM RESOLVED":
        abs(
            V_grid_max
            - stationary_Vmax
        )
        < 1.0e-4,

    "BETA 2.5 SUBCAUSTIC":
        zeta_by_beta[
            2.50
        ]
        < 1.0,

    "BETA 2.8 POSTCAUSTIC":
        zeta_by_beta[
            2.80
        ]
        > 1.0,
}


print()

print(
    "=== STRUCTURAL AUDIT ==="
)


for name, passed in checks.items():

    print(
        f"{name:28s}: "
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
    "lha13a0_stationary_points.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "id",
        "type",
        "s",
        "t",
        "V",
        "kappa_min",
        "kappa_max",
        "gradient_norm",
        "beta_c_local",
    ])

    w.writerows(
        stationary_rows
    )


with open(
    "lha13a0_caustic_scan.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "zeta_max",
        "invalid_area_fraction",
    ])

    w.writerows(
        caustic_rows
    )


with open(
    "lha13a0_symmetry_audit.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "transform",
        "potential_L1_difference",
        "potential_Linf_difference",
    ])

    w.writerows(
        symmetry_rows
    )


np.savez_compressed(

    "lha13a0_asymmetric_moire_geometry.npz",

    amplitudes=AMPLITUDES,
    phases=PHASES,

    G1=G1,
    G2=G2,
    G3=G3,

    A_direct=A_DIRECT,

    axis=axis,

    V_grid=V_grid,

    kappa_min=kappa_min,
    kappa_max=kappa_max,

    Hxy=H_grid[..., 0, 1],

    beta_c_global=beta_c_global,

    phase_invariant=PHI_INVARIANT,

    orientation_R2=orientation_R2,
)


print()

print(
    "saved: lha13a0_stationary_points.csv"
)

print(
    "saved: lha13a0_caustic_scan.csv"
)

print(
    "saved: lha13a0_symmetry_audit.csv"
)

print(
    "saved: lha13a0_asymmetric_moire_geometry.npz"
)
