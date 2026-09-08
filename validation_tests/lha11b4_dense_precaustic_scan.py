import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LHA-11B4
#
# Dense sub-caustic scan:
#
# exact periodic plane-wave quantum density
#                   vs
# tensorial RLH
#
# Goals:
#
# 1. resolve the non-monotonic LHA-11B3 error;
# 2. locate the rapid pre-caustic accuracy-loss regime;
# 3. track errors at minimum, saddle, maximum separately;
# 4. re-audit exact PW convergence (K2=64 vs 81)
#    over the entire new beta scan.
#
# Units:
#   hbar = M = V0 = |G| = 1.
# ============================================================


# ------------------------------------------------------------
# Beta grid
#
# Coarse-to-moderate scan first,
# then dense sampling approaching beta_c from below.
# ------------------------------------------------------------

BETAS = np.unique(
    np.concatenate([
        np.arange(
            0.50,
            1.81,
            0.10,
        ),

        np.arange(
            1.85,
            2.31,
            0.05,
        ),

        np.arange(
            2.32,
            2.51,
            0.02,
        ),

        np.arange(
            2.515,
            2.556,
            0.005,
        ),
    ])
)


K2_REFERENCE = 81
K2_CONTROL = 64

NGRID = 64

V0 = 1.0

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

G3 = -(G1+G2)

GS = np.array([
    G1,
    G2,
    G3,
])


# ============================================================
# Analytic caustic
# ============================================================

KAPPA_SADDLE_NEG = -1.5

BETA_C = (
    np.pi
    / np.sqrt(
        -KAPPA_SADDLE_NEG
    )
)

print(
    "=== LHA-11B4 "
    "DENSE PRE-CAUSTIC SCAN ==="
)

print()

print(
    f"beta_c(saddle) = "
    f"{BETA_C:.12f}"
)

print(
    f"number of beta points = "
    f"{len(BETAS)}"
)

print(
    f"largest beta = "
    f"{np.max(BETAS):.6f}"
)

print()


if np.max(BETAS) >= BETA_C:

    raise RuntimeError(
        "Dense scan must remain strictly sub-caustic."
    )


# ============================================================
# Fourier potential
# ============================================================

V_FOURIER = {
    (0, 0): 3.0*V0,

    (1, 0): -0.5*V0,
    (-1, 0): -0.5*V0,

    (0, 1): -0.5*V0,
    (0, -1): -0.5*V0,

    (1, 1): -0.5*V0,
    (-1, -1): -0.5*V0,
}


# ============================================================
# Plane-wave Hamiltonian
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


def solve_pw(
    K2_max,
):

    basis = build_basis(
        K2_max
    )

    nbasis = len(
        basis
    )

    index = {
        (
            n1,
            n2,
        ): i

        for i, (
            n1,
            n2,
            _
        ) in enumerate(
            basis
        )
    }


    Tmat = np.zeros(
        (
            nbasis,
            nbasis,
        ),
        dtype=float,
    )

    Vmat = np.zeros_like(
        Tmat
    )


    for i, (
        n1,
        n2,
        K2,
    ) in enumerate(
        basis
    ):

        Tmat[i, i] = (
            0.5*K2
        )

        for (
            dn1,
            dn2
        ), coeff in (
            V_FOURIER.items()
        ):

            j = index.get(
                (
                    n1-dn1,
                    n2-dn2,
                )
            )

            if j is not None:

                Vmat[
                    i,
                    j
                ] = coeff


    Hmat = (
        Tmat
        + Vmat
    )


    herm_error = float(
        np.max(
            np.abs(
                Hmat-Hmat.T
            )
        )
    )


    E, C = np.linalg.eigh(
        Hmat
    )


    VC = (
        Vmat @ C
    )

    V_eigen = np.sum(
        C*VC,
        axis=0,
    )


    return {
        "basis": basis,
        "H": Hmat,
        "Vmat": Vmat,
        "E": E,
        "C": C,
        "V_eigen": V_eigen,
        "hermiticity": herm_error,
    }


# ============================================================
# Reduced-coordinate real-space grid
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

coords = np.column_stack([
    S.ravel(),
    T.ravel(),
])


phase_grid = np.stack([
    2.0*np.pi*S,
    2.0*np.pi*T,
    -2.0*np.pi*(S+T),
], axis=0)


cos_grid = np.cos(
    phase_grid
)

sin_grid = np.sin(
    phase_grid
)


V = (
    3.0
    - np.sum(
        cos_grid,
        axis=0,
    )
)


gx = sum(
    sin_grid[i]
    * GS[i, 0]
    for i in range(3)
)

gy = sum(
    sin_grid[i]
    * GS[i, 1]
    for i in range(3)
)


gradient = np.stack([
    gx,
    gy,
], axis=-1)


Hxx = sum(
    cos_grid[i]
    * GS[i, 0]**2
    for i in range(3)
)

Hyy = sum(
    cos_grid[i]
    * GS[i, 1]**2
    for i in range(3)
)

Hxy = sum(
    cos_grid[i]
    * GS[i, 0]
    * GS[i, 1]
    for i in range(3)
)


Hgrid = np.empty(
    (
        NGRID,
        NGRID,
        2,
        2,
    )
)

Hgrid[..., 0, 0] = Hxx
Hgrid[..., 1, 1] = Hyy
Hgrid[..., 0, 1] = Hxy
Hgrid[..., 1, 0] = Hxy


kappa_grid, R_grid = (
    np.linalg.eigh(
        Hgrid
    )
)


# ============================================================
# Build PW wavefunctions once for each cutoff
# ============================================================

def prepare_realspace_pw(
    solution,
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
        ) in solution[
            "basis"
        ]
    ], dtype=float)


    phases = np.exp(
        2.0j*np.pi
        * (
            coords @ nvec.T
        )
    )


    psi = (
        phases
        @ solution[
            "C"
        ]
    )


    abspsi2 = (
        np.abs(
            psi
        )**2
    )


    return (
        nvec,
        abspsi2,
    )


print(
    "Solving K2=64 control..."
)

sol64 = solve_pw(
    K2_CONTROL
)

print(
    "Solving K2=81 reference..."
)

sol81 = solve_pw(
    K2_REFERENCE
)


print()

print(
    f"K2=64 NBASIS = "
    f"{len(sol64['basis'])}"
)

print(
    f"K2=81 NBASIS = "
    f"{len(sol81['basis'])}"
)

print(
    f"K2=64 Hermiticity = "
    f"{sol64['hermiticity']:.3e}"
)

print(
    f"K2=81 Hermiticity = "
    f"{sol81['hermiticity']:.3e}"
)

print()


nvec64, abspsi2_64 = (
    prepare_realspace_pw(
        sol64
    )
)

nvec81, abspsi2_81 = (
    prepare_realspace_pw(
        sol81
    )
)


# ============================================================
# Exact quantum density
# ============================================================

def exact_density(
    solution,
    abspsi2,
    beta,
):

    E = solution[
        "E"
    ]

    w = np.exp(
        -beta
        * (
            E-E[0]
        )
    )

    probs = (
        w
        / np.sum(w)
    )


    density = (
        abspsi2 @ probs
    )


    density /= np.mean(
        density
    )


    density = density.reshape(
        (
            NGRID,
            NGRID,
        )
    )


    V_matrix = float(
        probs
        @ solution[
            "V_eigen"
        ]
    )


    V_density = float(
        np.mean(
            V*density
        )
    )


    return (
        density,
        probs,
        V_matrix,
        V_density,
    )


# ============================================================
# Exact density at arbitrary high-symmetry point
# ============================================================

def exact_point_density(
    solution,
    probs,
    s,
    t,
):

    phase = np.array([
        np.exp(
            2.0j*np.pi
            * (
                n1*s
                + n2*t
            )
        )

        for (
            n1,
            n2,
            _
        ) in solution[
            "basis"
        ]
    ])


    psi_states = (
        phase
        @ solution[
            "C"
        ]
    )


    return float(
        np.sum(
            probs
            * np.abs(
                psi_states
            )**2
        )
    )


# ============================================================
# RLH spectral functions
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


    neg = (
        kappa < -1.0e-10
    )


    eta = (
        0.5
        * beta
        * np.sqrt(
            -kappa[neg]
        )
    )


    neg_flat = np.flatnonzero(
        neg
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
# RLH grid density
# ============================================================

def rlh_density(
    beta,
):

    Xi, F, valid = (
        spectral_Xi_F(
            kappa_grid,
            beta,
        )
    )


    if not np.all(
        valid
    ):

        return None


    g_mode = np.einsum(
        "...ji,...j->...i",
        R_grid,
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


    logp = (
        0.5*np.log(
            detXi
        )
        - beta*Phi
    )


    logp -= np.max(
        logp
    )


    p = np.exp(
        logp
    )


    p /= np.mean(
        p
    )


    return p


# ============================================================
# RLH at arbitrary point
# ============================================================

def geometry_point(
    s,
    t,
):

    phases = np.array([
        2.0*np.pi*s,
        2.0*np.pi*t,
        -2.0*np.pi*(s+t),
    ])


    cosp = np.cos(
        phases
    )

    sinp = np.sin(
        phases
    )


    Vp = (
        3.0
        - np.sum(
            cosp
        )
    )


    gp = np.sum(
        sinp[:, None]
        * GS,
        axis=0,
    )


    Hp = np.zeros(
        (
            2,
            2,
        ),
        dtype=float,
    )


    for i in range(3):

        Hp += (
            cosp[i]
            * np.outer(
                GS[i],
                GS[i],
            )
        )


    kp, Rp = np.linalg.eigh(
        Hp
    )


    return (
        Vp,
        gp,
        kp,
        Rp,
    )


def rlh_log_unnormalized_point(
    beta,
    s,
    t,
):

    Vp, gp, kp, Rp = (
        geometry_point(
            s,
            t,
        )
    )


    Xi, F, valid = (
        spectral_Xi_F(
            kp,
            beta,
        )
    )


    if not np.all(valid):

        return np.nan


    gmode = (
        Rp.T @ gp
    )


    Phi = (
        Vp
        + 0.5
        * np.sum(
            F*gmode*gmode
        )
    )


    return float(
        0.5
        * np.sum(
            np.log(Xi)
        )
        - beta*Phi
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


# True high-symmetry points.

POINTS = {
    "minimum": (
        0.0,
        0.0,
    ),

    "saddle": (
        0.5,
        0.0,
    ),

    "maximum": (
        1.0/3.0,
        1.0/3.0,
    ),
}


# ============================================================
# Main scan
# ============================================================

rows = []


max_cutoff_density_L1 = 0.0
max_cutoff_V_delta = 0.0
max_exact_V_consistency = 0.0

max_rlh_norm_error = 0.0

previous_l1 = None


print(
    "=== DENSE SCAN ==="
)

print()

print(
    " beta    zeta     L1(QM,RLH)   "
    "d<V>(%)    Pmin ratio  "
    "Psad ratio  Pmax ratio"
)

print(
    "------------------------------------------------"
    "--------------------------"
)


for beta in BETAS:

    # --------------------------------------------------------
    # Exact reference, K2=64 and 81
    # --------------------------------------------------------

    (
        exact64,
        probs64,
        V64,
        V64_density,
    ) = exact_density(
        sol64,
        abspsi2_64,
        beta,
    )


    (
        exact81,
        probs81,
        V81,
        V81_density,
    ) = exact_density(
        sol81,
        abspsi2_81,
        beta,
    )


    cutoff_L1 = L1(
        exact81,
        exact64,
    )


    cutoff_V_delta = abs(
        V81-V64
    )


    exact_V_consistency = abs(
        V81
        - V81_density
    )


    max_cutoff_density_L1 = max(
        max_cutoff_density_L1,
        cutoff_L1,
    )


    max_cutoff_V_delta = max(
        max_cutoff_V_delta,
        cutoff_V_delta,
    )


    max_exact_V_consistency = max(
        max_exact_V_consistency,
        exact_V_consistency,
    )


    # --------------------------------------------------------
    # RLH
    # --------------------------------------------------------

    rlh = rlh_density(
        beta
    )


    if rlh is None:

        raise RuntimeError(
            f"Unexpected super-caustic "
            f"point beta={beta}"
        )


    rlh_norm_error = abs(
        np.mean(
            rlh
        ) - 1.0
    )


    max_rlh_norm_error = max(
        max_rlh_norm_error,
        rlh_norm_error,
    )


    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    l1 = L1(
        exact81,
        rlh,
    )


    V_rlh = expectation(
        V,
        rlh,
    )


    V_error_pct = (
        100.0
        * (
            V_rlh-V81
        )
        / V81
    )


    zeta_max = (
        beta
        * np.sqrt(1.5)
        / np.pi
    )


    # --------------------------------------------------------
    # Exact density at true symmetry points
    # --------------------------------------------------------

    exact_points = {}

    for name, (
        sp,
        tp
    ) in POINTS.items():

        exact_points[name] = (
            exact_point_density(
                sol81,
                probs81,
                sp,
                tp,
            )
        )


    # --------------------------------------------------------
    # RLH values at exact symmetry points
    #
    # Need global normalization.
    #
    # We recover it from the same unnormalized RLH expression
    # evaluated on the 64x64 grid.
    # Instead of reconstructing it again, obtain point ratios
    # relative to the numerically normalized grid using a
    # common log normalization.
    # --------------------------------------------------------

    Xi_grid, F_grid, valid_grid = (
        spectral_Xi_F(
            kappa_grid,
            beta,
        )
    )


    g_mode_grid = np.einsum(
        "...ji,...j->...i",
        R_grid,
        gradient,
    )


    Phi_grid = (
        V
        + 0.5
        * np.sum(
            F_grid
            * g_mode_grid*g_mode_grid,
            axis=-1,
        )
    )


    logu_grid = (
        0.5
        * np.sum(
            np.log(
                Xi_grid
            ),
            axis=-1,
        )
        - beta*Phi_grid
    )


    shift = float(
        np.max(
            logu_grid
        )
    )


    Z_reduced = float(
        np.mean(
            np.exp(
                logu_grid-shift
            )
        )
    )


    log_norm = (
        shift
        + np.log(
            Z_reduced
        )
    )


    rlh_points = {}

    ratios = {}


    for name, (
        sp,
        tp
    ) in POINTS.items():

        logu = (
            rlh_log_unnormalized_point(
                beta,
                sp,
                tp,
            )
        )


        ppoint = np.exp(
            logu-log_norm
        )


        rlh_points[name] = (
            ppoint
        )


        ratios[name] = (
            ppoint
            / exact_points[name]
        )


    # --------------------------------------------------------
    # Local change diagnostic
    # --------------------------------------------------------

    if previous_l1 is None:

        delta_l1 = np.nan

    else:

        delta_l1 = (
            l1-previous_l1
        )


    previous_l1 = l1


    rows.append([
        beta,
        zeta_max,
        l1,
        0.5*l1,
        V81,
        V_rlh,
        V_error_pct,

        exact_points[
            "minimum"
        ],
        rlh_points[
            "minimum"
        ],
        ratios[
            "minimum"
        ],

        exact_points[
            "saddle"
        ],
        rlh_points[
            "saddle"
        ],
        ratios[
            "saddle"
        ],

        exact_points[
            "maximum"
        ],
        rlh_points[
            "maximum"
        ],
        ratios[
            "maximum"
        ],

        cutoff_L1,
        cutoff_V_delta,
        exact_V_consistency,
        rlh_norm_error,
        delta_l1,
    ])


    print(
        f"{beta:6.3f}  "
        f"{zeta_max:7.4f}  "
        f"{l1:11.6f}  "
        f"{V_error_pct:+8.3f}  "
        f"{ratios['minimum']:10.4f}  "
        f"{ratios['saddle']:10.4f}  "
        f"{ratios['maximum']:10.4f}"
    )


# ============================================================
# Convert for diagnostics
# ============================================================

arr = np.array(
    rows,
    dtype=float,
)


beta_arr = arr[:, 0]
zeta_arr = arr[:, 1]
l1_arr = arr[:, 2]
Verr_arr = arr[:, 6]

Pmin_ratio_arr = arr[:, 9]
Psad_ratio_arr = arr[:, 12]
Pmax_ratio_arr = arr[:, 15]


# ============================================================
# Characterize non-monotonicity
# ============================================================

imin = int(
    np.argmin(
        l1_arr
    )
)

imax_before = int(
    np.argmax(
        l1_arr
    )
)


# We also find the minimum only after beta >= 1,
# to avoid trivially identifying the classical end.

mask_post1 = (
    beta_arr >= 1.0
)

indices_post1 = np.where(
    mask_post1
)[0]

imin_post1 = indices_post1[
    np.argmin(
        l1_arr[
            mask_post1
        ]
    )
]


def first_crossing(
    values,
    threshold,
):

    idx = np.where(
        values >= threshold
    )[0]

    if len(idx) == 0:

        return (
            np.nan,
            np.nan,
        )

    i = idx[0]

    return (
        beta_arr[i],
        zeta_arr[i],
    )


L1_5_beta, L1_5_zeta = (
    first_crossing(
        l1_arr,
        0.05,
    )
)

L1_10_beta, L1_10_zeta = (
    first_crossing(
        l1_arr,
        0.10,
    )
)


V_5_beta, V_5_zeta = (
    first_crossing(
        np.abs(
            Verr_arr
        ),
        5.0,
    )
)

V_10_beta, V_10_zeta = (
    first_crossing(
        np.abs(
            Verr_arr
        ),
        10.0,
    )
)


# ============================================================
# Structural audit
# ============================================================

cutoff_density_pass = (
    max_cutoff_density_L1
    < 1.0e-5
)

cutoff_V_pass = (
    max_cutoff_V_delta
    < 1.0e-5
)

exact_consistency_pass = (
    max_exact_V_consistency
    < 1.0e-10
)

rlh_norm_pass = (
    max_rlh_norm_error
    < 1.0e-12
)

subcaustic_pass = (
    np.max(
        zeta_arr
    ) < 1.0
)


print()
print(
    "=== DENSE-SCAN CHARACTERIZATION ==="
)

print()

print(
    "global minimum L1:"
)

print(
    f"  beta = "
    f"{beta_arr[imin]:.6f}"
)

print(
    f"  zeta = "
    f"{zeta_arr[imin]:.6f}"
)

print(
    f"  L1   = "
    f"{l1_arr[imin]:.9f}"
)

print()

print(
    "minimum L1 for beta >= 1:"
)

print(
    f"  beta = "
    f"{beta_arr[imin_post1]:.6f}"
)

print(
    f"  zeta = "
    f"{zeta_arr[imin_post1]:.6f}"
)

print(
    f"  L1   = "
    f"{l1_arr[imin_post1]:.9f}"
)

print()


print(
    "descriptive threshold crossings "
    "(NOT pass/fail criteria):"
)

print(
    f"  L1 >= 0.05 : "
    f"beta={L1_5_beta}, "
    f"zeta={L1_5_zeta}"
)

print(
    f"  L1 >= 0.10 : "
    f"beta={L1_10_beta}, "
    f"zeta={L1_10_zeta}"
)

print(
    f"  |d<V>| >= 5%  : "
    f"beta={V_5_beta}, "
    f"zeta={V_5_zeta}"
)

print(
    f"  |d<V>| >= 10% : "
    f"beta={V_10_beta}, "
    f"zeta={V_10_zeta}"
)

print()


print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    "max exact density L1 "
    "(K2=64 vs 81) = "
    f"{max_cutoff_density_L1:.6e}"
)

print(
    "max exact <V> delta "
    "(K2=64 vs 81) = "
    f"{max_cutoff_V_delta:.6e}"
)

print(
    "max exact matrix/density "
    "<V> mismatch = "
    f"{max_exact_V_consistency:.6e}"
)

print(
    "max RLH norm error = "
    f"{max_rlh_norm_error:.6e}"
)

print(
    "max zeta = "
    f"{np.max(zeta_arr):.9f}"
)

print()


checks = {
    "EXACT DENSITY CUTOFF":
        cutoff_density_pass,

    "EXACT <V> CUTOFF":
        cutoff_V_pass,

    "EXACT MATRIX/DENSITY CONSISTENCY":
        exact_consistency_pass,

    "RLH NORMALIZATION":
        rlh_norm_pass,

    "ENTIRE SCAN SUB-CAUSTIC":
        subcaustic_pass,
}


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

print()

print(
    "IMPORTANT:"
)

print(
    "The 5% and 10% crossings above are "
    "descriptive diagnostics only."
)

print(
    "They are not being imposed as universal "
    "validity thresholds."
)


# ============================================================
# Save CSV
# ============================================================

with open(
    "lha11b4_dense_precaustic_scan.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "beta",
        "zeta_max",
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

        "exact_density_L1_K64_vs_K81",
        "exact_V_delta_K64_vs_K81",
        "exact_V_matrix_density_mismatch",
        "RLH_norm_error",
        "delta_L1_vs_previous_beta",
    ])

    writer.writerows(
        rows
    )


# ============================================================
# Figure 1 — density L1 vs zeta
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

ax.plot(
    zeta_arr,
    l1_arr,
    marker="o",
    markersize=3,
)

ax.axvline(
    1.0,
    linestyle="--",
)

ax.set_xlabel(
    r"$\zeta_{\max}$"
)

ax.set_ylabel(
    r"$L^1(P_{\rm RLH},P_{\rm QM})$"
)

ax.set_title(
    "Pre-caustic accuracy loss on a periodic 2D landscape"
)

fig.tight_layout()

fig.savefig(
    "lha11b4_L1_vs_zeta.png",
    dpi=300,
)

fig.savefig(
    "lha11b4_L1_vs_zeta.pdf",
)

plt.close(fig)


# ============================================================
# Figure 2 — observable error vs zeta
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

ax.plot(
    zeta_arr,
    Verr_arr,
    marker="o",
    markersize=3,
)

ax.axhline(
    0.0,
    linestyle="--",
)

ax.axvline(
    1.0,
    linestyle="--",
)

ax.set_xlabel(
    r"$\zeta_{\max}$"
)

ax.set_ylabel(
    r"RLH error in $\langle V\rangle$ (%)"
)

ax.set_title(
    "Observable-level pre-caustic error"
)

fig.tight_layout()

fig.savefig(
    "lha11b4_V_error_vs_zeta.png",
    dpi=300,
)

fig.savefig(
    "lha11b4_V_error_vs_zeta.pdf",
)

plt.close(fig)


# ============================================================
# Figure 3 — high-symmetry point ratios
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

ax.plot(
    zeta_arr,
    Pmin_ratio_arr,
    marker="o",
    markersize=3,
    label="minimum",
)

ax.plot(
    zeta_arr,
    Psad_ratio_arr,
    marker="o",
    markersize=3,
    label="saddle",
)

ax.plot(
    zeta_arr,
    Pmax_ratio_arr,
    marker="o",
    markersize=3,
    label="maximum",
)

ax.axhline(
    1.0,
    linestyle="--",
)

ax.axvline(
    1.0,
    linestyle="--",
)

ax.set_xlabel(
    r"$\zeta_{\max}$"
)

ax.set_ylabel(
    r"$P_{\rm RLH}/P_{\rm QM}$"
)

ax.set_title(
    "Where the periodic RLH loses spatial accuracy"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "lha11b4_high_symmetry_ratios.png",
    dpi=300,
)

fig.savefig(
    "lha11b4_high_symmetry_ratios.pdf",
)

plt.close(fig)


print()

print(
    "saved: "
    "lha11b4_dense_precaustic_scan.csv"
)

print(
    "saved: "
    "lha11b4_L1_vs_zeta.png/pdf"
)

print(
    "saved: "
    "lha11b4_V_error_vs_zeta.png/pdf"
)

print(
    "saved: "
    "lha11b4_high_symmetry_ratios.png/pdf"
)
