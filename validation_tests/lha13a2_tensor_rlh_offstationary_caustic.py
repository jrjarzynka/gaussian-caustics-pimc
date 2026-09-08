import csv
import numpy as np


# ============================================================
# LHA-13A2
#
# Tensorial RLH benchmark for the asymmetric moire-like
# periodic landscape.
#
# Exact target:
#   LHA-13A1 BZ ensemble.
#
# Questions:
#
# 1. Does quantitative accuracy deteriorate before zeta=1?
# 2. Does the first off-stationary caustic behave consistently
#    with the local Gaussian stability criterion?
# 3. What happens locally at inequivalent saddles, maximum,
#    and the generic caustic point?
#
# NO regularization for zeta >= 1.
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


N_CONTROL = 256
N_PROD = 512


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


B = np.column_stack([
    G1,
    G2,
])


A_DIRECT = (
    2.0*np.pi
    * np.linalg.inv(
        B.T
    )
)


# ============================================================
# Read stationary points
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


maximum = [
    r for r in stationary
    if r["type"] == "maximum"
][0]


saddles = sorted(
    [
        r for r in stationary
        if r["type"] == "saddle"
    ],
    key=lambda r: float(r["V"]),
)


POINTS = {

    "minimum": (
        float(minimum["s"]),
        float(minimum["t"]),
    ),

    "saddle1": (
        float(saddles[0]["s"]),
        float(saddles[0]["t"]),
    ),

    "saddle2": (
        float(saddles[1]["s"]),
        float(saddles[1]["t"]),
    ),

    "maximum": (
        float(maximum["s"]),
        float(maximum["t"]),
    ),
}


# ============================================================
# Generic first-caustic location from exact A1 file
# ============================================================


a1_npz = np.load(
    "lha13a1_exact_bz_reference.npz"
)


caustic_point = np.asarray(
    a1_npz["caustic_point"],
    dtype=float,
)


beta_c_global = float(
    a1_npz["beta_c_global"]
)


global_kappa_min = float(
    a1_npz["global_kappa_min"]
)


POINTS[
    "caustic_generic"
] = (
    float(caustic_point[0]),
    float(caustic_point[1]),
)


# ============================================================
# Energy zero
# ============================================================


s0 = float(
    minimum["s"]
)

t0 = float(
    minimum["t"]
)


theta_min = np.array([

    2.0*np.pi*s0
    + PHASES[0],

    2.0*np.pi*t0
    + PHASES[1],

    -2.0*np.pi*(s0+t0)
    + PHASES[2],
])


Umin = -float(
    np.sum(
        AMPLITUDES
        * np.cos(theta_min)
    )
)


# ============================================================
# Spectral RLH functions
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


    eps = 1.0e-9


    pos = kappa > eps
    neg = kappa < -eps
    zer = ~(pos | neg)


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
            eta >= np.pi/2.0
        ):

            raise RuntimeError(
                "Local Gaussian caustic crossed."
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


    return Xi, F


# ============================================================
# Local fields on reduced coordinate grid
# ============================================================


def grid_fields(
    N,
):

    axis = (
        np.arange(
            N,
            dtype=float,
        )
        / N
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

        -2.0*np.pi*(S+T)
        + PHASES[2],

    ], axis=-1)


    c = np.cos(theta)
    s = np.sin(theta)


    U = -np.sum(
        AMPLITUDES*c,
        axis=-1,
    )


    V = (
        U-Umin
    )


    grad = np.einsum(
        "...a,ai->...i",
        AMPLITUDES*s,
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


    return (
        V,
        grad,
        kappa,
        R,
    )


# ============================================================
# RLH grid and normalization
# ============================================================


def rlh_grid(
    beta,
    N,
):

    V, grad, kappa, R = (
        grid_fields(N)
    )


    zeta = (
        beta
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


    invalid = (
        np.max(
            zeta,
            axis=-1,
        )
        >= 1.0
    )


    invalid_fraction = float(
        np.mean(invalid)
    )


    if np.any(invalid):

        return {
            "valid": False,
            "zeta_max": zeta_max,
            "invalid_fraction":
                invalid_fraction,
        }


    Xi, F = spectral_Xi_F(
        kappa,
        beta,
    )


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
        - beta*Phi
    )


    shift = float(
        np.max(logu)
    )


    u = np.exp(
        logu-shift
    )


    norm = float(
        np.mean(u)
    )


    P = (
        u/norm
    )


    Vmean = float(
        np.mean(
            P*V
        )
    )


    return {
        "valid": True,

        "zeta_max":
            zeta_max,

        "invalid_fraction":
            invalid_fraction,

        "Vmean":
            Vmean,

        "shift":
            shift,

        "norm":
            norm,
    }


# ============================================================
# Analytic RLH density at one arbitrary point
# ============================================================


def point_rlh_density(
    beta,
    s,
    t,
    shift,
    norm,
):

    theta = np.array([

        2.0*np.pi*s
        + PHASES[0],

        2.0*np.pi*t
        + PHASES[1],

        -2.0*np.pi*(s+t)
        + PHASES[2],
    ])


    c = np.cos(theta)
    sn = np.sin(theta)


    U = -float(
        np.sum(
            AMPLITUDES*c
        )
    )


    V = (
        U-Umin
    )


    grad = np.sum(
        (
            AMPLITUDES*sn
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


    kappa, R = np.linalg.eigh(
        H
    )


    Xi, F = spectral_Xi_F(
        kappa,
        beta,
    )


    gmodal = (
        R.T @ grad
    )


    Phi = (
        V
        + 0.5
        * np.sum(
            F*gmodal*gmodal
        )
    )


    logu = (
        0.5
        * np.sum(
            np.log(Xi)
        )
        - beta*Phi
    )


    return float(
        np.exp(
            logu-shift
        )
        / norm
    )


# ============================================================
# Exact BZ table
# ============================================================


with open(
    "lha13a1_exact_bz_reference.csv",
    newline="",
) as f:

    exact_rows = list(
        csv.DictReader(f)
    )


def exact_row(
    beta,
):

    return min(
        exact_rows,
        key=lambda r:
            abs(
                float(r["beta"])
                - beta
            ),
    )


# ============================================================
# Run RLH
# ============================================================


rows = []


max_grid_V_delta = 0.0
max_grid_point_delta = 0.0


print(
    "=== LHA-13A2 "
    "TENSORIAL RLH THROUGH "
    "OFF-STATIONARY CAUSTIC ==="
)

print()

print(
    f"global kappa_min = "
    f"{global_kappa_min:+.12f}"
)

print(
    f"global beta_c    = "
    f"{beta_c_global:.12f}"
)

print(
    f"caustic point    = "
    f"({caustic_point[0]:.8f}, "
    f"{caustic_point[1]:.8f})"
)

print()


print(
    " beta    zeta_max    status     "
    "V_exact      V_RLH        Verr(%)"
)

print(
    "------------------------------------------------"
    "------------------"
)


for beta in BETAS:

    ex = exact_row(
        beta
    )


    V_exact = float(
        ex["V_BZ"]
    )


    r256 = rlh_grid(
        beta,
        N_CONTROL,
    )


    r512 = rlh_grid(
        beta,
        N_PROD,
    )


    if (
        not r512["valid"]
    ):

        rows.append([
            beta,

            r512[
                "zeta_max"
            ],

            "undefined",

            V_exact,

            np.nan,
            np.nan,

            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,

            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,

            r512[
                "invalid_fraction"
            ],

            np.nan,
            np.nan,
        ])


        print(
            f"{beta:5.2f}  "
            f"{r512['zeta_max']:10.6f}  "
            f"{'UNDEFINED':10s}  "
            f"{V_exact:10.7f}  "
            f"{'---':>10s}  "
            f"{'---':>9s}"
        )


        continue


    # --------------------------------------------------------
    # Grid convergence
    # --------------------------------------------------------


    V_grid_delta = abs(
        r512["Vmean"]
        - r256["Vmean"]
    )


    max_grid_V_delta = max(
        max_grid_V_delta,
        V_grid_delta,
    )


    # --------------------------------------------------------
    # Point densities at both resolutions
    # --------------------------------------------------------


    p256 = {}
    p512 = {}


    for name, (
        s,
        t,
    ) in POINTS.items():

        p256[name] = (
            point_rlh_density(
                beta,
                s,
                t,
                r256["shift"],
                r256["norm"],
            )
        )


        p512[name] = (
            point_rlh_density(
                beta,
                s,
                t,
                r512["shift"],
                r512["norm"],
            )
        )


    point_grid_delta = max(
        abs(
            p512[name]
            - p256[name]
        )
        for name in POINTS
    )


    max_grid_point_delta = max(
        max_grid_point_delta,
        point_grid_delta,
    )


    # --------------------------------------------------------
    # Exact point values
    # --------------------------------------------------------


    exact_points = {

        "minimum":
            float(
                ex[
                    "Pmin_BZ"
                ]
            ),

        "saddle1":
            float(
                ex[
                    "Psaddle1_BZ"
                ]
            ),

        "saddle2":
            float(
                ex[
                    "Psaddle2_BZ"
                ]
            ),

        "maximum":
            float(
                ex[
                    "Pmax_BZ"
                ]
            ),

        "caustic_generic":
            float(
                ex[
                    "Pcaustic_generic_BZ"
                ]
            ),
    }


    ratios = {
        name:
            p512[name]
            / exact_points[name]

        for name in POINTS
    }


    V_error_pct = (
        100.0
        * (
            r512["Vmean"]
            - V_exact
        )
        / V_exact
    )


    rows.append([
        beta,

        r512[
            "zeta_max"
        ],

        "valid",

        V_exact,

        r512[
            "Vmean"
        ],

        V_error_pct,

        p512[
            "minimum"
        ],

        p512[
            "saddle1"
        ],

        p512[
            "saddle2"
        ],

        p512[
            "maximum"
        ],

        p512[
            "caustic_generic"
        ],

        ratios[
            "minimum"
        ],

        ratios[
            "saddle1"
        ],

        ratios[
            "saddle2"
        ],

        ratios[
            "maximum"
        ],

        ratios[
            "caustic_generic"
        ],

        r512[
            "invalid_fraction"
        ],

        V_grid_delta,

        point_grid_delta,
    ])


    print(
        f"{beta:5.2f}  "
        f"{r512['zeta_max']:10.6f}  "
        f"{'valid':10s}  "
        f"{V_exact:10.7f}  "
        f"{r512['Vmean']:10.7f}  "
        f"{V_error_pct:+9.3f}"
    )


print()


# ============================================================
# Detailed point-ratio table
# ============================================================


print(
    "=== SUBCAUSTIC POINT DENSITY RATIOS "
    "RLH / EXACT ==="
)

print(
    " beta     Pmin      Psad1     "
    "Psad2     Pmax      Pcaustic"
)

print(
    "------------------------------------------------"
    "----------------"
)


for row in rows:

    if row[2] != "valid":
        continue


    print(
        f"{row[0]:5.2f}  "
        f"{row[11]:9.5f}  "
        f"{row[12]:9.5f}  "
        f"{row[13]:9.5f}  "
        f"{row[14]:9.5f}  "
        f"{row[15]:9.5f}"
    )


# ============================================================
# Structural audit
# ============================================================


precaustic_rows = [
    r for r in rows
    if r[2] == "valid"
]


postcaustic_rows = [
    r for r in rows
    if r[2] == "undefined"
]


grid_V_pass = (
    max_grid_V_delta
    < 1.0e-8
)


grid_point_pass = (
    max_grid_point_delta
    < 1.0e-8
)


all_pre_valid = all(
    r[0] < beta_c_global
    for r in precaustic_rows
)


all_post_invalid = all(
    r[0] > beta_c_global
    for r in postcaustic_rows
)


no_regularization = all(
    np.isnan(r[4])
    and np.isnan(r[5])
    for r in postcaustic_rows
)


print()

print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    f"max RLH <V> grid delta "
    f"{N_CONTROL}->{N_PROD} = "
    f"{max_grid_V_delta:.6e}"
)

print(
    f"max RLH point grid delta "
    f"{N_CONTROL}->{N_PROD} = "
    f"{max_grid_point_delta:.6e}"
)

print()


checks = {

    "RLH GRID <V> CONVERGENCE":
        grid_V_pass,

    "RLH POINT NORMALIZATION CONVERGENCE":
        grid_point_pass,

    "ALL SAMPLED PRECAUSTIC POINTS VALID":
        all_pre_valid,

    "ALL SAMPLED POSTCAUSTIC POINTS UNDEFINED":
        all_post_invalid,

    "NO POSTCAUSTIC REGULARIZATION":
        no_regularization,
}


for name, passed in (
    checks.items()
):

    print(
        f"{name:38s}: "
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
    "lha13a2_tensor_rlh_offstationary_caustic.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",

        "zeta_max",

        "RLH_status",

        "V_exact_BZ",

        "V_RLH",

        "V_error_pct",

        "Pmin_RLH",
        "Psaddle1_RLH",
        "Psaddle2_RLH",
        "Pmax_RLH",
        "Pcaustic_RLH",

        "Pmin_ratio_RLH_over_exact",
        "Psaddle1_ratio_RLH_over_exact",
        "Psaddle2_ratio_RLH_over_exact",
        "Pmax_ratio_RLH_over_exact",
        "Pcaustic_ratio_RLH_over_exact",

        "invalid_area_fraction",

        "V_grid_delta_256_to_512",

        "point_grid_delta_256_to_512",
    ])

    w.writerows(
        rows
    )


print()

print(
    "saved: "
    "lha13a2_tensor_rlh_offstationary_caustic.csv"
)
