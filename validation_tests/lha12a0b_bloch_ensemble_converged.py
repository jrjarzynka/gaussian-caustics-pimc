import csv
import numpy as np


# ============================================================
# LHA-12A0
#
# Gamma-only one-cell ED
#             vs
# Brillouin-zone averaged periodic ED
#
# Purpose:
# Determine which exact quantum reference must be used for
# comparison with an unwrapped PI-QMC path in an infinite
# periodic potential.
#
# Model:
#
# V(r) = 3
#        - cos(G1.r)
#        - cos(G2.r)
#        - cos(G3.r)
#
# hbar = M = V0 = |G| = 1
#
# Bloch Hamiltonian:
#
# H_k(G,G')
# =
# 1/2 |k+G|^2 delta_GG'
# + V_(G-G')
#
# We compare:
#
# 1. k = Gamma only
# 2. uniform integration over one reciprocal primitive cell
#
# Metrics:
#
#   <V>
#   P(min)
#   P(saddle)
#   P(max)
#
# No PI-QMC is run yet.
# ============================================================


BETAS = np.array([
    0.5,
    1.0,
    2.0,
    2.5,
])


# ------------------------------------------------------------
# Convergence ladders
# ------------------------------------------------------------

KGRID_SCAN = [
    8,
    12,
    16,
]

KGRID_CUTOFF_FOR_SCAN = 81


CUTOFF_SCAN = [
    49,
    64,
    81,
]

KGRID_FOR_CUTOFF_SCAN = 12


# Production setting after the two audits.

K2_PROD = 81
NK_PROD = 16


sqrt3 = np.sqrt(3.0)


# ============================================================
# Reciprocal geometry
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


# ------------------------------------------------------------
# Fourier coefficients
# ------------------------------------------------------------

V_FOURIER = {
    (0, 0): 3.0,

    (1, 0): -0.5,
    (-1, 0): -0.5,

    (0, 1): -0.5,
    (0, -1): -0.5,

    (1, 1): -0.5,
    (-1, -1): -0.5,
}


# ============================================================
# High-symmetry points in reduced coordinates
# ============================================================

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
# Plane-wave basis
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
# k-dependent Hamiltonian
# ============================================================

def prepare_basis_data(
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


    # Cartesian reciprocal vectors G.

    Gcart = (
        nvec[:, 0, None]*G1[None, :]
        +
        nvec[:, 1, None]*G2[None, :]
    )


    # Potential matrix independent of Bloch k.

    Vmat = np.zeros(
        (
            nbasis,
            nbasis,
        ),
        dtype=float,
    )


    for i, (
        n1,
        n2,
        _
    ) in enumerate(
        basis
    ):

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


    # Neighbor lists for cheap diagonal matrix elements
    # <state|V|state>.

    couplings = []

    for (
        dn1,
        dn2
    ), coeff in (
        V_FOURIER.items()
    ):

        if (
            dn1 == 0
            and dn2 == 0
        ):
            continue

        ii = []
        jj = []

        for i, (
            n1,
            n2,
            _
        ) in enumerate(
            basis
        ):

            j = index.get(
                (
                    n1-dn1,
                    n2-dn2,
                )
            )

            if j is not None:

                ii.append(i)
                jj.append(j)

        couplings.append(
            (
                np.array(
                    ii,
                    dtype=int,
                ),
                np.array(
                    jj,
                    dtype=int,
                ),
                coeff,
            )
        )


    # Plane-wave phases at selected real-space points.
    #
    # exp[i G.r]
    # =
    # exp[2*pi*i(n1*s+n2*t)]
    #
    # Global exp(i k.r) drops out of |psi|^2.

    point_phase = {}

    for name, (
        s,
        t
    ) in POINTS.items():

        point_phase[
            name
        ] = np.exp(
            2.0j*np.pi
            * (
                nvec[:, 0]*s
                + nvec[:, 1]*t
            )
        )


    return {
        "basis": basis,
        "nvec": nvec,
        "Gcart": Gcart,
        "Vmat": Vmat,
        "couplings": couplings,
        "point_phase": point_phase,
    }


def solve_at_k(
    prepared,
    ku,
    kv,
):

    # Bloch vector:
    #
    # k = ku G1 + kv G2

    kcart = (
        ku*G1
        + kv*G2
    )


    KG = (
        prepared[
            "Gcart"
        ]
        + kcart[None, :]
    )


    kinetic = (
        0.5
        * np.sum(
            KG*KG,
            axis=1,
        )
    )


    H = (
        prepared[
            "Vmat"
        ].copy()
    )

    H[
        np.diag_indices_from(H)
    ] += kinetic


    E, C = np.linalg.eigh(
        H
    )


    # --------------------------------------------------------
    # Potential expectation for every eigenstate.
    #
    # Diagonal Fourier term contributes exactly 3.
    # --------------------------------------------------------

    Vstate = np.full(
        len(E),
        3.0,
        dtype=float,
    )


    for (
        ii,
        jj,
        coeff
    ) in prepared[
        "couplings"
    ]:

        Vstate += (
            coeff
            * np.sum(
                C[ii, :]
                * C[jj, :],
                axis=0,
            )
        )


    # --------------------------------------------------------
    # Density of every eigenstate at selected points.
    #
    # |exp(i k.r)|^2 = 1,
    # so only the periodic plane-wave coefficients matter.
    # --------------------------------------------------------

    point_state_density = {}

    for name, phase in (
        prepared[
            "point_phase"
        ].items()
    ):

        psi = (
            phase @ C
        )

        point_state_density[
            name
        ] = (
            np.abs(
                psi
            )**2
        )


    return (
        E,
        Vstate,
        point_state_density,
    )


# ============================================================
# Ensemble calculation
# ============================================================

def ensemble(
    K2_max,
    nk,
    gamma_only=False,
):

    prepared = (
        prepare_basis_data(
            K2_max
        )
    )


    if gamma_only:

        kpoints = [
            (
                0.0,
                0.0,
            )
        ]

    else:

        # Midpoint Monkhorst-Pack grid over one reciprocal
        # primitive cell:
        #
        # -1/2 <= ku,kv < 1/2

        vals = (
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

        kpoints = [
            (
                ku,
                kv,
            )

            for ku in vals
            for kv in vals
        ]


    solved = []


    for (
        ku,
        kv
    ) in kpoints:

        solved.append(
            solve_at_k(
                prepared,
                ku,
                kv,
            )
        )


    global_E0 = min(
        float(
            np.min(
                item[0]
            )
        )

        for item in solved
    )


    result = {}


    for beta in BETAS:

        Z = 0.0

        V_num = 0.0

        P_num = {
            name: 0.0
            for name
            in POINTS
        }


        for (
            E,
            Vstate,
            point_state_density
        ) in solved:

            w = np.exp(
                -beta
                * (
                    E-global_E0
                )
            )


            Z += np.sum(
                w
            )


            V_num += np.sum(
                w*Vstate
            )


            for name in POINTS:

                P_num[name] += (
                    np.sum(
                        w
                        * point_state_density[
                            name
                        ]
                    )
                )


        result[
            beta
        ] = {
            "V": (
                V_num/Z
            ),

            "Pmin": (
                P_num[
                    "minimum"
                ]/Z
            ),

            "Psaddle": (
                P_num[
                    "saddle"
                ]/Z
            ),

            "Pmax": (
                P_num[
                    "maximum"
                ]/Z
            ),

            "Z_shifted": Z,
        }


    return (
        result,
        len(
            prepared[
                "basis"
            ]
        ),
    )


# ============================================================
# Helpers
# ============================================================

def metric_vector(
    result,
    beta,
):

    r = result[
        beta
    ]

    return np.array([
        r["V"],
        r["Pmin"],
        r["Psaddle"],
        r["Pmax"],
    ])


def max_abs_delta(
    a,
    b,
):

    vals = []

    for beta in BETAS:

        vals.append(
            np.max(
                np.abs(
                    metric_vector(
                        a,
                        beta,
                    )
                    -
                    metric_vector(
                        b,
                        beta,
                    )
                )
            )
        )

    return float(
        max(vals)
    )


# ============================================================
# 1. k-grid convergence
# ============================================================

print(
    "=== LHA-12A0 "
    "BLOCH-ENSEMBLE AUDIT ==="
)

print()

print(
    "=== k-GRID CONVERGENCE ==="
)

print(
    f"K2 cutoff = "
    f"{KGRID_CUTOFF_FOR_SCAN}"
)

print()


kgrid_results = {}

previous = None


for nk in KGRID_SCAN:

    result, nbasis = ensemble(
        KGRID_CUTOFF_FOR_SCAN,
        nk,
        gamma_only=False,
    )


    kgrid_results[
        nk
    ] = result


    if previous is None:

        delta = np.nan

    else:

        delta = max_abs_delta(
            result,
            previous,
        )


    print(
        f"Nk={nk:2d}x{nk:2d}  "
        f"NBASIS={nbasis:3d}  "
        f"max metric delta="
        f"{delta:.6e}"
    )


    previous = result


print()


# ============================================================
# 2. reciprocal cutoff convergence
# ============================================================

print(
    "=== RECIPROCAL CUTOFF CONVERGENCE ==="
)

print(
    f"k-grid = "
    f"{KGRID_FOR_CUTOFF_SCAN}x"
    f"{KGRID_FOR_CUTOFF_SCAN}"
)

print()


cutoff_results = {}

previous = None


for K2 in CUTOFF_SCAN:

    result, nbasis = ensemble(
        K2,
        KGRID_FOR_CUTOFF_SCAN,
        gamma_only=False,
    )


    cutoff_results[
        K2
    ] = result


    if previous is None:

        delta = np.nan

    else:

        delta = max_abs_delta(
            result,
            previous,
        )


    print(
        f"K2={K2:2d}  "
        f"NBASIS={nbasis:3d}  "
        f"max metric delta="
        f"{delta:.6e}"
    )


    previous = result


print()


# ============================================================
# 3. Production BZ ensemble
# ============================================================

print(
    "=== PRODUCTION COMPARISON ==="
)

print(
    f"BZ: K2={K2_PROD}, "
    f"Nk={NK_PROD}x{NK_PROD}"
)

print()


bz, nbasis_prod = ensemble(
    K2_PROD,
    NK_PROD,
    gamma_only=False,
)


gamma, _ = ensemble(
    K2_PROD,
    1,
    gamma_only=True,
)


rows = []


print(
    " beta      <V>_Gamma      <V>_BZ      "
    "dV(%)      "
    "Pmin G/BZ   Psad G/BZ   Pmax G/BZ"
)

print(
    "------------------------------------------------"
    "--------------------------------------"
)


for beta in BETAS:

    g = gamma[
        beta
    ]

    b = bz[
        beta
    ]


    dV_pct = (
        100.0
        * (
            b["V"]
            - g["V"]
        )
        / g["V"]
    )


    Pmin_ratio = (
        b["Pmin"]
        / g["Pmin"]
    )

    Psad_ratio = (
        b["Psaddle"]
        / g["Psaddle"]
    )

    Pmax_ratio = (
        b["Pmax"]
        / g["Pmax"]
    )


    print(
        f"{beta:5.2f}  "
        f"{g['V']:13.9f}  "
        f"{b['V']:11.9f}  "
        f"{dV_pct:+8.3f}  "
        f"{Pmin_ratio:9.5f}  "
        f"{Psad_ratio:9.5f}  "
        f"{Pmax_ratio:9.5f}"
    )


    rows.append([
        beta,

        g["V"],
        b["V"],
        dV_pct,

        g["Pmin"],
        b["Pmin"],
        Pmin_ratio,

        g["Psaddle"],
        b["Psaddle"],
        Psad_ratio,

        g["Pmax"],
        b["Pmax"],
        Pmax_ratio,
    ])


# ============================================================
# Convergence audit
# ============================================================

last_nk = KGRID_SCAN[-1]
prev_nk = KGRID_SCAN[-2]


final_kgrid_delta = max_abs_delta(
    kgrid_results[
        last_nk
    ],
    kgrid_results[
        prev_nk
    ],
)


last_K2 = CUTOFF_SCAN[-1]
prev_K2 = CUTOFF_SCAN[-2]


final_cutoff_delta = max_abs_delta(
    cutoff_results[
        last_K2
    ],
    cutoff_results[
        prev_K2
    ],
)


kgrid_pass = (
    final_kgrid_delta
    < 1.0e-5
)

cutoff_pass = (
    final_cutoff_delta
    < 1.0e-5
)


# Difference between ensembles is not a pass/fail condition.
# It is the scientific result of this audit.


print()

print(
    "=== CONVERGENCE AUDIT ==="
)

print(
    f"final k-grid delta "
    f"({prev_nk}->{last_nk}) = "
    f"{final_kgrid_delta:.6e}"
)

print(
    f"final cutoff delta "
    f"({prev_K2}->{last_K2}) = "
    f"{final_cutoff_delta:.6e}"
)

print()


print(
    "BZ k-GRID CONVERGENCE:",
    "PASS"
    if kgrid_pass
    else "FAIL"
)

print(
    "BZ BASIS CONVERGENCE:",
    "PASS"
    if cutoff_pass
    else "FAIL"
)


overall = (
    kgrid_pass
    and cutoff_pass
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
    "Gamma-vs-BZ differences are the "
    "physical output of this checkpoint, "
    "not a failure criterion."
)


# ============================================================
# CSV
# ============================================================

with open(
    "lha12a0b_bloch_ensemble_converged.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "beta",

        "V_gamma",
        "V_BZ",
        "V_BZ_minus_gamma_pct",

        "Pmin_gamma",
        "Pmin_BZ",
        "Pmin_BZ_over_gamma",

        "Psaddle_gamma",
        "Psaddle_BZ",
        "Psaddle_BZ_over_gamma",

        "Pmax_gamma",
        "Pmax_BZ",
        "Pmax_BZ_over_gamma",
    ])

    writer.writerows(
        rows
    )


print()

print(
    "saved: "
    "lha12a0b_bloch_ensemble_converged.csv"
)
