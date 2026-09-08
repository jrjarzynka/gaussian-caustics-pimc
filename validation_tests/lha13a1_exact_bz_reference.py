import csv
import numpy as np


# ============================================================
# LHA-13A1
#
# Exact Bloch/BZ quantum reference for the asymmetric
# moire-like periodic benchmark introduced in LHA-13A0.
#
# No RLH.
# No PI-QMC.
#
# Dimensionless units:
#   M = hbar = 1
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
# Read stationary points from A0
# ============================================================


with open(
    "lha13a0_stationary_points.csv",
    newline="",
) as f:

    stationary = list(
        csv.DictReader(f)
    )


def rows_of(kind):

    return [
        row
        for row in stationary
        if row["type"] == kind
    ]


minimum = rows_of(
    "minimum"
)[0]


maximum = rows_of(
    "maximum"
)[0]


saddles = sorted(
    rows_of("saddle"),
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
# Independently recover the generic global-caustic location
# on the same 512 x 512 geometry grid as A0.
# ============================================================


NGEOM = 512


axis = (
    np.arange(
        NGEOM,
        dtype=float,
    )
    / NGEOM
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


ct = np.cos(
    theta
)


GS = np.array([
    G1,
    G2,
    -(G1+G2),
])


Hgrid = np.einsum(
    "...a,ai,aj->...ij",
    AMPLITUDES*ct,
    GS,
    GS,
)


eig = np.linalg.eigvalsh(
    Hgrid
)


idx = np.unravel_index(
    np.argmin(
        eig[..., 0]
    ),
    eig[..., 0].shape,
)


CAUSTIC_POINT = (
    idx[1]/NGEOM,
    idx[0]/NGEOM,
)


POINTS[
    "caustic_generic"
] = CAUSTIC_POINT


global_kappa_min = float(
    eig[
        idx
    ][0]
)


beta_c_global = (
    np.pi
    / np.sqrt(
        -global_kappa_min
    )
)


# ============================================================
# Determine energy-zero shift from A0 minimum
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
        * np.cos(
            theta_min
        )
    )
)


V0 = -Umin


# ============================================================
# Fourier coefficients
#
# V(r)
# =
# V0
# -
# sum_j A_j cos(G_j.r + phi_j)
#
# G3 = -(G1+G2)
# ============================================================


V_FOURIER = {

    (0, 0):
        V0,

    (1, 0):
        -0.5
        * AMPLITUDES[0]
        * np.exp(
            1j*PHASES[0]
        ),

    (-1, 0):
        -0.5
        * AMPLITUDES[0]
        * np.exp(
            -1j*PHASES[0]
        ),

    (0, 1):
        -0.5
        * AMPLITUDES[1]
        * np.exp(
            1j*PHASES[1]
        ),

    (0, -1):
        -0.5
        * AMPLITUDES[1]
        * np.exp(
            -1j*PHASES[1]
        ),

    (-1, -1):
        -0.5
        * AMPLITUDES[2]
        * np.exp(
            1j*PHASES[2]
        ),

    (1, 1):
        -0.5
        * AMPLITUDES[2]
        * np.exp(
            -1j*PHASES[2]
        ),
}


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
    K2max,
):

    R = (
        int(
            np.ceil(
                2.0*np.sqrt(
                    K2max
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


            if K2 <= K2max:

                basis.append(
                    (
                        n1,
                        n2,
                        K2,
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


def prepare(
    K2max,
):

    basis = build_basis(
        K2max
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
    ], dtype=int)


    Gcart = (
        nvec[:, 0, None]
        * G1[None, :]
        +
        nvec[:, 1, None]
        * G2[None, :]
    )


    Vmat = np.zeros(
        (
            len(basis),
            len(basis),
        ),
        dtype=complex,
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


    point_phase = {}


    for name, (
        s,
        t,
    ) in POINTS.items():

        point_phase[
            name
        ] = np.exp(
            2.0j*np.pi
            * (
                nvec[:, 0]*s
                +
                nvec[:, 1]*t
            )
        )


    return {
        "basis": basis,
        "nvec": nvec,
        "Gcart": Gcart,
        "Vmat": Vmat,
        "point_phase": point_phase,
    }


# ============================================================
# Fourier reconstruction audit
# ============================================================


rng = np.random.default_rng(
    130101
)


max_fourier_error = 0.0


for _ in range(200):

    s, t = rng.random(2)


    direct = (

        V0

        - AMPLITUDES[0]
        * np.cos(
            2.0*np.pi*s
            + PHASES[0]
        )

        - AMPLITUDES[1]
        * np.cos(
            2.0*np.pi*t
            + PHASES[1]
        )

        - AMPLITUDES[2]
        * np.cos(
            -2.0*np.pi*(s+t)
            + PHASES[2]
        )
    )


    fourier = 0.0j


    for (
        n1,
        n2
    ), coeff in (
        V_FOURIER.items()
    ):

        fourier += (
            coeff
            * np.exp(
                2.0j*np.pi
                * (
                    n1*s
                    + n2*t
                )
            )
        )


    max_fourier_error = max(
        max_fourier_error,
        abs(
            fourier-direct
        ),
    )


# ============================================================
# Exact Bloch ensemble
# ============================================================


def ensemble(
    K2max,
    nk,
    gamma_only=False,
):

    p = prepare(
        K2max
    )


    hermiticity = float(
        np.max(
            np.abs(
                p["Vmat"]
                - p["Vmat"].conj().T
            )
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


    global_E0 = np.inf


    for ku, kv in kpoints:

        kcart = (
            ku*G1
            + kv*G2
        )


        KG = (
            p["Gcart"]
            + kcart[None, :]
        )


        H = p[
            "Vmat"
        ].copy()


        H[
            np.diag_indices_from(
                H
            )
        ] += (
            0.5
            * np.sum(
                KG*KG,
                axis=1,
            )
        )


        E, C = np.linalg.eigh(
            H
        )


        global_E0 = min(
            global_E0,
            float(E[0]),
        )


        VC = (
            p["Vmat"]
            @ C
        )


        Vstate = np.real(
            np.sum(
                np.conj(C)
                * VC,
                axis=0,
            )
        )


        point_state = {}


        for name, phase in (
            p[
                "point_phase"
            ].items()
        ):

            psi = (
                phase
                @ C
            )


            point_state[
                name
            ] = (
                np.abs(psi)**2
            )


        solved.append(
            (
                E,
                Vstate,
                point_state,
            )
        )


    result = {}


    for beta in BETAS:

        Z = 0.0
        Vnum = 0.0


        Pnum = {
            name: 0.0
            for name in POINTS
        }


        for (
            E,
            Vstate,
            point_state,
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


            Vnum += np.sum(
                w*Vstate
            )


            for name in POINTS:

                Pnum[
                    name
                ] += np.sum(
                    w
                    * point_state[
                        name
                    ]
                )


        result[
            beta
        ] = {

            "V":
                Vnum/Z,

            **{
                name:
                    Pnum[name]/Z

                for name in POINTS
            },
        }


    return (
        result,
        len(
            p["basis"]
        ),
        hermiticity,
    )


def metric_delta(
    a,
    b,
):

    maximum = 0.0
    where = None


    for beta in BETAS:

        for key in a[
            beta
        ]:

            d = abs(
                a[beta][key]
                - b[beta][key]
            )


            if d > maximum:

                maximum = float(d)

                where = (
                    beta,
                    key,
                )


    return (
        maximum,
        where,
    )


# ============================================================
# Run convergence ladder
# ============================================================


print(
    "=== LHA-13A1 "
    "EXACT ASYMMETRIC BZ REFERENCE ==="
)

print()


print(
    f"V0 shift                 = "
    f"{V0:.12f}"
)

print(
    f"generic caustic point    = "
    f"({CAUSTIC_POINT[0]:.8f}, "
    f"{CAUSTIC_POINT[1]:.8f})"
)

print(
    f"global kappa_min         = "
    f"{global_kappa_min:+.12f}"
)

print(
    f"global beta_c            = "
    f"{beta_c_global:.12f}"
)

print(
    f"Fourier reconstruction   = "
    f"{max_fourier_error:.6e}"
)

print()


control_basis, n81, h81 = ensemble(
    K2_CONTROL,
    NK_CONTROL,
)


prod_basis_same_k, n100, h100 = ensemble(
    K2_PROD,
    NK_CONTROL,
)


production, _, _ = ensemble(
    K2_PROD,
    NK_PROD,
)


gamma, _, _ = ensemble(
    K2_PROD,
    1,
    gamma_only=True,
)


basis_delta, basis_where = metric_delta(
    prod_basis_same_k,
    control_basis,
)


kgrid_delta, kgrid_where = metric_delta(
    production,
    prod_basis_same_k,
)


print(
    "=== NUMERICAL CONVERGENCE ==="
)

print(
    f"K2={K2_CONTROL}: "
    f"NBASIS={n81}"
)

print(
    f"K2={K2_PROD}: "
    f"NBASIS={n100}"
)

print()

print(
    f"max Hermiticity error    = "
    f"{max(h81,h100):.6e}"
)

print(
    f"basis delta "
    f"{K2_CONTROL}->{K2_PROD} "
    f"at Nk={NK_CONTROL} = "
    f"{basis_delta:.6e} "
    f"at {basis_where}"
)

print(
    f"k-grid delta "
    f"{NK_CONTROL}->{NK_PROD} "
    f"at K2={K2_PROD} = "
    f"{kgrid_delta:.6e} "
    f"at {kgrid_where}"
)

print()


# ============================================================
# Main exact table
# ============================================================


print(
    " beta     <V>_BZ       "
    "dV(BZ-Gamma)%    "
    "Pmin       Psad1      "
    "Psad2      Pmax       Pcaustic"
)

print(
    "------------------------------------------------"
    "----------------------------------------------"
)


rows = []


for beta in BETAS:

    b = production[
        beta
    ]


    g = gamma[
        beta
    ]


    dV_gamma_pct = (
        100.0
        * (
            b["V"]
            - g["V"]
        )
        / g["V"]
    )


    caustic_ratio = (
        b[
            "caustic_generic"
        ]
        / g[
            "caustic_generic"
        ]
    )


    rows.append([
        beta,

        b["V"],
        g["V"],
        dV_gamma_pct,

        b["minimum"],
        b["saddle1"],
        b["saddle2"],
        b["maximum"],
        b["caustic_generic"],

        caustic_ratio,
    ])


    print(
        f"{beta:5.2f}  "
        f"{b['V']:11.8f}  "
        f"{dV_gamma_pct:+12.6f}  "
        f"{b['minimum']:9.5f}  "
        f"{b['saddle1']:9.5f}  "
        f"{b['saddle2']:9.5f}  "
        f"{b['maximum']:9.5f}  "
        f"{b['caustic_generic']:9.5f}"
    )


# ============================================================
# Structural checks
# ============================================================


minimum_density = min(

    value

    for beta in BETAS

    for key, value in (
        production[
            beta
        ].items()
    )

    if key != "V"
)


checks = {

    "FOURIER RECONSTRUCTION":
        max_fourier_error
        < 1.0e-12,

    "COMPLEX HAMILTONIAN HERMITICITY":
        max(
            h81,
            h100,
        )
        < 1.0e-12,

    "RECIPROCAL BASIS CONVERGENCE":
        basis_delta
        < 1.0e-6,

    "BZ k-GRID CONVERGENCE":
        kgrid_delta
        < 1.0e-8,

    "POSITIVE POINT DENSITIES":
        minimum_density
        > 0.0,

    "CAUSTIC LOCATION OFF STATIONARY":
        all(

            np.linalg.norm(

                (
                    np.asarray(
                        CAUSTIC_POINT
                    )
                    -
                    np.asarray(
                        point
                    )
                    + 0.5
                )
                % 1.0
                - 0.5

            )
            > 0.05

            for name, point in POINTS.items()

            if name
            != "caustic_generic"
        ),
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


print()

print(
    "IMPORTANT:"
)

print(
    "Gamma-vs-BZ differences are physical "
    "diagnostics and are NOT a PASS criterion."
)

print(
    "Exact quantum observables remain defined "
    "on both sides of the local RLH caustic."
)


# ============================================================
# Save
# ============================================================


with open(
    "lha13a1_exact_bz_reference.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",

        "V_BZ",
        "V_Gamma",
        "V_BZ_minus_Gamma_pct",

        "Pmin_BZ",
        "Psaddle1_BZ",
        "Psaddle2_BZ",
        "Pmax_BZ",
        "Pcaustic_generic_BZ",

        "Pcaustic_BZ_over_Gamma",
    ])

    w.writerows(
        rows
    )


np.savez_compressed(

    "lha13a1_exact_bz_reference.npz",

    betas=BETAS,

    V_BZ=np.asarray([
        production[b]["V"]
        for b in BETAS
    ]),

    Pmin_BZ=np.asarray([
        production[b]["minimum"]
        for b in BETAS
    ]),

    Psaddle1_BZ=np.asarray([
        production[b]["saddle1"]
        for b in BETAS
    ]),

    Psaddle2_BZ=np.asarray([
        production[b]["saddle2"]
        for b in BETAS
    ]),

    Pmax_BZ=np.asarray([
        production[b]["maximum"]
        for b in BETAS
    ]),

    Pcaustic_BZ=np.asarray([
        production[b]["caustic_generic"]
        for b in BETAS
    ]),

    caustic_point=np.asarray(
        CAUSTIC_POINT
    ),

    global_kappa_min=global_kappa_min,

    beta_c_global=beta_c_global,

    K2_prod=K2_PROD,

    Nk_prod=NK_PROD,

    basis_delta=basis_delta,

    kgrid_delta=kgrid_delta,
)


print()

print(
    "saved: lha13a1_exact_bz_reference.csv"
)

print(
    "saved: lha13a1_exact_bz_reference.npz"
)
