import csv
import numpy as np


# ============================================================
# LHA-12A1
#
# Converged full Brillouin-zone averaged exact quantum density
# for the triangular periodic benchmark.
#
# This density will be the direct target for periodic PI-QMC.
#
# Production:
#   K2_max = 81
#   Nk     = 16 x 16
#
# Controls:
#   basis:  K2=64 -> 81 at Nk=12
#   k-grid: Nk=12 -> 16 at K2=81
#
# Reduced real-space grid:
#   64 x 64, endpoint excluded
#
# hbar = M = V0 = |G| = 1.
# ============================================================


BETAS = np.array([
    0.5,
    1.0,
    1.9,
    2.5,
])


NGRID = 64

K2_CONTROL = 64
K2_PROD = 81

NK_CONTROL = 12
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


V_FOURIER = {
    (0, 0): 3.0,

    (1, 0): -0.5,
    (-1, 0): -0.5,

    (0, 1): -0.5,
    (0, -1): -0.5,

    (1, 1): -0.5,
    (-1, -1): -0.5,
}


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
# Basis
# ============================================================

def K2_of(n1, n2):

    return (
        n1*n1
        + n2*n2
        - n1*n2
    )


def build_basis(K2_max):

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


def prepare(K2_max):

    basis = build_basis(
        K2_max
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


    Gcart = (
        nvec[:, 0, None]*G1[None, :]
        +
        nvec[:, 1, None]*G2[None, :]
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


    Vmat = np.zeros(
        (
            len(basis),
            len(basis),
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


    return {
        "basis": basis,
        "nvec": nvec,
        "Gcart": Gcart,
        "Vmat": Vmat,
    }


# ============================================================
# Bloch ensemble density matrix
# ============================================================

def bz_density_matrices(
    K2_max,
    nk,
    gamma_only=False,
):

    p = prepare(
        K2_max
    )

    N = len(
        p[
            "basis"
        ]
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
            p[
                "Gcart"
            ]
            + kcart[None, :]
        )


        H = (
            p[
                "Vmat"
            ].copy()
        )


        H[
            np.diag_indices_from(H)
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


        solved.append(
            (
                E,
                C,
            )
        )


    rhos = {}


    for beta in BETAS:

        rho = np.zeros(
            (
                N,
                N,
            ),
            dtype=float,
        )

        Z = 0.0


        for E, C in solved:

            w = np.exp(
                -beta
                * (
                    E-global_E0
                )
            )


            Z += np.sum(
                w
            )


            rho += (
                (C*w)
                @ C.T
            )


        rho /= Z


        rhos[
            beta
        ] = rho


    return (
        p,
        rhos,
    )


# ============================================================
# Density matrix -> periodic real-space density
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


V_real = (
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


def density_from_rho(
    nvec,
    rho,
):

    diff = (
        nvec[:, None, :]
        -
        nvec[None, :, :]
    )


    d1 = (
        diff[..., 0]
        .ravel()
    )

    d2 = (
        diff[..., 1]
        .ravel()
    )


    values = rho.ravel()


    m = int(
        max(
            np.max(
                np.abs(d1)
            ),
            np.max(
                np.abs(d2)
            ),
        )
    )


    coeff = np.zeros(
        (
            2*m+1,
            2*m+1,
        ),
        dtype=complex,
    )


    np.add.at(
        coeff,
        (
            d1+m,
            d2+m,
        ),
        values,
    )


    density = np.zeros(
        (
            NGRID,
            NGRID,
        ),
        dtype=complex,
    )


    inds = np.argwhere(
        np.abs(
            coeff
        )
        > 1.0e-15
    )


    for i, j in inds:

        dn1 = i-m
        dn2 = j-m


        density += (
            coeff[
                i,
                j
            ]
            * np.exp(
                2.0j*np.pi
                * (
                    dn1*S
                    + dn2*T
                )
            )
        )


    imag_residual = float(
        np.max(
            np.abs(
                density.imag
            )
        )
    )


    density = density.real


    density /= np.mean(
        density
    )


    return (
        density,
        imag_residual,
        coeff,
    )


def point_density_from_rho(
    nvec,
    rho,
    s,
    t,
):

    phase = np.exp(
        2.0j*np.pi
        * (
            nvec[:, 0]*s
            + nvec[:, 1]*t
        )
    )


    return float(
        np.real(
            phase
            @ rho
            @ np.conjugate(
                phase
            )
        )
    )


# ============================================================
# Symmetry
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


def symmetry_errors(
    density,
):

    out = {}


    for name, (
        sn,
        tn
    ) in TRANSFORMS.items():

        transformed = density[
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
            density
            - transformed
        )


        out[name] = (
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
        )


    return out


def L1(
    a,
    b,
):

    return float(
        np.mean(
            np.abs(
                a-b
            )
        )
    )


# ============================================================
# Build references
# ============================================================

print(
    "=== LHA-12A1 "
    "EXACT BZ DENSITY REFERENCE ==="
)

print()


print(
    "Building K2=64, Nk=12 control..."
)

p64, rho64 = (
    bz_density_matrices(
        K2_CONTROL,
        NK_CONTROL,
    )
)


print(
    "Building K2=81, Nk=12 control..."
)

p81_12, rho81_12 = (
    bz_density_matrices(
        K2_PROD,
        NK_CONTROL,
    )
)


print(
    "Building K2=81, Nk=16 production..."
)

p81_16, rho81_16 = (
    bz_density_matrices(
        K2_PROD,
        NK_PROD,
    )
)


print(
    "Building Gamma-only comparison..."
)

pgamma, rhogamma = (
    bz_density_matrices(
        K2_PROD,
        1,
        gamma_only=True,
    )
)


print()


# ============================================================
# Main audit
# ============================================================

rows = []

prod_densities = []
gamma_densities = []


max_basis_L1 = 0.0
max_kgrid_L1 = 0.0

max_norm_error = 0.0
max_negative_violation = 0.0

max_matrix_density_mismatch = 0.0

max_sym_L1 = 0.0
max_sym_Linf = 0.0

max_imag_residual = 0.0


print(
    " beta    basis-L1      kgrid-L1     "
    "Gamma/BZ-L1    <V>_BZ       "
    "Pmin       Psaddle      Pmax"
)

print(
    "------------------------------------------------"
    "---------------------------------------------"
)


for beta in BETAS:

    d64, _, _ = density_from_rho(
        p64[
            "nvec"
        ],
        rho64[
            beta
        ],
    )


    d81_12, _, _ = density_from_rho(
        p81_12[
            "nvec"
        ],
        rho81_12[
            beta
        ],
    )


    dprod, imag_residual, coeff = (
        density_from_rho(
            p81_16[
                "nvec"
            ],
            rho81_16[
                beta
            ],
        )
    )


    dgamma, _, _ = density_from_rho(
        pgamma[
            "nvec"
        ],
        rhogamma[
            beta
        ],
    )


    basis_L1 = L1(
        d81_12,
        d64,
    )


    kgrid_L1 = L1(
        dprod,
        d81_12,
    )


    gamma_BZ_L1 = L1(
        dprod,
        dgamma,
    )


    norm_error = abs(
        np.mean(
            dprod
        )
        - 1.0
    )


    min_density = float(
        np.min(
            dprod
        )
    )


    negative_violation = max(
        0.0,
        -min_density,
    )


    V_density = float(
        np.mean(
            dprod
            * V_real
        )
    )


    V_matrix = float(
        np.sum(
            rho81_16[
                beta
            ]
            * p81_16[
                "Vmat"
            ]
        )
    )


    V_mismatch = abs(
        V_density
        - V_matrix
    )


    Pmin = point_density_from_rho(
        p81_16[
            "nvec"
        ],
        rho81_16[
            beta
        ],
        *POINTS[
            "minimum"
        ],
    )


    Psaddle = point_density_from_rho(
        p81_16[
            "nvec"
        ],
        rho81_16[
            beta
        ],
        *POINTS[
            "saddle"
        ],
    )


    Pmax = point_density_from_rho(
        p81_16[
            "nvec"
        ],
        rho81_16[
            beta
        ],
        *POINTS[
            "maximum"
        ],
    )


    syms = symmetry_errors(
        dprod
    )


    sym_L1 = max(
        x[0]
        for x in syms.values()
    )


    sym_Linf = max(
        x[1]
        for x in syms.values()
    )


    max_basis_L1 = max(
        max_basis_L1,
        basis_L1,
    )


    max_kgrid_L1 = max(
        max_kgrid_L1,
        kgrid_L1,
    )


    max_norm_error = max(
        max_norm_error,
        norm_error,
    )


    max_negative_violation = max(
        max_negative_violation,
        negative_violation,
    )


    max_matrix_density_mismatch = max(
        max_matrix_density_mismatch,
        V_mismatch,
    )


    max_sym_L1 = max(
        max_sym_L1,
        sym_L1,
    )


    max_sym_Linf = max(
        max_sym_Linf,
        sym_Linf,
    )


    max_imag_residual = max(
        max_imag_residual,
        imag_residual,
    )


    print(
        f"{beta:5.2f}  "
        f"{basis_L1:12.4e}  "
        f"{kgrid_L1:12.4e}  "
        f"{gamma_BZ_L1:12.4e}  "
        f"{V_density:10.7f}  "
        f"{Pmin:10.6f}  "
        f"{Psaddle:10.6f}  "
        f"{Pmax:10.6f}"
    )


    rows.append([
        beta,

        basis_L1,
        kgrid_L1,
        gamma_BZ_L1,

        V_matrix,
        V_density,
        V_mismatch,

        Pmin,
        Psaddle,
        Pmax,

        norm_error,
        min_density,

        sym_L1,
        sym_Linf,

        imag_residual,
    ])


    prod_densities.append(
        dprod
    )


    gamma_densities.append(
        dgamma
    )


# ============================================================
# Structural PASS / FAIL
# ============================================================

checks = {
    "FULL-DENSITY BASIS CONVERGENCE":
        max_basis_L1
        < 1.0e-5,

    "FULL-DENSITY k-GRID CONVERGENCE":
        max_kgrid_L1
        < 1.0e-8,

    "DENSITY NORMALIZATION":
        max_norm_error
        < 1.0e-12,

    "DENSITY POSITIVITY":
        max_negative_violation
        < 1.0e-14,

    "MATRIX/DENSITY <V>":
        max_matrix_density_mismatch
        < 1.0e-10,

    "LATTICE SYMMETRY L1":
        max_sym_L1
        < 1.0e-8,

    "LATTICE SYMMETRY Linf":
        max_sym_Linf
        < 1.0e-7,

    "FOURIER IMAGINARY RESIDUAL":
        max_imag_residual
        < 1.0e-12,
}


print()

print(
    "=== STRUCTURAL AUDIT ==="
)


print(
    "max basis density L1       = "
    f"{max_basis_L1:.6e}"
)

print(
    "max k-grid density L1      = "
    f"{max_kgrid_L1:.6e}"
)

print(
    "max normalization error    = "
    f"{max_norm_error:.6e}"
)

print(
    "max negative violation     = "
    f"{max_negative_violation:.6e}"
)

print(
    "max <V> matrix/density err = "
    f"{max_matrix_density_mismatch:.6e}"
)

print(
    "max symmetry L1            = "
    f"{max_sym_L1:.6e}"
)

print(
    "max symmetry Linf          = "
    f"{max_sym_Linf:.6e}"
)

print(
    "max imaginary residual     = "
    f"{max_imag_residual:.6e}"
)

print()


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
    "lha12a1_exact_bz_density_reference.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "beta",

        "density_L1_K64_vs_K81_at_Nk12",
        "density_L1_Nk12_vs_Nk16_at_K81",
        "density_L1_Gamma_vs_BZ",

        "V_matrix_BZ",
        "V_density_BZ",
        "V_matrix_density_mismatch",

        "Pmin_BZ",
        "Psaddle_BZ",
        "Pmax_BZ",

        "density_norm_error",
        "density_min",

        "symmetry_L1_max",
        "symmetry_Linf_max",

        "fourier_imaginary_residual",
    ])

    writer.writerows(
        rows
    )


np.savez_compressed(
    "lha12a1_exact_bz_reference_K2_81_Nk16.npz",

    betas=BETAS,

    axis=axis,
    S=S,
    T=T,

    V=V_real,

    densities_BZ=np.stack(
        prod_densities,
        axis=0,
    ),

    densities_Gamma=np.stack(
        gamma_densities,
        axis=0,
    ),

    K2_max=K2_PROD,
    Nk=NK_PROD,

    G1=G1,
    G2=G2,
)


print()

print(
    "saved: "
    "lha12a1_exact_bz_density_reference.csv"
)

print(
    "saved: "
    "lha12a1_exact_bz_reference_K2_81_Nk16.npz"
)
