import csv
import numpy as np


# ============================================================
# LHA-11B1
#
# Symmetry-preserving plane-wave cutoff convergence for
# the triangular periodic potential
#
# V(r) = 3
#        - cos(G1.r)
#        - cos(G2.r)
#        - cos(G3.r)
#
# Units:
#   hbar = 1
#   M    = 1
#   V0   = 1
#   |G|  = 1
#
# Basis cutoff:
#
# |K|^2 = n1^2 + n2^2 - n1*n2 <= K2_MAX
#
# This preserves reciprocal-lattice symmetry much better
# than a rectangular |n1|,|n2| cutoff.
# ============================================================


V0 = 1.0

BETAS = [
    0.5,
    1.0,
    2.0,
    2.5,
]

K2_CUTOFFS = [
    4,
    9,
    16,
    25,
    36,
    49,
    64,
    81,
]

NGRID = 64

N_LOW = 10


# ============================================================
# Fourier coefficients
# ============================================================

V_FOURIER = {
    (0, 0): 3.0 * V0,

    (1, 0): -0.5 * V0,
    (-1, 0): -0.5 * V0,

    (0, 1): -0.5 * V0,
    (0, -1): -0.5 * V0,

    (1, 1): -0.5 * V0,
    (-1, -1): -0.5 * V0,
}


# ============================================================
# Reciprocal metric
#
# G1.G1 = 1
# G2.G2 = 1
# G1.G2 = -1/2
#
# Therefore:
#
# |n1 G1 + n2 G2|^2
# = n1^2 + n2^2 - n1*n2
# ============================================================

def K2_of(n1, n2):

    return (
        n1*n1
        + n2*n2
        - n1*n2
    )


def build_basis(K2_max):

    # Generous integer search range.
    R = (
        int(
            np.ceil(
                2.0*np.sqrt(K2_max)
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

    # Put reciprocal shells in deterministic order.
    basis.sort(
        key=lambda z: (
            z[2],
            z[0],
            z[1],
        )
    )

    return basis


# ============================================================
# Hamiltonian
# ============================================================

def build_hamiltonian(
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

    T = np.zeros(
        (
            nbasis,
            nbasis,
        ),
        dtype=float,
    )

    Vmat = np.zeros_like(
        T
    )

    for i, (
        n1,
        n2,
        K2,
    ) in enumerate(
        basis
    ):

        T[i, i] = (
            0.5*K2
        )

        for (
            dn1,
            dn2
        ), coeff in (
            V_FOURIER.items()
        ):

            # V_(K_i-K_j):
            #
            # (n1_i-n1_j,
            #  n2_i-n2_j)
            # = (dn1,dn2)

            j_key = (
                n1-dn1,
                n2-dn2,
            )

            j = index.get(
                j_key
            )

            if j is not None:

                Vmat[i, j] = (
                    coeff
                )

    H = (
        T
        + Vmat
    )

    return (
        basis,
        H,
        T,
        Vmat,
    )


# ============================================================
# Reduced-coordinate real-space grid
#
# r = s a1 + t a2
#
# G1.r = 2*pi*s
# G2.r = 2*pi*t
# G3.r = -2*pi*(s+t)
#
# Plane wave:
#
# exp[i K.r]
# =
# exp[2*pi*i(n1*s+n2*t)]
#
# The reduced density q(s,t) integrates to 1 over
# the unit square 0<=s,t<1.
# ============================================================

axis = (
    np.arange(
        NGRID,
        dtype=float,
    )
    / NGRID
)

S, TT = np.meshgrid(
    axis,
    axis,
    indexing="xy",
)

coords = np.column_stack([
    S.ravel(),
    TT.ravel(),
])


V_real_grid = (
    3.0
    - np.cos(
        2.0*np.pi*S
    )
    - np.cos(
        2.0*np.pi*TT
    )
    - np.cos(
        2.0*np.pi*(S+TT)
    )
)


# ============================================================
# Storage
# ============================================================

rows = []

solutions = {}

previous = None


print(
    "=== LHA-11B1 "
    "PLANE-WAVE CUTOFF CONVERGENCE ==="
)

print()

print(
    "cutoff metric:"
)

print(
    "|K|^2 = "
    "n1^2+n2^2-n1*n2"
)

print()


# ============================================================
# Scan cutoffs
# ============================================================

for K2_max in K2_CUTOFFS:

    (
        basis,
        H,
        Tmat,
        Vmat,
    ) = build_hamiltonian(
        K2_max
    )

    nbasis = len(
        basis
    )

    herm_error = np.max(
        np.abs(
            H-H.T
        )
    )

    E, C = np.linalg.eigh(
        H
    )

    gap = (
        E[1]-E[0]
    )

    # --------------------------------------------------------
    # Potential expectation in every eigenstate
    #
    # <n|V|n>
    # --------------------------------------------------------

    VC = (
        Vmat @ C
    )

    V_eigen = np.sum(
        C*VC,
        axis=0,
    )

    # --------------------------------------------------------
    # Real-space eigenfunctions in reduced coordinates
    #
    # phi_K(s,t)
    # =
    # exp[2*pi*i(n1*s+n2*t)]
    #
    # These are normalized on the unit reduced cell.
    # --------------------------------------------------------

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

    phases = np.exp(
        2.0j*np.pi
        * (
            coords @ nvec.T
        )
    )

    psi = (
        phases @ C
    )

    abspsi2 = (
        np.abs(psi)**2
    )

    current_densities = {}

    print(
        "========================================"
    )

    print(
        f"K2_MAX = {K2_max:3d}"
    )

    print(
        f"NBASIS = {nbasis:4d}"
    )

    print(
        f"E0     = {E[0]:.12f}"
    )

    print(
        f"gap    = {gap:.12f}"
    )

    print(
        f"Hermiticity = "
        f"{herm_error:.3e}"
    )

    if previous is not None:

        ncompare = min(
            N_LOW,
            len(E),
            len(
                previous["E"]
            ),
        )

        lowE_delta = np.max(
            np.abs(
                E[:ncompare]
                - previous["E"][
                    :ncompare
                ]
            )
        )

        print(
            "max delta first "
            f"{ncompare} energies = "
            f"{lowE_delta:.3e}"
        )

    else:

        lowE_delta = np.nan

    print()

    print(
        " beta    <V> matrix       "
        "<V> density      "
        "norm(raw)       "
        "L1 vs prev"
    )

    print(
        "------------------------------------------------"
        "----------------"
    )

    for beta in BETAS:

        w = np.exp(
            -beta
            * (
                E-E[0]
            )
        )

        probs = (
            w / np.sum(w)
        )

        V_matrix = float(
            probs @ V_eigen
        )

        density_raw = (
            abspsi2
            @ probs
        )

        norm_raw = float(
            np.mean(
                density_raw
            )
        )

        density = (
            density_raw
            / norm_raw
        )

        density = density.reshape(
            (
                NGRID,
                NGRID,
            )
        )

        V_density = float(
            np.mean(
                density
                * V_real_grid
            )
        )

        density_V_difference = (
            V_density
            - V_matrix
        )

        if previous is not None:

            prev_density = (
                previous[
                    "densities"
                ][beta]
            )

            L1_prev = float(
                np.mean(
                    np.abs(
                        density
                        - prev_density
                    )
                )
            )

            V_prev_delta = (
                V_matrix
                - previous[
                    "V_matrix"
                ][beta]
            )

        else:

            L1_prev = np.nan
            V_prev_delta = np.nan

        print(
            f"{beta:5.2f}  "
            f"{V_matrix:14.10f}  "
            f"{V_density:14.10f}  "
            f"{norm_raw:12.9f}  "
            f"{L1_prev:10.3e}"
        )

        rows.append([
            K2_max,
            nbasis,
            beta,
            E[0],
            gap,
            herm_error,
            lowE_delta,
            V_matrix,
            V_density,
            density_V_difference,
            norm_raw,
            V_prev_delta,
            L1_prev,
        ])

        current_densities[
            beta
        ] = density.copy()

    print()

    solutions[
        K2_max
    ] = {
        "basis": basis,
        "E": E.copy(),
        "C": C.copy(),
        "V_matrix": {
            beta: next(
                r[7]
                for r in reversed(rows)
                if (
                    r[0] == K2_max
                    and
                    r[2] == beta
                )
            )
            for beta in BETAS
        },
        "densities": {
            beta:
            current_densities[
                beta
            ].copy()
            for beta in BETAS
        },
    }

    previous = {
        "E": E.copy(),
        "V_matrix": solutions[
            K2_max
        ]["V_matrix"],
        "densities": current_densities,
    }


# ============================================================
# Final convergence audit:
#
# compare K2=64 vs K2=81
# ============================================================

prev_cut = K2_CUTOFFS[-2]
last_cut = K2_CUTOFFS[-1]

prev_sol = solutions[
    prev_cut
]

last_sol = solutions[
    last_cut
]


ncompare = min(
    N_LOW,
    len(
        prev_sol["E"]
    ),
    len(
        last_sol["E"]
    ),
)

final_energy_delta = float(
    np.max(
        np.abs(
            last_sol["E"][
                :ncompare
            ]
            - prev_sol["E"][
                :ncompare
            ]
        )
    )
)


final_V_deltas = []

final_L1_deltas = []

for beta in BETAS:

    final_V_deltas.append(
        abs(
            last_sol[
                "V_matrix"
            ][beta]
            - prev_sol[
                "V_matrix"
            ][beta]
        )
    )

    final_L1_deltas.append(
        float(
            np.mean(
                np.abs(
                    last_sol[
                        "densities"
                    ][beta]
                    - prev_sol[
                        "densities"
                    ][beta]
                )
            )
        )
    )


max_final_V_delta = max(
    final_V_deltas
)

max_final_L1_delta = max(
    final_L1_deltas
)


# Check density/matrix potential consistency
# at the largest cutoff.

last_rows = [
    r
    for r in rows
    if r[0] == last_cut
]

max_V_consistency_error = max(
    abs(
        r[9]
    )
    for r in last_rows
)

max_norm_error = max(
    abs(
        r[10]-1.0
    )
    for r in last_rows
)


# ============================================================
# PASS criteria
# ============================================================

energy_pass = (
    final_energy_delta
    < 1.0e-8
)

thermal_V_pass = (
    max_final_V_delta
    < 1.0e-5
)

density_pass = (
    max_final_L1_delta
    < 1.0e-5
)

consistency_pass = (
    max_V_consistency_error
    < 1.0e-10
)

normalization_pass = (
    max_norm_error
    < 1.0e-10
)


print(
    "=== FINAL CUTOFF AUDIT ==="
)

print(
    f"compare K2={prev_cut} "
    f"vs K2={last_cut}"
)

print()

print(
    "max delta first "
    f"{ncompare} energies = "
    f"{final_energy_delta:.6e}"
)

print(
    "max delta <V>       = "
    f"{max_final_V_delta:.6e}"
)

print(
    "max density L1      = "
    f"{max_final_L1_delta:.6e}"
)

print(
    "max matrix-density "
    "<V> mismatch       = "
    f"{max_V_consistency_error:.6e}"
)

print(
    "max raw norm error  = "
    f"{max_norm_error:.6e}"
)

print()


checks = {
    "LOW-ENERGY SPECTRUM":
        energy_pass,

    "THERMAL <V>":
        thermal_V_pass,

    "THERMAL DENSITY":
        density_pass,

    "MATRIX/DENSITY CONSISTENCY":
        consistency_pass,

    "DENSITY NORMALIZATION":
        normalization_pass,
}


for name, passed in (
    checks.items()
):

    print(
        f"{name:30s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


overall = all(
    checks.values()
)

print()

print(
    "OVERALL:",
    "PASS"
    if overall
    else "FAIL"
)


# ============================================================
# Save CSV
# ============================================================

with open(
    "lha11b1_plane_wave_cutoff_convergence.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "K2_MAX",
        "NBASIS",
        "beta",
        "E0",
        "gap",
        "hermiticity_error",
        "max_low_energy_delta_vs_prev",
        "V_matrix",
        "V_density",
        "V_density_minus_matrix",
        "density_raw_norm",
        "V_delta_vs_prev",
        "density_L1_vs_prev",
    ])

    writer.writerows(
        rows
    )


# ============================================================
# Save highest-cutoff exact reference
# for LHA-11B2
# ============================================================

final_basis = np.array([
    [
        n1,
        n2,
        K2,
    ]
    for (
        n1,
        n2,
        K2
    ) in last_sol[
        "basis"
    ]
], dtype=float)


density_stack = np.stack([
    last_sol[
        "densities"
    ][beta]
    for beta in BETAS
], axis=0)


np.savez_compressed(
    "lha11b1_exact_periodic_reference_K2_81.npz",
    betas=np.array(
        BETAS,
        dtype=float,
    ),
    axis=axis,
    S=S,
    T=TT,
    V_real=V_real_grid,
    basis=final_basis,
    energies=last_sol[
        "E"
    ],
    densities=density_stack,
)


print()

print(
    "saved: "
    "lha11b1_plane_wave_cutoff_convergence.csv"
)

print(
    "saved: "
    "lha11b1_exact_periodic_reference_K2_81.npz"
)
