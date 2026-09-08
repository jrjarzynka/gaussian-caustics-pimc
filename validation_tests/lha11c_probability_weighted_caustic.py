import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LHA-11C
#
# Exact-quantum probability carried by the locally
# Gaussian-invalid region of the triangular periodic landscape.
#
# NO RLH regularization is introduced.
#
# We measure:
#
#   f_invalid
#     = geometric fraction of primitive cell with zeta >= 1
#
#   p_invalid
#     = exact quantum probability in that region
#
# and separate invalid regions by Hessian signature:
#
#   indefinite      -> saddle-like
#   negative-definite -> maximum-like
#
# Units:
#   hbar = M = V0 = |G| = 1
# ============================================================


V0 = 1.0

K2_REFERENCE = 81
K2_CONTROL = 64

# Divisible by both 2 and 3:
# saddles and maxima lie exactly on the grid.
NGRID = 192

CHUNK = 2048


BETAS = np.array([
    2.5000,
    2.5500,
    2.5600,

    # Bracket first saddle caustic
    2.5650,
    2.5652,
    2.5700,

    2.6000,
    2.7000,
    2.8000,
    3.0000,
    3.3000,
    3.6000,

    # Bracket high-symmetry maximum caustic
    3.6275,
    3.6277,

    3.8000,
    4.0000,
    4.5000,
    5.0000,
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

G3 = -(G1+G2)

GS = np.array([
    G1,
    G2,
    G3,
])


# ============================================================
# Analytic high-symmetry caustics
# ============================================================

KAPPA_SADDLE_NEG = -1.5
KAPPA_MAXIMUM = -0.75


BETA_C_SADDLE = (
    np.pi
    / np.sqrt(
        -KAPPA_SADDLE_NEG
    )
)


BETA_C_MAXIMUM = (
    np.pi
    / np.sqrt(
        -KAPPA_MAXIMUM
    )
)


print(
    "=== LHA-11C "
    "PROBABILITY-WEIGHTED CAUSTIC AUDIT ==="
)

print()

print(
    f"beta_c saddle  = "
    f"{BETA_C_SADDLE:.12f}"
)

print(
    f"beta_c maximum = "
    f"{BETA_C_MAXIMUM:.12f}"
)

print()


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
# Plane-wave exact quantum reference
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


    H = (
        Tmat
        + Vmat
    )


    herm = float(
        np.max(
            np.abs(
                H-H.T
            )
        )
    )


    E, C = np.linalg.eigh(
        H
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
        "E": E,
        "C": C,
        "V_eigen": V_eigen,
        "hermiticity": herm,
    }


# ============================================================
# Fine real-space reduced-coordinate grid
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


# ============================================================
# Periodic potential and Hessian
# ============================================================

phase = np.stack([
    2.0*np.pi*S,
    2.0*np.pi*T,
    -2.0*np.pi*(S+T),
], axis=0)


cosp = np.cos(
    phase
)


V = (
    3.0
    - np.sum(
        cosp,
        axis=0,
    )
)


Hxx = sum(
    cosp[i]
    * GS[i, 0]**2
    for i in range(3)
)

Hyy = sum(
    cosp[i]
    * GS[i, 1]**2
    for i in range(3)
)

Hxy = sum(
    cosp[i]
    * GS[i, 0]
    * GS[i, 1]
    for i in range(3)
)


disc = np.sqrt(
    (Hxx-Hyy)**2
    + 4.0*Hxy**2
)


kappa_min = (
    0.5
    * (
        Hxx+Hyy-disc
    )
)

kappa_max = (
    0.5
    * (
        Hxx+Hyy+disc
    )
)


# Hessian-signature regions

negative_curvature_mask = (
    kappa_min < 0.0
)

indefinite_base_mask = (
    (kappa_min < 0.0)
    &
    (kappa_max > 0.0)
)

negative_definite_base_mask = (
    kappa_max < 0.0
)


# ============================================================
# Build exact eigenstate densities on the fine grid
# ============================================================

def prepare_abspsi2(
    solution,
):

    basis = solution[
        "basis"
    ]

    C = solution[
        "C"
    ]

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


    npoints = len(
        coords
    )

    nstates = C.shape[1]


    abspsi2 = np.empty(
        (
            npoints,
            nstates,
        ),
        dtype=np.float64,
    )


    for start in range(
        0,
        npoints,
        CHUNK,
    ):

        stop = min(
            start+CHUNK,
            npoints,
        )


        phase_chunk = np.exp(
            2.0j*np.pi
            * (
                coords[
                    start:stop
                ]
                @ nvec.T
            )
        )


        psi_chunk = (
            phase_chunk
            @ C
        )


        abspsi2[
            start:stop
        ] = (
            np.abs(
                psi_chunk
            )**2
        )


    return abspsi2


print(
    "Solving exact PW control K2=64..."
)

sol64 = solve_pw(
    K2_CONTROL
)

print(
    "Solving exact PW reference K2=81..."
)

sol81 = solve_pw(
    K2_REFERENCE
)

print()

print(
    f"K2=64 basis = "
    f"{len(sol64['basis'])}"
)

print(
    f"K2=81 basis = "
    f"{len(sol81['basis'])}"
)

print(
    f"Hermiticity 64 = "
    f"{sol64['hermiticity']:.3e}"
)

print(
    f"Hermiticity 81 = "
    f"{sol81['hermiticity']:.3e}"
)

print()


print(
    "Preparing real-space eigenstate densities..."
)

abspsi2_64 = prepare_abspsi2(
    sol64
)

abspsi2_81 = prepare_abspsi2(
    sol81
)

print(
    "Real-space exact reference prepared."
)

print()


# ============================================================
# Exact thermal density
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


    density_flat = (
        abspsi2
        @ probs
    )


    raw_norm = float(
        np.mean(
            density_flat
        )
    )


    density_flat = (
        density_flat
        / raw_norm
    )


    density = density_flat.reshape(
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
            density*V
        )
    )


    return (
        density,
        raw_norm,
        V_matrix,
        V_density,
    )


# ============================================================
# Integration helpers
# ============================================================

def probability_mass(
    density,
    mask,
):

    return float(
        np.mean(
            density
            * mask
        )
    )


def geometric_fraction(
    mask,
):

    return float(
        np.mean(
            mask
        )
    )


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


# ============================================================
# Main scan
# ============================================================

rows = []


max_cutoff_L1 = 0.0
max_cutoff_invalid_mass_delta = 0.0
max_V_consistency = 0.0
max_norm_error = 0.0


print(
    "=== CAUSTIC CROSSING SCAN ==="
)

print()

print(
    " beta    zeta_max   "
    "invalid area   exact invalid mass   "
    "mass/area"
)

print(
    "------------------------------------------------"
    "----------------"
)


for beta in BETAS:

    (
        density64,
        norm64,
        V64,
        V64_density,
    ) = exact_density(
        sol64,
        abspsi2_64,
        beta,
    )


    (
        density81,
        norm81,
        V81,
        V81_density,
    ) = exact_density(
        sol81,
        abspsi2_81,
        beta,
    )


    # --------------------------------------------------------
    # Local Gaussian instability map
    # --------------------------------------------------------

    zeta = (
        beta/np.pi
        * np.sqrt(
            np.maximum(
                0.0,
                -kappa_min,
            )
        )
    )


    invalid = (
        zeta >= 1.0
    )


    invalid_indefinite = (
        invalid
        &
        indefinite_base_mask
    )


    invalid_negative_definite = (
        invalid
        &
        negative_definite_base_mask
    )


    # --------------------------------------------------------
    # Geometric fractions
    # --------------------------------------------------------

    invalid_area = (
        geometric_fraction(
            invalid
        )
    )


    indefinite_area = (
        geometric_fraction(
            invalid_indefinite
        )
    )


    negative_definite_area = (
        geometric_fraction(
            invalid_negative_definite
        )
    )


    # --------------------------------------------------------
    # Exact quantum probability masses
    # --------------------------------------------------------

    invalid_mass81 = (
        probability_mass(
            density81,
            invalid,
        )
    )


    invalid_mass64 = (
        probability_mass(
            density64,
            invalid,
        )
    )


    indefinite_mass = (
        probability_mass(
            density81,
            invalid_indefinite,
        )
    )


    negative_definite_mass = (
        probability_mass(
            density81,
            invalid_negative_definite,
        )
    )


    negative_curvature_mass = (
        probability_mass(
            density81,
            negative_curvature_mask,
        )
    )


    # Mean exact density in invalid region,
    # relative to uniform density (=1).

    if invalid_area > 0.0:

        enrichment = (
            invalid_mass81
            / invalid_area
        )

    else:

        enrichment = np.nan


    # --------------------------------------------------------
    # Exact reference convergence
    # --------------------------------------------------------

    cutoff_L1 = L1(
        density81,
        density64,
    )


    invalid_mass_delta = abs(
        invalid_mass81
        - invalid_mass64
    )


    V_consistency = abs(
        V81
        - V81_density
    )


    norm_error = abs(
        norm81-1.0
    )


    max_cutoff_L1 = max(
        max_cutoff_L1,
        cutoff_L1,
    )


    max_cutoff_invalid_mass_delta = max(
        max_cutoff_invalid_mass_delta,
        invalid_mass_delta,
    )


    max_V_consistency = max(
        max_V_consistency,
        V_consistency,
    )


    max_norm_error = max(
        max_norm_error,
        norm_error,
    )


    zeta_global = float(
        np.max(
            zeta
        )
    )


    zeta_saddle = (
        beta*np.sqrt(1.5)
        / np.pi
    )


    zeta_maximum = (
        beta*np.sqrt(0.75)
        / np.pi
    )


    print(
        f"{beta:7.4f}  "
        f"{zeta_global:8.5f}  "
        f"{100*invalid_area:10.4f}%  "
        f"{100*invalid_mass81:14.6f}%  "
        f"{enrichment:9.5f}"
    )


    rows.append([
        beta,

        zeta_global,
        zeta_saddle,
        zeta_maximum,

        invalid_area,
        invalid_mass81,
        enrichment,

        indefinite_area,
        indefinite_mass,

        negative_definite_area,
        negative_definite_mass,

        geometric_fraction(
            negative_curvature_mask
        ),

        negative_curvature_mass,

        V81,

        cutoff_L1,
        invalid_mass_delta,
        V_consistency,
        norm_error,
    ])


# ============================================================
# Threshold audits
# ============================================================

arr = np.array(
    rows,
    dtype=float,
)


beta_arr = arr[:, 0]
invalid_area_arr = arr[:, 4]
invalid_mass_arr = arr[:, 5]


below_saddle = (
    beta_arr < BETA_C_SADDLE
)

above_saddle = (
    beta_arr > BETA_C_SADDLE
)


saddle_onset_pass = (
    np.all(
        invalid_area_arr[
            below_saddle
        ] == 0.0
    )
    and
    np.any(
        invalid_area_arr[
            above_saddle
        ] > 0.0
    )
)


# Exact reference convergence

cutoff_pass = (
    max_cutoff_L1
    < 1.0e-5
)

mass_convergence_pass = (
    max_cutoff_invalid_mass_delta
    < 1.0e-5
)

matrix_density_pass = (
    max_V_consistency
    < 1.0e-10
)

normalization_pass = (
    max_norm_error
    < 1.0e-10
)


print()

print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    "max exact density L1 "
    "(K2=64 vs 81) = "
    f"{max_cutoff_L1:.6e}"
)

print(
    "max invalid-mass delta "
    "(K2=64 vs 81) = "
    f"{max_cutoff_invalid_mass_delta:.6e}"
)

print(
    "max exact matrix-density "
    "<V> mismatch = "
    f"{max_V_consistency:.6e}"
)

print(
    "max exact raw norm error = "
    f"{max_norm_error:.6e}"
)

print()


checks = {
    "FIRST CAUSTIC ONSET":
        saddle_onset_pass,

    "EXACT DENSITY CUTOFF":
        cutoff_pass,

    "INVALID-MASS CUTOFF":
        mass_convergence_pass,

    "EXACT MATRIX/DENSITY":
        matrix_density_pass,

    "EXACT NORMALIZATION":
        normalization_pass,
}


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
# Save CSV
# ============================================================

with open(
    "lha11c_probability_weighted_caustic.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "beta",

        "zeta_global_max",
        "zeta_saddle",
        "zeta_highsym_maximum",

        "invalid_area_fraction",
        "exact_invalid_probability",
        "invalid_probability_over_area",

        "invalid_indefinite_area",
        "exact_invalid_indefinite_probability",

        "invalid_negative_definite_area",
        "exact_invalid_negative_definite_probability",

        "negative_curvature_area",
        "exact_negative_curvature_probability",

        "V_exact",

        "exact_density_L1_K64_vs_K81",
        "invalid_probability_delta_K64_vs_K81",
        "V_matrix_density_mismatch",
        "exact_raw_norm_error",
    ])

    writer.writerows(
        rows
    )


# ============================================================
# Figure 1
# Geometric invalid area vs exact probability mass
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.4)
)

ax.plot(
    beta_arr,
    100.0*arr[:, 4],
    marker="o",
    label="invalid cell area",
)

ax.plot(
    beta_arr,
    100.0*arr[:, 5],
    marker="o",
    label="exact QM probability in invalid region",
)

ax.axvline(
    BETA_C_SADDLE,
    linestyle="--",
    label="saddle caustic",
)

ax.axvline(
    BETA_C_MAXIMUM,
    linestyle=":",
    label="maximum-point caustic",
)

ax.set_xlabel(
    r"$\beta$"
)

ax.set_ylabel(
    "fraction (%)"
)

ax.set_title(
    "Geometric vs probability-weighted Gaussian breakdown"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "lha11c_invalid_area_vs_probability.png",
    dpi=300,
)

fig.savefig(
    "lha11c_invalid_area_vs_probability.pdf",
)

plt.close(fig)


# ============================================================
# Figure 2
# Mean exact density inside invalid region
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.4)
)

mask_nonzero = (
    arr[:, 4] > 0.0
)

ax.plot(
    beta_arr[
        mask_nonzero
    ],
    arr[
        mask_nonzero,
        6
    ],
    marker="o",
)

ax.axhline(
    1.0,
    linestyle="--",
)

ax.axvline(
    BETA_C_SADDLE,
    linestyle="--",
)

ax.set_xlabel(
    r"$\beta$"
)

ax.set_ylabel(
    r"$p_{\rm invalid}/f_{\rm invalid}$"
)

ax.set_title(
    "Quantum occupation of the Gaussian-invalid region"
)

fig.tight_layout()

fig.savefig(
    "lha11c_invalid_region_enrichment.png",
    dpi=300,
)

fig.savefig(
    "lha11c_invalid_region_enrichment.pdf",
)

plt.close(fig)


# ============================================================
# Figure 3
# Hessian-signature-resolved invalid probability
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.4)
)

ax.plot(
    beta_arr,
    100.0*arr[:, 8],
    marker="o",
    label="indefinite / saddle-like",
)

ax.plot(
    beta_arr,
    100.0*arr[:, 10],
    marker="o",
    label="negative-definite / maximum-like",
)

ax.axvline(
    BETA_C_SADDLE,
    linestyle="--",
)

ax.axvline(
    BETA_C_MAXIMUM,
    linestyle=":",
)

ax.set_xlabel(
    r"$\beta$"
)

ax.set_ylabel(
    "exact quantum probability (%)"
)

ax.set_title(
    "Where the Gaussian-invalid probability resides"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "lha11c_signature_resolved_invalid_probability.png",
    dpi=300,
)

fig.savefig(
    "lha11c_signature_resolved_invalid_probability.pdf",
)

plt.close(fig)


print()

print(
    "saved: "
    "lha11c_probability_weighted_caustic.csv"
)

print(
    "saved: "
    "lha11c_invalid_area_vs_probability.png/pdf"
)

print(
    "saved: "
    "lha11c_invalid_region_enrichment.png/pdf"
)

print(
    "saved: "
    "lha11c_signature_resolved_invalid_probability.png/pdf"
)
