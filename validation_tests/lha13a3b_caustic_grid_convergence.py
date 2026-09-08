import csv
import numpy as np
from scipy.optimize import minimize


# ============================================================
# LHA-13A3b
#
# Real-space grid convergence of the probability-weighted
# off-stationary caustic.
#
# NO RLH.
# NO PI-QMC.
#
# Exact BZ density is represented through its reciprocal-space
# Fourier coefficients, allowing cheap evaluation on
#
#   128^2, 256^2, 512^2
#
# grids after the BZ density matrix has been constructed once.
#
# The near-onset beta=2.712 point is intentionally NOT used
# as a strict geometric-convergence gate because the invalid
# set is initially sub-pixel/tiny.
# ============================================================


BETAS = np.array([
    2.712,
    2.72,
    2.80,
    3.00,
])


NGRIDS = [
    128,
    256,
    512,
]


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
    r
    for r in stationary
    if r["type"] == "minimum"
][0]


s0 = float(
    minimum["s"]
)

t0 = float(
    minimum["t"]
)


theta_min = np.array([
    2.0*np.pi*s0 + PHASES[0],
    2.0*np.pi*t0 + PHASES[1],
    -2.0*np.pi*(s0+t0) + PHASES[2],
])


Umin = -float(
    np.sum(
        AMPLITUDES
        * np.cos(theta_min)
    )
)


V0 = -Umin


# ============================================================
# Potential Fourier coefficients
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

            k2 = K2_of(
                n1,
                n2,
            )


            if k2 <= K2max:

                basis.append(
                    (
                        n1,
                        n2,
                        k2,
                    )
                )


    basis.sort(
        key=lambda x: (
            x[2],
            x[0],
            x[1],
        )
    )


    return basis


basis = build_basis(
    K2
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


NBASIS = len(
    basis
)


Gcart = (
    nvec[:, 0, None]
    * G1[None, :]
    +
    nvec[:, 1, None]
    * G2[None, :]
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
        NBASIS,
        NBASIS,
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


hermiticity = float(
    np.max(
        np.abs(
            Vmat
            - Vmat.conj().T
        )
    )
)


# ============================================================
# Continuous softest curvature
# ============================================================


def hessian_uv(
    uv,
):

    s, t = np.mod(
        np.asarray(
            uv,
            dtype=float,
        ),
        1.0,
    )


    theta = np.array([
        2.0*np.pi*s
        + PHASES[0],

        2.0*np.pi*t
        + PHASES[1],

        -2.0*np.pi*(s+t)
        + PHASES[2],
    ])


    H = np.einsum(
        "a,ai,aj->ij",
        AMPLITUDES
        * np.cos(theta),
        GS,
        GS,
    )


    return H


def soft_curvature(
    uv,
):

    return float(
        np.linalg.eigvalsh(
            hessian_uv(uv)
        )[0]
    )


# Initial point from LHA-13A1.
a1 = np.load(
    "lha13a1_exact_bz_reference.npz"
)


x0 = np.asarray(
    a1[
        "caustic_point"
    ],
    dtype=float,
)


opt = minimize(

    soft_curvature,

    x0=x0,

    method="Nelder-Mead",

    options={
        "xatol": 1.0e-13,
        "fatol": 1.0e-13,
        "maxiter": 10000,
    },
)


caustic_uv_cont = np.mod(
    opt.x,
    1.0,
)


kappa_min_cont = float(
    soft_curvature(
        caustic_uv_cont
    )
)


beta_c_cont = (
    np.pi
    / np.sqrt(
        -kappa_min_cont
    )
)


# ============================================================
# Construct exact BZ density matrices.
#
# We accumulate all requested beta values during one BZ loop.
# No large list of eigenvector matrices is retained.
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


rhos = [
    np.zeros(
        (
            NBASIS,
            NBASIS,
        ),
        dtype=complex,
    )

    for _ in BETAS
]


Z = np.zeros(
    len(BETAS),
    dtype=float,
)


print(
    "=== LHA-13A3b "
    "CAUSTIC GRID CONVERGENCE ==="
)

print()

print(
    f"NBASIS             = "
    f"{NBASIS}"
)

print(
    f"Nk                 = "
    f"{NK} x {NK}"
)

print(
    f"Hermiticity        = "
    f"{hermiticity:.6e}"
)

print()

print(
    "=== CONTINUOUS CAUSTIC REFINEMENT ==="
)

print(
    f"optimizer success  = "
    f"{opt.success}"
)

print(
    f"caustic s,t        = "
    f"({caustic_uv_cont[0]:.12f}, "
    f"{caustic_uv_cont[1]:.12f})"
)

print(
    f"kappa_min continuous = "
    f"{kappa_min_cont:+.12f}"
)

print(
    f"beta_c continuous    = "
    f"{beta_c_cont:.12f}"
)

print()


print(
    "Building exact BZ density matrices..."
)


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


        E, C = np.linalg.eigh(
            H
        )


        for ib, beta in enumerate(
            BETAS
        ):

            # Energies in this problem are modest and positive;
            # direct Boltzmann factors are numerically safe.

            w = np.exp(
                -beta*E
            )


            Z[ib] += np.sum(
                w
            )


            rhos[ib] += (
                (
                    C
                    * w[None, :]
                )
                @ C.conj().T
            )


for ib in range(
    len(BETAS)
):

    rhos[ib] /= Z[ib]


rho_hermiticity = max(
    float(
        np.max(
            np.abs(
                rho
                - rho.conj().T
            )
        )
    )

    for rho in rhos
)


rho_trace_error = max(
    abs(
        float(
            np.trace(
                rho
            ).real
        )
        - 1.0
    )

    for rho in rhos
)


print(
    "exact BZ density matrices built."
)

print(
    f"max rho Hermiticity = "
    f"{rho_hermiticity:.6e}"
)

print(
    f"max rho trace error = "
    f"{rho_trace_error:.6e}"
)

print()


# ============================================================
# rho_nm -> real-space Fourier density coefficients
#
# P(s,t)
# =
# sum_nm rho_nm
# exp[2pi i ((n1-m1)s + (n2-m2)t)]
#
# Therefore coefficient for d=n-m is
#
# c_d = sum_{n-m=d} rho_nm.
# ============================================================


diff = (
    nvec[:, None, :]
    - nvec[None, :, :]
)


d1 = diff[
    ...,
    0
]


d2 = diff[
    ...,
    1
]


DMAX = int(
    max(
        np.max(
            np.abs(d1)
        ),
        np.max(
            np.abs(d2)
        ),
    )
)


def density_fourier_coefficients(
    rho,
):

    coeff = np.zeros(
        (
            2*DMAX+1,
            2*DMAX+1,
        ),
        dtype=complex,
    )


    np.add.at(

        coeff,

        (
            (
                d2+DMAX
            ).ravel(),

            (
                d1+DMAX
            ).ravel(),
        ),

        rho.ravel(),
    )


    return coeff


density_coeffs = [
    density_fourier_coefficients(
        rho
    )

    for rho in rhos
]


# ============================================================
# Fourier density on arbitrary endpoint-excluded grid.
# ============================================================


def exact_density_on_grid(
    coeff,
    N,
):

    if N <= 2*DMAX:

        raise RuntimeError(
            f"N={N} too small for "
            f"Fourier support DMAX={DMAX}"
        )


    fft_coeff = np.zeros(
        (
            N,
            N,
        ),
        dtype=complex,
    )


    for i2 in range(
        2*DMAX+1
    ):

        n2 = i2-DMAX


        for i1 in range(
            2*DMAX+1
        ):

            n1 = i1-DMAX


            c = coeff[
                i2,
                i1
            ]


            if abs(c) < 1.0e-16:
                continue


            fft_coeff[
                n2 % N,
                n1 % N,
            ] += c


    # numpy ifft2 contains 1/N^2.
    P_complex = np.fft.ifft2(
        fft_coeff
        * (
            N*N
        )
    )


    imag_residual = float(
        np.max(
            np.abs(
                P_complex.imag
            )
        )
    )


    P_raw = P_complex.real


    raw_mean = float(
        np.mean(
            P_raw
        )
    )


    norm_error = abs(
        raw_mean-1.0
    )


    # Normalize only after recording the independent check.
    P = (
        P_raw/raw_mean
    )


    return (
        P,
        norm_error,
        imag_residual,
    )


# ============================================================
# Hessian / invalid masks on arbitrary grid
# ============================================================


def geometry_on_grid(
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


    H = np.einsum(
        "...a,ai,aj->...ij",
        AMPLITUDES
        * np.cos(theta),
        GS,
        GS,
    )


    eig = np.linalg.eigvalsh(
        H
    )


    kmin = eig[
        ...,
        0
    ]


    beta_c_grid = (
        np.pi
        / np.sqrt(
            -float(
                np.min(
                    kmin
                )
            )
        )
    )


    return (
        kmin,
        beta_c_grid,
    )


# ============================================================
# Load old 128-grid A3 output for independent reproduction
# ============================================================


with open(
    "lha13a3_probability_weighted_offstationary_caustic.csv",
    newline="",
) as f:

    old_rows = list(
        csv.DictReader(f)
    )


old_by_beta = {
    float(
        r["beta"]
    ): r

    for r in old_rows
}


# ============================================================
# Grid scan
# ============================================================


rows = []


max_norm_error = 0.0
max_imag_residual = 0.0
min_density = np.inf

max_old_f_delta = 0.0
max_old_p_delta = 0.0
max_old_R_delta = 0.0


results = {}


for N in NGRIDS:

    kappa_min_grid, beta_c_grid = (
        geometry_on_grid(
            N
        )
    )


    results[
        N
    ] = {}


    print(
        f"=== GRID {N} x {N} ==="
    )

    print(
        f"grid beta_c = "
        f"{beta_c_grid:.12f}"
    )

    print(
        f"grid-continuous beta_c delta = "
        f"{beta_c_grid-beta_c_cont:+.6e}"
    )

    print()

    print(
        " beta      invalid area"
        "       p_invalid"
        "          R=p/f"
    )

    print(
        "--------------------------------"
        "--------------------"
    )


    for ib, beta in enumerate(
        BETAS
    ):

        P, norm_error, imag_residual = (
            exact_density_on_grid(
                density_coeffs[ib],
                N,
            )
        )


        zeta = (
            beta
            / np.pi
            * np.sqrt(
                np.maximum(
                    0.0,
                    -kappa_min_grid
                )
            )
        )


        invalid = (
            zeta >= 1.0
        )


        negative = (
            kappa_min_grid < 0.0
        )


        f_invalid = float(
            np.mean(
                invalid
            )
        )


        p_invalid = float(
            np.mean(
                P*invalid
            )
        )


        if f_invalid > 0.0:

            R_invalid = (
                p_invalid
                / f_invalid
            )

        else:

            R_invalid = np.nan


        f_negative = float(
            np.mean(
                negative
            )
        )


        p_negative = float(
            np.mean(
                P*negative
            )
        )


        max_norm_error = max(
            max_norm_error,
            norm_error,
        )


        max_imag_residual = max(
            max_imag_residual,
            imag_residual,
        )


        min_density = min(
            min_density,
            float(
                np.min(P)
            ),
        )


        results[N][beta] = {
            "f": f_invalid,
            "p": p_invalid,
            "R": R_invalid,

            "f_negative":
                f_negative,

            "p_negative":
                p_negative,

            "beta_c_grid":
                beta_c_grid,
        }


        rows.append([
            N,
            beta,

            beta_c_grid,

            f_invalid,
            p_invalid,
            R_invalid,

            f_negative,
            p_negative,

            norm_error,
            imag_residual,

            float(
                np.min(P)
            ),

            float(
                np.max(P)
            ),
        ])


        print(
            f"{beta:6.3f}   "
            f"{100*f_invalid:12.7f}%   "
            f"{100*p_invalid:12.8f}%   "
            f"{R_invalid:11.7f}"
        )


        # --------------------------------------------
        # Reproduce the original N=128 calculation.
        # --------------------------------------------

        if (
            N == 128
            and beta in old_by_beta
        ):

            old = old_by_beta[
                beta
            ]


            old_f = float(
                old[
                    "invalid_area_fraction"
                ]
            )


            old_p = float(
                old[
                    "exact_invalid_probability"
                ]
            )


            old_R = float(
                old[
                    "invalid_probability_over_area"
                ]
            )


            max_old_f_delta = max(
                max_old_f_delta,
                abs(
                    f_invalid-old_f
                ),
            )


            max_old_p_delta = max(
                max_old_p_delta,
                abs(
                    p_invalid-old_p
                ),
            )


            if (
                np.isfinite(
                    R_invalid
                )
                and np.isfinite(
                    old_R
                )
            ):

                max_old_R_delta = max(
                    max_old_R_delta,
                    abs(
                        R_invalid-old_R
                    ),
                )


    print()


# ============================================================
# 256 -> 512 convergence
# ============================================================


print(
    "=== 256 -> 512 CONVERGENCE ==="
)

print(
    " beta       rel df"
    "          rel dp"
    "          rel dR"
)

print(
    "--------------------------------------------"
)


conv_rows = []


for beta in BETAS:

    a = results[
        256
    ][beta]


    b = results[
        512
    ][beta]


    def relative_delta(
        x_old,
        x_new,
    ):

        if x_new == 0.0:

            return np.nan


        return abs(
            x_new-x_old
        ) / abs(x_new)


    rel_f = relative_delta(
        a["f"],
        b["f"],
    )


    rel_p = relative_delta(
        a["p"],
        b["p"],
    )


    rel_R = relative_delta(
        a["R"],
        b["R"],
    )


    conv_rows.append([
        beta,
        rel_f,
        rel_p,
        rel_R,
    ])


    print(
        f"{beta:6.3f}   "
        f"{rel_f:11.5e}   "
        f"{rel_p:11.5e}   "
        f"{rel_R:11.5e}"
    )


# ============================================================
# Structural audit
# ============================================================


# beta=2.712 is deliberately excluded from the strict
# mask-geometry convergence gate: immediately after onset the
# invalid set is comparable to individual grid cells.
#
# R=p/f is the more robust physical quantity.
# ============================================================


stable_R_betas = [
    2.72,
    2.80,
    3.00,
]


stable_geometry_betas = [
    2.80,
    3.00,
]


max_rel_R_stable = max(

    next(
        row[3]
        for row in conv_rows
        if abs(
            row[0]-beta
        ) < 1.0e-12
    )

    for beta in stable_R_betas
)


max_rel_f_stable = max(

    next(
        row[1]
        for row in conv_rows
        if abs(
            row[0]-beta
        ) < 1.0e-12
    )

    for beta in stable_geometry_betas
)


max_rel_p_stable = max(

    next(
        row[2]
        for row in conv_rows
        if abs(
            row[0]-beta
        ) < 1.0e-12
    )

    for beta in stable_geometry_betas
)


beta_c_512 = results[
    512
][BETAS[0]][
    "beta_c_grid"
]


beta_c_512_error = abs(
    beta_c_512
    - beta_c_cont
)


checks = {

    "HAMILTONIAN HERMITICITY":
        hermiticity
        < 1.0e-12,

    "DENSITY-MATRIX HERMITICITY":
        rho_hermiticity
        < 1.0e-12,

    "DENSITY-MATRIX TRACE":
        rho_trace_error
        < 1.0e-12,

    "FOURIER DENSITY NORMALIZATION":
        max_norm_error
        < 1.0e-12,

    "FOURIER DENSITY REALITY":
        max_imag_residual
        < 1.0e-12,

    "EXACT DENSITY POSITIVE":
        min_density
        > 0.0,

    "REPRODUCE OLD N128 f":
        max_old_f_delta
        < 1.0e-12,

    "REPRODUCE OLD N128 p":
        max_old_p_delta
        < 1.0e-10,

    "REPRODUCE OLD N128 R":
        max_old_R_delta
        < 1.0e-8,

    "CONTINUOUS CAUSTIC OPTIMIZATION":
        bool(
            opt.success
        ),

    "N512 CAUSTIC LOCATION":
        beta_c_512_error
        < 1.0e-4,

    # Numerical, not physical thresholds:
    "R INVALID GRID CONVERGENCE":
        max_rel_R_stable
        < 0.03,

    "GEOMETRIC AREA CONVERGENCE":
        max_rel_f_stable
        < 0.02,

    "INVALID PROBABILITY CONVERGENCE":
        max_rel_p_stable
        < 0.02,
}


print()

print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    f"max density norm error    = "
    f"{max_norm_error:.6e}"
)

print(
    f"max density imag residual = "
    f"{max_imag_residual:.6e}"
)

print(
    f"minimum exact density     = "
    f"{min_density:.6e}"
)

print()

print(
    f"old N128 max df           = "
    f"{max_old_f_delta:.6e}"
)

print(
    f"old N128 max dp           = "
    f"{max_old_p_delta:.6e}"
)

print(
    f"old N128 max dR           = "
    f"{max_old_R_delta:.6e}"
)

print()

print(
    f"continuous beta_c         = "
    f"{beta_c_cont:.12f}"
)

print(
    f"N512 beta_c               = "
    f"{beta_c_512:.12f}"
)

print(
    f"|delta beta_c|            = "
    f"{beta_c_512_error:.6e}"
)

print()

print(
    f"max stable rel dR         = "
    f"{max_rel_R_stable:.6e}"
)

print(
    f"max stable rel df         = "
    f"{max_rel_f_stable:.6e}"
)

print(
    f"max stable rel dp         = "
    f"{max_rel_p_stable:.6e}"
)

print()


for name, passed in (
    checks.items()
):

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
    "beta=2.712 is an onset diagnostic, "
    "not a strict geometric-convergence point."
)

print(
    "The physically important post-caustic "
    "quantity is R_invalid = p_invalid/f_invalid."
)


# ============================================================
# Save
# ============================================================


with open(
    "lha13a3b_caustic_grid_convergence.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "NGRID",
        "beta",

        "beta_c_grid",

        "invalid_area_fraction",
        "exact_invalid_probability",
        "invalid_probability_over_area",

        "negative_curvature_area_fraction",
        "exact_negative_curvature_probability",

        "density_norm_error",
        "density_imag_residual",

        "density_min",
        "density_max",
    ])

    w.writerows(
        rows
    )


with open(
    "lha13a3b_caustic_grid_convergence_summary.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "relative_delta_f_256_to_512",
        "relative_delta_p_256_to_512",
        "relative_delta_R_256_to_512",
    ])

    w.writerows(
        conv_rows
    )


np.savez_compressed(

    "lha13a3b_caustic_grid_convergence.npz",

    betas=BETAS,

    ngrids=np.asarray(
        NGRIDS,
        dtype=int,
    ),

    caustic_uv_continuous=
        caustic_uv_cont,

    kappa_min_continuous=
        kappa_min_cont,

    beta_c_continuous=
        beta_c_cont,

    max_rel_R_stable=
        max_rel_R_stable,

    max_rel_f_stable=
        max_rel_f_stable,

    max_rel_p_stable=
        max_rel_p_stable,
)


print()

print(
    "saved: "
    "lha13a3b_caustic_grid_convergence.csv"
)

print(
    "saved: "
    "lha13a3b_caustic_grid_convergence_summary.csv"
)

print(
    "saved: "
    "lha13a3b_caustic_grid_convergence.npz"
)
