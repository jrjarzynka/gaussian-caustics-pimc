import csv
import numpy as np


BETAS = [0.5, 1.0, 1.9, 2.5]

NBIN = 16
N_VALIDATE = 64
N_HI_1 = 256
N_HI_2 = 512

sqrt3 = np.sqrt(3.0)

G = np.array([
    [1.0, 0.0],
    [-0.5, sqrt3/2.0],
    [-0.5, -sqrt3/2.0],
])


# ============================================================
# RLH density
# ============================================================

def rlh_density(beta, N, midpoint=False):

    if midpoint:
        axis = (
            np.arange(N, dtype=float) + 0.5
        ) / N
    else:
        axis = (
            np.arange(N, dtype=float)
        ) / N

    S, T = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    theta = np.stack([
        2.0*np.pi*S,
        2.0*np.pi*T,
        -2.0*np.pi*(S+T),
    ], axis=-1)

    c = np.cos(theta)
    s = np.sin(theta)

    V = (
        3.0
        - c[..., 0]
        - c[..., 1]
        - c[..., 2]
    )

    grad = np.einsum(
        "...a,ai->...i",
        s,
        G,
    )

    H = np.einsum(
        "...a,ai,aj->...ij",
        c,
        G,
        G,
    )

    kappa, R = np.linalg.eigh(H)

    Xi = np.empty_like(kappa)
    F = np.empty_like(kappa)

    eps = 1.0e-9

    pos = kappa > eps
    neg = kappa < -eps
    zer = ~(pos | neg)

    if np.any(pos):
        xi = (
            0.5
            * beta
            * np.sqrt(kappa[pos])
        )

        Xi[pos] = (
            np.tanh(xi) / xi
        )

        F[pos] = (
            Xi[pos] - 1.0
        ) / kappa[pos]

    if np.any(neg):
        eta = (
            0.5
            * beta
            * np.sqrt(-kappa[neg])
        )

        if np.max(eta) >= np.pi/2:
            raise RuntimeError(
                f"beta={beta}: caustic crossed"
            )

        Xi[neg] = (
            np.tan(eta) / eta
        )

        F[neg] = (
            Xi[neg] - 1.0
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

    # modal gradient R^T g
    gmodal = np.einsum(
        "...ia,...i->...a",
        R,
        grad,
    )

    quadratic = np.sum(
        F * gmodal*gmodal,
        axis=-1,
    )

    Phi = (
        V
        + 0.5*quadratic
    )

    logP = (
        0.5*np.sum(
            np.log(Xi),
            axis=-1,
        )
        - beta*Phi
    )

    logP -= np.max(logP)

    P = np.exp(logP)

    P /= np.mean(P)

    zeta = (
        beta/np.pi
        * np.sqrt(
            np.maximum(
                0.0,
                -kappa,
            )
        )
    )

    return (
        P,
        V,
        float(np.max(zeta)),
    )


# ============================================================
# Exact 64-grid density -> exact bin averages
# ============================================================

def Fourier_bin_average(Pfine, nbin):

    nfine = Pfine.shape[0]

    F = (
        np.fft.fft2(Pfine)
        / (nfine*nfine)
    )

    freq = (
        np.fft.fftfreq(nfine)
        * nfine
    ).astype(int)

    centers = (
        np.arange(nbin, dtype=float)
        + 0.5
    ) / nbin

    out = np.zeros(
        (nbin, nbin),
        dtype=complex,
    )

    for i2, n2 in enumerate(freq):

        phase_t = np.exp(
            2j*np.pi*n2*centers
        )

        sinc2 = np.sinc(
            n2/nbin
        )

        for i1, n1 in enumerate(freq):

            coeff = F[i2, i1]

            if abs(coeff) < 1e-15:
                continue

            phase_s = np.exp(
                2j*np.pi*n1*centers
            )

            sinc1 = np.sinc(
                n1/nbin
            )

            out += (
                coeff
                * sinc1*sinc2
                * phase_t[:, None]
                * phase_s[None, :]
            )

    imag = float(
        np.max(np.abs(out.imag))
    )

    out = out.real
    out /= np.mean(out)

    return out, imag


# ============================================================
# High-resolution RLH -> histogram-bin averages
# ============================================================

def block_bin_average(P, nbin):

    N = P.shape[0]

    if N % nbin != 0:
        raise ValueError(
            "N must be divisible by nbin"
        )

    m = N // nbin

    out = (
        P
        .reshape(
            nbin,
            m,
            nbin,
            m,
        )
        .mean(axis=(1, 3))
    )

    out /= np.mean(out)

    return out


def L1(a, b):
    return float(
        np.mean(
            np.abs(a-b)
        )
    )


# ============================================================
# Inputs
# ============================================================

exact = np.load(
    "lha12a1_exact_bz_reference_K2_81_Nk16.npz"
)

betas_exact = np.asarray(
    exact["betas"],
    dtype=float,
)

BZ = np.asarray(
    exact["densities_BZ"],
    dtype=float,
)

Gamma = np.asarray(
    exact["densities_Gamma"],
    dtype=float,
)


# B4 rows
with open(
    "lha11b4_dense_precaustic_scan.csv",
    newline="",
) as f:

    b4_rows = list(
        csv.DictReader(f)
    )


def b4_row(beta):

    return min(
        b4_rows,
        key=lambda r: abs(
            float(r["beta"]) - beta
        ),
    )


# C1 rows
with open(
    "lha12c1_multibeta_pimc_summary.csv",
    newline="",
) as f:

    c1_rows = list(
        csv.DictReader(f)
    )


def c1_row(beta):

    return min(
        c1_rows,
        key=lambda r: abs(
            float(r["beta"]) - beta
        ),
    )


# ============================================================
# Validation + harmonized comparison
# ============================================================

rows = []

print(
    "=== LHA-12C2a "
    "HARMONIZED DENSITY COMPARISON ==="
)

print()


for beta in BETAS:

    ie = int(
        np.argmin(
            np.abs(
                betas_exact-beta
            )
        )
    )

    if abs(
        betas_exact[ie]-beta
    ) > 1e-12:
        raise RuntimeError(
            f"missing exact beta={beta}"
        )

    b4 = b4_row(beta)
    c1 = c1_row(beta)

    # --------------------------------------------------------
    # Reconstruct exactly the old 64-grid RLH
    # --------------------------------------------------------

    P64, V64, zeta = rlh_density(
        beta,
        N_VALIDATE,
        midpoint=False,
    )

    V_RLH_64 = float(
        np.mean(
            P64*V64
        )
    )

    L1_gamma_64 = L1(
        P64,
        Gamma[ie],
    )

    V_B4 = float(
        b4["V_RLH"]
    )

    L1_B4 = float(
        b4["density_L1"]
    )

    dV_validation = abs(
        V_RLH_64-V_B4
    )

    dL1_validation = abs(
        L1_gamma_64-L1_B4
    )

    # --------------------------------------------------------
    # Correct exact BZ 16x16 bin target
    # --------------------------------------------------------

    Pexact16, imag = (
        Fourier_bin_average(
            BZ[ie],
            NBIN,
        )
    )

    # --------------------------------------------------------
    # RLH direct high-resolution bin integration
    # --------------------------------------------------------

    P256, _, _ = rlh_density(
        beta,
        N_HI_1,
        midpoint=True,
    )

    P512, _, _ = rlh_density(
        beta,
        N_HI_2,
        midpoint=True,
    )

    R256 = block_bin_average(
        P256,
        NBIN,
    )

    R512 = block_bin_average(
        P512,
        NBIN,
    )

    bin_convergence = L1(
        R512,
        R256,
    )

    L1_RLH_bin16 = L1(
        R512,
        Pexact16,
    )

    L1_PIMC_bin16 = float(
        c1["density_L1_pooled"]
    )

    rows.append([
        beta,
        zeta,

        V_RLH_64,
        V_B4,
        dV_validation,

        L1_gamma_64,
        L1_B4,
        dL1_validation,

        L1_PIMC_bin16,
        L1_RLH_bin16,

        bin_convergence,
        imag,
    ])

    print(
        f"beta={beta:3.1f}  "
        f"zeta={zeta:.6f}"
    )

    print(
        f"  B4 validation: "
        f"dV={dV_validation:.3e}, "
        f"dL1={dL1_validation:.3e}"
    )

    print(
        f"  RLH bin convergence "
        f"256->512 L1={bin_convergence:.3e}"
    )

    print(
        f"  harmonized 16x16:"
    )

    print(
        f"    PI-QMC L1 = "
        f"{L1_PIMC_bin16:.8f}"
    )

    print(
        f"    RLH    L1 = "
        f"{L1_RLH_bin16:.8f}"
    )

    print()


# ============================================================
# Structural audit
# ============================================================

A = np.asarray(
    rows,
    dtype=float,
)

max_dV = float(
    np.max(A[:, 4])
)

max_dL1 = float(
    np.max(A[:, 7])
)

max_bin_conv = float(
    np.max(A[:, 10])
)

max_imag = float(
    np.max(A[:, 11])
)


print(
    "=== STRUCTURAL AUDIT ==="
)

print(
    f"max B4 V reconstruction error   = "
    f"{max_dV:.6e}"
)

print(
    f"max B4 L1 reconstruction error  = "
    f"{max_dL1:.6e}"
)

print(
    f"max RLH bin 256->512 L1        = "
    f"{max_bin_conv:.6e}"
)

print(
    f"max exact-bin imag residual     = "
    f"{max_imag:.6e}"
)

print()


reconstruction_pass = (
    max_dV < 1e-9
    and max_dL1 < 1e-9
)

bin_pass = (
    max_bin_conv < 1e-4
)

imag_pass = (
    max_imag < 1e-12
)


print(
    "B4 RLH RECONSTRUCTION:",
    "PASS"
    if reconstruction_pass
    else "FAIL"
)

print(
    "RLH BIN INTEGRATION:",
    "PASS"
    if bin_pass
    else "FAIL"
)

print(
    "EXACT BIN FOURIER REALITY:",
    "PASS"
    if imag_pass
    else "FAIL"
)

print()

print(
    "STRUCTURAL OVERALL:",
    "PASS"
    if (
        reconstruction_pass
        and bin_pass
        and imag_pass
    )
    else "FAIL"
)


# ============================================================
# Save manuscript-ready density comparison
# ============================================================

with open(
    "lha12c2a_harmonized_density_comparison.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "zeta_max",

        "V_RLH_reconstructed",
        "V_RLH_B4",
        "V_RLH_reconstruction_abs_error",

        "L1_RLH_Gamma64_reconstructed",
        "L1_RLH_B4",
        "L1_RLH_reconstruction_abs_error",

        "L1_PIMC_bin16_vs_exact_BZ",
        "L1_RLH_bin16_vs_exact_BZ",

        "RLH_bin16_L1_256_vs_512",
        "exact_bin_imag_residual",
    ])

    w.writerows(rows)


print()

print(
    "saved: "
    "lha12c2a_harmonized_density_comparison.csv"
)
