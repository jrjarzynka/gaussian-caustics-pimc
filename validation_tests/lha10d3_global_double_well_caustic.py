import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


# ============================================================
# LHA-10D3
#
# Globally stable quartic double well containing a local
# negative-curvature region.
#
# Units:
#   hbar = 1
#   M    = 1
#
# V(x) = -a*x^2/2 + g*x^4/4 + a^2/(4g)
#
# V''(0) = -a.
#
# Therefore the first local Gaussian caustic at x=0 occurs at
#
# beta_c * sqrt(a) = pi.
# ============================================================


A = 1.0
G = 0.1

BETA_C = np.pi / np.sqrt(A)

BETAS = [
    0.50,
    1.00,
    2.00,
    2.80,
    3.00,
    3.10,
    3.13,
    3.14,
    3.15,
    3.30,
]

XMAX = 8.0
N_GRID = 4001
N_STATES = 300


x = np.linspace(
    -XMAX,
    XMAX,
    N_GRID,
)

dx = x[1] - x[0]


# ------------------------------------------------------------
# Potential and derivatives
# ------------------------------------------------------------

V_SHIFT = A*A/(4.0*G)

V = (
    -0.5*A*x*x
    + 0.25*G*x**4
    + V_SHIFT
)

dV = (
    -A*x
    + G*x**3
)

ddV = (
    -A
    + 3.0*G*x*x
)


# ------------------------------------------------------------
# Exact quantum reference
# ------------------------------------------------------------

diag = (
    1.0/dx**2
    + V
)

off = np.full(
    N_GRID-1,
    -0.5/dx**2,
)

E, U = eigh_tridiagonal(
    diag,
    off,
    select="i",
    select_range=(
        0,
        N_STATES-1,
    ),
)

state_pdf = (
    U*U
) / dx


def normalize(pdf):

    return (
        pdf
        / np.trapezoid(
            pdf,
            x,
        )
    )


def exact_pdf(beta):

    w = np.exp(
        -beta*(E-E[0])
    )

    tail_indicator = np.exp(
        -beta*(E[-1]-E[0])
    )

    w /= np.sum(w)

    pdf = (
        state_pdf @ w
    )

    return (
        normalize(pdf),
        tail_indicator,
    )


# ------------------------------------------------------------
# Spectral Xi_beta(kappa) and F_beta(kappa)
# ------------------------------------------------------------

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

    # Near-flat direction.
    small = (
        np.abs(kappa) < 1.0e-9
    )

    # Series:
    #
    # Xi = 1 - beta^2*kappa/12
    #          + beta^4*kappa^2/120 + ...
    #
    # F = (Xi-1)/kappa

    Xi[small] = (
        1.0
        - beta**2*kappa[small]/12.0
        + beta**4*kappa[small]**2/120.0
    )

    F[small] = (
        -beta**2/12.0
        + beta**4*kappa[small]/120.0
    )

    # Positive curvature.
    pos = (
        kappa >= 1.0e-9
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

    # Negative curvature.
    neg = (
        kappa <= -1.0e-9
    )

    idx = np.where(
        neg
    )[0]

    eta = (
        0.5
        * beta
        * np.sqrt(
            -kappa[neg]
        )
    )

    stable = (
        eta < np.pi/2.0
    )

    good = idx[stable]
    bad = idx[~stable]

    eta_good = eta[stable]

    Xi_good = (
        np.tan(eta_good)
        / eta_good
    )

    Xi[good] = Xi_good

    F[good] = (
        Xi_good-1.0
    ) / kappa[good]

    valid[bad] = False

    Xi[bad] = np.nan
    F[bad] = np.nan

    return Xi, F, valid


# ------------------------------------------------------------
# RLH distribution
# ------------------------------------------------------------

def rlh_pdf(beta):

    Xi, F, valid = (
        spectral_Xi_F(
            ddV,
            beta,
        )
    )

    if not np.all(valid):

        return (
            None,
            Xi,
            F,
            valid,
        )

    Phi = (
        V
        + 0.5*dV*dV*F
    )

    log_pdf = (
        0.5*np.log(Xi)
        - beta*Phi
    )

    # Numerical stabilization.
    log_pdf -= np.max(
        log_pdf
    )

    pdf = np.exp(
        log_pdf
    )

    pdf = normalize(
        pdf
    )

    return (
        pdf,
        Xi,
        F,
        valid,
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def moment2(pdf):

    return np.trapezoid(
        x*x*pdf,
        x,
    )


def L1(p, q):

    return np.trapezoid(
        np.abs(p-q),
        x,
    )


rows = []

curves = {}

print(
    "=== LHA-10D3 GLOBAL DOUBLE WELL ==="
)

print()
print(
    f"a              = {A:.8f}"
)

print(
    f"g              = {G:.8f}"
)

print(
    f"barrier height = {V_SHIFT:.8f}"
)

print(
    f"beta_c         = {BETA_C:.12f}"
)

print()


for beta in BETAS:

    exact, tail = exact_pdf(
        beta
    )

    rlh, Xi, F, valid = (
        rlh_pdf(
            beta
        )
    )

    zeta_saddle = (
        beta*np.sqrt(A)/np.pi
    )

    invalid_fraction = (
        1.0
        - np.mean(valid)
    )

    exact_x2 = moment2(
        exact
    )

    center = (
        len(x)//2
    )

    print(
        "----------------------------------------"
    )

    print(
        f"beta          = {beta:.6f}"
    )

    print(
        f"zeta_saddle   = {zeta_saddle:.9f}"
    )

    print(
        f"invalid grid  = "
        f"{100*invalid_fraction:.4f}%"
    )

    print(
        f"ED tail ind.  = {tail:.3e}"
    )

    if rlh is not None:

        rlh_x2 = moment2(
            rlh
        )

        x2_error = (
            100.0
            * (rlh_x2-exact_x2)
            / exact_x2
        )

        l1err = L1(
            rlh,
            exact,
        )

        center_ratio = (
            rlh[center]
            / exact[center]
        )

        Xi_center = (
            Xi[center]
        )

        print(
            f"exact <x2>    = "
            f"{exact_x2:.10f}"
        )

        print(
            f"RLH   <x2>    = "
            f"{rlh_x2:.10f}"
        )

        print(
            f"x2 error      = "
            f"{x2_error:+.6f}%"
        )

        print(
            f"L1            = "
            f"{l1err:.8f}"
        )

        print(
            f"Xi(0)         = "
            f"{Xi_center:.8f}"
        )

        print(
            f"P_RLH(0)/P_QM(0) = "
            f"{center_ratio:.8f}"
        )

        status = "VALID"

        curves[beta] = (
            exact.copy(),
            rlh.copy(),
        )

    else:

        rlh_x2 = np.nan
        x2_error = np.nan
        l1err = np.nan
        center_ratio = np.nan
        Xi_center = np.nan

        print(
            "RLH status     = "
            "BEYOND FIRST LOCAL CAUSTIC"
        )

        status = "INVALID"

        curves[beta] = (
            exact.copy(),
            None,
        )

    rows.append([
        beta,
        zeta_saddle,
        exact_x2,
        rlh_x2,
        x2_error,
        l1err,
        Xi_center,
        center_ratio,
        invalid_fraction,
        tail,
        status,
    ])

    print()


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

with open(
    "lha10d3_global_double_well_caustic.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "zeta_saddle",
        "x2_exact",
        "x2_RLH",
        "x2_error_pct",
        "L1_RLH_exact",
        "Xi_saddle",
        "P_RLH0_over_P_exact0",
        "invalid_grid_fraction",
        "ED_tail_indicator",
        "RLH_status",
    ])

    w.writerows(rows)


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

plot_betas = [
    1.0,
    2.8,
    3.10,
    3.14,
    3.15,
]

fig, ax = plt.subplots(
    figsize=(8.0, 5.6)
)

for beta in plot_betas:

    exact, rlh = curves[
        beta
    ]

    ax.plot(
        x,
        exact,
        label=(
            rf"exact $\beta={beta:g}$"
        ),
    )

    if rlh is not None:

        ax.plot(
            x,
            rlh,
            linestyle="--",
            label=(
                rf"RLH $\beta={beta:g}$"
            ),
        )


ax.set_xlim(
    -5.0,
    5.0,
)

ax.set_xlabel(
    r"$x$"
)

ax.set_ylabel(
    r"$P(x)$"
)

ax.set_title(
    "Global double well approaching the local Gaussian caustic"
)

ax.legend(
    fontsize=8,
    ncol=2,
)

fig.tight_layout()

fig.savefig(
    "lha10d3_global_double_well_caustic.png",
    dpi=300,
)

fig.savefig(
    "lha10d3_global_double_well_caustic.pdf",
)


# ------------------------------------------------------------
# Threshold check
# ------------------------------------------------------------

below = [
    r
    for r in rows
    if (
        r[0] < BETA_C
        and r[-1] == "VALID"
    )
]

above = [
    r
    for r in rows
    if (
        r[0] > BETA_C
        and r[-1] == "INVALID"
    )
]

threshold_pass = (
    len(below) > 0
    and len(above) > 0
)

print(
    "=== THRESHOLD CHECK ==="
)

print(
    "all sampled beta < beta_c valid:",
    all(
        r[-1] == "VALID"
        for r in rows
        if r[0] < BETA_C
    )
)

print(
    "all sampled beta > beta_c invalid:",
    all(
        r[-1] == "INVALID"
        for r in rows
        if r[0] > BETA_C
    )
)

print()

print(
    "GLOBAL CAUSTIC TEST:",
    "PASS"
    if threshold_pass
    else "FAIL"
)

print()

print(
    "saved: "
    "lha10d3_global_double_well_caustic.csv"
)

print(
    "saved: "
    "lha10d3_global_double_well_caustic.png"
)

print(
    "saved: "
    "lha10d3_global_double_well_caustic.pdf"
)
