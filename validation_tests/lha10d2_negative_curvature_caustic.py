import csv
import numpy as np


# ============================================================
# LHA-10D2
#
# Spectral validation of the negative-curvature continuation
# of the starting-point local Gaussian kernel.
#
# Units:
#   hbar = 1
#   M    = 1
#
# Fluctuation operator:
#
#   L = -d^2/dtau^2 + kappa
#
# on tau in [0,beta], with Dirichlet boundary conditions.
#
# For kappa < 0:
#
#   kappa = -nu^2
#
# and the continuum stability condition is
#
#   beta*nu < pi
#
# or equivalently
#
#   zeta = beta*nu/pi < 1.
#
# ============================================================


BETA = 5.0

ZETAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    0.90,
    0.97,
    0.99,
    0.999,
    1.001,
    1.05,
]

NMODES_LIST = [
    50,
    200,
    1000,
    5000,
]


def kappa_from_zeta(zeta):

    if zeta == 0.0:
        return 0.0

    nu = (
        zeta
        * np.pi
        / BETA
    )

    return -(nu**2)


def analytic_prefactor(zeta):

    if zeta == 0.0:
        return 1.0

    x = np.pi*zeta

    if x >= np.pi:
        return np.nan

    return np.sqrt(
        x / np.sin(x)
    )


def analytic_Xi(zeta):

    if zeta == 0.0:
        return 1.0

    eta = (
        0.5
        * np.pi
        * zeta
    )

    if eta >= np.pi/2:
        return np.nan

    return (
        np.tan(eta)
        / eta
    )


def finite_product_prefactor(
    kappa,
    nmodes,
):

    n = np.arange(
        1,
        nmodes+1,
        dtype=float,
    )

    lambda_free = (
        n*np.pi/BETA
    )**2

    lambda_curved = (
        lambda_free
        + kappa
    )

    n_negative = np.sum(
        lambda_curved <= 0.0
    )

    if n_negative > 0:
        return (
            np.nan,
            lambda_curved[0],
            int(n_negative),
        )

    # Gaussian determinant ratio:
    #
    # D_N =
    # prod_n [
    # lambda_curved/lambda_free
    # ]^(-1/2)
    #
    # Use logs for numerical stability.

    logD = (
        -0.5
        * np.sum(
            np.log(
                lambda_curved
                / lambda_free
            )
        )
    )

    return (
        np.exp(logD),
        lambda_curved[0],
        0,
    )


rows = []

print(
    "=== LHA-10D2 "
    "NEGATIVE-CURVATURE CAUSTIC ==="
)

print()

kappa_c = (
    -(np.pi/BETA)**2
)

print(
    f"beta       = {BETA:.8f}"
)

print(
    f"kappa_c    = {kappa_c:.12f}"
)

print()


max_rel_error_stable = 0.0

for zeta in ZETAS:

    kappa = kappa_from_zeta(
        zeta
    )

    D_exact = analytic_prefactor(
        zeta
    )

    Xi = analytic_Xi(
        zeta
    )

    print(
        "----------------------------------------"
    )

    print(
        f"zeta  = {zeta:.6f}"
    )

    print(
        f"kappa = {kappa:.12f}"
    )

    if np.isfinite(D_exact):

        print(
            f"D_exact = {D_exact:.10f}"
        )

        print(
            f"Xi      = {Xi:.10f}"
        )

    else:

        print(
            "D_exact = divergent / unstable"
        )

        print(
            "Xi      = beyond first caustic"
        )

    for nmodes in NMODES_LIST:

        D_num, lambda1, nneg = (
            finite_product_prefactor(
                kappa,
                nmodes,
            )
        )

        if np.isfinite(D_num):

            relerr = abs(
                D_num-D_exact
            ) / D_exact

            max_rel_error_stable = max(
                max_rel_error_stable,
                relerr
                if nmodes == max(NMODES_LIST)
                else 0.0,
            )

            print(
                f"N={nmodes:5d}  "
                f"lambda1={lambda1:+.8e}  "
                f"D_N={D_num:.10f}  "
                f"relerr={relerr:.3e}"
            )

        else:

            print(
                f"N={nmodes:5d}  "
                f"lambda1={lambda1:+.8e}  "
                f"NEGATIVE MODES={nneg}"
            )

        rows.append([
            BETA,
            zeta,
            kappa,
            nmodes,
            lambda1,
            nneg,
            D_exact,
            D_num,
            (
                abs(D_num-D_exact)/D_exact
                if (
                    np.isfinite(D_num)
                    and np.isfinite(D_exact)
                )
                else np.nan
            ),
            Xi,
        ])

    print()


with open(
    "lha10d2_negative_curvature_caustic.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "zeta",
        "kappa",
        "nmodes",
        "lambda1",
        "negative_modes",
        "analytic_prefactor",
        "finite_product_prefactor",
        "relative_error",
        "Xi_negative_curvature",
    ])

    w.writerows(rows)


# ------------------------------------------------------------
# Explicit threshold audit
# ------------------------------------------------------------

eps = 1.0e-6

z_below = 1.0 - eps
z_above = 1.0 + eps

k_below = kappa_from_zeta(
    z_below
)

k_above = kappa_from_zeta(
    z_above
)

_, lambda_below, nneg_below = (
    finite_product_prefactor(
        k_below,
        5000,
    )
)

_, lambda_above, nneg_above = (
    finite_product_prefactor(
        k_above,
        5000,
    )
)


print(
    "=== THRESHOLD AUDIT ==="
)

print(
    f"zeta={z_below:.6f}: "
    f"lambda1={lambda_below:+.12e}, "
    f"negative_modes={nneg_below}"
)

print(
    f"zeta={z_above:.6f}: "
    f"lambda1={lambda_above:+.12e}, "
    f"negative_modes={nneg_above}"
)


threshold_pass = (
    lambda_below > 0.0
    and nneg_below == 0
    and lambda_above < 0.0
    and nneg_above >= 1
)

print()

print(
    "CAUSTIC THRESHOLD:",
    "PASS"
    if threshold_pass
    else "FAIL"
)

print()

print(
    "OVERALL:",
    "PASS"
    if threshold_pass
    else "FAIL"
)

print()

print(
    "saved: "
    "lha10d2_negative_curvature_caustic.csv"
)
