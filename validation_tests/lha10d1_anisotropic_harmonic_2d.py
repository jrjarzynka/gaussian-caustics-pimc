import csv
import numpy as np


# ============================================================
# LHA-10D1
#
# Exact validation of the tensorial multidimensional RLH
# for a rotated anisotropic 2D harmonic oscillator.
#
# Dimensionless units:
#   hbar = 1
#   M    = 1
#
# V(r) = 1/2 r^T H r
#
# Eigenvalues of H are omega_a^2.
# ============================================================


BETAS = [
    0.2,
    1.0,
    5.0,
    20.0,
]

OMEGA_RATIOS = [
    1.0,
    1.5,
    3.0,
    6.0,
]

ANGLES_DEG = [
    0.0,
    23.0,
    57.0,
]


def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s],
        [s,  c],
    ])


def xi_factor(beta, omega):

    z = 0.5 * beta * omega

    if abs(z) < 1.0e-8:
        return (
            1.0
            - z*z/3.0
            + 2.0*z**4/15.0
        )

    return np.tanh(z) / z


def build_H(
    omega1,
    omega2,
    theta,
):

    R = rotation(theta)

    Hdiag = np.diag([
        omega1**2,
        omega2**2,
    ])

    return (
        R @ Hdiag @ R.T
    )


def exact_quantum_covariance(
    H,
    beta,
):

    kappa, R = np.linalg.eigh(H)

    omega = np.sqrt(kappa)

    sigma2 = np.array([
        1.0
        / (
            2.0*w
            * np.tanh(
                0.5*beta*w
            )
        )
        for w in omega
    ])

    return (
        R
        @ np.diag(sigma2)
        @ R.T
    )


def Xi_matrix(
    H,
    beta,
):

    kappa, R = np.linalg.eigh(H)

    omega = np.sqrt(kappa)

    Xi_vals = np.array([
        xi_factor(beta, w)
        for w in omega
    ])

    Xi = (
        R
        @ np.diag(Xi_vals)
        @ R.T
    )

    return Xi, Xi_vals


def tensor_rlh_covariance(
    H,
    beta,
):

    Xi, Xi_vals = Xi_matrix(
        H,
        beta,
    )

    # For an exactly harmonic potential:
    #
    # Phi_RLH =
    # 1/2 r^T H Xi r
    #
    # Hence:
    #
    # Sigma_RLH = (beta H Xi)^(-1)

    precision = (
        beta
        * H
        @ Xi
    )

    Sigma = np.linalg.inv(
        precision
    )

    return Sigma, Xi, Xi_vals


def scalar_trace_covariance(
    H,
    beta,
):

    Xi, _ = Xi_matrix(
        H,
        beta,
    )

    xi_bar = (
        0.5
        * np.trace(Xi)
    )

    return np.linalg.inv(
        beta
        * xi_bar
        * H
    )


def scalar_geom_covariance(
    H,
    beta,
):

    Xi, _ = Xi_matrix(
        H,
        beta,
    )

    xi_bar = np.sqrt(
        np.linalg.det(Xi)
    )

    return np.linalg.inv(
        beta
        * xi_bar
        * H
    )


def rel_fro_error(
    test,
    reference,
):

    return (
        np.linalg.norm(
            test-reference,
            ord="fro",
        )
        /
        np.linalg.norm(
            reference,
            ord="fro",
        )
    )


def moment_r2(Sigma):
    return np.trace(Sigma)


def rel_r2_error(
    test,
    reference,
):

    return (
        (
            moment_r2(test)
            - moment_r2(reference)
        )
        /
        moment_r2(reference)
    )


rows = []

max_tensor_error = 0.0
max_rotation_error = 0.0

print(
    "=== LHA-10D1 "
    "ANISOTROPIC HARMONIC 2D ==="
)
print()

for beta in BETAS:

    print(
        "========================================"
    )

    print(
        f"beta* = {beta:g}"
    )

    print(
        "========================================"
    )

    for ratio in OMEGA_RATIOS:

        omega1 = 1.0
        omega2 = ratio

        # Principal-axis reference.
        H0 = build_H(
            omega1,
            omega2,
            0.0,
        )

        Sigma0, _, _ = (
            tensor_rlh_covariance(
                H0,
                beta,
            )
        )

        for angle_deg in ANGLES_DEG:

            theta = np.deg2rad(
                angle_deg
            )

            H = build_H(
                omega1,
                omega2,
                theta,
            )

            exact = (
                exact_quantum_covariance(
                    H,
                    beta,
                )
            )

            tensor, Xi, Xi_vals = (
                tensor_rlh_covariance(
                    H,
                    beta,
                )
            )

            scalar_trace = (
                scalar_trace_covariance(
                    H,
                    beta,
                )
            )

            scalar_geom = (
                scalar_geom_covariance(
                    H,
                    beta,
                )
            )

            tensor_err = rel_fro_error(
                tensor,
                exact,
            )

            trace_err = rel_fro_error(
                scalar_trace,
                exact,
            )

            geom_err = rel_fro_error(
                scalar_geom,
                exact,
            )

            tensor_r2_err = rel_r2_error(
                tensor,
                exact,
            )

            trace_r2_err = rel_r2_error(
                scalar_trace,
                exact,
            )

            geom_r2_err = rel_r2_error(
                scalar_geom,
                exact,
            )

            # Rotation covariance test:
            #
            # Sigma(theta)
            # must equal
            # R Sigma(0) R^T.

            R = rotation(theta)

            rotated_reference = (
                R
                @ Sigma0
                @ R.T
            )

            rotation_err = rel_fro_error(
                tensor,
                rotated_reference,
            )

            max_tensor_error = max(
                max_tensor_error,
                tensor_err,
            )

            max_rotation_error = max(
                max_rotation_error,
                rotation_err,
            )

            rows.append([
                beta,
                ratio,
                angle_deg,
                Xi_vals[0],
                Xi_vals[1],
                exact[0, 0],
                exact[1, 1],
                exact[0, 1],
                tensor_err,
                tensor_r2_err,
                trace_err,
                trace_r2_err,
                geom_err,
                geom_r2_err,
                rotation_err,
            ])

        # Only print one angle because scalar errors
        # must be rotation invariant.

        subset = rows[-len(ANGLES_DEG)]

        print(
            f"omega2/omega1={ratio:4.1f}  "
            f"Xi=({subset[3]:.6f}, "
            f"{subset[4]:.6f})  "
            f"tensor={subset[8]:.3e}  "
            f"trace={100*subset[10]:7.3f}%  "
            f"geom={100*subset[12]:7.3f}%"
        )

    print()


with open(
    "lha10d1_anisotropic_harmonic_2d.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "beta_star",
        "omega_ratio",
        "angle_deg",
        "Xi1",
        "Xi2",
        "exact_x2",
        "exact_y2",
        "exact_xy",
        "tensor_cov_rel_error",
        "tensor_r2_rel_error",
        "trace_cov_rel_error",
        "trace_r2_rel_error",
        "geom_cov_rel_error",
        "geom_r2_rel_error",
        "rotation_rel_error",
    ])

    writer.writerows(rows)


print(
    "=== GLOBAL CHECKS ==="
)

print(
    "max tensor covariance relative error = "
    f"{max_tensor_error:.6e}"
)

print(
    "max rotation covariance error        = "
    f"{max_rotation_error:.6e}"
)

tensor_pass = (
    max_tensor_error < 1.0e-12
)

rotation_pass = (
    max_rotation_error < 1.0e-12
)

print()

print(
    "TENSOR HARMONIC EXACTNESS:",
    "PASS"
    if tensor_pass
    else "FAIL"
)

print(
    "ROTATIONAL COVARIANCE:",
    "PASS"
    if rotation_pass
    else "FAIL"
)

print()

if tensor_pass and rotation_pass:

    print(
        "OVERALL: PASS"
    )

else:

    print(
        "OVERALL: FAIL"
    )


print()
print(
    "saved: "
    "lha10d1_anisotropic_harmonic_2d.csv"
)
