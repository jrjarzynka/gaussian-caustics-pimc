import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


# ============================================================
# LHA-10D4
#
# Rotated 2D globally stable double well:
#
# V(u,v) =
#   -a u^2/2 + g u^4/4 + a^2/(4g)
#   + omega_perp^2 v^2/2
#
# Principal coordinates (u,v) are rotated relative to
# laboratory coordinates (x,y).
#
# Exact quantum PDF:
#
# P_QM(u,v) = P_DW_exact(u) P_HO_exact(v)
#
# RLH is evaluated tensorially in the laboratory frame.
#
# Units:
#   hbar = 1
#   M    = 1
# ============================================================


A = 1.0
G = 0.1
OMEGA_PERP = 1.7

ANGLES_DEG = [
    0.0,
    31.0,
    67.0,
]

ZETAS = [
    0.30,
    0.65,
    0.90,
    0.97,
    0.995,
    1.01,
]

# zeta = beta*sqrt(A)/pi
# with A=1:
# beta = pi*zeta


# ------------------------------------------------------------
# 1D exact double-well reference
# ------------------------------------------------------------

UMAX = 8.0
NU = 5001
N_STATES = 350

u_ref = np.linspace(
    -UMAX,
    UMAX,
    NU,
)

du = u_ref[1] - u_ref[0]

V_SHIFT = A*A/(4.0*G)

V_dw_ref = (
    -0.5*A*u_ref**2
    + 0.25*G*u_ref**4
    + V_SHIFT
)

diag = (
    1.0/du**2
    + V_dw_ref
)

off = np.full(
    NU-1,
    -0.5/du**2,
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
) / du


def normalize_1d(p):

    return (
        p
        / np.trapezoid(
            p,
            u_ref,
        )
    )


def exact_dw_pdf(beta):

    w = np.exp(
        -beta*(E-E[0])
    )

    w /= np.sum(w)

    p = state_pdf @ w

    return normalize_1d(p)


# ------------------------------------------------------------
# Exact harmonic transverse PDF
# ------------------------------------------------------------

def exact_ho_pdf(v, beta):

    sigma2 = (
        1.0
        /
        (
            2.0
            * OMEGA_PERP
            * np.tanh(
                0.5
                * beta
                * OMEGA_PERP
            )
        )
    )

    return (
        np.exp(
            -v*v/(2.0*sigma2)
        )
        /
        np.sqrt(
            2.0*np.pi*sigma2
        )
    )


# ------------------------------------------------------------
# Spectral Xi_beta(kappa), F_beta(kappa)
# ------------------------------------------------------------

def spectral_Xi_F(kappa, beta):

    kappa = np.asarray(
        kappa,
        dtype=float,
    )

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

    small = (
        np.abs(kappa)
        < 1.0e-8
    )

    # Series around kappa = 0:
    #
    # Xi =
    # 1 - beta^2 kappa/12
    #   + beta^4 kappa^2/120 + ...
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

    # Positive curvature

    pos = (
        kappa > 1.0e-8
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

    # Negative curvature

    neg = (
        kappa < -1.0e-8
    )

    eta = (
        0.5
        * beta
        * np.sqrt(
            -kappa[neg]
        )
    )

    neg_indices = np.where(
        neg
    )[0]

    stable = (
        eta < np.pi/2.0
    )

    good = neg_indices[stable]
    bad = neg_indices[~stable]

    eta_good = eta[stable]

    Xi_good = (
        np.tan(eta_good)
        / eta_good
    )

    Xi[good] = Xi_good

    F[good] = (
        Xi_good-1.0
    ) / kappa[good]

    Xi[bad] = np.nan
    F[bad] = np.nan

    valid[bad] = False

    return Xi, F, valid


# ------------------------------------------------------------
# Laboratory grid
# ------------------------------------------------------------

XMAX = 7.0
NXY = 601

axis = np.linspace(
    -XMAX,
    XMAX,
    NXY,
)

dxy = axis[1] - axis[0]

X, Y = np.meshgrid(
    axis,
    axis,
    indexing="xy",
)


def rotation(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s],
        [s,  c],
    ])


def lab_to_principal(
    X,
    Y,
    theta,
):

    c = np.cos(theta)
    s = np.sin(theta)

    # [u,v]^T = R^T [x,y]^T

    u = (
        c*X
        + s*Y
    )

    v = (
        -s*X
        + c*Y
    )

    return u, v


def normalize_2d(pdf):

    Z = np.sum(
        pdf
    ) * dxy*dxy

    return pdf/Z


def integrate_2d(field):

    return (
        np.sum(field)
        * dxy*dxy
    )


# ------------------------------------------------------------
# Exact 2D PDF
# ------------------------------------------------------------

def exact_2d_pdf(
    u,
    v,
    beta,
    p_dw_ref,
):

    p_u = np.interp(
        u.ravel(),
        u_ref,
        p_dw_ref,
        left=0.0,
        right=0.0,
    ).reshape(
        u.shape
    )

    p_v = exact_ho_pdf(
        v,
        beta,
    )

    return normalize_2d(
        p_u*p_v
    )


# ------------------------------------------------------------
# Tensorial RLH in laboratory coordinates
# ------------------------------------------------------------

def tensor_rlh_lab(
    X,
    Y,
    theta,
    beta,
):

    c = np.cos(theta)
    s = np.sin(theta)

    u, v = lab_to_principal(
        X,
        Y,
        theta,
    )

    # Potential

    V_dw = (
        -0.5*A*u*u
        + 0.25*G*u**4
        + V_SHIFT
    )

    V_trans = (
        0.5
        * OMEGA_PERP**2
        * v*v
    )

    V = (
        V_dw
        + V_trans
    )

    # Principal gradients

    gu = (
        -A*u
        + G*u**3
    )

    gv = (
        OMEGA_PERP**2
        * v
    )

    # Principal Hessian eigenvalues

    kappa_u = (
        -A
        + 3.0*G*u*u
    )

    kappa_v = np.full_like(
        u,
        OMEGA_PERP**2,
    )

    # Spectral functions

    Xi_u, F_u, valid_u = (
        spectral_Xi_F(
            kappa_u.ravel(),
            beta,
        )
    )

    Xi_v, F_v, valid_v = (
        spectral_Xi_F(
            kappa_v.ravel(),
            beta,
        )
    )

    Xi_u = Xi_u.reshape(
        u.shape
    )

    Xi_v = Xi_v.reshape(
        u.shape
    )

    F_u = F_u.reshape(
        u.shape
    )

    F_v = F_v.reshape(
        u.shape
    )

    valid = (
        valid_u.reshape(u.shape)
        &
        valid_v.reshape(u.shape)
    )

    # --------------------------------------------------------
    # Rotate gradient to laboratory frame:
    #
    # g_xy = R g_uv
    # --------------------------------------------------------

    gx = (
        c*gu
        - s*gv
    )

    gy = (
        s*gu
        + c*gv
    )

    # --------------------------------------------------------
    # F(H) in laboratory frame:
    #
    # F_lab = R diag(Fu,Fv) R^T
    # --------------------------------------------------------

    Fxx = (
        c*c*F_u
        + s*s*F_v
    )

    Fyy = (
        s*s*F_u
        + c*c*F_v
    )

    Fxy = (
        c*s
        * (F_u-F_v)
    )

    # Tensor contraction in laboratory coordinates:

    contraction_lab = (
        Fxx*gx*gx
        + 2.0*Fxy*gx*gy
        + Fyy*gy*gy
    )

    # Independent principal-axis expression.
    # Must agree pointwise to machine precision.

    contraction_principal = (
        F_u*gu*gu
        + F_v*gv*gv
    )

    good = (
        valid
        & np.isfinite(
            contraction_lab
        )
        & np.isfinite(
            contraction_principal
        )
    )

    if np.any(good):

        denom = np.maximum(
            1.0,
            np.abs(
                contraction_principal[good]
            )
        )

        contraction_error = np.max(
            np.abs(
                contraction_lab[good]
                - contraction_principal[good]
            )
            / denom
        )

    else:

        contraction_error = np.nan

    # Mixed Hessian component in laboratory frame.
    # Useful diagnostic showing that the rotated test really
    # has V_xy != 0.

    Hxy = (
        c*s
        * (kappa_u-kappa_v)
    )

    # If any spatial point is beyond the first caustic,
    # the derived global RLH PDF is not defined.

    if not np.all(valid):

        return {
            "pdf": None,
            "valid": valid,
            "u": u,
            "v": v,
            "kappa_u": kappa_u,
            "Xi_u": Xi_u,
            "contraction_error": contraction_error,
            "max_abs_Hxy": np.max(
                np.abs(Hxy)
            ),
        }

    Phi = (
        V
        + 0.5*contraction_lab
    )

    detXi = (
        Xi_u*Xi_v
    )

    log_pdf = (
        0.5*np.log(detXi)
        - beta*Phi
    )

    log_pdf -= np.max(
        log_pdf
    )

    pdf = np.exp(
        log_pdf
    )

    pdf = normalize_2d(
        pdf
    )

    return {
        "pdf": pdf,
        "valid": valid,
        "u": u,
        "v": v,
        "kappa_u": kappa_u,
        "Xi_u": Xi_u,
        "contraction_error": contraction_error,
        "max_abs_Hxy": np.max(
            np.abs(Hxy)
        ),
    }


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def l1_distance(p, q):

    return integrate_2d(
        np.abs(p-q)
    )


def expectation(
    observable,
    pdf,
):

    return integrate_2d(
        observable*pdf
    )


# ------------------------------------------------------------
# Main scan
# ------------------------------------------------------------

rows = []

rotation_groups = {}

max_tensor_contraction_error = 0.0

print(
    "=== LHA-10D4 "
    "ROTATED 2D STABILIZED SADDLE ==="
)

print()

print(
    f"a            = {A:.8f}"
)

print(
    f"g            = {G:.8f}"
)

print(
    f"omega_perp   = {OMEGA_PERP:.8f}"
)

print(
    f"barrier      = {V_SHIFT:.8f}"
)

print()


for zeta in ZETAS:

    beta = (
        np.pi*zeta/np.sqrt(A)
    )

    p_dw_ref = exact_dw_pdf(
        beta
    )

    print(
        "========================================"
    )

    print(
        f"zeta = {zeta:.6f}"
    )

    print(
        f"beta = {beta:.10f}"
    )

    print(
        "========================================"
    )

    group = []

    for angle_deg in ANGLES_DEG:

        theta = np.deg2rad(
            angle_deg
        )

        u, v = lab_to_principal(
            X,
            Y,
            theta,
        )

        exact = exact_2d_pdf(
            u,
            v,
            beta,
            p_dw_ref,
        )

        result = tensor_rlh_lab(
            X,
            Y,
            theta,
            beta,
        )

        rlh = result["pdf"]

        valid = result["valid"]

        contraction_error = (
            result[
                "contraction_error"
            ]
        )

        if np.isfinite(
            contraction_error
        ):

            max_tensor_contraction_error = max(
                max_tensor_contraction_error,
                contraction_error,
            )

        invalid_fraction = (
            1.0
            - np.mean(valid)
        )

        invalid_exact_mass = (
            integrate_2d(
                exact*(~valid)
            )
        )

        max_abs_Hxy = (
            result[
                "max_abs_Hxy"
            ]
        )

        exact_u2 = expectation(
            u*u,
            exact,
        )

        exact_v2 = expectation(
            v*v,
            exact,
        )

        exact_r2 = expectation(
            X*X+Y*Y,
            exact,
        )

        if rlh is not None:

            rlh_u2 = expectation(
                u*u,
                rlh,
            )

            rlh_v2 = expectation(
                v*v,
                rlh,
            )

            rlh_r2 = expectation(
                X*X+Y*Y,
                rlh,
            )

            u2_err = (
                100.0
                * (rlh_u2-exact_u2)
                / exact_u2
            )

            v2_err = (
                100.0
                * (rlh_v2-exact_v2)
                / exact_v2
            )

            r2_err = (
                100.0
                * (rlh_r2-exact_r2)
                / exact_r2
            )

            l1 = l1_distance(
                rlh,
                exact,
            )

            status = "VALID"

        else:

            rlh_u2 = np.nan
            rlh_v2 = np.nan
            rlh_r2 = np.nan

            u2_err = np.nan
            v2_err = np.nan
            r2_err = np.nan
            l1 = np.nan

            status = "INVALID"

        print(
            f"theta={angle_deg:5.1f} deg  "
            f"status={status:7s}  "
            f"Hxy_max={max_abs_Hxy:9.4f}  "
            f"tensor-check={contraction_error:.3e}"
        )

        if rlh is not None:

            print(
                f"    "
                f"L1={l1:.6f}  "
                f"du2={u2_err:+8.3f}%  "
                f"dv2={v2_err:+8.3f}%  "
                f"dr2={r2_err:+8.3f}%"
            )

        else:

            print(
                f"    "
                f"invalid grid="
                f"{100*invalid_fraction:.4f}%  "
                f"exact probability in invalid region="
                f"{100*invalid_exact_mass:.4f}%"
            )

        row = [
            zeta,
            beta,
            angle_deg,
            status,
            max_abs_Hxy,
            contraction_error,
            invalid_fraction,
            invalid_exact_mass,
            exact_u2,
            rlh_u2,
            u2_err,
            exact_v2,
            rlh_v2,
            v2_err,
            exact_r2,
            rlh_r2,
            r2_err,
            l1,
        ]

        rows.append(row)

        group.append(row)

    rotation_groups[zeta] = group

    print()


# ------------------------------------------------------------
# Rotation-invariance audit
# ------------------------------------------------------------

print(
    "=== ROTATION-INVARIANCE AUDIT ==="
)

max_l1_spread = 0.0
max_u2_error_spread = 0.0
max_r2_error_spread = 0.0


for zeta in ZETAS:

    group = rotation_groups[
        zeta
    ]

    valid_rows = [
        r
        for r in group
        if r[3] == "VALID"
    ]

    if not valid_rows:
        continue

    l1_values = np.array([
        r[17]
        for r in valid_rows
    ])

    u2_values = np.array([
        r[10]
        for r in valid_rows
    ])

    r2_values = np.array([
        r[16]
        for r in valid_rows
    ])

    l1_spread = (
        np.max(l1_values)
        - np.min(l1_values)
    )

    u2_spread = (
        np.max(u2_values)
        - np.min(u2_values)
    )

    r2_spread = (
        np.max(r2_values)
        - np.min(r2_values)
    )

    max_l1_spread = max(
        max_l1_spread,
        l1_spread,
    )

    max_u2_error_spread = max(
        max_u2_error_spread,
        u2_spread,
    )

    max_r2_error_spread = max(
        max_r2_error_spread,
        r2_spread,
    )

    print(
        f"zeta={zeta:.3f}  "
        f"L1 spread={l1_spread:.3e}  "
        f"u2-error spread={u2_spread:.3e} pp  "
        f"r2-error spread={r2_spread:.3e} pp"
    )


# ------------------------------------------------------------
# Structural PASS / FAIL
# ------------------------------------------------------------

subcaustic_valid = all(
    r[3] == "VALID"
    for r in rows
    if r[0] < 1.0
)

supercaustic_invalid = all(
    r[3] == "INVALID"
    for r in rows
    if r[0] > 1.0
)

tensor_pass = (
    max_tensor_contraction_error
    < 1.0e-12
)

rotation_pass = (
    max_l1_spread < 2.0e-3
    and max_u2_error_spread < 0.2
    and max_r2_error_spread < 0.2
)


print()
print(
    "=== GLOBAL CHECKS ==="
)

print(
    "max tensor lab/principal contraction error = "
    f"{max_tensor_contraction_error:.6e}"
)

print(
    "max L1 rotation spread                    = "
    f"{max_l1_spread:.6e}"
)

print(
    "max u2-error rotation spread              = "
    f"{max_u2_error_spread:.6e} pp"
)

print(
    "max r2-error rotation spread              = "
    f"{max_r2_error_spread:.6e} pp"
)

print()

print(
    "SUB-CAUSTIC DOMAIN:",
    "PASS"
    if subcaustic_valid
    else "FAIL"
)

print(
    "SUPER-CAUSTIC DOMAIN:",
    "PASS"
    if supercaustic_invalid
    else "FAIL"
)

print(
    "TENSOR ROTATION IDENTITY:",
    "PASS"
    if tensor_pass
    else "FAIL"
)

print(
    "OBSERVABLE ROTATION INVARIANCE:",
    "PASS"
    if rotation_pass
    else "FAIL"
)

overall = (
    subcaustic_valid
    and supercaustic_invalid
    and tensor_pass
    and rotation_pass
)

print()

print(
    "OVERALL:",
    "PASS"
    if overall
    else "FAIL"
)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

with open(
    "lha10d4_rotated_2d_stabilized_saddle.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "zeta",
        "beta",
        "angle_deg",
        "status",
        "max_abs_Hxy",
        "tensor_contraction_rel_error",
        "invalid_grid_fraction",
        "exact_probability_invalid_region",
        "u2_exact",
        "u2_RLH",
        "u2_error_pct",
        "v2_exact",
        "v2_RLH",
        "v2_error_pct",
        "r2_exact",
        "r2_RLH",
        "r2_error_pct",
        "L1_RLH_exact",
    ])

    w.writerows(rows)


# ------------------------------------------------------------
# Simple validation figure
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.6, 5.2)
)

for angle_deg in ANGLES_DEG:

    z_plot = []
    l1_plot = []

    for r in rows:

        if (
            r[2] == angle_deg
            and r[3] == "VALID"
        ):

            z_plot.append(
                r[0]
            )

            l1_plot.append(
                r[17]
            )

    ax.plot(
        z_plot,
        l1_plot,
        marker="o",
        label=(
            rf"$\theta={angle_deg:g}^\circ$"
        ),
    )


ax.axvline(
    1.0,
    linestyle="--",
)

ax.set_xlabel(
    r"local instability index $\zeta$"
)

ax.set_ylabel(
    r"$L^1(P_{\rm RLH},P_{\rm QM})$"
)

ax.set_title(
    "Rotational covariance of the 2D tensorial RLH"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "lha10d4_rotated_2d_stabilized_saddle.png",
    dpi=300,
)

fig.savefig(
    "lha10d4_rotated_2d_stabilized_saddle.pdf",
)


print()
print(
    "saved: "
    "lha10d4_rotated_2d_stabilized_saddle.csv"
)

print(
    "saved: "
    "lha10d4_rotated_2d_stabilized_saddle.png"
)

print(
    "saved: "
    "lha10d4_rotated_2d_stabilized_saddle.pdf"
)
