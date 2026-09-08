from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ============================================================
# FIGURE 1 — Tensorial SPLH and Gaussian caustic
#
# Panels:
#   (a) coordinate-invariant tensorial construction
#   (b) exact anisotropic-harmonic validation
#   (c) negative-curvature spectral response
#   (d) first Gaussian fluctuation mode and caustic
#
# Uses validated LHA-10D1 / LHA-10D2 outputs.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
OUT = ROOT / "figures"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


D1 = DATA / "lha10d1_anisotropic_harmonic_2d.csv"
D2 = DATA / "lha10d2_negative_curvature_caustic.csv"


# ============================================================
# Helpers
# ============================================================


def read_csv(path):

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fcol(rows, key):

    return np.array([
        float(r[key])
        for r in rows
    ])


d1 = read_csv(D1)
d2 = read_csv(D2)


# ============================================================
# Panel (b) dataset
#
# Use the strongest anisotropy / rotated case:
# omega_ratio = 6, theta = 57 deg.
# ============================================================


harm = [
    r for r in d1
    if (
        abs(float(r["omega_ratio"]) - 6.0) < 1e-12
        and
        abs(float(r["angle_deg"]) - 57.0) < 1e-12
    )
]


if not harm:
    raise RuntimeError(
        "Could not find omega_ratio=6, angle=57 validation branch."
    )


harm.sort(
    key=lambda r: float(r["beta_star"])
)


beta_h = fcol(
    harm,
    "beta_star"
)


err_tensor = np.abs(
    fcol(
        harm,
        "tensor_cov_rel_error"
    )
)


err_trace = np.abs(
    fcol(
        harm,
        "trace_cov_rel_error"
    )
)


err_geom = np.abs(
    fcol(
        harm,
        "geom_cov_rel_error"
    )
)


# Machine-zero values cannot appear on a log axis.
floor = 1e-16


err_tensor_plot = np.maximum(
    err_tensor,
    floor
)


# ============================================================
# Panel (c,d) dataset
#
# D2 contains repeated zeta values at several finite-product
# truncations. Xi and lambda1 are analytic / truncation
# independent, so retain one record per zeta, preferring the
# largest nmodes as the most stringent stored check.
# ============================================================


best_by_zeta = {}


for r in d2:

    z = float(r["zeta"])
    n = int(r["nmodes"])

    if (
        z not in best_by_zeta
        or
        n > int(best_by_zeta[z]["nmodes"])
    ):
        best_by_zeta[z] = r


caustic_rows = list(
    best_by_zeta.values()
)


caustic_rows.sort(
    key=lambda r: float(r["zeta"])
)


zeta = fcol(
    caustic_rows,
    "zeta"
)


lambda1 = fcol(
    caustic_rows,
    "lambda1"
)


Xi_neg = np.array([
    float(r["Xi_negative_curvature"])
    if r["Xi_negative_curvature"].lower() != "nan"
    else np.nan
    for r in caustic_rows
])


mask_pre = (
    zeta < 1.0
)


# ============================================================
# Numerical audit
# ============================================================


print(
    "=== FIGURE 1 NUMERICAL AUDIT ==="
)

print()

print(
    "anisotropic branch:"
)

print(
    f"  beta* range              = "
    f"{beta_h.min():.6g} .. {beta_h.max():.6g}"
)

print(
    f"  max tensor covariance err= "
    f"{np.max(err_tensor):.6e}"
)

print(
    f"  max trace-scalar err     = "
    f"{np.max(err_trace):.6f}"
)

print(
    f"  max geom-scalar err      = "
    f"{np.max(err_geom):.6f}"
)

print()

i_last = int(
    np.argmax(beta_h)
)


print(
    f"at beta*={beta_h[i_last]:.1f}:"
)

print(
    f"  tensor error             = "
    f"{err_tensor[i_last]:.6e}"
)

print(
    f"  trace scalar error       = "
    f"{err_trace[i_last]:.6f}"
)

print(
    f"  geom scalar error        = "
    f"{err_geom[i_last]:.6f}"
)

print()

# Find closest stored points below / above caustic.

below = np.where(
    zeta < 1.0
)[0]


above = np.where(
    zeta > 1.0
)[0]


if len(below) == 0 or len(above) == 0:
    raise RuntimeError(
        "D2 must contain data on both sides of zeta=1."
    )


ib = below[
    np.argmax(
        zeta[below]
    )
]


ia = above[
    np.argmin(
        zeta[above]
    )
]


print(
    "caustic bracket:"
)

print(
    f"  below: zeta={zeta[ib]:.6f}, "
    f"lambda1={lambda1[ib]:+.6e}"
)

print(
    f"  above: zeta={zeta[ia]:.6f}, "
    f"lambda1={lambda1[ia]:+.6e}"
)

print()


# ============================================================
# Publication style
# ============================================================


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 9.4,
    "axes.labelsize": 9.8,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 7.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig = plt.figure(
    figsize=(
        7.25,
        5.55,
    )
)


gs = fig.add_gridspec(
    2,
    2,
    hspace=0.38,
    wspace=0.34,
)


ax_a = fig.add_subplot(
    gs[0, 0]
)

ax_b = fig.add_subplot(
    gs[0, 1]
)

ax_c = fig.add_subplot(
    gs[1, 0]
)

ax_d = fig.add_subplot(
    gs[1, 1]
)


# ============================================================
# Panel (a)
# Tensorial construction schematic
# ============================================================


ax_a.set_xlim(
    0.0,
    1.0,
)

ax_a.set_ylim(
    0.0,
    1.0,
)

ax_a.axis(
    "off"
)


ax_a.set_title(
    "(a) Tensorial local-harmonic construction"
)


# Classical/local quadratic ellipse

ellipse_local = Ellipse(
    (0.23, 0.54),
    width=0.31,
    height=0.16,
    angle=32,
    fill=False,
    linewidth=1.4,
)


ax_a.add_patch(
    ellipse_local
)


# principal axes

theta = np.deg2rad(
    32.0
)


e1 = np.array([
    np.cos(theta),
    np.sin(theta),
])


e2 = np.array([
    -np.sin(theta),
    np.cos(theta),
])


origin = np.array([
    0.23,
    0.54,
])


for vec, label, scale in [
    (
        e1,
        r"$\Lambda_1$",
        0.18,
    ),
    (
        e2,
        r"$\Lambda_2$",
        0.12,
    ),
]:

    end = (
        origin
        + scale*vec
    )

    ax_a.annotate(
        "",
        xy=end,
        xytext=origin,
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.0,
        },
    )

    ax_a.text(
        end[0] + 0.015,
        end[1] + 0.015,
        label,
        fontsize=8.5,
    )


ax_a.text(
    0.23,
    0.78,
    r"$W=\mathsf{M}^{-1/2}H\mathsf{M}^{-1/2}$" + "\n" + r"$=R\,{\rm diag}(\Lambda_a)\,R^T$",
    ha="center",
    fontsize=9.0,
)


# Arrow to quantum tensor map

ax_a.annotate(
    "",
    xy=(0.58, 0.54),
    xytext=(0.40, 0.54),
    arrowprops={
        "arrowstyle": "->",
        "lw": 1.2,
    },
)


ax_a.text(
    0.49,
    0.59,
    "spectral\nmapping",
    ha="center",
    va="bottom",
    fontsize=7.6,
)


# Quantum-smearing ellipse

ellipse_q = Ellipse(
    (0.76, 0.54),
    width=0.23,
    height=0.33,
    angle=32,
    fill=False,
    linewidth=1.5,
)


ax_a.add_patch(
    ellipse_q
)


ax_a.text(
    0.76,
    0.79,
    (
        r"$\Xi_\beta(W)$"
        "\n"
        r"$=R\,{\rm diag}(\Xi_a)\,R^T$"
    ),
    ha="center",
    fontsize=8.7,
)


ax_a.text(
    0.50,
    0.20,
    (
        "Quantum smearing follows the local mass-weighted\n"
        "normal modes and is generally tensorial."
    ),
    ha="center",
    va="center",
    fontsize=8.4,
)


# ============================================================
# Panel (b)
# Anisotropic harmonic exactness
# ============================================================


ax_b.loglog(
    beta_h,
    err_tensor_plot,
    marker="o",
    linewidth=1.3,
    label="Tensorial",
)


ax_b.loglog(
    beta_h,
    err_trace,
    marker="s",
    linewidth=1.3,
    label="Scalar: trace",
)


ax_b.loglog(
    beta_h,
    err_geom,
    marker="^",
    linewidth=1.3,
    label="Scalar: geometric",
)


ax_b.set_xlabel(
    r"Inverse temperature $\beta^*$"
)


ax_b.set_ylabel(
    "Relative covariance error"
)


ax_b.set_title(
    r"(b) Rotated anisotropic harmonic oscillator"
)


ax_b.legend(
    frameon=False,
    loc="upper left",
)


ax_b.grid(
    which="both",
    linewidth=0.35,
    alpha=0.28,
)


ax_b.text(
    0.97,
    0.06,
    (
        r"$\omega_2/\omega_1=6$"
        "\n"
        r"$\theta=57^\circ$"
    ),
    transform=ax_b.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.8,
)


# ============================================================
# Panel (c)
# Negative-curvature spectral response
#
# eta = pi*zeta/2, therefore:
#
# Xi = tan(pi*zeta/2)/(pi*zeta/2).
# ============================================================


ax_c.plot(
    zeta[mask_pre],
    Xi_neg[mask_pre],
    marker="o",
    linewidth=1.3,
)


# Analytic guide only, from the validated formula.

zg = np.linspace(
    0.0,
    0.985,
    500,
)


eta = (
    0.5*np.pi*zg
)


Xi_guide = np.ones_like(
    zg
)


nz = (
    zg > 0.0
)


Xi_guide[nz] = (
    np.tan(
        eta[nz]
    )
    / eta[nz]
)


ax_c.plot(
    zg,
    Xi_guide,
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
)


ax_c.axvline(
    1.0,
    linestyle=":",
    linewidth=1.2,
)


ax_c.axvspan(
    1.0,
    1.08,
    alpha=0.08,
)


ax_c.text(
    1.012,
    0.92,
    "undefined",
    transform=ax_c.get_xaxis_transform(),
    rotation=90,
    va="top",
    fontsize=7.6,
)


ax_c.text(
    0.04,
    0.91,
    (
        r"$\Xi(\Lambda<0)="
        r"\tan(\pi\zeta/2)/(\pi\zeta/2)$"
    ),
    transform=ax_c.transAxes,
    ha="left",
    va="top",
    fontsize=8.2,
)


ax_c.set_xlim(
    0.0,
    1.08,
)


ax_c.set_xlabel(
    r"Negative-curvature parameter $\zeta$"
)


ax_c.set_ylabel(
    r"Spectral response $\Xi$"
)


ax_c.set_title(
    "(c) Negative curvature approaches a caustic"
)


# ============================================================
# Panel (d)
# First fluctuation eigenvalue
# ============================================================


ax_d.plot(
    zeta,
    lambda1,
    marker="o",
    linewidth=1.3,
)


ax_d.axhline(
    0.0,
    linewidth=0.9,
)


ax_d.axvline(
    1.0,
    linestyle=":",
    linewidth=1.2,
)


ax_d.axvspan(
    1.0,
    1.08,
    alpha=0.08,
)


ax_d.annotate(
    r"$\lambda_1=0$",
    xy=(
        1.0,
        0.0,
    ),
    xytext=(
        0.76,
        0.11,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=8.2,
)


ax_d.text(
    0.04,
    0.91,
    (
        r"$\lambda^{\rm fl}_{n,a}=(n\pi/\beta\hbar)^2+\Lambda_a$"
        "\n"
        r"$\zeta<1\ \Longleftrightarrow\ \lambda_1>0$"
    ),
    transform=ax_d.transAxes,
    ha="left",
    va="top",
    fontsize=8.1,
)


ax_d.set_xlim(
    0.0,
    1.08,
)


ax_d.set_xlabel(
    r"Negative-curvature parameter $\zeta$"
)


ax_d.set_ylabel(
    r"First fluctuation eigenvalue $\lambda_1$"
)


ax_d.set_title(
    "(d) Gaussian existence criterion"
)


# ============================================================
# Save
# ============================================================


fig.subplots_adjust(
    top=0.97,
    bottom=0.09,
    left=0.09,
    right=0.98,
)


pdf = (
    OUT
    / "Fig1_tensorial_SPLH_caustic.pdf"
)


png = (
    OUT
    / "Fig1_tensorial_SPLH_caustic.png"
)


fig.savefig(
    pdf,
    bbox_inches="tight",
)


fig.savefig(
    png,
    dpi=400,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# Provenance summary
# ============================================================


prov = (
    OUT
    / "Fig1_key_data.csv"
)


with prov.open(
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "quantity",
        "value",
    ])

    w.writerow([
        "harmonic_omega_ratio",
        6.0,
    ])

    w.writerow([
        "harmonic_angle_deg",
        57.0,
    ])

    w.writerow([
        "max_tensor_cov_rel_error",
        np.max(err_tensor),
    ])

    w.writerow([
        "max_trace_scalar_cov_rel_error",
        np.max(err_trace),
    ])

    w.writerow([
        "max_geom_scalar_cov_rel_error",
        np.max(err_geom),
    ])

    w.writerow([
        "caustic_zeta",
        1.0,
    ])


print(
    f"saved: {pdf.relative_to(ROOT)}"
)

print(
    f"saved: {png.relative_to(ROOT)}"
)

print(
    f"saved: {prov.relative_to(ROOT)}"
)
