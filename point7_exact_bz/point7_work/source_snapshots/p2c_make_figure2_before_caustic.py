from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# P2C — Publication Figure 2
#
# "Accuracy fails before the caustic"
#
# IMPORTANT METHODOLOGICAL DISTINCTION:
#
# Dense LHA-11 curves:
#     Gamma-sector exact reference.
#     Used only as a visual guide to the dense trend.
#
# Quantitative markers:
#     LHA-12 BZ-averaged exact reference.
#     These are the publication anchors.
#
# No interpolation is used to invent BZ data between anchors.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "validation_tests"

OUT = ROOT / "publication_figures"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


dense_file = (
    DATA
    / "lha11b4_dense_precaustic_scan.csv"
)

bz_file = (
    DATA
    / "lha12c0a_exact_P64_multibeta_converged.csv"
)

harm_file = (
    DATA
    / "lha12c2a_harmonized_density_comparison.csv"
)


# ============================================================
# Helpers
# ============================================================


def read_csv(path):

    with path.open(
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def col(rows, name):

    return np.array([
        float(r[name])
        for r in rows
    ])


# ============================================================
# Read sources
# ============================================================


dense = read_csv(
    dense_file
)

bz = read_csv(
    bz_file
)

harm = read_csv(
    harm_file
)


beta_dense = col(
    dense,
    "beta",
)

zeta_dense = col(
    dense,
    "zeta_max",
)

energy_err_dense = col(
    dense,
    "V_error_pct",
)

density_L1_dense = col(
    dense,
    "density_L1",
)


beta_bz = col(
    bz,
    "beta",
)

V_exact_bz = col(
    bz,
    "V_continuum",
)

trotter_bias_pct = col(
    bz,
    "finiteP_bias_pct",
)


beta_h = col(
    harm,
    "beta",
)

zeta_h = col(
    harm,
    "zeta_max",
)

V_splh = col(
    harm,
    "V_splh_reconstructed",
)

density_L1_bz = col(
    harm,
    "L1_splh_bin16_vs_exact_BZ",
)


# ============================================================
# Exact join check
# ============================================================


if not np.allclose(
    beta_bz,
    beta_h,
    rtol=0.0,
    atol=1e-12,
):

    raise RuntimeError(
        "BZ energy and harmonized-density "
        "beta grids do not match."
    )


energy_err_bz = (
    100.0
    * (
        V_splh
        - V_exact_bz
    )
    / V_exact_bz
)


# ============================================================
# Caustic
# ============================================================


beta_c = (
    np.pi
    / np.sqrt(1.5)
)


zeta_c = 1.0


# ============================================================
# Anchor beta = 2.5
# ============================================================


i_anchor = int(
    np.argmin(
        np.abs(
            beta_bz - 2.5
        )
    )
)


if abs(
    beta_bz[i_anchor] - 2.5
) > 1e-12:

    raise RuntimeError(
        "beta=2.5 anchor not found."
    )


anchor_beta = beta_bz[
    i_anchor
]

anchor_zeta = zeta_h[
    i_anchor
]

anchor_energy = energy_err_bz[
    i_anchor
]

anchor_L1 = density_L1_bz[
    i_anchor
]

anchor_trotter = trotter_bias_pct[
    i_anchor
]


# ============================================================
# Numerical publication audit
# ============================================================


print(
    "=== FIGURE 2 NUMERICAL AUDIT ==="
)

print()

print(
    f"beta_c                  = "
    f"{beta_c:.12f}"
)

print(
    f"anchor beta             = "
    f"{anchor_beta:.6f}"
)

print(
    f"anchor zeta_max         = "
    f"{anchor_zeta:.9f}"
)

print(
    f"anchor splh V error      = "
    f"{anchor_energy:+.6f}%"
)

print(
    f"anchor density L1       = "
    f"{anchor_L1:.9f}"
)

print(
    f"anchor P64 Trotter bias = "
    f"{anchor_trotter:+.8f}%"
)

print()


if not (
    anchor_zeta < 1.0
):

    raise RuntimeError(
        "Central anchor is not pre-caustic."
    )


# ============================================================
# Figure geometry
# ============================================================


plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig = plt.figure(
    figsize=(
        7.15,
        6.45,
    )
)


gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[
        1.0,
        0.88,
    ],
    hspace=0.34,
    wspace=0.32,
)


ax_a = fig.add_subplot(
    gs[0, 0]
)

ax_b = fig.add_subplot(
    gs[0, 1]
)

ax_c = fig.add_subplot(
    gs[1, :]
)


# ============================================================
# Panel (a)
# Signed energy error
# ============================================================


ax_a.plot(
    beta_dense,
    energy_err_dense,
    linestyle="--",
    linewidth=1.25,
    label=r"Dense $\Gamma$-sector guide",
)


ax_a.plot(
    beta_bz,
    energy_err_bz,
    marker="o",
    linestyle="none",
    markersize=5.5,
    label="BZ exact anchors",
)


ax_a.axhline(
    0.0,
    linewidth=0.8,
    linestyle="-",
)


ax_a.axvline(
    beta_c,
    linewidth=1.15,
    linestyle=":",
)


ax_a.plot(
    [anchor_beta],
    [anchor_energy],
    marker="o",
    linestyle="none",
    markersize=7.5,
    markerfacecolor="C1",
    markeredgecolor="black",
    markeredgewidth=0.8,
)


ax_a.annotate(
    (
        r"$\beta=2.5$"
        "\n"
        r"$\zeta_{\max}=0.9746<1$"
        "\n"
        r"$\Delta V_{\rm splh}=+17.39\%$"
    ),
    xy=(
        anchor_beta,
        anchor_energy,
    ),
    xytext=(
        1.55,
        15.0,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=8.2,
)


ax_a.text(
    beta_c - 0.025,
    ax_a.get_ylim()[1]
    if ax_a.get_ylim()[1] > 0
    else 1.0,
    "",
)


ax_a.set_xlabel(
    r"Inverse temperature $\beta$"
)

ax_a.set_ylabel(
    r"splh energy error (\%)"
)

ax_a.set_title(
    "(a) Energy accuracy"
)


ax_a.legend(
    frameon=False,
    loc="upper left",
)


# ============================================================
# Panel (b)
# Density L1
# ============================================================


ax_b.plot(
    beta_dense,
    density_L1_dense,
    linestyle="--",
    linewidth=1.25,
    label=r"Dense $\Gamma$-sector guide",
)


ax_b.plot(
    beta_h,
    density_L1_bz,
    marker="o",
    linestyle="none",
    markersize=5.5,
    label="BZ exact anchors",
)


ax_b.axvline(
    beta_c,
    linewidth=1.15,
    linestyle=":",
)


ax_b.plot(
    [anchor_beta],
    [anchor_L1],
    marker="o",
    linestyle="none",
    markersize=7.5,
    markeredgecolor="black",
    markeredgewidth=0.8,
)


ax_b.annotate(
    (
        r"$L^1=0.115$"
        "\n"
        r"while $\zeta_{\max}<1$"
    ),
    xy=(
        anchor_beta,
        anchor_L1,
    ),
    xytext=(
        1.45,
        0.105,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=8.2,
)


ax_b.set_xlabel(
    r"Inverse temperature $\beta$"
)

ax_b.set_ylabel(
    r"Density error $L^1$"
)

ax_b.set_title(
    "(b) Distribution accuracy"
)


# ============================================================
# Panel (c)
# Error hierarchy
#
# Absolute values are intentional here because this panel
# compares numerical scales, not signs.
# ============================================================


abs_splh = np.abs(
    energy_err_bz
)

abs_trotter = np.abs(
    trotter_bias_pct
)


ax_c.semilogy(
    beta_bz,
    abs_splh,
    marker="o",
    linewidth=1.25,
    label=r"$|\Delta V_{\rm splh}|$",
)


ax_c.semilogy(
    beta_bz,
    abs_trotter,
    marker="s",
    linewidth=1.25,
    label=r"$|\Delta V_{P=64}-\Delta V_{\infty}|$",
)


ax_c.axvline(
    beta_c,
    linewidth=1.15,
    linestyle=":",
)


ratio_anchor = (
    abs_splh[i_anchor]
    / abs_trotter[i_anchor]
)


ax_c.annotate(
    (
        r"$\beta=2.5:$ "
        + f"{ratio_anchor:.0f}"
        + r"$\times$ separation"
    ),
    xy=(
        anchor_beta,
        abs_splh[i_anchor],
    ),
    xytext=(
        1.42,
        3.0,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=8.3,
)


ax_c.set_xlabel(
    r"Inverse temperature $\beta$"
)

ax_c.set_ylabel(
    r"Absolute energy error (\%)"
)

ax_c.set_title(
    "(c) Approximation error versus primitive Trotter error"
)


ax_c.legend(
    frameon=False,
    ncol=2,
    loc="lower left",
)


# ============================================================
# Common caustic labels
# ============================================================


for ax in [
    ax_a,
    ax_b,
    ax_c,
]:

    ymin, ymax = ax.get_ylim()

    if ax.get_yscale() == "log":
        y_text = ymax / 1.15
    else:
        y_text = ymax - 0.055*(ymax-ymin)

    ax.text(
        beta_c + 0.018,
        y_text,
        r"$\beta_c$",
        ha="left",
        va="top",
        fontsize=8.2,
    )

    ax.set_xlim(
        0.43,
        beta_c + 0.12,
    )


# ============================================================
# Panel labels are already embedded in titles.
# Clean layout.
# ============================================================




fig.subplots_adjust(
    top=0.975,
    bottom=0.09,
    left=0.10,
    right=0.985,
)


# ============================================================
# Save
# ============================================================


pdf = (
    OUT
    / "Fig2_before_caustic.pdf"
)

png = (
    OUT
    / "Fig2_before_caustic.png"
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


plt.close(fig)


# ============================================================
# Save exact plotted anchor data for provenance
# ============================================================


source_csv = (
    OUT
    / "Fig2_BZ_anchor_data.csv"
)


with source_csv.open(
    "w",
    newline="",
) as f:

    import csv as csv_module

    w = csv_module.writer(f)

    w.writerow([
        "beta",
        "zeta_max",
        "V_exact_BZ",
        "V_splh",
        "splh_energy_error_pct",
        "splh_density_L1_vs_exact_BZ",
        "P64_Trotter_bias_pct",
    ])

    for i in range(
        len(beta_bz)
    ):

        w.writerow([
            beta_bz[i],
            zeta_h[i],
            V_exact_bz[i],
            V_splh[i],
            energy_err_bz[i],
            density_L1_bz[i],
            trotter_bias_pct[i],
        ])


print(
    f"saved: {pdf.relative_to(ROOT)}"
)

print(
    f"saved: {png.relative_to(ROOT)}"
)

print(
    f"saved: {source_csv.relative_to(ROOT)}"
)
