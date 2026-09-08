from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# FIGURE 4 — FINAL PUBLICATION CANDIDATE
#
# Exact finite-P  <->  PI-QMC  <->  tensorial RLH
#
# Uses only validated P0B publication arrays.
# No interpolation, smoothing, or re-binning.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation_tests"
OUT = ROOT / "publication_figures"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


NPZ = DATA / "lhap0b_final_publication_statistics.npz"
CSV = DATA / "lhap0b_final_publication_statistics.csv"
A2  = DATA / "lha13a2_tensor_rlh_offstationary_caustic.csv"


# ============================================================
# Final validated arrays
# ============================================================


d = np.load(NPZ)


P_exact = np.asarray(
    d["P_exact_P64_bin_ts"],
    dtype=float,
)

P_pimc = np.asarray(
    d["P_PIMC_bin_ts"],
    dtype=float,
)

P_rlh = np.asarray(
    d["P_RLH_bin_ts"],
    dtype=float,
)

P_cont = np.asarray(
    d["P_exact_continuum_bin_ts"],
    dtype=float,
)


if not (
    P_exact.shape
    == P_pimc.shape
    == P_rlh.shape
    == P_cont.shape
):
    raise RuntimeError(
        "Density-array shapes differ."
    )


NBIN = P_exact.shape[0]


if P_exact.shape != (
    NBIN,
    NBIN,
):
    raise RuntimeError(
        "Expected square registered density bins."
    )


# ============================================================
# Exact bin edges
#
# IMPORTANT:
# these are the actual 16x16 bin boundaries.
# No half-bin shift and no visual interpolation.
# ============================================================


edges = np.linspace(
    0.0,
    1.0,
    NBIN + 1,
)


S_edge, T_edge = np.meshgrid(
    edges,
    edges,
)


# ============================================================
# Final scalar statistics
# ============================================================


with CSV.open(
    newline=""
) as f:

    row = next(
        csv.DictReader(f)
    )


def F(name):
    return float(row[name])


L1_trotter = F(
    "exact_P64_vs_continuum_L1"
)

L1_pimc = F(
    "PIMC_vs_exact_P64_L1"
)

L1_rlh = F(
    "SPLH_vs_exact_P64_L1"
)

boot_lo = F(
    "density_boot_q025"
)

boot_med = F(
    "density_boot_median"
)

boot_hi = F(
    "density_boot_q975"
)

V_exact_P = F(
    "exact_finiteP_V"
)

V_pimc = F(
    "pimc_V_mean"
)

V_z = F(
    "pimc_V_z"
)


# ============================================================
# beta=2.70 SPLH anchor
# ============================================================


with A2.open(
    newline=""
) as f:

    a2_rows = list(
        csv.DictReader(f)
    )


a2 = min(
    a2_rows,
    key=lambda r:
        abs(
            float(r["beta"])
            - 2.70
        ),
)


beta = float(
    a2["beta"]
)

zeta_max = float(
    a2["zeta_max"]
)

V_exact_cont = float(
    a2["V_exact_BZ"]
)

V_rlh = float(
    a2["V_SPLH"]
)

V_rlh_error_pct = float(
    a2["V_error_pct"]
)


V_trotter_bias_pct = (
    100.0
    * (
        V_exact_P
        - V_exact_cont
    )
    / V_exact_cont
)


# ============================================================
# Independent reconstruction audit
# ============================================================


def L1(A, B):

    return float(
        np.mean(
            np.abs(
                A-B
            )
        )
    )


check_trotter = L1(
    P_exact,
    P_cont,
)

check_pimc = L1(
    P_pimc,
    P_exact,
)

check_rlh = L1(
    P_splh,
    P_exact,
)


for name, calc, saved in [
    (
        "Trotter",
        check_trotter,
        L1_trotter,
    ),
    (
        "PI-QMC",
        check_pimc,
        L1_pimc,
    ),
    (
        "SPLH",
        check_splh,
        L1_splh,
    ),
]:

    if abs(
        calc-saved
    ) > 1e-12:

        raise RuntimeError(
            f"{name} L1 reconstruction mismatch."
        )


# ============================================================
# Residual maps
# ============================================================


D_pimc = (
    P_pimc
    - P_exact
)


D_rlh = (
    P_splh
    - P_exact
)


res_lim = float(
    max(
        np.max(
            np.abs(D_pimc)
        ),
        np.max(
            np.abs(D_rlh)
        ),
    )
)


# ============================================================
# Shared density scale
# ============================================================


dens_min = 0.0


dens_max = float(
    max(
        np.max(P_exact),
        np.max(P_pimc),
        np.max(P_rlh),
    )
)


# ============================================================
# Numerical audit
# ============================================================


print(
    "=== FIGURE 4 FINAL NUMERICAL AUDIT ==="
)

print()

print(
    f"beta                       = "
    f"{beta:.6f}"
)

print(
    f"zeta_max                   = "
    f"{zeta_max:.9f}"
)

print()

print(
    f"L1 exact P64 vs continuum = "
    f"{L1_trotter:.9e}"
)

print(
    f"L1 PI-QMC vs exact P64     = "
    f"{L1_pimc:.9f}"
)

print(
    f"L1 SPLH vs exact P64        = "
    f"{L1_splh:.9f}"
)

print()

print(
    f"PI-QMC bootstrap 95%       = "
    f"[{boot_lo:.9f}, "
    f"{boot_hi:.9f}]"
)

print(
    f"bootstrap median           = "
    f"{boot_med:.9f}"
)

print()

print(
    f"SPLH / PI-QMC L1            = "
    f"{L1_splh/L1_pimc:.6f}"
)

print(
    f"SPLH / bootstrap upper      = "
    f"{L1_splh/boot_hi:.6f}"
)

print(
    f"SPLH / Trotter L1           = "
    f"{L1_splh/L1_trotter:.2f}"
)

print()

print(
    f"SPLH energy error           = "
    f"{V_splh_error_pct:+.6f}%"
)

print(
    f"P64 Trotter energy bias    = "
    f"{V_trotter_bias_pct:+.8f}%"
)

print(
    f"PI-QMC energy z            = "
    f"{V_z:+.6f}"
)


# ============================================================
# Publication style
#
# Keep usetex off while finalizing layout.
# Vector PDF remains publication quality.
# ============================================================


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 9.4,
    "axes.labelsize": 9.8,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Layout
# ============================================================


fig = plt.figure(
    figsize=(
        7.25,
        5.45,
    )
)


gs = fig.add_gridspec(
    2,
    3,
    hspace=0.38,
    wspace=0.42,
    width_ratios=[
        1.0,
        1.0,
        1.08,
    ],
)


ax_a = fig.add_subplot(
    gs[0, 0]
)

ax_b = fig.add_subplot(
    gs[0, 1]
)

ax_c = fig.add_subplot(
    gs[0, 2]
)

ax_d = fig.add_subplot(
    gs[1, 0]
)

ax_e = fig.add_subplot(
    gs[1, 1]
)

ax_f = fig.add_subplot(
    gs[1, 2]
)


# ============================================================
# Helper for exact registered bin maps
# ============================================================


def map_panel(
    ax,
    arr,
    *,
    title,
    cmap,
    vmin,
    vmax,
    ylabel=False,
):

    im = ax.pcolormesh(
        S_edge,
        T_edge,
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=False,
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        r"$s$"
    )

    if ylabel:

        ax.set_ylabel(
            r"$t$"
        )

    else:

        ax.set_ylabel(
            ""
        )

    ax.set_title(
        title
    )

    ax.set_xticks([
        0.0,
        0.5,
        1.0,
    ])

    ax.set_yticks([
        0.0,
        0.5,
        1.0,
    ])

    return im


# ============================================================
# Top row: identical density scale
# ============================================================


im_density = map_panel(
    ax_a,
    P_exact,
    title=r"(a) Exact $P=64$",
    cmap="viridis",
    vmin=dens_min,
    vmax=dens_max,
    ylabel=True,
)


map_panel(
    ax_b,
    P_pimc,
    title="(b) PI-QMC",
    cmap="viridis",
    vmin=dens_min,
    vmax=dens_max,
)


map_panel(
    ax_c,
    P_splh,
    title="(c) Tensorial SPLH",
    cmap="viridis",
    vmin=dens_min,
    vmax=dens_max,
)


divider_c = make_axes_locatable(
    ax_c
)


cax_density = divider_c.append_axes(
    "right",
    size="4.5%",
    pad=0.06,
)


cb_density = fig.colorbar(
    im_density,
    cax=cax_density,
)


cb_density.set_label(
    "Normalized density"
)


# ============================================================
# Bottom row: identical symmetric residual scale
# ============================================================


im_res = map_panel(
    ax_d,
    D_pimc,
    title=r"(d) PI-QMC $-$ exact",
    cmap="RdBu_r",
    vmin=-res_lim,
    vmax=res_lim,
    ylabel=True,
)


map_panel(
    ax_e,
    D_rlh,
    title=r"(e) SPLH $-$ exact",
    cmap="RdBu_r",
    vmin=-res_lim,
    vmax=res_lim,
)


divider_e = make_axes_locatable(
    ax_e
)


cax_res = divider_e.append_axes(
    "right",
    size="3.3%",
    pad=0.045,
)


cb_res = fig.colorbar(
    im_res,
    cax=cax_res,
)


cb_res.set_label(
    r"$P-P_{\rm exact}$"
)


# ============================================================
# Panel (f): simple hierarchy
#
# No energy box.
# No extra physics text.
# Caption carries beta, zeta and energy information.
# ============================================================


x = np.array([
    0,
    1,
    2,
])


vals = np.array([
    L1_trotter,
    L1_pimc,
    L1_splh,
])


ax_f.scatter(
    x,
    vals,
    s=52,
    zorder=3,
)


# whole-chain bootstrap only belongs to PI-QMC

ax_f.vlines(
    1,
    boot_lo,
    boot_hi,
    linewidth=2.0,
    zorder=2,
)


ax_f.scatter(
    [1],
    [boot_med],
    marker="_",
    s=110,
    zorder=4,
)


ax_f.text(
    0.92,
    boot_hi*1.06,
    "95% whole-chain\nbootstrap",
    ha="center",
    va="bottom",
    fontsize=7.2,
)


ax_f.set_yscale(
    "log"
)


ax_f.set_xticks(
    x
)


ax_f.set_xticklabels([
    "Trotter\n$P=64$",
    "PI-QMC",
    "SPLH",
])


ax_f.set_ylabel(
    ""
)

ax_f.text(
    0.03,
    0.96,
    r"Density $L^1$",
    transform=ax_f.transAxes,
    ha="left",
    va="top",
    fontsize=8.0,
)


ax_f.set_title(
    "(f) Quantitative hierarchy"
)


ax_f.grid(
    axis="y",
    which="both",
    linewidth=0.35,
    alpha=0.30,
)


# exact numerical labels

labels = [
    f"{L1_trotter:.2e}",
    f"{L1_pimc:.4f}",
    f"{L1_splh:.4f}",
]


for xi, yi, text in zip(
    x,
    vals,
    labels,
):

    ax_f.text(
        xi,
        yi*1.32,
        text,
        ha="center",
        va="bottom",
        fontsize=7.4,
    )


ax_f.set_ylim(
    L1_trotter/3.0,
    L1_splh*2.7,
)


# ============================================================
# Final spacing
# ============================================================


fig.subplots_adjust(
    top=0.97,
    bottom=0.11,
    left=0.075,
    right=0.965,
)


# ============================================================
# Save
# ============================================================


pdf = (
    OUT
    / "Fig4_exact_pimc_splh_FINAL.pdf"
)


png = (
    OUT
    / "Fig4_exact_pimc_splh_FINAL.png"
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


print()

print(
    f"saved: {pdf.relative_to(ROOT)}"
)

print(
    f"saved: {png.relative_to(ROOT)}"
)
