from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# P2E — Publication Figure 4
#
# Final independent bridge:
#
# exact finite-P QM  <->  PI-QMC  <->  tensorial RLH
#
# Central pre-caustic anchor:
#
# beta = 2.70
# P = 64
# zeta_max ~ 0.9957 < 1
#
# Uses ONLY final P0B publication ensemble.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "validation_tests"

OUT = ROOT / "publication_figures"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


npz_file = (
    DATA
    / "lhap0b_final_publication_statistics.npz"
)

csv_file = (
    DATA
    / "lhap0b_final_publication_statistics.csv"
)

a2_file = (
    DATA
    / "lha13a2_tensor_rlh_offstationary_caustic.csv"
)


# ============================================================
# Read final P0B arrays
# ============================================================


d = np.load(
    npz_file
)


P_pimc = np.asarray(
    d["P_PIMC_bin_ts"],
    dtype=float,
)

P_exact = np.asarray(
    d["P_exact_P64_bin_ts"],
    dtype=float,
)

P_cont = np.asarray(
    d["P_exact_continuum_bin_ts"],
    dtype=float,
)

P_rlh = np.asarray(
    d["P_RLH_bin_ts"],
    dtype=float,
)


if not (
    P_pimc.shape
    == P_exact.shape
    == P_cont.shape
    == P_rlh.shape
):

    raise RuntimeError(
        "Density shapes do not match."
    )


NBIN = P_exact.shape[0]


# ============================================================
# Read final scalar statistics
# ============================================================


with csv_file.open(
    newline=""
) as f:

    summary = next(
        csv.DictReader(f)
    )


def F(name):

    return float(
        summary[name]
    )


L1_trotter = F(
    "exact_P64_vs_continuum_L1"
)

L1_pimc = F(
    "PIMC_vs_exact_P64_L1"
)

L1_rlh = F(
    "RLH_vs_exact_P64_L1"
)

L1_time = F(
    "time_half_L1"
)

L1_seed = F(
    "seed_half_L1"
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

V_sem = F(
    "pimc_V_sem"
)

V_z = F(
    "pimc_V_z"
)

V_trotter_bias_pct = None


# ============================================================
# Read RLH beta=2.70 anchor
# ============================================================


with a2_file.open(
    newline=""
) as f:

    a2rows = list(
        csv.DictReader(f)
    )


a2 = min(
    a2rows,
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
    a2["V_RLH"]
)

V_rlh_error_pct = float(
    a2["V_error_pct"]
)


# finite-P energy bias directly from exact values

V_trotter_bias_pct = (
    100.0
    * (
        V_exact_P
        - V_exact_cont
    )
    / V_exact_cont
)


# ============================================================
# Independent numerical checks
# ============================================================


def L1(A, B):

    return float(
        np.mean(
            np.abs(
                A-B
            )
        )
    )


check_pimc = L1(
    P_pimc,
    P_exact,
)

check_rlh = L1(
    P_rlh,
    P_exact,
)

check_trotter = L1(
    P_exact,
    P_cont,
)


if abs(
    check_pimc-L1_pimc
) > 1e-12:

    raise RuntimeError(
        "PIMC L1 reconstruction mismatch."
    )


if abs(
    check_rlh-L1_rlh
) > 1e-12:

    raise RuntimeError(
        "RLH L1 reconstruction mismatch."
    )


if abs(
    check_trotter-L1_trotter
) > 1e-12:

    raise RuntimeError(
        "Trotter density L1 reconstruction mismatch."
    )


# ============================================================
# Residuals
# ============================================================


D_pimc = (
    P_pimc
    - P_exact
)

D_rlh = (
    P_rlh
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


density_min = float(
    min(
        np.min(P_exact),
        np.min(P_pimc),
        np.min(P_rlh),
    )
)


density_max = float(
    max(
        np.max(P_exact),
        np.max(P_pimc),
        np.max(P_rlh),
    )
)


# ============================================================
# Audit
# ============================================================


print(
    "=== FIGURE 4 NUMERICAL AUDIT ==="
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
    f"L1 PIMC vs exact P64       = "
    f"{L1_pimc:.9f}"
)

print(
    f"L1 RLH vs exact P64        = "
    f"{L1_rlh:.9f}"
)

print(
    f"L1 exact P64 vs continuum = "
    f"{L1_trotter:.9e}"
)

print()

print(
    f"whole-chain bootstrap 95%  = "
    f"[{boot_lo:.9f}, "
    f"{boot_hi:.9f}]"
)

print(
    f"RLH / bootstrap upper      = "
    f"{L1_rlh/boot_hi:.6f}"
)

print(
    f"RLH / PIMC L1              = "
    f"{L1_rlh/L1_pimc:.6f}"
)

print()

print(
    f"V exact continuum          = "
    f"{V_exact_cont:.12f}"
)

print(
    f"V exact P64                = "
    f"{V_exact_P:.12f}"
)

print(
    f"V PI-QMC                   = "
    f"{V_pimc:.12f}"
)

print(
    f"V RLH                      = "
    f"{V_rlh:.12f}"
)

print()

print(
    f"RLH energy error           = "
    f"{V_rlh_error_pct:+.6f}%"
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
# Plot setup — cleaner publication layout
# ============================================================


plt.rcParams.update({
    "font.size": 9.2,
    "axes.labelsize": 9.8,
    "axes.titlesize": 10.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig = plt.figure(
    figsize=(
        7.45,
        6.75,
    )
)


gs = fig.add_gridspec(
    2,
    3,
    hspace=0.42,
    wspace=0.30,
)


ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])


extent = (
    0.0,
    1.0,
    0.0,
    1.0,
)


# ============================================================
# Density maps
# ============================================================


density_arrays = [
    (
        ax_a,
        P_exact,
        r"(a) Exact $P=64$",
    ),
    (
        ax_b,
        P_pimc,
        "(b) PI-QMC",
    ),
    (
        ax_c,
        P_rlh,
        "(c) Tensorial RLH",
    ),
]


density_im = None


for ax, arr, title in density_arrays:

    density_im = ax.imshow(
        arr,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        vmin=density_min,
        vmax=density_max,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        r"$s$"
    )


ax_a.set_ylabel(
    r"$t$"
)


# Remove redundant y labels/ticks from middle/right top panels.

for ax in [
    ax_b,
    ax_c,
]:

    ax.set_ylabel("")


# Compact vertical colorbar attached to final density panel only.
# All three panels use exactly the same limits.

cb_density = fig.colorbar(
    density_im,
    ax=ax_c,
    orientation="vertical",
    fraction=0.046,
    pad=0.035,
)


cb_density.set_label(
    "Normalized density"
)


# ============================================================
# Residual maps
# ============================================================


residual_arrays = [
    (
        ax_d,
        D_pimc,
        r"(d) PI-QMC $-$ exact",
    ),
    (
        ax_e,
        D_rlh,
        r"(e) RLH $-$ exact",
    ),
]


res_im = None


for ax, arr, title in residual_arrays:

    res_im = ax.imshow(
        arr,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-res_lim,
        vmax=res_lim,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        r"$s$"
    )


ax_d.set_ylabel(
    r"$t$"
)

ax_e.set_ylabel(
    ""
)


cb_res = fig.colorbar(
    res_im,
    ax=ax_e,
    orientation="vertical",
    fraction=0.046,
    pad=0.035,
)


cb_res.set_label(
    "Density residual"
)


# ============================================================
# Panel (f): horizontal quantitative hierarchy
# ============================================================


categories = [
    "Trotter\n$P=64$",
    "PI-QMC",
    "RLH",
]


y = np.array([
    2.0,
    1.0,
    0.0,
])


values = np.array([
    L1_trotter,
    L1_pimc,
    L1_rlh,
])


ax_f.scatter(
    values,
    y,
    s=55,
    zorder=3,
)


# Whole-chain bootstrap interval for PI-QMC.
# Horizontal error bar because discrepancy is on x-axis.

ax_f.hlines(
    1.0,
    boot_lo,
    boot_hi,
    linewidth=2.2,
    zorder=2,
)


ax_f.scatter(
    [boot_med],
    [1.0],
    marker="|",
    s=100,
    zorder=4,
)


ax_f.text(
    boot_hi * 1.08,
    1.0,
    "95% whole-chain\nbootstrap",
    ha="left",
    va="center",
    fontsize=7.3,
)


ax_f.set_xscale(
    "log"
)


ax_f.set_yticks(
    y
)


ax_f.set_yticklabels(
    categories
)


ax_f.set_xlabel(
    r"Density discrepancy $L^1$"
)


ax_f.set_title(
    "(f) Error hierarchy"
)


ax_f.grid(
    axis="x",
    which="both",
    linewidth=0.4,
    alpha=0.30,
)


# Numeric values, positioned consistently.

for yi, xi in zip(
    y,
    values,
):

    if xi < 1e-3:

        text = (
            f"{xi:.2e}"
        )

    else:

        text = (
            f"{xi:.4f}"
        )

    ax_f.text(
        xi * 1.12,
        yi + 0.13,
        text,
        ha="left",
        va="bottom",
        fontsize=7.6,
    )


# Key energy information only.
# Keep this concise; density is the main point of panel f.

energy_text = (
    r"$\beta=2.70$"
    "\n"
    r"$\zeta_{\max}=0.9957<1$"
    "\n"
    r"$\Delta V_{\rm RLH}=+12.45\%$"
    "\n"
    r"$P=64$ Trotter: $-0.0173\%$"
)


ax_f.text(
    0.98,
    0.97,
    energy_text,
    transform=ax_f.transAxes,
    ha="right",
    va="top",
    fontsize=7.45,
    bbox={
        "boxstyle":
            "round,pad=0.30",

        "facecolor":
            "white",

        "edgecolor":
            "0.65",

        "linewidth":
            0.6,

        "alpha":
            0.92,
    },
)


# Give room to the bootstrap label on the right.

xmin = (
    L1_trotter / 2.5
)

xmax = (
    L1_rlh * 2.4
)


ax_f.set_xlim(
    xmin,
    xmax,
)


ax_f.set_ylim(
    -0.55,
    2.55,
)


# ============================================================
# Save
# ============================================================


fig.subplots_adjust(
    top=0.975,
    bottom=0.09,
    left=0.075,
    right=0.96,
)


pdf = (
    OUT
    / "Fig4_exact_pimc_rlh_v2.pdf"
)

png = (
    OUT
    / "Fig4_exact_pimc_rlh_v2.png"
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

# ============================================================
# Publication provenance table
# ============================================================


prov = (
    OUT
    / "Fig4_key_data.csv"
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

    values_out = [
        (
            "beta",
            beta,
        ),
        (
            "zeta_max",
            zeta_max,
        ),
        (
            "density_L1_exactP64_vs_continuum",
            L1_trotter,
        ),
        (
            "density_L1_PIMC_vs_exactP64",
            L1_pimc,
        ),
        (
            "density_L1_RLH_vs_exactP64",
            L1_rlh,
        ),
        (
            "density_bootstrap_q025",
            boot_lo,
        ),
        (
            "density_bootstrap_median",
            boot_med,
        ),
        (
            "density_bootstrap_q975",
            boot_hi,
        ),
        (
            "V_exact_continuum",
            V_exact_cont,
        ),
        (
            "V_exact_P64",
            V_exact_P,
        ),
        (
            "V_PIMC",
            V_pimc,
        ),
        (
            "V_RLH",
            V_rlh,
        ),
        (
            "V_RLH_error_pct",
            V_rlh_error_pct,
        ),
        (
            "V_P64_Trotter_bias_pct",
            V_trotter_bias_pct,
        ),
        (
            "V_PIMC_z",
            V_z,
        ),
    ]

    w.writerows(
        values_out
    )


print()

print(
    f"saved: {pdf.relative_to(ROOT)}"
)

print(
    f"saved: {png.relative_to(ROOT)}"
)

print(
    f"saved: {prov.relative_to(ROOT)}"
)
