from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# P2D — Publication Figure 3
#
# Off-stationary first caustic and probability-weighted
# interpretation of Gaussian-invalid regions.
#
# No new physics is calculated here. All quantities are
# reconstructed from validated LHA-13 outputs.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
OUT = ROOT / "figures"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


geom_file = DATA / "lha13a0_asymmetric_moire_geometry.npz"

stationary_file = DATA / "lha13a0_stationary_points.csv"

prob_file = DATA / "lha13a3_probability_weighted_offstationary_caustic.csv"


# ============================================================
# Read geometry
# ============================================================


g = np.load(
    geom_file
)


A = np.asarray(
    g["amplitudes"],
    dtype=float,
)

phi = np.asarray(
    g["phases"],
    dtype=float,
)

axis = np.asarray(
    g["axis"],
    dtype=float,
)

V_raw = np.asarray(
    g["V_grid"],
    dtype=float,
)

kmin_raw = np.asarray(
    g["kappa_min"],
    dtype=float,
)

beta_c_saved = float(
    g["beta_c_global"]
)


# ============================================================
# Stationary points
# ============================================================


with stationary_file.open(
    newline=""
) as f:

    stationary = list(
        csv.DictReader(f)
    )


minimum_row = [
    r for r in stationary
    if r["type"] == "minimum"
][0]


s_min = float(
    minimum_row["s"]
)

t_min = float(
    minimum_row["t"]
)


# ============================================================
# Analytic potential only for array-orientation audit.
#
# Reduced coordinates:
#
# theta1 =  2 pi s + phi1
# theta2 =  2 pi t + phi2
# theta3 = -2 pi(s+t) + phi3
# ============================================================


def U_analytic(s, t):

    return -(
        A[0]
        * np.cos(
            2*np.pi*s
            + phi[0]
        )
        +
        A[1]
        * np.cos(
            2*np.pi*t
            + phi[1]
        )
        +
        A[2]
        * np.cos(
            -2*np.pi*(s+t)
            + phi[2]
        )
    )


Umin = float(
    U_analytic(
        s_min,
        t_min,
    )
)


def V_analytic(s, t):

    return (
        U_analytic(s, t)
        - Umin
    )


# ============================================================
# Determine stored grid orientation instead of assuming it.
#
# Convention needed for plotting:
#
#   array[t_index, s_index]
#
# so x = s, y = t.
# ============================================================


S_ts = axis[None, :]
T_ts = axis[:, None]


V_candidate_ts = V_analytic(
    S_ts,
    T_ts,
)


S_st = axis[:, None]
T_st = axis[None, :]


V_candidate_st = V_analytic(
    S_st,
    T_st,
)


err_ts = float(
    np.max(
        np.abs(
            V_raw
            - V_candidate_ts
        )
    )
)


err_st = float(
    np.max(
        np.abs(
            V_raw
            - V_candidate_st
        )
    )
)


if err_ts <= err_st:

    layout = "axis0=t, axis1=s"

    V = V_raw

    kmin = kmin_raw

else:

    layout = "axis0=s, axis1=t"

    V = V_raw.T

    kmin = kmin_raw.T


orientation_error = min(
    err_ts,
    err_st,
)


# ============================================================
# Global first-caustic point
# ============================================================


it, js = np.unravel_index(
    np.argmin(kmin),
    kmin.shape,
)


s_caustic = float(
    axis[js]
)

t_caustic = float(
    axis[it]
)


kappa_global = float(
    kmin[it, js]
)


beta_c_calc = float(
    np.pi
    / np.sqrt(
        -kappa_global
    )
)


V_caustic = float(
    V[it, js]
)


# ============================================================
# Pre-caustic stability field
# ============================================================


BETA_PRE = 2.70


zeta = (
    BETA_PRE
    / np.pi
    * np.sqrt(
        np.maximum(
            0.0,
            -kmin,
        )
    )
)


zeta_max = float(
    np.max(zeta)
)


# ============================================================
# Stationary local caustics
# ============================================================


finite_stationary = []


for row in stationary:

    b = row[
        "beta_c_local"
    ]

    if b.lower() == "inf":
        continue

    finite_stationary.append({
        "id":
            int(row["id"]),

        "type":
            row["type"],

        "s":
            float(row["s"]),

        "t":
            float(row["t"]),

        "V":
            float(row["V"]),

        "beta_c":
            float(b),
    })


finite_stationary.sort(
    key=lambda x: x["beta_c"]
)


# ============================================================
# Probability-weighted invalid-region data
# ============================================================


with prob_file.open(
    newline=""
) as f:

    prob_rows = list(
        csv.DictReader(f)
    )


beta_prob = np.array([
    float(r["beta"])
    for r in prob_rows
])


invalid_area = np.array([
    float(r["invalid_area_fraction"])
    for r in prob_rows
])


invalid_prob = np.array([
    float(r["exact_invalid_probability"])
    for r in prob_rows
])


ratio_prob_area = np.array([
    float(r["invalid_probability_over_area"])
    if r["invalid_probability_over_area"].lower() != "nan"
    else np.nan
    for r in prob_rows
])


# ============================================================
# Publication numerical audit
# ============================================================


print(
    "=== FIGURE 3 NUMERICAL AUDIT ==="
)

print()

print(
    f"array layout                    = {layout}"
)

print(
    f"orientation max abs error       = "
    f"{orientation_error:.6e}"
)

print()

print(
    f"global kappa_min                = "
    f"{kappa_global:.12f}"
)

print(
    f"beta_c from global kappa        = "
    f"{beta_c_calc:.12f}"
)

print(
    f"beta_c stored                   = "
    f"{beta_c_saved:.12f}"
)

print(
    f"|delta beta_c|                  = "
    f"{abs(beta_c_calc-beta_c_saved):.6e}"
)

print()

print(
    f"global caustic reduced coords   = "
    f"(s,t)=({s_caustic:.9f}, "
    f"{t_caustic:.9f})"
)

print(
    f"V at global caustic point       = "
    f"{V_caustic:.9f}"
)

print()

print(
    f"zeta_max(beta=2.70)             = "
    f"{zeta_max:.9f}"
)

print()

print(
    "stationary-point local caustics:"
)

for x in finite_stationary:

    print(
        f"  id={x['id']} "
        f"{x['type']:8s} "
        f"beta_c={x['beta_c']:.9f} "
        f"V={x['V']:.6f}"
    )


stationary_beta_min = min(
    x["beta_c"]
    for x in finite_stationary
)


if not (
    beta_c_calc
    < stationary_beta_min
):

    raise RuntimeError(
        "Global caustic is not earlier "
        "than all stationary-point caustics."
    )


if not (
    zeta_max < 1.0
):

    raise RuntimeError(
        "beta=2.70 should still be pre-caustic."
    )


# beta = 3.0 anchor for panel (d)

i3 = int(
    np.argmin(
        np.abs(
            beta_prob-3.0
        )
    )
)


if abs(
    beta_prob[i3]-3.0
) > 1e-12:

    raise RuntimeError(
        "beta=3.0 probability anchor missing."
    )


print()

print(
    f"beta=3 invalid area             = "
    f"{100*invalid_area[i3]:.6f}%"
)

print(
    f"beta=3 exact invalid probability= "
    f"{100*invalid_prob[i3]:.6f}%"
)

print(
    f"beta=3 probability/area ratio   = "
    f"{ratio_prob_area[i3]:.6f}"
)


# ============================================================
# Figure setup
# ============================================================


plt.rcParams.update({
    "font.size": 9.2,
    "axes.labelsize": 9.8,
    "axes.titlesize": 10,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig = plt.figure(
    figsize=(
        7.2,
        6.55,
    )
)


gs = fig.add_gridspec(
    2,
    2,
    hspace=0.34,
    wspace=0.30,
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
# Panel (a): asymmetric potential
# ============================================================


im_a = ax_a.contourf(
    axis,
    axis,
    V,
    levels=30,
)


cb_a = fig.colorbar(
    im_a,
    ax=ax_a,
    pad=0.025,
)


cb_a.set_label(
    r"$V(s,t)$"
)


marker_map = {
    "minimum":
        "o",

    "saddle":
        "^",

    "maximum":
        "s",
}


used_labels = set()


for row in stationary:

    typ = row["type"]

    label = typ.capitalize()

    if label in used_labels:
        label = None
    else:
        used_labels.add(label)

    ax_a.scatter(
        float(row["s"]),
        float(row["t"]),
        marker=marker_map.get(
            typ,
            "o",
        ),
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        label=label,
    )


ax_a.scatter(
    s_caustic,
    t_caustic,
    marker="*",
    s=115,
    facecolors="white",
    edgecolors="black",
    linewidths=1.0,
    label="_nolegend_",
)


ax_a.annotate(
    "off-stationary\nfirst caustic",
    xy=(
        s_caustic,
        t_caustic,
    ),
    xytext=(
        0.64,
        0.57,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=7.7,
)


ax_a.set_xlabel(
    r"Reduced coordinate $s$"
)

ax_a.set_ylabel(
    r"Reduced coordinate $t$"
)

ax_a.set_title(
    "(a) Asymmetric periodic landscape"
)


ax_a.legend(
    frameon=False,
    loc="upper left",
)


ax_a.set_aspect(
    "equal"
)


# ============================================================
# Panel (b): pre-caustic zeta field
# ============================================================


im_b = ax_b.contourf(
    axis,
    axis,
    zeta,
    levels=np.linspace(
        0.0,
        1.0,
        31,
    ),
)


cb_b = fig.colorbar(
    im_b,
    ax=ax_b,
    pad=0.025,
)


cb_b.set_label(
    r"$\zeta(s,t)$ at $\beta=2.70$"
)

cb_b.set_ticks(
    np.linspace(
        0.0,
        1.0,
        6,
    )
)


ax_b.scatter(
    s_caustic,
    t_caustic,
    marker="*",
    s=115,
    facecolors="white",
    edgecolors="black",
    linewidths=1.0,
)


ax_b.annotate(
    (
        r"$\zeta_{\max}=0.9957<1$"
        "\n"
        r"$\beta_c=2.7117$"
    ),
    xy=(
        s_caustic,
        t_caustic,
    ),
    xytext=(
        0.61,
        0.58,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=7.8,
)


ax_b.set_xlabel(
    r"Reduced coordinate $s$"
)

ax_b.set_ylabel(
    r"Reduced coordinate $t$"
)

ax_b.set_title(
    "(b) Local stability just before the caustic"
)


ax_b.set_aspect(
    "equal"
)


# ============================================================
# Panel (c): global versus stationary caustic
# ============================================================


entries = [
    (
        "Global",
        beta_c_calc,
    )
]


for x in finite_stationary:

    if x["type"] == "maximum":

        label = "Maximum"

    elif x["type"] == "saddle":

        label = (
            f"Saddle {x['id']}"
        )

    else:

        label = (
            f"{x['type']} {x['id']}"
        )

    entries.append(
        (
            label,
            x["beta_c"],
        )
    )


entries.sort(
    key=lambda x: x[1]
)


labels_c = [
    x[0]
    for x in entries
]


values_c = np.array([
    x[1]
    for x in entries
])


ypos = np.arange(
    len(entries)
)


ax_c.scatter(
    values_c,
    ypos,
    s=45,
)


for y, value in zip(
    ypos,
    values_c,
):

    ax_c.text(
        value + 0.025,
        y,
        f"{value:.3f}",
        va="center",
        fontsize=8.0,
    )


ax_c.set_yticks(
    ypos
)

ax_c.set_yticklabels(
    labels_c
)

ax_c.invert_yaxis()


ax_c.set_xlabel(
    r"Local caustic inverse temperature $\beta_c$"
)

ax_c.set_title(
    "(c) Stationary-point survey misses the first caustic"
)


ax_c.set_xlim(
    beta_c_calc - 0.12,
    max(values_c) + 0.30,
)


# ============================================================
# Panel (d): geometry vs physical occupation
# ============================================================


mask_area = (
    invalid_area > 0.0
)


mask_prob = (
    invalid_prob > 0.0
)


ax_d.semilogy(
    beta_prob[mask_area],
    invalid_area[mask_area],
    marker="o",
    linewidth=1.3,
    label="Gaussian-invalid area",
)


ax_d.semilogy(
    beta_prob[mask_prob],
    invalid_prob[mask_prob],
    marker="s",
    linewidth=1.3,
    label="Exact probability in invalid region",
)


ax_d.axvline(
    beta_c_calc,
    linestyle=":",
    linewidth=1.1,
)


ax_d.annotate(
    (
        r"$\beta=3.0$"
        "\n"
        f"area = {100*invalid_area[i3]:.1f}%"
        "\n"
        f"prob. = {100*invalid_prob[i3]:.3f}%"
    ),
    xy=(
        beta_prob[i3],
        invalid_prob[i3],
    ),
    xytext=(
        2.76,
        0.009,
    ),
    arrowprops={
        "arrowstyle": "->",
        "lw": 0.8,
    },
    fontsize=7.6,
)


ymin, ymax = ax_d.get_ylim()


ax_d.text(
    beta_c_calc + 0.012,
    ymax/1.3,
    r"$\beta_c$",
    ha="left",
    va="top",
    fontsize=8.0,
)


ax_d.set_xlabel(
    r"Inverse temperature $\beta$"
)

ax_d.set_ylabel(
    "Fraction"
)

ax_d.set_title(
    "(d) Invalid geometry is weakly occupied"
)


ax_d.text(
    3.07,
    0.19,
    "Gaussian-invalid area",
    fontsize=7.5,
)

ax_d.text(
    3.07,
    1.1e-3,
    "Exact probability",
    fontsize=7.5,
)


# ============================================================
# Save
# ============================================================


fig.subplots_adjust(
    top=0.975,
    bottom=0.09,
    left=0.10,
    right=0.98,
)


pdf = OUT / "Fig3_offstationary_caustic.pdf"

png = OUT / "Fig3_offstationary_caustic.png"


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
# Save plotted key data for provenance
# ============================================================


summary = (
    OUT
    / "Fig3_key_data.csv"
)


with summary.open(
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "quantity",
        "value",
    ])

    w.writerow([
        "beta_c_global",
        beta_c_calc,
    ])

    w.writerow([
        "s_caustic",
        s_caustic,
    ])

    w.writerow([
        "t_caustic",
        t_caustic,
    ])

    w.writerow([
        "kappa_min_global",
        kappa_global,
    ])

    w.writerow([
        "V_caustic",
        V_caustic,
    ])

    w.writerow([
        "zeta_max_beta2p70",
        zeta_max,
    ])

    for x in finite_stationary:

        w.writerow([
            (
                f"beta_c_stationary_"
                f"{x['id']}_"
                f"{x['type']}"
            ),
            x["beta_c"],
        ])

    w.writerow([
        "invalid_area_beta3",
        invalid_area[i3],
    ])

    w.writerow([
        "exact_invalid_probability_beta3",
        invalid_prob[i3],
    ])


print()

print(
    f"saved: {pdf.relative_to(ROOT)}"
)

print(
    f"saved: {png.relative_to(ROOT)}"
)

print(
    f"saved: {summary.relative_to(ROOT)}"
)
