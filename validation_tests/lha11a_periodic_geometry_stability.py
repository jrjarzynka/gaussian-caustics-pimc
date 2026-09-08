import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LHA-11A
#
# Geometry and local-Gaussian stability audit for a genuine
# 2D triangular-periodic landscape:
#
# V(r) = V0 [3 - cos(G1.r) - cos(G2.r) - cos(G3.r)]
#
# with G1 + G2 + G3 = 0 and |Ga| = G.
#
# Units:
#   hbar = 1
#   M    = 1
#
# No PI-QMC.
# No quantum diagonalization.
#
# This checkpoint tests only:
#   * periodic geometry
#   * stationary-point topology
#   * Hessian eigenvalues
#   * local instability index zeta
#   * onset of the first Gaussian caustic
# ============================================================


V0 = 1.0
G_MAG = 1.0

# Divisible by 2 and 3 so that all high-symmetry points
# lie exactly on the grid.
N = 600


BETA_SAMPLES = [
    0.50,
    1.00,
    2.00,
    2.50,
    2.56,
    2.57,
    3.00,
    3.60,
    3.63,
    4.00,
    5.00,
]

# Dense scan only for the stability-fraction figure.
BETA_CURVE = np.linspace(
    0.2,
    5.0,
    241,
)

# Representative map:
# saddles are already super-caustic,
# while the high-symmetry maxima have not yet crossed
# their own local caustic.
BETA_MAP = 3.0


# ============================================================
# Reciprocal and direct lattice
# ============================================================

sqrt3 = np.sqrt(3.0)

G1 = G_MAG * np.array([
    1.0,
    0.0,
])

G2 = G_MAG * np.array([
    -0.5,
    sqrt3/2.0,
])

G3 = -(G1 + G2)

GS = np.array([
    G1,
    G2,
    G3,
])


# Reciprocal primitive basis B = [G1 G2].
# Direct basis A satisfies:
#
#     B^T A = 2 pi I

Bmat = np.column_stack([
    G1,
    G2,
])

Amat = (
    2.0*np.pi
    * np.linalg.inv(Bmat).T
)

a1 = Amat[:, 0]
a2 = Amat[:, 1]


# Reduced primitive-cell coordinates:
#
# r = s a1 + t a2,
# 0 <= s,t < 1

s = np.arange(
    N,
    dtype=float,
) / N

t = np.arange(
    N,
    dtype=float,
) / N

S, T = np.meshgrid(
    s,
    t,
    indexing="xy",
)

X = (
    a1[0]*S
    + a2[0]*T
)

Y = (
    a1[1]*S
    + a2[1]*T
)


# ============================================================
# Potential, gradient, Hessian
# ============================================================

phase = np.stack([
    g[0]*X + g[1]*Y
    for g in GS
], axis=0)

cos_phase = np.cos(
    phase
)

sin_phase = np.sin(
    phase
)


V = V0 * (
    3.0
    - np.sum(
        cos_phase,
        axis=0,
    )
)


# grad V = V0 sum sin(G.r) G

gx = V0 * sum(
    sin_phase[i] * GS[i, 0]
    for i in range(3)
)

gy = V0 * sum(
    sin_phase[i] * GS[i, 1]
    for i in range(3)
)

grad_norm = np.sqrt(
    gx*gx + gy*gy
)


# Hessian:
#
# H = V0 sum cos(G.r) G G^T

Hxx = V0 * sum(
    cos_phase[i] * GS[i, 0]**2
    for i in range(3)
)

Hyy = V0 * sum(
    cos_phase[i] * GS[i, 1]**2
    for i in range(3)
)

Hxy = V0 * sum(
    cos_phase[i]
    * GS[i, 0]
    * GS[i, 1]
    for i in range(3)
)


# Exact eigenvalues of symmetric 2x2 Hessian.

disc = np.sqrt(
    (Hxx-Hyy)**2
    + 4.0*Hxy**2
)

kappa_min = 0.5 * (
    Hxx + Hyy - disc
)

kappa_max = 0.5 * (
    Hxx + Hyy + disc
)


# ============================================================
# Analytic high-symmetry expectations
# ============================================================

CURV_SCALE = (
    V0 * G_MAG**2
)

# Minimum:
#
# H = (3/2) V0 G^2 I

K_MINIMUM = (
    1.5 * CURV_SCALE
)

# Saddle:
#
# eigenvalues = (-3/2, +1/2) V0 G^2

K_SADDLE_NEG = (
    -1.5 * CURV_SCALE
)

K_SADDLE_POS = (
    0.5 * CURV_SCALE
)

# Maximum:
#
# H = -(3/4) V0 G^2 I

K_MAXIMUM = (
    -0.75 * CURV_SCALE
)


# First global caustic occurs at the saddles:
#
# beta_c sqrt(-kappa) = pi

BETA_C_SADDLE = (
    np.pi
    / np.sqrt(
        -K_SADDLE_NEG
    )
)

# The high-symmetry maxima cross their local caustic later.

BETA_C_MAXIMUM = (
    np.pi
    / np.sqrt(
        -K_MAXIMUM
    )
)


# ============================================================
# Stationary-point audit
# ============================================================

GRAD_TOL = (
    1.0e-10
    * V0
    * G_MAG
)

CURV_TOL = (
    1.0e-9
    * CURV_SCALE
)

stationary_mask = (
    grad_norm < GRAD_TOL
)

indices = np.argwhere(
    stationary_mask
)

stationary_points = []

for row, col in indices:

    km = float(
        kappa_min[row, col]
    )

    kp = float(
        kappa_max[row, col]
    )

    if km > CURV_TOL:

        kind = "minimum"

    elif kp < -CURV_TOL:

        kind = "maximum"

    elif (
        km < -CURV_TOL
        and kp > CURV_TOL
    ):

        kind = "saddle"

    else:

        kind = "degenerate"

    stationary_points.append({
        "type": kind,
        "s": float(S[row, col]),
        "t": float(T[row, col]),
        "x": float(X[row, col]),
        "y": float(Y[row, col]),
        "V": float(V[row, col]),
        "kappa_min": km,
        "kappa_max": kp,
    })


# ============================================================
# Local Gaussian stability
# ============================================================

def zeta_map(beta):

    # M = hbar = 1:
    #
    # zeta =
    # beta/pi sqrt(max(0,-kappa_min))

    return (
        beta/np.pi
        * np.sqrt(
            np.maximum(
                0.0,
                -kappa_min,
            )
        )
    )


def invalid_fraction(beta):

    zeta = zeta_map(
        beta
    )

    return float(
        np.mean(
            zeta >= 1.0
        )
    )


negative_curvature_fraction = float(
    np.mean(
        kappa_min < 0.0
    )
)

negative_definite_fraction = float(
    np.mean(
        kappa_max < 0.0
    )
)


# ============================================================
# Print geometry
# ============================================================

print(
    "=== LHA-11A PERIODIC 2D "
    "GEOMETRY + STABILITY AUDIT ==="
)

print()

print(
    "=== LATTICE ==="
)

print(
    "G1 =",
    G1,
)

print(
    "G2 =",
    G2,
)

print(
    "G3 =",
    G3,
)

print()

print(
    "a1 =",
    a1,
)

print(
    "a2 =",
    a2,
)

print()

print(
    "G1+G2+G3 =",
    G1+G2+G3,
)

print()


print(
    "=== GLOBAL LANDSCAPE ==="
)

print(
    f"V min              = "
    f"{np.min(V):.12f}"
)

print(
    f"V max              = "
    f"{np.max(V):.12f}"
)

print(
    f"global kappa_min   = "
    f"{np.min(kappa_min):.12f}"
)

print(
    f"global kappa_max   = "
    f"{np.max(kappa_max):.12f}"
)

print(
    f"max |Hxy|          = "
    f"{np.max(np.abs(Hxy)):.12f}"
)

print()

print(
    "fraction with >=1 negative Hessian mode = "
    f"{100*negative_curvature_fraction:.4f}%"
)

print(
    "fraction with 2 negative Hessian modes  = "
    f"{100*negative_definite_fraction:.4f}%"
)

print()


print(
    "=== ANALYTIC CAUSTICS ==="
)

print(
    f"beta_c saddle  = "
    f"{BETA_C_SADDLE:.12f}"
)

print(
    f"beta_c maximum = "
    f"{BETA_C_MAXIMUM:.12f}"
)

print()


# ============================================================
# Print stationary points
# ============================================================

print(
    "=== STATIONARY POINTS IN ONE "
    "PRIMITIVE CELL ==="
)

print()

counts = {
    "minimum": 0,
    "saddle": 0,
    "maximum": 0,
    "degenerate": 0,
}

for i, p in enumerate(
    stationary_points,
    start=1,
):

    counts[p["type"]] += 1

    print(
        f"{i:2d}. "
        f"{p['type']:8s}  "
        f"(s,t)=("
        f"{p['s']:.6f},"
        f"{p['t']:.6f})  "
        f"V={p['V']:.8f}  "
        f"kappa=("
        f"{p['kappa_min']:+.8f}, "
        f"{p['kappa_max']:+.8f})"
    )

print()

print(
    "counts:",
    counts,
)

print()


# ============================================================
# Beta scan
# ============================================================

summary_rows = []

print(
    "=== TEMPERATURE / beta STABILITY SCAN ==="
)

print()

print(
    " beta      zeta_max    invalid cell (%)"
)

print(
    "----------------------------------------"
)


for beta in BETA_SAMPLES:

    zeta = zeta_map(
        beta
    )

    zeta_max = float(
        np.max(zeta)
    )

    frac = invalid_fraction(
        beta
    )

    print(
        f"{beta:6.3f}    "
        f"{zeta_max:9.6f}    "
        f"{100*frac:12.6f}"
    )

    summary_rows.append([
        beta,
        zeta_max,
        frac,
        negative_curvature_fraction,
        negative_definite_fraction,
    ])


print()


# ============================================================
# Formal PASS / FAIL
# ============================================================

expected_counts = {
    "minimum": 1,
    "saddle": 3,
    "maximum": 2,
    "degenerate": 0,
}


topology_pass = (
    counts == expected_counts
)


extrema_pass = (
    abs(np.min(V)-0.0) < 1.0e-12
    and
    abs(np.max(V)-4.5*V0) < 1.0e-10
    and
    abs(
        np.min(kappa_min)
        - K_SADDLE_NEG
    ) < 1.0e-10
    and
    abs(
        np.max(kappa_max)
        - K_MINIMUM
    ) < 1.0e-10
)


signature_pass = True

for p in stationary_points:

    if p["type"] == "minimum":

        ok = (
            abs(
                p["kappa_min"]
                - K_MINIMUM
            ) < 1.0e-10
            and
            abs(
                p["kappa_max"]
                - K_MINIMUM
            ) < 1.0e-10
        )

    elif p["type"] == "saddle":

        ok = (
            abs(
                p["kappa_min"]
                - K_SADDLE_NEG
            ) < 1.0e-10
            and
            abs(
                p["kappa_max"]
                - K_SADDLE_POS
            ) < 1.0e-10
        )

    elif p["type"] == "maximum":

        ok = (
            abs(
                p["kappa_min"]
                - K_MAXIMUM
            ) < 1.0e-10
            and
            abs(
                p["kappa_max"]
                - K_MAXIMUM
            ) < 1.0e-10
        )

    else:

        ok = False

    signature_pass &= ok


# Explicitly bracket the analytically predicted first caustic.

below_beta = 2.56
above_beta = 2.57

below_fraction = invalid_fraction(
    below_beta
)

above_fraction = invalid_fraction(
    above_beta
)

caustic_pass = (
    below_beta < BETA_C_SADDLE
    and
    above_beta > BETA_C_SADDLE
    and
    below_fraction == 0.0
    and
    above_fraction > 0.0
)


print(
    "=== GLOBAL CHECKS ==="
)

print(
    "GEOMETRY EXTREMA:",
    "PASS"
    if extrema_pass
    else "FAIL"
)

print(
    "STATIONARY TOPOLOGY:",
    "PASS"
    if topology_pass
    else "FAIL"
)

print(
    "HESSIAN SIGNATURES:",
    "PASS"
    if signature_pass
    else "FAIL"
)

print(
    "FIRST CAUSTIC ONSET:",
    "PASS"
    if caustic_pass
    else "FAIL"
)

overall = (
    extrema_pass
    and topology_pass
    and signature_pass
    and caustic_pass
)

print()

print(
    "OVERALL:",
    "PASS"
    if overall
    else "FAIL"
)

print()


# ============================================================
# CSV — beta summary
# ============================================================

with open(
    "lha11a_periodic_geometry_stability.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "beta",
        "zeta_max",
        "invalid_cell_fraction",
        "negative_curvature_fraction",
        "negative_definite_fraction",
    ])

    w.writerows(
        summary_rows
    )


# ============================================================
# CSV — stationary points
# ============================================================

with open(
    "lha11a_periodic_stationary_points.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)

    w.writerow([
        "type",
        "s",
        "t",
        "x",
        "y",
        "V",
        "kappa_min",
        "kappa_max",
    ])

    for p in stationary_points:

        w.writerow([
            p["type"],
            p["s"],
            p["t"],
            p["x"],
            p["y"],
            p["V"],
            p["kappa_min"],
            p["kappa_max"],
        ])


# ============================================================
# Save raw geometry for LHA-11B
# ============================================================

np.savez_compressed(
    "lha11a_periodic_geometry.npz",
    X=X,
    Y=Y,
    S=S,
    T=T,
    V=V,
    gx=gx,
    gy=gy,
    Hxx=Hxx,
    Hxy=Hxy,
    Hyy=Hyy,
    kappa_min=kappa_min,
    kappa_max=kappa_max,
    a1=a1,
    a2=a2,
    G1=G1,
    G2=G2,
    G3=G3,
)


# ============================================================
# Figure 1 — local instability map
# ============================================================

zeta_plot = zeta_map(
    BETA_MAP
)

fig, ax = plt.subplots(
    figsize=(7.2, 6.2)
)

mesh = ax.pcolormesh(
    X,
    Y,
    zeta_plot,
    shading="auto",
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
)

cbar.set_label(
    rf"$\zeta(\mathbf{{r}})$ at "
    rf"$\beta={BETA_MAP:g}$"
)


# Negative-curvature boundary:
#
# kappa_min = 0

ax.contour(
    X,
    Y,
    kappa_min,
    levels=[0.0],
    linewidths=1.0,
)


# First-caustic boundary:
#
# zeta = 1

ax.contour(
    X,
    Y,
    zeta_plot,
    levels=[1.0],
    linewidths=1.6,
)


for p in stationary_points:

    if p["type"] == "minimum":

        marker = "o"
        label = "min"

    elif p["type"] == "saddle":

        marker = "^"
        label = "saddle"

    else:

        marker = "x"
        label = "max"

    ax.scatter(
        p["x"],
        p["y"],
        marker=marker,
        s=45,
    )

    ax.annotate(
        label,
        (
            p["x"],
            p["y"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )


ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    r"$x$"
)

ax.set_ylabel(
    r"$y$"
)

ax.set_title(
    "Periodic 2D local-Gaussian stability map"
)

fig.tight_layout()

fig.savefig(
    "lha11a_periodic_zeta_map.png",
    dpi=300,
)

fig.savefig(
    "lha11a_periodic_zeta_map.pdf",
)

plt.close(fig)


# ============================================================
# Figure 2 — invalid cell fraction vs beta
# ============================================================

fractions = np.array([
    invalid_fraction(beta)
    for beta in BETA_CURVE
])


fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

ax.plot(
    BETA_CURVE,
    100.0*fractions,
)

ax.axvline(
    BETA_C_SADDLE,
    linestyle="--",
    label=(
        r"first saddle caustic"
    ),
)

ax.axvline(
    BETA_C_MAXIMUM,
    linestyle=":",
    label=(
        r"maximum-point caustic"
    ),
)

ax.set_xlabel(
    r"$\beta$"
)

ax.set_ylabel(
    "Gaussian-invalid area of primitive cell (%)"
)

ax.set_title(
    "Growth of the local-caustic region"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "lha11a_invalid_fraction_vs_beta.png",
    dpi=300,
)

fig.savefig(
    "lha11a_invalid_fraction_vs_beta.pdf",
)

plt.close(fig)


print(
    "saved: "
    "lha11a_periodic_geometry_stability.csv"
)

print(
    "saved: "
    "lha11a_periodic_stationary_points.csv"
)

print(
    "saved: "
    "lha11a_periodic_geometry.npz"
)

print(
    "saved: "
    "lha11a_periodic_zeta_map.png/pdf"
)

print(
    "saved: "
    "lha11a_invalid_fraction_vs_beta.png/pdf"
)
