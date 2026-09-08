#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication_figures"

# Validated TB-03 production values (10 independent chains per P).
P = np.array([64, 128, 256], dtype=float)

p_inv = np.array([0.88960, 0.91607, 0.92485])
ci_lo = np.array([0.88892, 0.91569, 0.92400])
ci_hi = np.array([0.89020, 0.91645, 0.92547])

zeta50 = np.array([17.33, 17.31, 17.32])
p_zeta_gt5 = np.array([0.87028, 0.89919, 0.90877])

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 9.2,
    "axes.labelsize": 9.4,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.25, 2.45),
    constrained_layout=True,
)

# (a) Finite-P stabilization.
ax = axes[0]
yerr = np.vstack([p_inv - ci_lo, ci_hi - p_inv])

ax.errorbar(
    P, p_inv,
    yerr=yerr,
    marker="o",
    linewidth=1.2,
    capsize=3,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.0,
)

ax.text(
    0.04, 0.17,
    r"predeclared criterion:" "\n" r"lower 95% CI $>0.5$",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.1,
)

ax.set_xscale("log", base=2)
ax.set_xticks(P)
ax.set_xticklabels(["64", "128", "256"])
ax.set_ylim(0.45, 0.95)
ax.set_xlabel(r"Bead count $P$")
ax.set_ylabel(r"$\hat p_{\rm inv}^{(P)}$")
ax.set_title(r"(a) Invalid occupation")
ax.grid(linewidth=0.35, alpha=0.28)

# (b) Most of the invalid weight is far beyond threshold.
ax = axes[1]
x = np.arange(P.size)
width = 0.34

ax.bar(
    x - width/2,
    p_inv,
    width=width,
    label=r"$P(\zeta\geq1)$",
)

ax.bar(
    x + width/2,
    p_zeta_gt5,
    width=width,
    label=r"$P(\zeta>5)$",
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.0,
)

ax.set_xticks(x)
ax.set_xticklabels(["64", "128", "256"])
ax.set_ylim(0.0, 1.0)
ax.set_xlabel(r"Bead count $P$")
ax.set_ylabel("Bead probability")
ax.set_title(r"(b) Deep post-caustic weight")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", linewidth=0.35, alpha=0.28)

# (c) Median instability parameter.
ax = axes[2]

ax.plot(
    P,
    zeta50,
    marker="o",
    linewidth=1.2,
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
)

ax.text(
    0.96, 0.13,
    r"caustic: $\zeta=1$",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.4,
)

ax.set_xscale("log", base=2)
ax.set_xticks(P)
ax.set_xticklabels(["64", "128", "256"])
ax.set_ylim(0.0, 19.0)
ax.set_xlabel(r"Bead count $P$")
ax.set_ylabel(r"Median $\zeta$")
ax.set_title(r"(c) Instability depth")
ax.grid(linewidth=0.35, alpha=0.28)

OUT.mkdir(parents=True, exist_ok=True)
pdf = OUT / "Fig5_interacting_gaussian_invalid.pdf"
png = OUT / "Fig5_interacting_gaussian_invalid.png"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=400, bbox_inches="tight")
plt.close(fig)

print(f"saved: {pdf}")
print(f"saved: {png}")
