#!/usr/bin/env python3
"""Compact, non-decorative summary of P=256 Hessian attribution."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))


def read_rows():
    with (HERE / "mode_attribution_distributions_P256.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def get(rows, condition, metric):
    row = next(r for r in rows if r["condition"] == condition and r["metric"] == metric)
    return {k: float(row[f"pooled_{k}"]) for k in ("p05", "p25", "median", "p75", "p95")}


def draw_summary(ax, y, stats, color, label=None):
    ax.plot([stats["p05"], stats["p95"]], [y, y], color=color, lw=1.3)
    ax.plot([stats["p25"], stats["p75"]], [y, y], color=color, lw=6, solid_capstyle="butt")
    ax.plot(stats["median"], y, marker="o", color="white", markeredgecolor=color,
            markeredgewidth=1.3, markersize=5, label=label)


def main():
    rows = read_rows()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.88, wspace=0.28)
    colors = {"all": "#4C78A8", "zeta_ge_1": "#F58518", "zeta_gt_5": "#B33A3A"}
    labels = {"all": "all beads", "zeta_ge_1": r"$\zeta\geq1$", "zeta_gt_5": r"$\zeta>5$"}
    metrics_a = [("c_moire", r"$c_{\rm moir\acute e}$"),
                 ("c_BLK", r"$c_{\rm BLK}$"), ("c_wall", r"$c_{\rm wall}$")]
    offsets = {"all": 0.22, "zeta_ge_1": 0.0, "zeta_gt_5": -0.22}
    for i, (metric, label) in enumerate(metrics_a):
        for condition in ("all", "zeta_ge_1", "zeta_gt_5"):
            draw_summary(ax1, i + offsets[condition], get(rows, condition, metric), colors[condition],
                         labels[condition] if i == 0 else None)
    ax1.axvline(0, color="0.5", lw=0.8)
    ax1.set_yticks(range(3), [x[1] for x in metrics_a])
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Rayleigh contribution (eV nm$^{-2}m_0^{-1}$)")
    ax1.set_title("(a) Curvature along the full unstable mode", loc="left", fontsize=10)
    ax1.legend(frameon=False, fontsize=8, loc="best")

    metrics_b = [("O_rho", r"$O_\rho$"), ("O_theta", r"$O_\theta$"), ("O_COM", r"$O_{\rm COM}$")]
    for i, (metric, label) in enumerate(metrics_b):
        for condition in ("all", "zeta_gt_5"):
            off = 0.13 if condition == "all" else -0.13
            draw_summary(ax2, i + off, get(rows, condition, metric), colors[condition],
                         labels[condition] if i == 0 else None)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_yticks(range(3), [x[1] for x in metrics_b])
    ax2.invert_yaxis()
    ax2.set_xlabel("squared mode overlap")
    ax2.set_title("(b) Character of the full unstable mode", loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="best")
    fig.text(0.5, 0.055, "dot: median; thick: 25--75%; thin: 5--95%; pooled bead probability",
             ha="center", fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color="0.9", lw=0.7)
        ax.tick_params(labelsize=8)
    fig.savefig(HERE / "mode_attribution_diagnostic.png", dpi=240)
    fig.savefig(HERE / "mode_attribution_diagnostic.pdf")


if __name__ == "__main__":
    main()
