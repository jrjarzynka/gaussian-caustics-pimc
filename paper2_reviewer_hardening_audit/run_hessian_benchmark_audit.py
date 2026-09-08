#!/usr/bin/env python3
"""Independent converged Hessian-field audit for the two Paper-2 benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root


HERE = Path(__file__).resolve().parent
SQRT3 = np.sqrt(3.0)
G = np.array([[1.0, 0.0], [-0.5, SQRT3 / 2.0], [-0.5, -SQRT3 / 2.0]])
B = np.column_stack((G[0], G[1]))
A_DIRECT = 2.0 * np.pi * np.linalg.inv(B.T)


def fields(uv: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray):
    uv = np.asarray(uv, dtype=float)
    s = uv[..., 0]
    t = uv[..., 1]
    theta = np.stack((2 * np.pi * s + phases[0],
                      2 * np.pi * t + phases[1],
                      -2 * np.pi * (s + t) + phases[2]), axis=-1)
    c = np.cos(theta)
    sn = np.sin(theta)
    grad = np.einsum("...a,ai->...i", amplitudes * sn, G)
    H = np.einsum("...a,ai,aj->...ij", amplitudes * c, G, G)
    eig = np.linalg.eigvalsh(H)
    return grad, H, eig


def lambda_min(uv: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray) -> float:
    return float(fields(np.mod(uv, 1.0), amplitudes, phases)[2][0])


def scan(N: int, amplitudes: np.ndarray, phases: np.ndarray):
    axis = np.arange(N, dtype=float) / N
    S, T = np.meshgrid(axis, axis, indexing="xy")
    uv = np.stack((S, T), axis=-1)
    eig = fields(uv, amplitudes, phases)[2]
    idx = np.unravel_index(np.argmin(eig[..., 0]), eig[..., 0].shape)
    return float(eig[idx][0]), np.array([idx[1] / N, idx[0] / N]), float(np.mean(eig[..., 0] < 0))


def optimized_global(amplitudes: np.ndarray, phases: np.ndarray):
    _, coarse, _ = scan(256, amplitudes, phases)
    starts = [coarse]
    axis = np.linspace(0.0, 1.0, 12, endpoint=False)
    starts += [np.array([s, t]) for s in axis for t in axis]
    candidates = []
    for start in starts:
        result = minimize(
            lambda x: lambda_min(x, amplitudes, phases), start,
            method="Nelder-Mead",
            options={"xatol": 1e-13, "fatol": 1e-14, "maxiter": 3000},
        )
        uv = np.mod(result.x, 1.0)
        candidates.append((lambda_min(uv, amplitudes, phases), uv, result.success, result.nfev))
    candidates.sort(key=lambda x: x[0])
    return candidates[0], candidates


def stationary_points(amplitudes: np.ndarray, phases: np.ndarray):
    found: list[np.ndarray] = []
    for s in np.linspace(0.0, 1.0, 20, endpoint=False):
        for t in np.linspace(0.0, 1.0, 20, endpoint=False):
            res = root(lambda uv: fields(np.mod(uv, 1), amplitudes, phases)[0], [s, t], tol=1e-12)
            uv = np.mod(res.x, 1.0)
            if res.success and np.linalg.norm(fields(uv, amplitudes, phases)[0]) < 1e-10:
                delta = lambda a, b: np.linalg.norm((a - b + 0.5) % 1.0 - 0.5)
                if all(delta(uv, old) > 1e-8 for old in found):
                    found.append(uv)
    rows = []
    for uv in found:
        grad, H, eig = fields(uv, amplitudes, phases)
        kind = "minimum" if eig[0] > 0 else ("maximum" if eig[1] < 0 else "saddle")
        r = A_DIRECT @ uv
        rows.append({
            "type": kind, "s": uv[0], "t": uv[1], "x": r[0], "y": r[1],
            "lambda_min": eig[0], "lambda_max": eig[1],
            "gradient_norm": np.linalg.norm(grad),
            "beta_c_local": np.pi / np.sqrt(-eig[0]) if eig[0] < 0 else np.inf,
        })
    rows.sort(key=lambda r: ({"minimum": 0, "saddle": 1, "maximum": 2}[r["type"]], r["lambda_min"]))
    return rows


def main() -> None:
    tri_A = np.ones(3)
    tri_phi = np.zeros(3)
    asym_A = np.array([1.0, 0.78, 0.62])
    asym_phi = np.array([0.0, 0.23, 0.71])
    convergence = []
    for name, aa, pp in (("triangular", tri_A, tri_phi), ("asymmetric", asym_A, asym_phi)):
        for N in (128, 256, 512, 1024, 2048):
            lm, uv, frac = scan(N, aa, pp)
            convergence.append({"benchmark": name, "N": N, "grid_lambda_min": lm,
                                "grid_s": uv[0], "grid_t": uv[1],
                                "negative_eigenvalue_area_fraction": frac})
    best_tri, _ = optimized_global(tri_A, tri_phi)
    best_asym, asym_candidates = optimized_global(asym_A, asym_phi)
    rows = []
    for name, best, aa, pp in (("triangular", best_tri, tri_A, tri_phi),
                               ("asymmetric", best_asym, asym_A, asym_phi)):
        lm, uv, success, nfev = best
        grad, H, eig = fields(uv, aa, pp)
        r = A_DIRECT @ uv
        rows.append({
            "benchmark": name, "lambda_min_optimized": lm,
            "beta_c_global": np.pi / np.sqrt(-lm),
            "s": uv[0], "t": uv[1], "x": r[0], "y": r[1],
            "potential_gradient_norm_at_lambda_min": np.linalg.norm(grad),
            "optimizer_success": success, "optimizer_nfev": nfev,
            "hessian_other_eigenvalue": eig[1],
        })

    with (HERE / "hessian_scan_convergence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=convergence[0].keys()); w.writeheader(); w.writerows(convergence)
    with (HERE / "hessian_global_minima.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    tri_stationary = stationary_points(tri_A, tri_phi)
    stationary = stationary_points(asym_A, asym_phi)
    with (HERE / "hessian_triangular_stationary_points.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tri_stationary[0].keys()); w.writeheader(); w.writerows(tri_stationary)
    with (HERE / "hessian_asymmetric_stationary_points.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stationary[0].keys()); w.writeheader(); w.writerows(stationary)
    report = {
        "direct_lattice_columns": A_DIRECT.tolist(),
        "reciprocal_vectors": G.tolist(),
        "asymmetric_multistart_candidate_spread_first_20": [float(x[0]) for x in asym_candidates[:20]],
        "canonical_recommendation": rows[1],
        "precision_policy": "quote lambda_min and beta_c to 7 significant decimal digits after grid and optimizer convergence",
    }
    (HERE / "hessian_benchmark_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"global": rows, "triangular_stationary": tri_stationary,
                      "asymmetric_stationary": stationary}, indent=2))


if __name__ == "__main__":
    main()
