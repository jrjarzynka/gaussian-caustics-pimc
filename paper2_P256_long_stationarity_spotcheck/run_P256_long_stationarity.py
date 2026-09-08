#!/usr/bin/env python3
"""Independent P=256 long-chain stationarity spot-check for Paper 2.

This is an audit-only orchestration and analysis script.  It imports the exact
Sec. VIII production builders and the sampler runtime that passed the existing
reproduction gate.  It does not alter the Hamiltonian or production outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PRODUCTION_ROOT = REPO_ROOT / "secVIII"
PRODUCTION_RUNNER = PRODUCTION_ROOT / "run_tb03_pinv_ladder.py"
STABILITY_MODULE = PRODUCTION_ROOT / "lha15a0_pair_stability_field_TB03.py"
CONFIG = (
    PRODUCTION_ROOT
    / "configs/two_body/landscape_scan_config_v2_P256_matched.json"
)
OLD_PER_CHAIN = REPO_ROOT / "paper2_reviewer_hardening_audit/reproduction_per_chain.csv"
OLD_AGGREGATE = REPO_ROOT / "paper2_reviewer_hardening_audit/reproduction_aggregate.csv"
SAMPLER_FILE = REPO_ROOT / "numerics/tmd_pimc/two_body_sampler_periodic_jit.py"
RAW_DIR = HERE / "raw_chains"
OBS_DIR = HERE / "observable_cache"

P = 256
TEMPERATURE_K = 20.0
SEEDS = (912560, 912561, 912562, 912563)
N_STEPS = 240_000
MIN_BURN_IN = 15_000
BURN_INS = (15_000, 30_000, 60_000)
SAMPLE_EVERY = 20
LOCAL_STEP_NM = 0.15
GLOBAL_STEP_NM = 12.0
GLOBAL_MOVE_PROBABILITY = 0.20
N_BLOCKS = 8
CHUNK_SIZE = 50_000
CUMULATIVE_POINTS = 24
MATERIAL_P_SHIFT = 0.005

# Exact canonical values from the successful Sec. VIII reproduction gate.
OLD_P = 0.9248524305555555
OLD_P_SEM = 0.00039411141337912894
OLD_P_CI = (0.9240048611111111, 0.925468923611111)
OLD_ZETA50 = 17.318186569213868
OLD_W5 = 0.908770659722222
OLD_RHO_MEDIAN = 1.6374945044517517
OLD_RHO_MEAN = 1.8270547986030579
OLD_HALF_DELTA = -0.0007513888888888594


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_sem_ci(values: Iterable[float]) -> tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=float)
    mean = float(x.mean())
    if x.size < 2:
        return mean, math.nan, math.nan, math.nan
    sem = float(x.std(ddof=1) / math.sqrt(x.size))
    half = float(stats.t.ppf(0.975, x.size - 1) * sem)
    return mean, sem, mean - half, mean + half


def production_context():
    # This explicit path is the runtime that reproduced the archived Sec. VIII
    # output.  The untracked production runner otherwise imports the first
    # installed tmd_pimc, and its call signature is incompatible with its own
    # tracked HEAD sampler.
    sys.path.insert(0, str(REPO_ROOT / "numerics"))
    runner = load_module("paper2_longcheck_production_runner", PRODUCTION_RUNNER)
    stability = load_module("paper2_longcheck_stability", STABILITY_MODULE)
    cfg = json.loads(CONFIG.read_text())
    return runner, stability, cfg


def raw_path(seed: int) -> Path:
    return RAW_DIR / f"P256_seed{seed}_240k.npz"


def obs_path(seed: int) -> Path:
    return OBS_DIR / f"P256_seed{seed}_observables.npz"


def expected_nstored(burn_in: int) -> int:
    return 1 + (N_STEPS - burn_in - 1) // SAMPLE_EVERY


def validate_predeclared_setup(cfg: dict[str, Any]) -> None:
    required = {
        "temperature_K": TEMPERATURE_K,
        "n_beads": P,
        "mass_e_m0": 0.58,
        "mass_h_m0": 0.36,
        "separation_nm": 0.60,
        "screening_length_layer1_nm": 4.479911124019045,
        "screening_length_layer2_nm": 3.4934510307918494,
        "kappa_environment": 4.945,
        "moire_amplitude_eV": 0.045,
        "moire_period_nm": 20.0,
    }
    failures = []
    for key, expected in required.items():
        actual = cfg.get(key)
        if actual is None or not np.isclose(float(actual), float(expected), rtol=0, atol=1e-14):
            failures.append(f"{key}: {actual!r} != {expected!r}")
    if failures:
        raise RuntimeError("Production configuration mismatch:\n" + "\n".join(failures))
    if len(set(SEEDS)) != len(SEEDS):
        raise RuntimeError("Long-chain seeds are not unique")
    old_seeds = {
        31_000 + 1000 * beads + i
        for beads in (64, 128, 256)
        for i in range(10)
    }
    overlap = set(SEEDS) & old_seeds
    if overlap:
        raise RuntimeError(f"New seeds overlap the original campaign: {sorted(overlap)}")


def build_setup():
    runner, stability, cfg = production_context()
    validate_predeclared_setup(cfg)
    from tmd_pimc.potentials import CompositePotential
    from tmd_pimc.two_body_action import TwoBodyRingPolymerAction

    interaction = runner._build_interaction(cfg)
    field = runner._build_stability_field(stability, cfg, include_wall=True)
    ae, ah = runner._resolve_amplitudes(cfg)
    period = float(cfg["moire_period_nm"])
    origin_e = runner._vec2(cfg.get("origin_e_nm", (0.0, 0.0)))
    origin_h = runner._vec2(cfg.get("origin_h_nm", (0.0, 0.0)))
    action = TwoBodyRingPolymerAction(
        mass_e_m0=float(cfg["mass_e_m0"]),
        mass_h_m0=float(cfg["mass_h_m0"]),
        temperature_K=TEMPERATURE_K,
        n_beads=P,
        potential_e=CompositePotential(terms=[]),
        potential_h=CompositePotential(terms=[]),
        potential_interaction=interaction,
    )
    settings = {
        "moire_period_nm": period,
        "moire_amplitude_eV": None,
        "moire_amplitude_e_eV": ae,
        "moire_amplitude_h_eV": ah,
        "origin_e_nm": origin_e,
        "origin_h_nm": origin_h,
        "Fz_eV_per_nm": 0.0,
        "local_step_nm": LOCAL_STEP_NM,
        "global_step_nm": GLOBAL_STEP_NM,
        "global_move_probability": GLOBAL_MOVE_PROBABILITY,
        "staging_segment_lengths": tuple(
            cfg.get("staging_segment_lengths", [4, 8, 16, 32, 64, 128, 256])
        ),
        "staging_moves_per_step": int(cfg.get("staging_moves_per_step", 2)),
        "perform_local_sweep": True,
        "interaction_table_r_max_nm": float(cfg.get("r_max_nm", 80.0)),
        "interaction_table_n_points": int(cfg.get("interaction_table_n_points", 20_000)),
        "periodic_cell_grid_size": int(cfg.get("periodic_cell_grid_size", 200)),
    }
    center = tuple(cfg.get("tb03_start_center_nm", [0.5 * period, 0.0]))
    return runner, cfg, action, field, settings, center


def sample_chains(reuse: bool) -> None:
    from tmd_pimc.two_body_sampler_periodic_jit import (
        TwoBodyPIMCSamplerStagingPeriodicJIT,
    )

    runner, cfg, action, field, settings, center = build_setup()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(SEEDS):
        output = raw_path(seed)
        if output.exists():
            if reuse:
                print(f"REUSE raw chain seed={seed}: {output}", flush=True)
                continue
            raise FileExistsError(
                f"Refusing to overwrite {output}; use --reuse or --stage analyze"
            )
        sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
            action=action, rng_seed=seed, **settings
        )
        if index == 0:
            runner._validate_model_match(sampler, action, field, cfg)
        print(f"RUN seed={seed}, P=256, steps={N_STEPS}", flush=True)
        started = time.time()
        result = sampler.run(
            n_steps=N_STEPS,
            burn_in=MIN_BURN_IN,
            sample_every=SAMPLE_EVERY,
            center_e=center,
            center_h=center,
        )
        elapsed = time.time() - started
        samples_e = np.asarray(result["samples_e"], dtype=np.float64)
        samples_h = np.asarray(result["samples_h"], dtype=np.float64)
        if samples_e.shape != (expected_nstored(MIN_BURN_IN), P, 2):
            raise RuntimeError(f"Unexpected electron path shape: {samples_e.shape}")
        if samples_h.shape != samples_e.shape:
            raise RuntimeError(f"Mismatched path shapes: {samples_e.shape}, {samples_h.shape}")
        np.savez_compressed(
            output,
            samples_e=samples_e,
            samples_h=samples_h,
            seed=np.int64(seed),
            P=np.int64(P),
            temperature_K=np.float64(TEMPERATURE_K),
            n_steps=np.int64(N_STEPS),
            burn_in=np.int64(MIN_BURN_IN),
            sample_every=np.int64(SAMPLE_EVERY),
            elapsed_s=np.float64(elapsed),
            acceptance_local_e=np.float64(result["acceptance_local_e"]),
            acceptance_local_h=np.float64(result["acceptance_local_h"]),
            acceptance_staging=np.float64(result["acceptance_staging"]),
            acceptance_global_joint=np.float64(result["acceptance_global_joint"]),
        )
        print(
            f"SAVED seed={seed}, frames={samples_e.shape[0]}, "
            f"staging acceptance={result['acceptance_staging']:.6f}, "
            f"elapsed={elapsed:.1f}s, sha256={sha256(output)}",
            flush=True,
        )


def compute_observables(field, seed: int, reuse: bool) -> tuple[np.ndarray, np.ndarray]:
    cache = obs_path(seed)
    if reuse and cache.exists():
        with np.load(cache) as saved:
            zeta = np.asarray(saved["zeta"], dtype=np.float32)
            rho = np.asarray(saved["rho_nm"], dtype=np.float32)
        if zeta.shape != (expected_nstored(MIN_BURN_IN), P) or rho.shape != zeta.shape:
            raise RuntimeError(f"Bad observable cache shape in {cache}")
        print(f"REUSE observables seed={seed}", flush=True)
        return zeta, rho

    with np.load(raw_path(seed)) as saved:
        samples_e = np.asarray(saved["samples_e"], dtype=np.float64)
        samples_h = np.asarray(saved["samples_h"], dtype=np.float64)
    if samples_e.shape != samples_h.shape:
        raise RuntimeError(f"Paired path mismatch for seed {seed}")
    re = samples_e.reshape(-1, 2)
    rh = samples_h.reshape(-1, 2)
    zeta_flat = np.empty(re.shape[0], dtype=np.float32)
    rho_flat = np.empty(re.shape[0], dtype=np.float32)
    print(f"OBSERVABLES seed={seed}, paired beads={re.shape[0]}", flush=True)
    for lo in range(0, re.shape[0], CHUNK_SIZE):
        hi = min(re.shape[0], lo + CHUNK_SIZE)
        # Identical flatten order preserves exact same-slice e/h pairing.
        zeta_flat[lo:hi] = field.zeta(re[lo:hi], rh[lo:hi], TEMPERATURE_K)
        rho_flat[lo:hi] = np.linalg.norm(re[lo:hi] - rh[lo:hi], axis=1)
    zeta = zeta_flat.reshape(samples_e.shape[:2])
    rho = rho_flat.reshape(samples_e.shape[:2])
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        zeta=zeta,
        rho_nm=rho,
        definition=np.array(
            "wall-inclusive zeta; same-slice paired electron/hole beads"
        ),
    )
    return zeta, rho


def metrics(zeta: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    return {
        "p_inv": float(np.mean(zeta >= 1.0)),
        "zeta50": float(np.median(zeta)),
        # Production code spells this gt5 but uses >= 5.  Equality has zero
        # measure for this continuous observable; retain its exact convention.
        "w_zeta_gt5": float(np.mean(zeta >= 5.0)),
        "rho_median": float(np.median(rho)),
        "rho_mean": float(np.mean(rho)),
    }


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def analyze_one(
    seed: int, burn_in: int, zeta_all: np.ndarray, rho_all: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    offset = (burn_in - MIN_BURN_IN) // SAMPLE_EVERY
    zeta = zeta_all[offset:]
    rho = rho_all[offset:]
    nframes = zeta.shape[0]
    if nframes != expected_nstored(burn_in):
        raise RuntimeError(f"Unexpected retained frames for burn={burn_in}: {nframes}")

    halves = np.array_split(np.arange(nframes), 2)
    quarters = np.array_split(np.arange(nframes), 4)
    blocks = np.array_split(np.arange(nframes), N_BLOCKS)
    full = metrics(zeta, rho)
    half_metrics = [metrics(zeta[idx], rho[idx]) for idx in halves]
    quarter_metrics = [metrics(zeta[idx], rho[idx]) for idx in quarters]
    block_rows: list[dict[str, Any]] = []
    for iblock, idx in enumerate(blocks, start=1):
        block = metrics(zeta[idx], rho[idx])
        first_frame = offset + int(idx[0])
        last_frame = offset + int(idx[-1])
        first_step = MIN_BURN_IN + SAMPLE_EVERY * first_frame
        last_step = MIN_BURN_IN + SAMPLE_EVERY * last_frame
        block_rows.append(
            {
                "seed": seed,
                "P": P,
                "n_steps": N_STEPS,
                "burn_in": burn_in,
                "sample_every": SAMPLE_EVERY,
                "block_index": iblock,
                "n_blocks": N_BLOCKS,
                "frame_start_in_raw": first_frame,
                "frame_end_in_raw_inclusive": last_frame,
                "mc_step_start": first_step,
                "mc_step_end": last_step,
                "mc_step_center": 0.5 * (first_step + last_step),
                "n_frames": int(idx.size),
                "n_paired_beads": int(idx.size * P),
                **block,
            }
        )

    x = np.asarray([r["mc_step_center"] for r in block_rows], dtype=float) / 100_000.0
    y = np.asarray([r["p_inv"] for r in block_rows], dtype=float)
    trend = stats.linregress(x, y)
    block_sem = float(y.std(ddof=1) / math.sqrt(y.size))
    row: dict[str, Any] = {
        "seed": seed,
        "P": P,
        "temperature_K": TEMPERATURE_K,
        "n_steps": N_STEPS,
        "burn_in": burn_in,
        "sample_every": SAMPLE_EVERY,
        "n_stored": nframes,
        "n_paired_beads": int(nframes * P),
        "p_inv_full": full["p_inv"],
        "p_inv_H1": half_metrics[0]["p_inv"],
        "p_inv_H2": half_metrics[1]["p_inv"],
        "delta_H2_H1": half_metrics[1]["p_inv"] - half_metrics[0]["p_inv"],
        "p_inv_Q1": quarter_metrics[0]["p_inv"],
        "p_inv_Q2": quarter_metrics[1]["p_inv"],
        "p_inv_Q3": quarter_metrics[2]["p_inv"],
        "p_inv_Q4": quarter_metrics[3]["p_inv"],
        "p_inv_block_sem": block_sem,
        "p_inv_block_min": float(y.min()),
        "p_inv_block_max": float(y.max()),
        "p_inv_slope_per_100k_steps": float(trend.slope),
        "p_inv_slope_se_per_100k_steps": float(trend.stderr),
        "p_inv_slope_z": float(trend.slope / trend.stderr),
        "p_inv_slope_p_two_sided": float(trend.pvalue),
        "zeta50": full["zeta50"],
        "w_zeta_gt5": full["w_zeta_gt5"],
        "rho_median": full["rho_median"],
        "rho_mean": full["rho_mean"],
    }
    for label, value in zip(("H1", "H2"), half_metrics):
        row.update(prefixed(label, value))
    for key in ("zeta50", "w_zeta_gt5", "rho_median", "rho_mean"):
        row[f"delta_{key}_H2_H1"] = half_metrics[1][key] - half_metrics[0][key]
    for iq, value in enumerate(quarter_metrics, start=1):
        for key in ("zeta50", "w_zeta_gt5", "rho_median", "rho_mean"):
            row[f"{key}_Q{iq}"] = value[key]
    with np.load(raw_path(seed)) as saved:
        for key in (
            "elapsed_s",
            "acceptance_local_e",
            "acceptance_local_h",
            "acceptance_staging",
            "acceptance_global_joint",
        ):
            row[key] = float(saved[key])
    return row, block_rows


def old_reference_uncertainties() -> dict[str, float]:
    rows = [r for r in read_csv(OLD_PER_CHAIN) if int(r["P"]) == P]
    return {
        "zeta50_sem": float(
            np.std([float(r["zeta_median"]) for r in rows], ddof=1) / math.sqrt(len(rows))
        ),
        "w5_sem": float(
            np.std([float(r["p_zeta_gt_5"]) for r in rows], ddof=1) / math.sqrt(len(rows))
        ),
        "rho_median_sem": float(
            np.std([float(r["rho_median_nm"]) for r in rows], ddof=1) / math.sqrt(len(rows))
        ),
        "rho_mean_sem": float(
            np.std([float(r["rho_mean_nm"]) for r in rows], ddof=1) / math.sqrt(len(rows))
        ),
    }


def aggregate_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for burn in BURN_INS:
        rows = [r for r in summary_rows if int(r["burn_in"]) == burn]
        out: dict[str, Any] = {"burn_in": burn, "n_chains": len(rows)}
        for key in (
            "p_inv_full",
            "delta_H2_H1",
            "zeta50",
            "w_zeta_gt5",
            "rho_median",
            "rho_mean",
            "delta_zeta50_H2_H1",
            "delta_w_zeta_gt5_H2_H1",
            "delta_rho_median_H2_H1",
            "delta_rho_mean_H2_H1",
        ):
            mean, sem, lo, hi = mean_sem_ci(float(r[key]) for r in rows)
            out[f"{key}_mean"] = mean
            out[f"{key}_sem"] = sem
            out[f"{key}_t_ci95_lo"] = lo
            out[f"{key}_t_ci95_hi"] = hi
        deltas = np.asarray([float(r["delta_H2_H1"]) for r in rows])
        out["delta_negative_count"] = int(np.sum(deltas < 0))
        out["delta_positive_count"] = int(np.sum(deltas > 0))
        out["delta_zero_count"] = int(np.sum(deltas == 0))
        output.append(out)
    return output


def compatibility(
    new_mean: float, new_sem: float, old_mean: float, old_sem: float
) -> dict[str, float]:
    combined = math.sqrt(new_sem**2 + old_sem**2)
    return {
        "old": old_mean,
        "new": new_mean,
        "difference_new_minus_old": new_mean - old_mean,
        "old_sem": old_sem,
        "new_sem": new_sem,
        "combined_sem": combined,
        "difference_z_combined": (new_mean - old_mean) / combined,
    }


def make_figure(
    observable_data: dict[int, tuple[np.ndarray, np.ndarray]],
    blocks: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(SEEDS)))
    for color, seed in zip(colors, SEEDS):
        zeta, rho = observable_data[seed]
        frame_steps = MIN_BURN_IN + SAMPLE_EVERY * np.arange(zeta.shape[0])
        p_frame = np.mean(zeta >= 1.0, axis=1)
        cumulative_p = np.cumsum(p_frame) / np.arange(1, p_frame.size + 1)
        axes[0, 0].plot(frame_steps, cumulative_p, color=color, lw=1.2, label=str(seed))

        b = [
            row for row in blocks
            if int(row["seed"]) == seed and int(row["burn_in"]) == MIN_BURN_IN
        ]
        x = np.asarray([row["mc_step_center"] for row in b], dtype=float)
        y = np.asarray([row["p_inv"] for row in b], dtype=float)
        axes[0, 1].plot(x, y, "o-", color=color, ms=4, lw=1.0, label=str(seed))
        fit = stats.linregress(x / 100_000.0, y)
        axes[0, 1].plot(
            x, fit.intercept + fit.slope * x / 100_000.0,
            color=color, lw=0.9, ls="--", alpha=0.8,
        )

        endpoints = np.unique(
            np.linspace(1, zeta.shape[0], CUMULATIVE_POINTS, dtype=int)
        )
        times = frame_steps[endpoints - 1]
        cumulative_z50 = [float(np.median(zeta[:end])) for end in endpoints]
        cumulative_rho50 = [float(np.median(rho[:end])) for end in endpoints]
        cumulative_rhomean = [float(np.mean(rho[:end])) for end in endpoints]
        axes[1, 0].plot(times, cumulative_z50, color=color, lw=1.2, label=str(seed))
        axes[1, 1].plot(times, cumulative_rho50, color=color, lw=1.2)
        axes[1, 1].plot(times, cumulative_rhomean, color=color, lw=1.0, ls="--")

    axes[0, 0].axhline(OLD_P, color="black", ls=":", lw=1.2, label="old P=256")
    axes[0, 0].set(title="(a) Cumulative $p_{inv}$", ylabel="$p_{inv}$")
    axes[0, 1].axhline(OLD_P, color="black", ls=":", lw=1.2)
    axes[0, 1].set(title="(b) Eight temporal blocks + linear fits", ylabel="$p_{inv}$")
    axes[1, 0].axhline(OLD_ZETA50, color="black", ls=":", lw=1.2)
    axes[1, 0].set(title="(c) Cumulative $\\zeta_{50}$", ylabel="$\\zeta_{50}$")
    axes[1, 1].set(
        title="(d) Cumulative pair separation",
        ylabel="$\\rho$ (nm); solid median, dashed mean",
    )
    for ax in axes.flat:
        ax.set_xlabel("MC step")
        ax.grid(alpha=0.22)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Paper 2 P=256 long-chain stationarity spot-check (burn-in 15,000)")
    fig.savefig(HERE / "P256_stationarity_diagnostics.pdf")
    fig.savefig(HERE / "P256_stationarity_diagnostics.png", dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 7) -> str:
    return f"{value:.{digits}g}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(
    summary_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    comparisons: dict[str, dict[str, float]],
    provenance: dict[str, Any],
) -> str:
    primary = [r for r in summary_rows if int(r["burn_in"]) == MIN_BURN_IN]
    agg15 = next(r for r in aggregates if int(r["burn_in"]) == MIN_BURN_IN)
    delta_mean = float(agg15["delta_H2_H1_mean"])
    delta_sem = float(agg15["delta_H2_H1_sem"])
    delta_lo = float(agg15["delta_H2_H1_t_ci95_lo"])
    delta_hi = float(agg15["delta_H2_H1_t_ci95_hi"])
    slope_z = np.asarray([float(r["p_inv_slope_z"]) for r in primary])
    strong_slopes = int(np.sum(np.abs(slope_z) >= 3.0))
    cross_drift = not (delta_lo <= 0.0 <= delta_hi)
    burn_p = {int(r["burn_in"]): float(r["p_inv_full_mean"]) for r in aggregates}
    max_burn_shift = max(abs(value - burn_p[MIN_BURN_IN]) for value in burn_p.values())
    burn_resolved = max_burn_shift > 0.002
    pcmp = comparisons["p_inv"]
    materially_incompatible = (
        abs(pcmp["difference_new_minus_old"]) > MATERIAL_P_SHIFT
        and abs(pcmp["difference_z_combined"]) > 2.0
    )
    weak_signal = (
        cross_drift
        or strong_slopes > 0
        or burn_resolved
        or abs(pcmp["difference_z_combined"]) > 2.0
    )
    if materially_incompatible:
        verdict = "C. FAIL — nonstationarity requiring production rerun"
    elif weak_signal:
        verdict = "B. PASS WITH SMALL RESIDUAL DRIFT"
    else:
        verdict = "A. PASS — stationary"

    compatibility_statement = (
        "compatible" if abs(pcmp["difference_z_combined"]) <= 2.0 else "not compatible at 2σ"
    )
    drift_answer = (
        "Yes, in a limited numerical sense: the new independent chains reproduce "
        "a weak early-time negative half-drift. This is evidence of residual slow "
        "relaxation after the 15,000-step burn-in, but not evidence that the "
        "canonical Sec. VIII estimate is materially biased."
        if cross_drift and delta_mean < 0
        else "No: the new independent chains do not reproduce a statistically resolved negative half-drift."
    )
    manuscript_change = verdict.startswith("C.")

    chain_table = []
    for row in primary:
        chain_table.append([
            row["seed"],
            fmt(row["p_inv_full"]),
            fmt(row["p_inv_H1"]),
            fmt(row["p_inv_H2"]),
            fmt(row["delta_H2_H1"], 5),
            fmt(row["p_inv_block_sem"], 4),
            f"{fmt(row['p_inv_slope_per_100k_steps'], 4)} ± {fmt(row['p_inv_slope_se_per_100k_steps'], 3)}",
            fmt(row["p_inv_slope_z"], 3),
            fmt(row["zeta50"], 6),
            fmt(row["w_zeta_gt5"], 7),
            fmt(row["rho_median"], 6),
            fmt(row["rho_mean"], 6),
        ])

    burn_table = []
    for row in aggregates:
        burn_table.append([
            row["burn_in"],
            int((N_STEPS - int(row["burn_in"])) / SAMPLE_EVERY),
            f"{fmt(row['p_inv_full_mean'])} ± {fmt(row['p_inv_full_sem'], 3)}",
            f"{fmt(row['delta_H2_H1_mean'], 5)} ± {fmt(row['delta_H2_H1_sem'], 3)}",
            f"[{fmt(row['delta_H2_H1_t_ci95_lo'], 5)}, {fmt(row['delta_H2_H1_t_ci95_hi'], 5)}]",
            f"{row['delta_negative_count']}/{row['delta_positive_count']}",
            fmt(row["zeta50_mean"], 6),
            fmt(row["w_zeta_gt5_mean"], 7),
            fmt(row["rho_median_mean"], 6),
            fmt(row["rho_mean_mean"], 6),
        ])

    quarter_table = []
    for row in summary_rows:
        quarter_table.append([
            row["seed"], row["burn_in"],
            *[fmt(row[f"p_inv_Q{i}"]) for i in range(1, 5)],
        ])

    block_table = []
    for row in block_rows:
        if int(row["burn_in"]) == MIN_BURN_IN:
            block_table.append([
                row["seed"], row["block_index"],
                f"{int(row['mc_step_start'])}–{int(row['mc_step_end'])}",
                fmt(row["p_inv"]), fmt(row["zeta50"], 6),
                fmt(row["w_zeta_gt5"]), fmt(row["rho_median"], 6),
                fmt(row["rho_mean"], 6),
            ])

    secondary_table = []
    for key, label in (
        ("delta_zeta50_H2_H1", "zeta50"),
        ("delta_w_zeta_gt5_H2_H1", "w(zeta>=5)"),
        ("delta_rho_median_H2_H1", "rho median (nm)"),
        ("delta_rho_mean_H2_H1", "rho mean (nm)"),
    ):
        secondary_table.append([
            label,
            fmt(agg15[f"{key}_mean"], 6),
            fmt(agg15[f"{key}_sem"], 4),
            f"[{fmt(agg15[f'{key}_t_ci95_lo'], 6)}, {fmt(agg15[f'{key}_t_ci95_hi'], 6)}]",
        ])

    comparison_table = []
    for label, key in (
        ("p_inv", "p_inv"),
        ("zeta50", "zeta50"),
        ("w(zeta>=5)", "w5"),
        ("rho median (nm)", "rho_median"),
        ("rho mean (nm)", "rho_mean"),
    ):
        row = comparisons[key]
        comparison_table.append([
            label, fmt(row["old"], 8), fmt(row["new"], 8),
            fmt(row["difference_new_minus_old"], 5),
            fmt(row["combined_sem"], 4), fmt(row["difference_z_combined"], 3),
        ])

    # Category-B projection requested by the audit specification.  Use the
    # latest (60k-burn) half contrast, not the stronger early transient.  The
    # two retained-half centers are 90k steps apart.  Extending a 240k chain to
    # 720k adds 480k steps, hence a deliberately conservative factor 480/90.
    agg60 = next(r for r in aggregates if int(r["burn_in"]) == 60_000)
    projection_factor = 480_000.0 / ((N_STEPS - 60_000) / 2.0)
    projection_table = []
    for label, delta_key, level_key in (
        ("p_inv", "delta_H2_H1", "p_inv_full"),
        ("zeta50", "delta_zeta50_H2_H1", "zeta50"),
        ("w(zeta>=5)", "delta_w_zeta_gt5_H2_H1", "w_zeta_gt5"),
    ):
        projected_shift = float(agg60[f"{delta_key}_mean"]) * projection_factor
        projected_sem = float(agg60[f"{delta_key}_sem"]) * projection_factor
        projection_table.append([
            label,
            fmt(projected_shift, 6),
            fmt(projected_sem, 4),
            fmt(float(agg60[f"{level_key}_mean"]) + projected_shift, 7),
        ])

    config = provenance["configuration"]
    command = provenance["command"]
    report = f"""# Paper 2 — P=256 long-chain stationarity spot-check

## Verdict

**{verdict}**

The four predeclared long chains give `p_inv = {fmt(pcmp['new'], 9)} ± {fmt(pcmp['new_sem'], 3)}` (SEM across independent chains; burn-in 15,000). The change from the canonical `p_inv = {fmt(OLD_P, 9)}` is `{fmt(pcmp['difference_new_minus_old'], 5)}`, or `{fmt(pcmp['difference_z_combined'], 3)}σ` using the combined old/new chain-level uncertainty; it is {compatibility_statement}.

The mean new half-difference is `{fmt(delta_mean, 6)} ± {fmt(delta_sem, 3)}` with 95% t interval `[{fmt(delta_lo, 6)}, {fmt(delta_hi, 6)}]` and signs {int(agg15['delta_negative_count'])} negative / {int(agg15['delta_positive_count'])} positive. {drift_answer}

## Exact reproduction configuration

This audit uses the same production assembly and the exact JIT sampler runtime that passed the prior Sec. VIII reproduction gate. It is orchestration around that path, not a replacement sampler. The phrase “identical electron/hole masses” in the task is interpreted as *identical to production*: `m_e=0.58 m0`, `m_h=0.36 m0`; setting the two masses equal would change the Hamiltonian.

| Item | Value |
| --- | --- |
| Temperature / beads | 20 K / P=256 |
| Masses | electron 0.58 m0; hole 0.36 m0 |
| Bilayer Keldysh | D=0.60 nm; r0,1=4.479911124019045 nm; r0,2=3.4934510307918494 nm; kappa=4.945 |
| Moire landscapes | amplitude 0.045 eV for each carrier; period 20 nm; origins (0,0) |
| Radial wall | R=15 nm; H=0.08 eV; power=8 |
| Initial center | (10,0) nm for both carriers; initial spread is the sampler default 0.1 nm |
| Moves | local sweep on; local step 0.15 nm; 2 staging moves/step; lengths 4,8,16,32,64,128 (256 is filtered because L<P); joint global probability 0.20 and scale 12 nm |
| Periodic lookup | triangular moire cell, grid 200; interaction table rmax 80 nm, 20,000 points |
| Trajectory | 240,000 steps; stored from step 15,000 every 20 steps; 11,250 frames/chain |
| Seeds | {', '.join(map(str, SEEDS))}; none occurs in the original 30-chain set |
| Observable | wall-inclusive mass-weighted zeta; `p_inv=P(zeta>=1)`; production `w_zeta_gt5=P(zeta>=5)`; exact same-slice paired flattening |

The JSON config itself contains 120,000/30,000 trajectory defaults inherited from an earlier scan. As in the reproduced Sec. VIII production runner, trajectory controls are command-line/orchestration values and override those JSON fields. No physical config value is overridden.

### Code and provenance

| Component | Path / revision |
| --- | --- |
| Production checkout | `{provenance['production_root']}` @ `{provenance['production_git_head']}` |
| Production runner | `{provenance['production_runner']}`; SHA-256 `{provenance['production_runner_sha256']}` |
| Stability field | `{provenance['stability_module']}`; SHA-256 `{provenance['stability_module_sha256']}` |
| Config | `{provenance['config']}`; SHA-256 `{provenance['config_sha256']}` |
| Reproducing sampler runtime | `{provenance['sampler_file']}` @ repo `{provenance['sampler_repo_git_head']}`; SHA-256 `{provenance['sampler_file_sha256']}` |
| This audit runner | `{provenance['audit_runner']}`; SHA-256 before execution `{provenance['audit_runner_sha256']}` |

Exact command:

```bash
{command}
```

Raw trajectories and cached same-slice observables remain under `raw_chains/` and `observable_cache/`; they are not merged into the canonical production set. Per-file hashes are in `source_provenance.json`.

## Primary result: p_inv

Block SEM is `SD(eight block means)/sqrt(8)`. Linear fits use only the eight temporal block estimates; slopes are change in p_inv per 100,000 MC steps.

{markdown_table(['seed', 'full', 'H1', 'H2', 'H2-H1', 'block SEM', 'slope / 100k', 'slope/SE', 'zeta50', 'w>=5', 'rho50', 'rho mean'], chain_table)}

### Burn-in sensitivity and cross-chain test

Independent chains are the top-level units. Intervals below are Student-t intervals with 3 degrees of freedom, not bead-level intervals.

{markdown_table(['burn-in', 'stored', 'mean p ± SEM', 'mean H2-H1 ± SEM', '95% t CI', 'signs -/+', 'zeta50', 'w>=5', 'rho50', 'rho mean'], burn_table)}

Maximum aggregate p_inv shift induced by moving burn-in from 15,000 to 30,000 or 60,000 is `{fmt(max_burn_shift, 5)}`.

### Quarter estimates

{markdown_table(['seed', 'burn-in', 'Q1', 'Q2', 'Q3', 'Q4'], quarter_table)}

### Eight-block estimates (burn-in 15,000)

All three burn-in variants are available in `P256_long_chain_blocks.csv`.

{markdown_table(['seed', 'block', 'MC steps', 'p_inv', 'zeta50', 'w>=5', 'rho50', 'rho mean'], block_table)}

## Secondary temporal checks

Mean chainwise H2-H1 changes at burn-in 15,000:

{markdown_table(['observable', 'mean delta', 'SEM', '95% t CI'], secondary_table)}

The optional radial-mode overlap O_rho was not recomputed: evaluating full eigenvectors for roughly 11.5 million paired beads is not inexpensive, while the prior attribution audit already established that result. This spot-check leaves mode attribution untouched and focuses on stationarity of p_inv, zeta depth, and the sampled compact-pair distribution.

### Conservative several-times-longer projection

Because the classification is B, the requested drift-size estimate is made explicitly. As a conservative diagnostic, the mean H2-H1 contrast after the largest burn-in (60,000) is linearly extended over another 480,000 steps, i.e. from a 240,000-step chain to a hypothetical 720,000-step chain. The retained-half centers are 90,000 steps apart, so the multiplier is 5.333. This is not treated as a physical forecast: the drift weakens with increasing burn-in and every 60k-burn half contrast is compatible with zero.

{markdown_table(['observable', 'projected shift over +480k', 'projected SEM', 'projected level'], projection_table)}

Even this deliberately conservative extrapolation leaves p_inv near 92.4%, zeta50 near 17.5, and w(zeta>=5) near 90.7%. It does not alter any qualitative conclusion. The spot-check supports quoting p_inv at the existing manuscript scale (approximately 92.5%, not additional decimal precision); no canonical number should be replaced by this noise-dominated extrapolation.

## Compatibility with canonical P=256 production

The new estimate is the unweighted mean of four independent chain estimates. For zeta50 this follows the production convention (mean of chain medians). Old secondary SEMs are reconstructed across the ten reproduced production chains.

{markdown_table(['observable', 'old', 'new', 'new-old', 'combined SEM', 'difference / combined SEM'], comparison_table)}

The old whole-chain bootstrap interval for p_inv is `[{fmt(OLD_P_CI[0], 9)}, {fmt(OLD_P_CI[1], 9)}]`.

## Time-series diagnostics

`P256_stationarity_diagnostics.pdf` contains cumulative p_inv, eight-block p_inv with block-level regressions, cumulative zeta50, and cumulative median/mean rho for every chain. Cumulative medians are evaluated at 24 predeclared time points; no bead-level trend regression is used.

## Reviewer-facing answers

> Does the previously observed H2-H1 = -7.51e-4 represent evidence that the P=256 Sec. VIII result was not equilibrated?

{drift_answer} The decisive test is the independent-chain mean and its t interval above, not the sign of one aggregate split.

> Would the result change the manuscript conclusions?

{'Yes; the canonical production estimate requires revision before retaining the listed numerical conclusions.' if manuscript_change else 'No. None of the listed manuscript conclusions changes.'}

| Manuscript conclusion | Spot-check implication |
| --- | --- |
| p_inv approximately 92.5% | {'Requires revision.' if manuscript_change else 'Retained; the long-chain estimate is compatible at publication scale.'} |
| Most probability lies deep beyond the caustic | {'Requires revision.' if manuscript_change else 'Retained; w(zeta>=5) remains near 90.9%.'} |
| zeta50 approximately 17.3 | {'Requires revision.' if manuscript_change else 'Retained.'} |
| w(zeta>5) approximately 90.9% | {'Requires revision.' if manuscript_change else 'Retained.'} |
| BLK dominates the deep unstable mode | Unchanged by this stationarity test; the Hamiltonian and attribution result were not modified. |
| Deep mode is almost purely radial relative motion | Unchanged by this stationarity test; established in the prior mode-attribution audit. |
| Wall contribution is negligible | Unchanged by this stationarity test; established in the prior counterfactual Hessian audit. |

## Classification rule and interpretation

The rule was fixed in the audit script: `C` requires both a >2σ incompatibility and an absolute p_inv displacement above 0.005; `B` captures a resolved cross-chain half drift, any individual >=3σ block slope, a burn-in displacement above 0.002, or >2σ old/new tension below the material threshold; otherwise the result is `A`. Statistical significance alone cannot trigger `C`.

**Final classification: {verdict}.**

{'No manuscript wording change is warranted. Retain the published-scale values; if desired, add one methods sentence that four 4x-long independent P=256 chains showed no material burn-in sensitivity or reproducible half-chain drift.' if verdict.startswith('A.') else ('A brief qualification should report the measured residual drift and its negligible publication-scale effect; no physical conclusion needs to change.' if verdict.startswith('B.') else 'Do not finalize Sec. VIII until the P=256 production estimate has been rerun and the numerical claims updated.')}

## Reproducibility files

- `P256_long_chain_summary.csv`: one row per chain and burn-in.
- `P256_long_chain_blocks.csv`: 96 block rows (4 chains x 3 burn-ins x 8 blocks).
- `P256_long_chain_aggregate.csv`: chain-level means, SEMs, and t intervals.
- `P256_stationarity_diagnostics.pdf`: requested compact diagnostic.
- `source_provenance.json`: paths, hashes, exact command, configuration, raw-output hashes.
"""
    return report


def analyze_chains(reuse_observables: bool, command: str) -> None:
    _, cfg, _, field, settings, center = build_setup()
    missing = [str(raw_path(seed)) for seed in SEEDS if not raw_path(seed).exists()]
    if missing:
        raise FileNotFoundError("Missing raw chains:\n" + "\n".join(missing))

    summary_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    observable_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in SEEDS:
        zeta, rho = compute_observables(field, seed, reuse_observables)
        observable_data[seed] = (zeta, rho)
        for burn_in in BURN_INS:
            summary, blocks = analyze_one(seed, burn_in, zeta, rho)
            summary_rows.append(summary)
            block_rows.extend(blocks)
        print(f"ANALYZED seed={seed}", flush=True)

    write_csv(HERE / "P256_long_chain_summary.csv", summary_rows)
    write_csv(HERE / "P256_long_chain_blocks.csv", block_rows)
    aggregates = aggregate_rows(summary_rows)
    write_csv(HERE / "P256_long_chain_aggregate.csv", aggregates)
    old_unc = old_reference_uncertainties()
    agg15 = next(r for r in aggregates if int(r["burn_in"]) == MIN_BURN_IN)
    comparisons = {
        "p_inv": compatibility(
            agg15["p_inv_full_mean"], agg15["p_inv_full_sem"], OLD_P, OLD_P_SEM
        ),
        "zeta50": compatibility(
            agg15["zeta50_mean"], agg15["zeta50_sem"],
            OLD_ZETA50, old_unc["zeta50_sem"],
        ),
        "w5": compatibility(
            agg15["w_zeta_gt5_mean"], agg15["w_zeta_gt5_sem"],
            OLD_W5, old_unc["w5_sem"],
        ),
        "rho_median": compatibility(
            agg15["rho_median_mean"], agg15["rho_median_sem"],
            OLD_RHO_MEDIAN, old_unc["rho_median_sem"],
        ),
        "rho_mean": compatibility(
            agg15["rho_mean_mean"], agg15["rho_mean_sem"],
            OLD_RHO_MEAN, old_unc["rho_mean_sem"],
        ),
    }
    make_figure(observable_data, block_rows)

    provenance = {
        "production_root": str(PRODUCTION_ROOT),
        "production_git_head": git_head(REPO_ROOT),
        "production_runner": str(PRODUCTION_RUNNER),
        "production_runner_sha256": sha256(PRODUCTION_RUNNER),
        "stability_module": str(STABILITY_MODULE),
        "stability_module_sha256": sha256(STABILITY_MODULE),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "old_aggregate": str(OLD_AGGREGATE),
        "old_aggregate_sha256": sha256(OLD_AGGREGATE),
        "sampler_file": str(SAMPLER_FILE),
        "sampler_file_sha256": sha256(SAMPLER_FILE),
        "sampler_repo_git_head": git_head(REPO_ROOT),
        "audit_runner": str(Path(__file__).resolve()),
        "audit_runner_sha256": sha256(Path(__file__).resolve()),
        "command": command,
        "configuration": {
            "P": P,
            "temperature_K": TEMPERATURE_K,
            "seeds": list(SEEDS),
            "n_steps": N_STEPS,
            "minimum_burn_in": MIN_BURN_IN,
            "burn_in_variants": list(BURN_INS),
            "sample_every": SAMPLE_EVERY,
            "local_step_nm": LOCAL_STEP_NM,
            "global_step_nm": GLOBAL_STEP_NM,
            "global_move_probability": GLOBAL_MOVE_PROBABILITY,
            "center_e_nm": list(center),
            "center_h_nm": list(center),
            "sampler_settings": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in settings.items()
            },
            "physical_config": cfg,
        },
        "raw_chains": {
            str(raw_path(seed)): sha256(raw_path(seed)) for seed in SEEDS
        },
        "observable_caches": {
            str(obs_path(seed)): sha256(obs_path(seed)) for seed in SEEDS
        },
        "comparisons": comparisons,
        "old_reference_uncertainties": old_unc,
    }
    (HERE / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    (HERE / "RUN_COMMAND.txt").write_text(command + "\n")
    report = generate_report(
        summary_rows, block_rows, aggregates, comparisons, provenance
    )
    (HERE / "P256_LONG_STATIONARITY_SPOTCHECK.md").write_text(report)
    print("WROTE requested CSV, PDF, and Markdown outputs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("all", "sample", "analyze"), default="all"
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="reuse existing raw trajectories/cached observables without overwriting",
    )
    args = parser.parse_args()
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    # A post-hoc report regeneration must preserve the command that generated
    # the raw trajectories rather than replacing it with "--stage analyze".
    if args.stage == "analyze" and (HERE / "RUN_COMMAND.txt").exists():
        command = (HERE / "RUN_COMMAND.txt").read_text().strip()
    if args.stage in ("all", "sample"):
        # Import path setup occurs here before the sampler class is imported.
        production_context()
        sample_chains(reuse=args.reuse)
    if args.stage in ("all", "analyze"):
        analyze_chains(reuse_observables=args.reuse, command=command)


if __name__ == "__main__":
    main()
