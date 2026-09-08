#!/usr/bin/env python3
"""Reproduce and attribute the Paper-2 Sec. VIII interacting-pair result.

This is an audit-only program.  It imports the exact production checkout and
runner used for TB-03, reconstructs the paths with the original deterministic
seeds, preserves chain identity, and refuses to perform attribution unless the
published aggregate is reproduced to manuscript precision.

No production file is modified.  All generated paths and tables are written
below this script's directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
AUDIT_REPO_ROOT = HERE.parent
SAMPLER_RUNTIME_ROOT = AUDIT_REPO_ROOT
PRODUCTION_ROOT = AUDIT_REPO_ROOT / "secVIII"
PRODUCTION_RUNNER = PRODUCTION_ROOT / "run_tb03_pinv_ladder.py"
STABILITY_MODULE = PRODUCTION_ROOT / "lha15a0_pair_stability_field_TB03.py"
CONFIG = PRODUCTION_ROOT / "configs/two_body/landscape_scan_config_v2_P256_matched.json"
REFERENCE_DIR = PRODUCTION_ROOT / "results"
REFERENCE_AGGREGATE = REFERENCE_DIR / "tb03_aggregate.csv"
CHAIN_DIR = HERE / "reconstructed_chains"

P_VALUES = (64, 128, 256)
N_CHAINS = 10
SEED_BASE = 31000
N_STEPS = 60000
BURN_IN = 15000
SAMPLE_EVERY = 20
LOCAL_STEP_NM = 0.15
GLOBAL_STEP_NM = 12.0
GLOBAL_MOVE_PROBABILITY = 0.20
CHUNK_SIZE = 50000
N_BOOTSTRAP = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def bootstrap_chain_mean(
    values: Iterable[float], seed: int, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=float)
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return mean, sem, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def production_context():
    # The untracked production runner first imported any installed tmd_pimc and
    # only fell back to --project-root.  Its per-carrier constructor arguments
    # are incompatible with the tracked sampler at PRODUCTION_ROOT HEAD.  The
    # local editable runtime below is the only located implementation matching
    # that call path, and is therefore tested against the archived aggregate.
    sys.path.insert(0, str(SAMPLER_RUNTIME_ROOT / "numerics"))
    runner = load_module("paper2_tb03_production_runner", PRODUCTION_RUNNER)
    stability = load_module("paper2_tb03_production_stability", STABILITY_MODULE)
    cfg = json.loads(CONFIG.read_text())
    return runner, stability, cfg


def chain_path(P: int, seed: int) -> Path:
    return CHAIN_DIR / f"P{P}_seed{seed}.npz"


def reconstruct_chains(reuse: bool) -> list[dict[str, Any]]:
    runner, stability, cfg = production_context()
    from tmd_pimc.potentials import CompositePotential
    from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
    from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT

    T = float(cfg["temperature_K"])
    interaction = runner._build_interaction(cfg)
    field = runner._build_stability_field(stability, cfg, include_wall=True)
    field_no_wall = runner._build_stability_field(stability, cfg, include_wall=False)
    ae, ah = runner._resolve_amplitudes(cfg)
    period = float(cfg["moire_period_nm"])
    origin_e = runner._vec2(cfg.get("origin_e_nm", (0.0, 0.0)))
    origin_h = runner._vec2(cfg.get("origin_h_nm", (0.0, 0.0)))
    stage_lengths = tuple(cfg.get("staging_segment_lengths", [4, 8, 16, 32, 64, 128, 256]))
    stage_moves = int(cfg.get("staging_moves_per_step", 2))
    grid_size = int(cfg.get("periodic_cell_grid_size", 200))
    center = tuple(cfg.get("tb03_start_center_nm", [0.5 * period, 0.0]))

    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for P in P_VALUES:
        action = TwoBodyRingPolymerAction(
            mass_e_m0=float(cfg["mass_e_m0"]),
            mass_h_m0=float(cfg["mass_h_m0"]),
            temperature_K=T,
            n_beads=P,
            potential_e=CompositePotential(terms=[]),
            potential_h=CompositePotential(terms=[]),
            potential_interaction=interaction,
        )
        for ichain in range(N_CHAINS):
            seed = SEED_BASE + 1000 * P + ichain
            output = chain_path(P, seed)
            if reuse and output.exists():
                with np.load(output) as saved:
                    samples_e = saved["samples_e"]
                    samples_h = saved["samples_h"]
                    acceptance = {
                        k: float(saved[k]) for k in (
                            "acceptance_local_e", "acceptance_local_h",
                            "acceptance_staging", "acceptance_global_joint",
                        )
                    }
                elapsed = 0.0
                status = "reused"
            else:
                sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
                    action=action,
                    moire_period_nm=period,
                    moire_amplitude_eV=None,
                    moire_amplitude_e_eV=ae,
                    moire_amplitude_h_eV=ah,
                    origin_e_nm=origin_e,
                    origin_h_nm=origin_h,
                    Fz_eV_per_nm=0.0,
                    local_step_nm=LOCAL_STEP_NM,
                    global_step_nm=GLOBAL_STEP_NM,
                    global_move_probability=GLOBAL_MOVE_PROBABILITY,
                    rng_seed=seed,
                    staging_segment_lengths=stage_lengths,
                    staging_moves_per_step=stage_moves,
                    perform_local_sweep=True,
                    interaction_table_r_max_nm=float(cfg.get("r_max_nm", 80.0)),
                    interaction_table_n_points=int(cfg.get("interaction_table_n_points", 20000)),
                    periodic_cell_grid_size=grid_size,
                )
                if ichain == 0:
                    runner._validate_model_match(sampler, action, field, cfg)
                start = time.time()
                result = sampler.run(
                    n_steps=N_STEPS,
                    burn_in=BURN_IN,
                    sample_every=SAMPLE_EVERY,
                    center_e=center,
                    center_h=center,
                )
                elapsed = time.time() - start
                samples_e = np.asarray(result["samples_e"], dtype=np.float64)
                samples_h = np.asarray(result["samples_h"], dtype=np.float64)
                acceptance = {
                    k: float(result[k]) for k in (
                        "acceptance_local_e", "acceptance_local_h",
                        "acceptance_staging", "acceptance_global_joint",
                    )
                }
                np.savez_compressed(
                    output,
                    samples_e=samples_e,
                    samples_h=samples_h,
                    P=np.int64(P),
                    seed=np.int64(seed),
                    n_steps=np.int64(N_STEPS),
                    burn_in=np.int64(BURN_IN),
                    sample_every=np.int64(SAMPLE_EVERY),
                    **{k: np.float64(v) for k, v in acceptance.items()},
                )
                status = "reconstructed"

            summary = runner._chain_summary(
                field, field_no_wall, samples_e, samples_h, T, CHUNK_SIZE
            )
            row = {
                "P": P,
                "seed": seed,
                "n_frames": int(samples_e.shape[0]),
                **summary,
                **acceptance,
                "elapsed_s": elapsed,
                "path": str(output),
                "path_sha256": sha256(output),
                "status": status,
            }
            rows.append(row)
            print(
                f"{status:13s} P={P:3d} seed={seed} "
                f"p_inv={row['p_invalid']:.9f} z50={row['zeta_median']:.6f}"
            )
    write_csv(HERE / "reproduction_per_chain.csv", rows)
    return rows


def reproduce_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = {int(r["P"]): r for r in read_csv(REFERENCE_AGGREGATE)}
    aggregate: list[dict[str, Any]] = []
    failures: list[str] = []
    for P in P_VALUES:
        rr = [r for r in rows if int(r["P"]) == P]
        p, p_sem, p_lo, p_hi = bootstrap_chain_mean(
            [r["p_invalid"] for r in rr], SEED_BASE + P
        )
        pnw, pnw_sem, pnw_lo, pnw_hi = bootstrap_chain_mean(
            [r["p_invalid_no_wall_diagnostic"] for r in rr], SEED_BASE + P + 1
        )
        actual = {
            "P": P,
            "n_chains": len(rr),
            "p_invalid": p,
            "p_invalid_chain_sem": p_sem,
            "p_invalid_ci95_lo": p_lo,
            "p_invalid_ci95_hi": p_hi,
            "p_invalid_no_wall_diagnostic": pnw,
            "p_invalid_no_wall_sem": pnw_sem,
            "p_invalid_no_wall_ci95_lo": pnw_lo,
            "p_invalid_no_wall_ci95_hi": pnw_hi,
            "zeta_median_chain_mean": float(np.mean([r["zeta_median"] for r in rr])),
            "p_zeta_gt_5_chain_mean": float(np.mean([r["p_zeta_gt_5"] for r in rr])),
            "rho_median_nm_chain_mean": float(np.mean([r["rho_median_nm"] for r in rr])),
            "rho_mean_nm_chain_mean": float(np.mean([r["rho_mean_nm"] for r in rr])),
        }
        ref = reference[P]
        checks = {
            "p_invalid": 5e-10,
            "p_invalid_ci95_lo": 5e-10,
            "p_invalid_ci95_hi": 5e-10,
            "p_invalid_no_wall_diagnostic": 5e-10,
            "zeta_median_chain_mean": 5e-5,
            "p_zeta_gt_5_chain_mean": 5e-10,
            "rho_median_nm_chain_mean": 5e-5,
            "rho_mean_nm_chain_mean": 5e-5,
        }
        for key, tol in checks.items():
            delta = float(actual[key]) - float(ref[key])
            actual[f"reference_{key}"] = float(ref[key])
            actual[f"delta_{key}"] = delta
            if abs(delta) > tol:
                failures.append(f"P={P} {key}: delta={delta:.12g} > {tol:g}")
        aggregate.append(actual)

    write_csv(HERE / "reproduction_aggregate.csv", aggregate)
    p256 = next(r for r in aggregate if r["P"] == 256)
    report = {
        "pass": not failures,
        "failures": failures,
        "wall_removal_delta_p_invalid_P256": (
            p256["p_invalid_no_wall_diagnostic"] - p256["p_invalid"]
        ),
        "source_provenance": {
            "production_root": str(PRODUCTION_ROOT),
            "production_git_head": git_head(AUDIT_REPO_ROOT),
            "runner": str(PRODUCTION_RUNNER),
            "runner_sha256": sha256(PRODUCTION_RUNNER),
            "stability_module": str(STABILITY_MODULE),
            "stability_module_sha256": sha256(STABILITY_MODULE),
            "config": str(CONFIG),
            "config_sha256": sha256(CONFIG),
            "reference_aggregate": str(REFERENCE_AGGREGATE),
            "reference_aggregate_sha256": sha256(REFERENCE_AGGREGATE),
            "sampler_runtime_root": str(SAMPLER_RUNTIME_ROOT),
            "sampler_runtime_git_head": git_head(SAMPLER_RUNTIME_ROOT),
            "sampler_runtime_file": str(SAMPLER_RUNTIME_ROOT / "numerics/tmd_pimc/two_body_sampler_periodic_jit.py"),
            "sampler_runtime_file_sha256": sha256(SAMPLER_RUNTIME_ROOT / "numerics/tmd_pimc/two_body_sampler_periodic_jit.py"),
            "production_HEAD_sampler_sha256_incompatible": sha256(SAMPLER_RUNTIME_ROOT / "numerics/tmd_pimc/two_body_sampler_periodic_jit.py"),
        },
        "sampler": {
            "P_values": list(P_VALUES), "n_chains_per_P": N_CHAINS,
            "seed_formula": "31000 + 1000*P + chain_index (0..9)",
            "n_steps": N_STEPS, "burn_in": BURN_IN,
            "sample_every": SAMPLE_EVERY, "n_stored_frames": 2250,
            "local_step_nm": LOCAL_STEP_NM,
            "global_step_nm": GLOBAL_STEP_NM,
            "global_move_probability": GLOBAL_MOVE_PROBABILITY,
        },
        "important_limitation": (
            "The production runner explicitly did not save raw path arrays. "
            "Audit paths were deterministically reconstructed from the exact "
            "production code, configuration, and seeds; they are not original "
            "archived path files."
        ),
    }
    (HERE / "reproduction_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError("Sec. VIII reproduction gate failed:\n" + "\n".join(failures))
    return report


def mass_weight(H: np.ndarray, invsqrt_mass: np.ndarray) -> np.ndarray:
    return H * invsqrt_mass[None, :, None] * invsqrt_mass[None, None, :]


def interaction_block(Hrel: np.ndarray) -> np.ndarray:
    out = np.zeros(Hrel.shape[:-2] + (4, 4), dtype=float)
    out[..., :2, :2] = Hrel
    out[..., 2:, 2:] = Hrel
    out[..., :2, 2:] = -Hrel
    out[..., 2:, :2] = -Hrel
    return out


def moire_block(He: np.ndarray, Hh: np.ndarray) -> np.ndarray:
    out = np.zeros(He.shape[:-2] + (4, 4), dtype=float)
    out[..., :2, :2] = He
    out[..., 2:, 2:] = Hh
    return out


def wall_hessian_finite_difference(
    d: np.ndarray, radius_nm: float, height_eV: float, power: int,
    rel: float = 3e-3, floor: float = 1e-4,
) -> np.ndarray:
    """Wall-only Hessian using exactly the production radial FD stencil."""
    d = np.asarray(d, dtype=float)
    rho = np.linalg.norm(d, axis=-1)
    safe = np.maximum(rho, 1e-12)
    h = np.maximum(floor, np.abs(rho) * rel)
    wall = lambda x: height_eV * (np.abs(x) / radius_nm) ** power
    vp, vm, v0 = wall(rho + h), wall(rho - h), wall(rho)
    v1 = (vp - vm) / (2 * h)
    v2 = (vp - 2 * v0 + vm) / h**2
    transverse = np.where(rho > 1e-9, v1 / safe, v2)
    nhat = d / safe[..., None]
    nn = np.einsum("...i,...j->...ij", nhat, nhat)
    eye = np.broadcast_to(np.eye(2), nn.shape)
    return v2[..., None, None] * nn + transverse[..., None, None] * (eye - nn)


def zeta_from_lambda(lam_min: np.ndarray, temperature_K: float, stability) -> np.ndarray:
    beta = 1.0 / (stability.KB_EV_PER_K * temperature_K)
    return (beta / np.pi) * np.sqrt(
        np.maximum(0.0, -lam_min) * stability.HBAR2_OVER_M0
    )


def add_distribution(row: dict[str, Any], prefix: str, values: np.ndarray, mask: np.ndarray) -> None:
    x = np.asarray(values)[mask]
    row[f"{prefix}_n"] = int(x.size)
    row[f"{prefix}_mean"] = float(np.mean(x)) if x.size else math.nan
    for q, label in ((5, "p05"), (25, "p25"), (50, "median"), (75, "p75"), (95, "p95")):
        row[f"{prefix}_{label}"] = float(np.percentile(x, q)) if x.size else math.nan


def initial_positive_iat(x: np.ndarray) -> tuple[float, float]:
    """FFT autocorrelation with a conservative initial-positive cutoff."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4 or np.all(x == 0):
        return 1.0, float(n)
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conjugate(f), n=m)[:n]
    acov /= np.arange(n, 0, -1)
    rho = acov / acov[0]
    positive = np.flatnonzero(rho[1:] <= 0.0)
    stop = int(positive[0] + 1) if positive.size else n
    tau = max(1.0, 1.0 + 2.0 * float(np.sum(rho[1:stop])))
    return tau, float(n / tau)


def analyze_chain(
    P: int, seed: int, field, field_no_wall, stability, temperature_K: float
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    with np.load(chain_path(P, seed)) as saved:
        se = np.asarray(saved["samples_e"], dtype=float)
        sh = np.asarray(saved["samples_h"], dtype=float)
    nframes = se.shape[0]
    re = se.reshape(-1, 2)
    rh = sh.reshape(-1, 2)
    n = re.shape[0]

    me = float(field.mass_e_m0)
    mh = float(field.mass_h_m0)
    mu = me * mh / (me + mh)
    invmass = np.array([1 / math.sqrt(me), 1 / math.sqrt(me),
                        1 / math.sqrt(mh), 1 / math.sqrt(mh)])
    qcomx = np.array([math.sqrt(me), 0.0, math.sqrt(mh), 0.0]) / math.sqrt(me + mh)
    qcomy = np.array([0.0, math.sqrt(me), 0.0, math.sqrt(mh)]) / math.sqrt(me + mh)

    names = (
        "lambda_min", "c_moire", "c_BLK", "c_wall",
        "O_rho", "O_theta", "O_COM", "overlap_closure", "zeta", "rho",
    )
    arrays = {name: np.empty(n, dtype=np.float32) for name in names}
    scenarios = {name: np.empty(n, dtype=np.float32) for name in (
        "full", "full-no-wall", "full-no-BLK", "BLK-only", "moire-only", "wall-only"
    )}
    closure_max = 0.0
    closure_sumsq = 0.0
    eig_match_max = 0.0
    ortho_max = 0.0
    overlap_closure_max = 0.0

    for lo in range(0, n, CHUNK_SIZE):
        hi = min(n, lo + CHUNK_SIZE)
        a = re[lo:hi]
        b = rh[lo:hi]
        d = a - b
        rho = np.linalg.norm(d, axis=1)
        safe = np.maximum(rho, 1e-15)
        rhat = d / safe[:, None]
        that = np.column_stack((-rhat[:, 1], rhat[:, 0]))

        Hmoire = moire_block(field.landscape_e.hessian(a), field.landscape_h.hessian(b))
        Hblkrel = field_no_wall.interaction.hessian(d)
        Hwallrel = wall_hessian_finite_difference(
            d,
            radius_nm=float(field.interaction.wall_radius_nm),
            height_eV=float(field.interaction.wall_height_eV),
            power=int(field.interaction.wall_power),
        )
        Wm = mass_weight(Hmoire, invmass)
        Wb = mass_weight(interaction_block(Hblkrel), invmass)
        Ww = mass_weight(interaction_block(Hwallrel), invmass)
        Wfull = Wm + Wb + Ww

        lam, vec = np.linalg.eigh(Wfull)
        lmin = lam[:, 0]
        qmin = vec[:, :, 0]
        cm = np.einsum("ni,nij,nj->n", qmin, Wm, qmin)
        cb = np.einsum("ni,nij,nj->n", qmin, Wb, qmin)
        cw = np.einsum("ni,nij,nj->n", qmin, Ww, qmin)
        closure = lmin - cm - cb - cw
        closure_max = max(closure_max, float(np.max(np.abs(closure))))
        closure_sumsq += float(np.sum(closure * closure))
        # A small subset per chunk verifies exact equivalence to the original
        # wall-inclusive production call without repeating expensive BLK
        # special functions for every bead.
        ncheck = min(32, hi - lo)
        direct = field.mass_weighted_eigenvalues(a[:ncheck], b[:ncheck])[:, 0]
        eig_match_max = max(eig_match_max, float(np.max(np.abs(lmin[:ncheck] - direct))))

        qrho = math.sqrt(mu) * np.column_stack((
            rhat[:, 0] / math.sqrt(me), rhat[:, 1] / math.sqrt(me),
            -rhat[:, 0] / math.sqrt(mh), -rhat[:, 1] / math.sqrt(mh),
        ))
        qtheta = math.sqrt(mu) * np.column_stack((
            that[:, 0] / math.sqrt(me), that[:, 1] / math.sqrt(me),
            -that[:, 0] / math.sqrt(mh), -that[:, 1] / math.sqrt(mh),
        ))
        # Gram test for the four analytic basis vectors.
        basis = np.stack((qrho, qtheta,
                          np.broadcast_to(qcomx, qrho.shape),
                          np.broadcast_to(qcomy, qrho.shape)), axis=2)
        gram = np.einsum("nik,nil->nkl", basis, basis)
        ortho_max = max(ortho_max, float(np.max(np.abs(gram - np.eye(4)))))
        orho = np.einsum("ni,ni->n", qmin, qrho) ** 2
        otheta = np.einsum("ni,ni->n", qmin, qtheta) ** 2
        ocom = (qmin @ qcomx) ** 2 + (qmin @ qcomy) ** 2
        oclosure = orho + otheta + ocom
        overlap_closure_max = max(overlap_closure_max, float(np.max(np.abs(oclosure - 1))))

        arrays["lambda_min"][lo:hi] = lmin
        arrays["c_moire"][lo:hi] = cm
        arrays["c_BLK"][lo:hi] = cb
        arrays["c_wall"][lo:hi] = cw
        arrays["O_rho"][lo:hi] = orho
        arrays["O_theta"][lo:hi] = otheta
        arrays["O_COM"][lo:hi] = ocom
        arrays["overlap_closure"][lo:hi] = oclosure
        arrays["zeta"][lo:hi] = zeta_from_lambda(lmin, temperature_K, stability)
        arrays["rho"][lo:hi] = rho

        scenarios["full"][lo:hi] = zeta_from_lambda(lmin, temperature_K, stability)
        scenarios["full-no-wall"][lo:hi] = zeta_from_lambda(
            np.linalg.eigvalsh(Wm + Wb)[:, 0], temperature_K, stability)
        scenarios["full-no-BLK"][lo:hi] = zeta_from_lambda(
            np.linalg.eigvalsh(Wm + Ww)[:, 0], temperature_K, stability)
        # For a central interaction the two nonzero mass-weighted eigenvalues
        # are (1/me+1/mh) times the two eigenvalues of H_rel; the other two are
        # exact translations.  Moire-only is block diagonal in carrier space.
        mass_factor = 1.0 / me + 1.0 / mh
        lblk = np.minimum(0.0, mass_factor * np.linalg.eigvalsh(Hblkrel)[:, 0])
        lwall = np.minimum(0.0, mass_factor * np.linalg.eigvalsh(Hwallrel)[:, 0])
        lm_e = np.linalg.eigvalsh(Hmoire[:, :2, :2] / me)[:, 0]
        lm_h = np.linalg.eigvalsh(Hmoire[:, 2:, 2:] / mh)[:, 0]
        lmoire = np.minimum(lm_e, lm_h)
        scenarios["BLK-only"][lo:hi] = zeta_from_lambda(lblk, temperature_K, stability)
        scenarios["moire-only"][lo:hi] = zeta_from_lambda(lmoire, temperature_K, stability)
        scenarios["wall-only"][lo:hi] = zeta_from_lambda(lwall, temperature_K, stability)

    z = arrays["zeta"]
    all_mask = np.ones(n, dtype=bool)
    invalid = z >= 1.0
    deep = z > 5.0
    row: dict[str, Any] = {
        "row_type": "chain", "P": P, "seed": seed,
        "n_frames": nframes, "n_paired_beads": n,
        "p_invalid": float(invalid.mean()),
        "p_zeta_gt_5": float(deep.mean()),
        "zeta_median": float(np.median(z)),
        "zeta_mean": float(np.mean(z)),
        "rho_mean_nm": float(np.mean(arrays["rho"])),
        "rho_median_nm": float(np.median(arrays["rho"])),
        "rayleigh_closure_max_abs": closure_max,
        "rayleigh_closure_rms": math.sqrt(closure_sumsq / n),
        "direct_full_eigenvalue_match_max_abs": eig_match_max,
        "mode_basis_orthonormality_max_abs": ortho_max,
        "mode_overlap_closure_max_abs": overlap_closure_max,
    }
    for condition, mask in (("all", all_mask), ("zeta_ge_1", invalid), ("zeta_gt_5", deep)):
        for metric in ("lambda_min", "c_moire", "c_BLK", "c_wall",
                       "O_rho", "O_theta", "O_COM"):
            add_distribution(row, f"{metric}_{condition}", arrays[metric], mask)

    counter_rows: list[dict[str, Any]] = []
    for name, zz in scenarios.items():
        counter_rows.append({
            "row_type": "chain", "P": P, "seed": seed, "scenario": name,
            "n_paired_beads": n,
            "p_invalid": float(np.mean(zz >= 1.0)),
            "p_zeta_gt_5": float(np.mean(zz > 5.0)),
            "zeta_median": float(np.median(zz)),
            "zeta_mean": float(np.mean(zz)),
        })

    frame_fraction = invalid.reshape(nframes, P).mean(axis=1)
    tau, ess = initial_positive_iat(frame_fraction)
    half = nframes // 2
    block_size = 50
    nblock = nframes // block_size
    block_means = frame_fraction[:nblock * block_size].reshape(nblock, block_size).mean(axis=1)
    diagnostics = {
        "P": P, "seed": seed, "n_frames": nframes,
        "threshold_fraction_mean": float(frame_fraction.mean()),
        "first_half_mean": float(frame_fraction[:half].mean()),
        "second_half_mean": float(frame_fraction[half:].mean()),
        "second_minus_first": float(frame_fraction[half:].mean() - frame_fraction[:half].mean()),
        "iat_frames_initial_positive": tau,
        "ess_frames_initial_positive": ess,
        "block_size_frames": block_size,
        "n_blocks": nblock,
        "block_mean_sd": float(block_means.std(ddof=1)),
        "block_mean_min": float(block_means.min()),
        "block_mean_max": float(block_means.max()),
    }
    return row, counter_rows, {**arrays, **{f"scenario::{k}": v for k, v in scenarios.items()}}, diagnostics


def aggregate_attribution(
    mode_rows: list[dict[str, Any]], counter_rows: list[dict[str, Any]],
    collected_by_P: dict[int, list[dict[str, np.ndarray]]], diagnostics: list[dict[str, Any]],
) -> None:
    p256_rows = [r for r in mode_rows if r["P"] == 256]
    aggregate: dict[str, Any] = {
        "row_type": "aggregate_mean_of_chain_statistics", "P": 256,
        "seed": "ALL", "n_frames": sum(r["n_frames"] for r in p256_rows),
        "n_paired_beads": sum(r["n_paired_beads"] for r in p256_rows),
    }
    statistic_keys = [k for k, v in p256_rows[0].items()
                      if k not in aggregate and k not in ("row_type", "seed", "P")
                      and isinstance(v, (int, float, np.floating))]
    for i, key in enumerate(statistic_keys):
        vals = [float(r[key]) for r in p256_rows]
        mean, sem, lo, hi = bootstrap_chain_mean(vals, 840000 + i)
        aggregate[key] = mean
        aggregate[f"{key}_chain_boot_ci95_lo"] = lo
        aggregate[f"{key}_chain_boot_ci95_hi"] = hi
    write_csv(HERE / "mode_attribution_P256.csv", p256_rows + [aggregate])

    # Pooled distributions are point summaries; uncertainty columns are
    # whole-chain bootstrap intervals for the mean of per-chain statistics.
    dist_rows: list[dict[str, Any]] = []
    data256 = collected_by_P[256]
    zpool = np.concatenate([x["zeta"] for x in data256])
    for condition, mask in (
        ("all", np.ones(zpool.size, dtype=bool)),
        ("zeta_ge_1", zpool >= 1.0),
        ("zeta_gt_5", zpool > 5.0),
    ):
        offset = 0
        chain_masks: list[np.ndarray] = []
        for chain in data256:
            zz = chain["zeta"]
            if condition == "all": chain_masks.append(np.ones(zz.size, dtype=bool))
            elif condition == "zeta_ge_1": chain_masks.append(zz >= 1.0)
            else: chain_masks.append(zz > 5.0)
            offset += zz.size
        for metric in ("lambda_min", "c_moire", "c_BLK", "c_wall", "O_rho", "O_theta", "O_COM"):
            pooled = np.concatenate([chain[metric] for chain in data256])[mask]
            row = {"P": 256, "condition": condition, "metric": metric, "n": pooled.size,
                   "pooled_mean": float(pooled.mean())}
            for q, label in ((5, "p05"), (25, "p25"), (50, "median"), (75, "p75"), (95, "p95")):
                row[f"pooled_{label}"] = float(np.percentile(pooled, q))
            chain_means = [float(np.mean(chain[metric][cmask])) for chain, cmask in zip(data256, chain_masks)]
            chain_medians = [float(np.median(chain[metric][cmask])) for chain, cmask in zip(data256, chain_masks)]
            _, _, row["chain_boot_mean_ci95_lo"], row["chain_boot_mean_ci95_hi"] = bootstrap_chain_mean(
                chain_means, 850000 + len(dist_rows)
            )
            _, _, row["chain_boot_mean_of_medians_ci95_lo"], row["chain_boot_mean_of_medians_ci95_hi"] = bootstrap_chain_mean(
                chain_medians, 860000 + len(dist_rows)
            )
            dist_rows.append(row)
    write_csv(HERE / "mode_attribution_distributions_P256.csv", dist_rows)

    counter_out: list[dict[str, Any]] = []
    scenarios = ("full", "full-no-wall", "full-no-BLK", "BLK-only", "moire-only", "wall-only")
    for P in P_VALUES:
        for iscen, scenario in enumerate(scenarios):
            rr = [r for r in counter_rows if r["P"] == P and r["scenario"] == scenario]
            pooled = np.concatenate([x[f"scenario::{scenario}"] for x in collected_by_P[P]])
            p, psem, plo, phi = bootstrap_chain_mean([r["p_invalid"] for r in rr], 870000 + P + iscen)
            w, wsem, wlo, whi = bootstrap_chain_mean([r["p_zeta_gt_5"] for r in rr], 880000 + P + iscen)
            cm, cmsem, cmlo, cmhi = bootstrap_chain_mean([r["zeta_median"] for r in rr], 890000 + P + iscen)
            counter_out.append({
                "P": P, "scenario": scenario, "n_chains": len(rr),
                "ensemble_note": "same physical full-Hamiltonian ensemble; Hessian-only reclassification",
                "p_invalid": p, "p_invalid_chain_sem": psem,
                "p_invalid_chain_boot_ci95_lo": plo, "p_invalid_chain_boot_ci95_hi": phi,
                "p_zeta_gt_5": w, "p_zeta_gt_5_chain_sem": wsem,
                "p_zeta_gt_5_chain_boot_ci95_lo": wlo, "p_zeta_gt_5_chain_boot_ci95_hi": whi,
                "zeta_pooled_median": float(np.median(pooled)),
                "zeta_mean_of_chain_medians": cm,
                "zeta_mean_of_chain_medians_sem": cmsem,
                "zeta_mean_of_chain_medians_boot_ci95_lo": cmlo,
                "zeta_mean_of_chain_medians_boot_ci95_hi": cmhi,
            })
    write_csv(HERE / "counterfactual_stability.csv", counter_out)

    # Leave-one-chain-out influence on p_inv, supplementing the time-series ESS.
    for P in P_VALUES:
        rr = [d for d in diagnostics if d["P"] == P]
        vals = np.array([d["threshold_fraction_mean"] for d in rr])
        full = float(vals.mean())
        for i, row in enumerate(rr):
            loo = float(np.delete(vals, i).mean())
            row["chain_mean_spread_sd"] = float(vals.std(ddof=1))
            row["leave_one_out_aggregate_p_invalid"] = loo
            row["leave_one_out_minus_full"] = loo - full
            row["max_abs_leave_one_out_shift_for_P"] = float(np.max(np.abs(
                np.array([np.delete(vals, j).mean() for j in range(vals.size)]) - full
            )))
    write_csv(HERE / "chain_diagnostics.csv", diagnostics)

    numerical_checks = {
        "rayleigh_closure_max_abs": max(r["rayleigh_closure_max_abs"] for r in mode_rows),
        "rayleigh_closure_rms_max_across_chains": max(r["rayleigh_closure_rms"] for r in mode_rows),
        "direct_full_eigenvalue_match_max_abs": max(r["direct_full_eigenvalue_match_max_abs"] for r in mode_rows),
        "mode_basis_orthonormality_max_abs": max(r["mode_basis_orthonormality_max_abs"] for r in mode_rows),
        "mode_overlap_closure_max_abs": max(r["mode_overlap_closure_max_abs"] for r in mode_rows),
    }
    (HERE / "attribution_numerical_checks.json").write_text(json.dumps(numerical_checks, indent=2) + "\n")


def analyze_all() -> None:
    runner, stability, cfg = production_context()
    T = float(cfg["temperature_K"])
    field = runner._build_stability_field(stability, cfg, include_wall=True)
    field_no_wall = runner._build_stability_field(stability, cfg, include_wall=False)
    mode_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    collected: dict[int, list[dict[str, np.ndarray]]] = {P: [] for P in P_VALUES}
    for P in P_VALUES:
        for ichain in range(N_CHAINS):
            seed = SEED_BASE + 1000 * P + ichain
            print(f"attributing P={P} seed={seed}")
            row, cr, arrays, diag = analyze_chain(P, seed, field, field_no_wall, stability, T)
            mode_rows.append(row)
            counter_rows.extend(cr)
            diagnostics.append(diag)
            collected[P].append(arrays)
    aggregate_attribution(mode_rows, counter_rows, collected, diagnostics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("all", "reproduce", "analyze"), default="all")
    ap.add_argument("--reuse", action="store_true", help="reuse audit paths if present")
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba_cache"))
    if args.stage in ("all", "reproduce"):
        rows = reconstruct_chains(reuse=args.reuse)
        reproduce_gate(rows)
    if args.stage in ("all", "analyze"):
        gate = json.loads((HERE / "reproduction_gate.json").read_text())
        if not gate.get("pass"):
            raise RuntimeError("Attribution refused: reproduction gate is not PASS")
        analyze_all()


if __name__ == "__main__":
    main()
