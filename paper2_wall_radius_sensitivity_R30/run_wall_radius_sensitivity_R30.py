#!/usr/bin/env python3
"""Predeclared R_wall=30 nm sensitivity campaign for Paper 2, Sec. VIII.

Scientific purpose
------------------
Test whether the *equilibrium occupation* of the Gaussian-invalid region is
robust to a deliberately large change of the auxiliary radial confinement.
This is a new equilibrated Hamiltonian, not a Hessian-only reclassification.

The script deliberately reuses the archived, validated Sec. VIII production
builders and the validated periodic two-body staging sampler.  The only
Hamiltonian change is

    wall_radius_nm: 15.0 -> 30.0

with wall_height_eV=0.08 and wall_power=8 unchanged.  P=256, T=20 K, proposal
scales, staging settings, interaction table, moire fields, masses, and BLK
interaction are unchanged.

Four new independent 240k-step chains are used.  Two start from the compact
pair minimum; two start with the carriers in equivalent moire minima separated
by one Bravais vector of length 20 nm, i.e. beyond the isolated-BLK outer
zeta=1 crossing (~18.7 nm).  The primary analysis uses a 60k burn-in.  The raw
chains are stored from 15k onward so 15k/30k/60k burn-in sensitivity can be
reported without rerunning or selecting a burn-in after looking at the result.

Top-level uncertainty units are independent chains, never beads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
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
CONFIG = PRODUCTION_ROOT / "configs/two_body/landscape_scan_config_v2_P256_matched.json"
SAMPLER_FILE = REPO_ROOT / "numerics/tmd_pimc/two_body_sampler_periodic_jit.py"

RAW_DIR = HERE / "raw_chains"
OBS_DIR = HERE / "observable_cache"
RESULTS_DIR = HERE / "results"

# Frozen source-of-truth hashes from the archived Sec. VIII reproduction gate.
EXPECTED_HASHES = {
    "production_runner": "63de5e2ba07f577a00627f22b28686ba1c9f57b66c5a5deee01411d99db8697b",
    "stability_module": "f423834e37c640bf855e799e75080a34cc0fa2dd21a90dd0390390efb719d926",
    "physical_config": "9d7779ad7da3554fda7118b645547d7b251141286e738afc5f5254cfc2163719",
    "sampler": "34b33ae11fefe2c25ec9e6dd438fd9b6d7c18ad8c3c16510b71ce94faed82d2c",
}

# Predeclared production protocol.
P = 256
TEMPERATURE_K = 20.0
WALL_RADIUS_NM = 30.0
WALL_HEIGHT_EV = 0.08
WALL_POWER = 8
OUTER_BLK_CROSSING_NM = 18.7

SEEDS = (930300, 930301, 930302, 930303)
START_CLASSES = ("compact", "compact", "dissociated", "dissociated")
N_STEPS = 240_000
RAW_BURN_IN = 15_000
PRIMARY_BURN_IN = 60_000
BURN_INS = (15_000, 30_000, 60_000)
SAMPLE_EVERY = 20
LOCAL_STEP_NM = 0.15
GLOBAL_STEP_NM = 12.0
GLOBAL_MOVE_PROBABILITY = 0.20
CHUNK_SIZE = 50_000
N_BLOCKS = 8

# Canonical Rwall=15 nm reference, used only for transparent comparison.
BASELINE_RWALL_NM = 15.0
BASELINE_PINV_P256 = 0.9248524305555555
BASELINE_PINV_CI95 = (0.9240048611111111, 0.925468923611111)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean_sem_tci(values: Iterable[float]) -> tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=float)
    mean = float(np.mean(x))
    if x.size < 2:
        return mean, math.nan, math.nan, math.nan
    sem = float(np.std(x, ddof=1) / math.sqrt(x.size))
    half = float(stats.t.ppf(0.975, x.size - 1) * sem)
    return mean, sem, mean - half, mean + half


def expected_nstored(burn_in: int, n_steps: int = N_STEPS, sample_every: int = SAMPLE_EVERY) -> int:
    return 1 + (n_steps - burn_in - 1) // sample_every


def verify_source_truth() -> dict[str, str]:
    paths = {
        "production_runner": PRODUCTION_RUNNER,
        "stability_module": STABILITY_MODULE,
        "physical_config": CONFIG,
        "sampler": SAMPLER_FILE,
    }
    actual: dict[str, str] = {}
    failures: list[str] = []
    for key, path in paths.items():
        if not path.exists():
            failures.append(f"missing {key}: {path}")
            continue
        digest = sha256(path)
        actual[key] = digest
        if digest != EXPECTED_HASHES[key]:
            failures.append(
                f"{key} hash mismatch:\n"
                f"  expected {EXPECTED_HASHES[key]}\n"
                f"  actual   {digest}\n"
                f"  file     {path}"
            )
    if failures:
        raise RuntimeError(
            "Archived Sec. VIII source-of-truth verification failed.\n"
            "Do not run production until this is audited:\n\n" + "\n".join(failures)
        )
    return actual


def production_context():
    # Force the archived/released runtime packaged with this repository.
    sys.path.insert(0, str(REPO_ROOT / "numerics"))
    runner = load_module("paper2_R30_production_runner", PRODUCTION_RUNNER)
    stability = load_module("paper2_R30_stability", STABILITY_MODULE)
    cfg = json.loads(CONFIG.read_text())
    return runner, stability, cfg


def validate_baseline_config(cfg: dict[str, Any]) -> None:
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
    failures: list[str] = []
    for key, expected in required.items():
        actual = cfg.get(key)
        if actual is None or not np.isclose(float(actual), float(expected), rtol=0, atol=1e-14):
            failures.append(f"{key}: {actual!r} != {expected!r}")
    if failures:
        raise RuntimeError("Baseline physical configuration mismatch:\n" + "\n".join(failures))
    if len(set(SEEDS)) != len(SEEDS):
        raise RuntimeError("Sensitivity seeds are not unique")
    old_production_seeds = {31_000 + 1000 * p + i for p in (64, 128, 256) for i in range(10)}
    old_long_seeds = {912560, 912561, 912562, 912563}
    overlap = set(SEEDS) & (old_production_seeds | old_long_seeds)
    if overlap:
        raise RuntimeError(f"Sensitivity seeds overlap an archived campaign: {sorted(overlap)}")


def build_setup():
    verify_source_truth()
    runner, stability, cfg = production_context()
    validate_baseline_config(cfg)

    # The sole Hamiltonian change in this campaign.
    cfg = dict(cfg)
    cfg["wall_radius_nm"] = WALL_RADIUS_NM
    cfg["wall_height_eV"] = WALL_HEIGHT_EV
    cfg["wall_power"] = WALL_POWER

    from tmd_pimc.potentials import CompositePotential
    from tmd_pimc.two_body_action import TwoBodyRingPolymerAction

    interaction = runner._build_interaction(cfg)
    field = runner._build_stability_field(stability, cfg, include_wall=True)
    field_no_wall = runner._build_stability_field(stability, cfg, include_wall=False)
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

    compact = np.asarray(cfg.get("tb03_start_center_nm", [0.5 * period, 0.0]), dtype=float)
    if compact.shape != (2,):
        raise RuntimeError(f"Bad tb03_start_center_nm: {compact}")

    # (0, period) is an exact Bravais vector of the analytic moire field:
    # G1·a2=0, G2·a2=+2pi, G3·a2=-2pi.  Thus both particles start in
    # equivalent one-body minima while their relative separation is exactly L=20 nm.
    bravais_a2 = np.array([0.0, period], dtype=float)
    starts = {
        "compact": (tuple(compact), tuple(compact)),
        "dissociated": (tuple(compact), tuple(compact + bravais_a2)),
    }
    return runner, cfg, action, field, field_no_wall, settings, starts


def raw_path(seed: int) -> Path:
    return RAW_DIR / f"R30_P256_seed{seed}_240k.npz"


def obs_path(seed: int) -> Path:
    return OBS_DIR / f"R30_P256_seed{seed}_observables.npz"


def print_protocol(cfg: dict[str, Any], starts: dict[str, tuple[tuple[float, float], tuple[float, float]]]) -> None:
    beta_wall_volume_ratio = (WALL_RADIUS_NM / BASELINE_RWALL_NM) ** 2
    print("=" * 80)
    print("Paper 2: R_wall sensitivity campaign")
    print("=" * 80)
    print(f"T, P                    : {TEMPERATURE_K:g} K, {P}")
    print(f"wall                     : 0.08 eV * (rho/{WALL_RADIUS_NM:g} nm)^8")
    print(f"baseline wall             : 0.08 eV * (rho/{BASELINE_RWALL_NM:g} nm)^8")
    print(f"asymptotic wall-volume ratio: {beta_wall_volume_ratio:g}x")
    print(f"steps/raw burn/sample     : {N_STEPS}/{RAW_BURN_IN}/{SAMPLE_EVERY}")
    print(f"primary burn-in           : {PRIMARY_BURN_IN}")
    print(f"burn-in audit             : {BURN_INS}")
    print(f"local/global/prob         : {LOCAL_STEP_NM:g} nm / {GLOBAL_STEP_NM:g} nm / {GLOBAL_MOVE_PROBABILITY:g}")
    print(f"seeds                     : {SEEDS}")
    for seed, cls in zip(SEEDS, START_CLASSES):
        ce, ch = starts[cls]
        print(f"  seed {seed}: {cls:11s} center_e={ce}, center_h={ch}, rho0={np.linalg.norm(np.asarray(ce)-np.asarray(ch)):.6g} nm")
    print(f"interaction table r_max   : {float(cfg.get('r_max_nm',80.0)):g} nm")
    print("Primary qualitative gate  : lower 95% chain-level t-CI for p_inv > 0.5")
    print("Mixing diagnostic          : compact-start and dissociated-start chains should converge")
    print()


def dry_run() -> None:
    runner, cfg, action, field, field_no_wall, settings, starts = build_setup()
    from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT
    print_protocol(cfg, starts)
    sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(action=action, rng_seed=SEEDS[0], **settings)
    runner._validate_model_match(sampler, action, field, cfg)

    rho = np.array([0.0, 5.0, 15.0, OUTER_BLK_CROSSING_NM, 20.0, 30.0, 40.0])
    wall = WALL_HEIGHT_EV * (rho / WALL_RADIUS_NM) ** WALL_POWER
    kbt = 8.617333e-5 * TEMPERATURE_K
    print("Selected R=30 wall energies:")
    for r, v in zip(rho, wall):
        print(f"  rho={r:6.2f} nm  Vwall={v: .8e} eV  Vwall/kBT={v/kbt: .5g}")
    print("\nDRY RUN PASSED. No Markov chain was launched.")


def sample_chains(reuse: bool, smoke: bool) -> None:
    runner, cfg, action, field, field_no_wall, settings, starts = build_setup()
    from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT
    print_protocol(cfg, starts)

    seeds = SEEDS if not smoke else (SEEDS[0], SEEDS[2])
    classes = START_CLASSES if not smoke else ("compact", "dissociated")
    n_steps = N_STEPS if not smoke else 2_000
    burn = RAW_BURN_IN if not smoke else 500
    every = SAMPLE_EVERY if not smoke else 50
    raw_dir = RAW_DIR if not smoke else HERE / "smoke_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index, (seed, start_class) in enumerate(zip(seeds, classes)):
        output = (raw_dir / f"R30_P256_seed{seed}_{'240k' if not smoke else 'smoke'}.npz")
        if output.exists():
            if reuse:
                print(f"REUSE raw chain seed={seed}: {output}", flush=True)
                continue
            raise FileExistsError(f"Refusing to overwrite {output}; use --reuse or delete it explicitly")

        sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(action=action, rng_seed=seed, **settings)
        if index == 0:
            runner._validate_model_match(sampler, action, field, cfg)
        center_e, center_h = starts[start_class]
        print(f"RUN seed={seed}, class={start_class}, P={P}, steps={n_steps}", flush=True)
        t0 = time.time()
        result = sampler.run(
            n_steps=n_steps,
            burn_in=burn,
            sample_every=every,
            center_e=center_e,
            center_h=center_h,
        )
        elapsed = time.time() - t0
        se = np.asarray(result["samples_e"], dtype=np.float64)
        sh = np.asarray(result["samples_h"], dtype=np.float64)
        expected = expected_nstored(burn, n_steps=n_steps, sample_every=every)
        if se.shape != (expected, P, 2):
            raise RuntimeError(f"Unexpected electron path shape for seed {seed}: {se.shape}, expected {(expected,P,2)}")
        if sh.shape != se.shape:
            raise RuntimeError(f"Mismatched path shapes for seed {seed}: {se.shape} vs {sh.shape}")
        np.savez_compressed(
            output,
            samples_e=se,
            samples_h=sh,
            seed=np.int64(seed),
            start_class=np.array(start_class),
            center_e_nm=np.asarray(center_e, dtype=np.float64),
            center_h_nm=np.asarray(center_h, dtype=np.float64),
            P=np.int64(P),
            temperature_K=np.float64(TEMPERATURE_K),
            wall_radius_nm=np.float64(WALL_RADIUS_NM),
            wall_height_eV=np.float64(WALL_HEIGHT_EV),
            wall_power=np.int64(WALL_POWER),
            n_steps=np.int64(n_steps),
            burn_in=np.int64(burn),
            sample_every=np.int64(every),
            elapsed_s=np.float64(elapsed),
            acceptance_local_e=np.float64(result["acceptance_local_e"]),
            acceptance_local_h=np.float64(result["acceptance_local_h"]),
            acceptance_staging=np.float64(result["acceptance_staging"]),
            acceptance_global_joint=np.float64(result["acceptance_global_joint"]),
        )
        print(
            f"SAVED {output}\n"
            f"  frames={se.shape[0]} staging_acc={result['acceptance_staging']:.6f} "
            f"global_acc={result['acceptance_global_joint']:.6f} elapsed={elapsed:.1f}s\n"
            f"  sha256={sha256(output)}",
            flush=True,
        )

    if smoke:
        print("SMOKE sampling completed. Production outputs were not touched.")


def compute_observables(field, field_no_wall, seed: int, reuse: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = obs_path(seed)
    if reuse and cache.exists():
        with np.load(cache) as saved:
            zeta = np.asarray(saved["zeta_R30"], dtype=np.float32)
            zeta_nw = np.asarray(saved["zeta_no_wall"], dtype=np.float32)
            rho = np.asarray(saved["rho_nm"], dtype=np.float32)
        return zeta, zeta_nw, rho

    rp = raw_path(seed)
    if not rp.exists():
        raise FileNotFoundError(f"Missing raw production chain: {rp}")
    with np.load(rp) as saved:
        se = np.asarray(saved["samples_e"], dtype=np.float64)
        sh = np.asarray(saved["samples_h"], dtype=np.float64)
    if se.shape != sh.shape:
        raise RuntimeError(f"Paired path mismatch for seed {seed}")
    re = se.reshape(-1, 2)
    rh = sh.reshape(-1, 2)
    n = re.shape[0]
    z = np.empty(n, dtype=np.float32)
    znw = np.empty(n, dtype=np.float32)
    rr = np.empty(n, dtype=np.float32)
    print(f"OBSERVABLES seed={seed}, paired same-slice beads={n}", flush=True)
    for lo in range(0, n, CHUNK_SIZE):
        hi = min(n, lo + CHUNK_SIZE)
        a = re[lo:hi]
        b = rh[lo:hi]
        z[lo:hi] = field.zeta(a, b, TEMPERATURE_K)
        znw[lo:hi] = field_no_wall.zeta(a, b, TEMPERATURE_K)
        rr[lo:hi] = np.linalg.norm(a - b, axis=1)
    z = z.reshape(se.shape[:2])
    znw = znw.reshape(se.shape[:2])
    rr = rr.reshape(se.shape[:2])
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        zeta_R30=z,
        zeta_no_wall=znw,
        rho_nm=rr,
        definition=np.array("Rwall=30 equilibrium samples; same-slice paired e/h beads"),
    )
    return z, znw, rr


def metrics(z: np.ndarray, znw: np.ndarray, rho: np.ndarray) -> dict[str, float]:
    return {
        "p_inv": float(np.mean(z >= 1.0)),
        "p_inv_no_wall_hessian": float(np.mean(znw >= 1.0)),
        "zeta50": float(np.median(z)),
        "zeta95": float(np.percentile(z, 95.0)),
        "zeta_max": float(np.max(z)),
        "p_zeta_ge_5": float(np.mean(z >= 5.0)),
        "rho_median_nm": float(np.median(rho)),
        "rho_mean_nm": float(np.mean(rho)),
        "rho_p95_nm": float(np.percentile(rho, 95.0)),
        "rho_p99_nm": float(np.percentile(rho, 99.0)),
        "rho_max_nm": float(np.max(rho)),
        "p_rho_gt_15nm": float(np.mean(rho > 15.0)),
        "p_rho_gt_18p7nm": float(np.mean(rho > OUTER_BLK_CROSSING_NM)),
        "p_rho_gt_20nm": float(np.mean(rho > 20.0)),
        "p_rho_gt_30nm": float(np.mean(rho > 30.0)),
    }


def analyze(reuse: bool) -> None:
    runner, cfg, action, field, field_no_wall, settings, starts = build_setup()
    print_protocol(cfg, starts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_chain: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    burn_rows: list[dict[str, Any]] = []

    for seed, start_class in zip(SEEDS, START_CLASSES):
        z_all, znw_all, rho_all = compute_observables(field, field_no_wall, seed, reuse)
        expected_raw = expected_nstored(RAW_BURN_IN)
        if z_all.shape != (expected_raw, P) or znw_all.shape != z_all.shape or rho_all.shape != z_all.shape:
            raise RuntimeError(f"Bad observable shape for seed {seed}: {z_all.shape}, {znw_all.shape}, {rho_all.shape}")

        for burn in BURN_INS:
            offset = (burn - RAW_BURN_IN) // SAMPLE_EVERY
            z = z_all[offset:]
            znw = znw_all[offset:]
            rho = rho_all[offset:]
            mm = metrics(z, znw, rho)
            burn_rows.append({
                "seed": seed,
                "start_class": start_class,
                "burn_in": burn,
                "n_frames": int(z.shape[0]),
                "n_paired_beads": int(z.size),
                **mm,
            })

        offset = (PRIMARY_BURN_IN - RAW_BURN_IN) // SAMPLE_EVERY
        z = z_all[offset:]
        znw = znw_all[offset:]
        rho = rho_all[offset:]
        if z.shape[0] != expected_nstored(PRIMARY_BURN_IN):
            raise RuntimeError(f"Unexpected retained frames for seed {seed}: {z.shape[0]}")
        mm = metrics(z, znw, rho)

        halves = np.array_split(np.arange(z.shape[0]), 2)
        mhalf = [metrics(z[idx], znw[idx], rho[idx]) for idx in halves]
        row = {
            "seed": seed,
            "start_class": start_class,
            "P": P,
            "temperature_K": TEMPERATURE_K,
            "wall_radius_nm": WALL_RADIUS_NM,
            "n_steps": N_STEPS,
            "primary_burn_in": PRIMARY_BURN_IN,
            "sample_every": SAMPLE_EVERY,
            "n_frames": int(z.shape[0]),
            "n_paired_beads": int(z.size),
            **mm,
            "p_inv_first_half": mhalf[0]["p_inv"],
            "p_inv_second_half": mhalf[1]["p_inv"],
            "delta_p_inv_second_minus_first": mhalf[1]["p_inv"] - mhalf[0]["p_inv"],
            "rho_mean_first_half_nm": mhalf[0]["rho_mean_nm"],
            "rho_mean_second_half_nm": mhalf[1]["rho_mean_nm"],
            "delta_rho_mean_second_minus_first_nm": mhalf[1]["rho_mean_nm"] - mhalf[0]["rho_mean_nm"],
        }
        per_chain.append(row)

        blocks = np.array_split(np.arange(z.shape[0]), N_BLOCKS)
        for ib, idx in enumerate(blocks, 1):
            bm = metrics(z[idx], znw[idx], rho[idx])
            block_rows.append({
                "seed": seed,
                "start_class": start_class,
                "block": ib,
                "n_blocks": N_BLOCKS,
                "n_frames": int(idx.size),
                "n_paired_beads": int(idx.size * P),
                **bm,
            })

        print(
            f"seed={seed} {start_class:11s} p_inv={row['p_inv']:.6f} "
            f"no-wall={row['p_inv_no_wall_hessian']:.6f} z50={row['zeta50']:.3f} "
            f"rho50={row['rho_median_nm']:.3f} rhoMean={row['rho_mean_nm']:.3f} "
            f"P(rho>18.7)={row['p_rho_gt_18p7nm']:.5f} half-delta={row['delta_p_inv_second_minus_first']:+.6f}",
            flush=True,
        )

    write_csv(RESULTS_DIR / "R30_per_chain_primary.csv", per_chain)
    write_csv(RESULTS_DIR / "R30_burnin_sensitivity.csv", burn_rows)
    write_csv(RESULTS_DIR / "R30_primary_blocks.csv", block_rows)

    p = np.asarray([r["p_inv"] for r in per_chain], dtype=float)
    p_nw = np.asarray([r["p_inv_no_wall_hessian"] for r in per_chain], dtype=float)
    z50 = np.asarray([r["zeta50"] for r in per_chain], dtype=float)
    w5 = np.asarray([r["p_zeta_ge_5"] for r in per_chain], dtype=float)
    rho50 = np.asarray([r["rho_median_nm"] for r in per_chain], dtype=float)
    rhom = np.asarray([r["rho_mean_nm"] for r in per_chain], dtype=float)
    pr18 = np.asarray([r["p_rho_gt_18p7nm"] for r in per_chain], dtype=float)

    p_mean, p_sem, p_lo, p_hi = mean_sem_tci(p)
    pnw_mean, pnw_sem, pnw_lo, pnw_hi = mean_sem_tci(p_nw)
    compact = [r for r in per_chain if r["start_class"] == "compact"]
    diss = [r for r in per_chain if r["start_class"] == "dissociated"]
    compact_p = float(np.mean([r["p_inv"] for r in compact]))
    diss_p = float(np.mean([r["p_inv"] for r in diss]))
    compact_rho = float(np.mean([r["rho_mean_nm"] for r in compact]))
    diss_rho = float(np.mean([r["rho_mean_nm"] for r in diss]))

    summary = {
        "protocol": {
            "P": P,
            "temperature_K": TEMPERATURE_K,
            "wall_radius_nm": WALL_RADIUS_NM,
            "wall_height_eV": WALL_HEIGHT_EV,
            "wall_power": WALL_POWER,
            "n_chains": len(per_chain),
            "seeds": list(SEEDS),
            "start_classes": list(START_CLASSES),
            "n_steps": N_STEPS,
            "raw_burn_in": RAW_BURN_IN,
            "primary_burn_in": PRIMARY_BURN_IN,
            "sample_every": SAMPLE_EVERY,
            "local_step_nm": LOCAL_STEP_NM,
            "global_step_nm": GLOBAL_STEP_NM,
            "global_move_probability": GLOBAL_MOVE_PROBABILITY,
            "asymptotic_dissociated_wall_volume_ratio_vs_R15": (WALL_RADIUS_NM / BASELINE_RWALL_NM) ** 2,
            "primary_gate": "lower 95% chain-level Student-t CI for p_inv > 0.5",
        },
        "primary": {
            "p_inv_chain_mean": p_mean,
            "p_inv_chain_sem": p_sem,
            "p_inv_t95_lo": p_lo,
            "p_inv_t95_hi": p_hi,
            "predominantly_invalid_gate_pass": bool(p_lo > 0.5),
            "p_inv_no_wall_hessian_chain_mean": pnw_mean,
            "p_inv_no_wall_hessian_chain_sem": pnw_sem,
            "p_inv_no_wall_hessian_t95_lo": pnw_lo,
            "p_inv_no_wall_hessian_t95_hi": pnw_hi,
            "delta_wall_hessian_classification_mean": p_mean - pnw_mean,
            "zeta50_chain_mean": float(np.mean(z50)),
            "p_zeta_ge_5_chain_mean": float(np.mean(w5)),
            "rho_median_nm_chain_mean": float(np.mean(rho50)),
            "rho_mean_nm_chain_mean": float(np.mean(rhom)),
            "p_rho_gt_18p7nm_chain_mean": float(np.mean(pr18)),
            "rho_max_nm_all_chains": float(max(r["rho_max_nm"] for r in per_chain)),
        },
        "initialization_mixing_diagnostic": {
            "compact_start_p_inv_mean": compact_p,
            "dissociated_start_p_inv_mean": diss_p,
            "diss_minus_compact_p_inv": diss_p - compact_p,
            "compact_start_rho_mean_nm": compact_rho,
            "dissociated_start_rho_mean_nm": diss_rho,
            "diss_minus_compact_rho_mean_nm": diss_rho - compact_rho,
            "interpretation": "Diagnostic only; two chains per start class are not treated as a separate inferential sample.",
        },
        "baseline_R15_reference": {
            "p_inv_P256": BASELINE_PINV_P256,
            "ci95": list(BASELINE_PINV_CI95),
            "delta_R30_minus_R15": p_mean - BASELINE_PINV_P256,
            "note": "R15 reference is the archived canonical production result; R30 is a separately equilibrated Hamiltonian.",
        },
        "source_hashes": verify_source_truth(),
    }
    (RESULTS_DIR / "R30_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report_lines = [
        "# Rwall=30 nm equilibrium sensitivity result",
        "",
        f"Primary analysis: P={P}, T={TEMPERATURE_K:g} K, 4 independent chains, {N_STEPS} steps/chain, primary burn-in {PRIMARY_BURN_IN}, sample every {SAMPLE_EVERY}.",
        "",
        f"- p_inv = {p_mean:.8f} +/- {p_sem:.8f} (SEM across chains)",
        f"- 95% Student-t CI = [{p_lo:.8f}, {p_hi:.8f}]",
        f"- lower-CI > 0.5 gate: {'PASS' if p_lo > 0.5 else 'FAIL'}",
        f"- no-wall-Hessian reclassification on the same R30 samples = {pnw_mean:.8f}",
        f"- zeta50 chain mean = {np.mean(z50):.6f}",
        f"- P(zeta>=5) chain mean = {np.mean(w5):.8f}",
        f"- rho median chain mean = {np.mean(rho50):.6f} nm",
        f"- rho mean chain mean = {np.mean(rhom):.6f} nm",
        f"- P(rho>18.7 nm) chain mean = {np.mean(pr18):.8f}",
        f"- max rho across retained samples = {max(r['rho_max_nm'] for r in per_chain):.6f} nm",
        f"- compact-start p_inv mean = {compact_p:.8f}",
        f"- dissociated-start p_inv mean = {diss_p:.8f}",
        f"- dissociated minus compact start-class difference = {diss_p-compact_p:+.8f}",
        f"- canonical R15 p_inv reference = {BASELINE_PINV_P256:.8f}",
        f"- R30 minus R15 p_inv = {p_mean-BASELINE_PINV_P256:+.8f}",
        "",
        "Independent chains are the top-level statistical units. Beads enter only as within-chain estimators of same-slice observables.",
        "",
    ]
    report = "\n".join(report_lines)
    (RESULTS_DIR / "R30_RESULT.md").write_text(report)

    print("\n" + "=" * 80)
    print("PRIMARY R30 RESULT")
    print("=" * 80)
    print(report)
    print(f"Wrote: {RESULTS_DIR / 'R30_summary.json'}")
    print(f"Wrote: {RESULTS_DIR / 'R30_per_chain_primary.csv'}")
    print(f"Wrote: {RESULTS_DIR / 'R30_burnin_sensitivity.csv'}")
    print(f"Wrote: {RESULTS_DIR / 'R30_primary_blocks.csv'}")
    print(f"Wrote: {RESULTS_DIR / 'R30_RESULT.md'}")


def write_provenance() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": "Paper2 SecVIII Rwall=30 nm equilibrium sensitivity",
        "created_before_results": True,
        "expected_source_hashes": EXPECTED_HASHES,
        "actual_source_hashes": verify_source_truth(),
        "scientific_differences_from_canonical_R15": [
            "wall_radius_nm: 15.0 -> 30.0",
            "four new independent 240000-step P=256 chains",
            "two compact and two intentionally dissociated initial conditions",
            "primary burn-in 60000; archived frames begin after 15000 for predeclared burn-in audit",
        ],
        "unchanged": [
            "T=20 K", "P=256", "wall_height_eV=0.08", "wall_power=8",
            "masses", "BLK parameters", "moire amplitudes/period/origins",
            "validated periodic two-body staging sampler", "local step 0.15 nm",
            "global step 12 nm", "global move probability 0.20", "sample every 20",
            "same-slice electron-hole pairing", "full 4D mass-weighted Hessian definition",
        ],
        "seeds": list(SEEDS),
        "start_classes": list(START_CLASSES),
        "primary_gate": "lower 95% chain-level Student-t CI for p_inv > 0.5",
        "baseline_R15_p_inv": BASELINE_PINV_P256,
    }
    (RESULTS_DIR / "source_provenance.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dry-run", "sample", "analyze", "all"), default="dry-run")
    ap.add_argument("--reuse", action="store_true", help="reuse existing raw chains/observable caches")
    ap.add_argument("--smoke", action="store_true", help="run 2 short chains (compact+dissociated); never used for scientific result")
    args = ap.parse_args()

    if args.smoke:
        sample_chains(reuse=args.reuse, smoke=True)
        return

    if args.stage == "dry-run":
        dry_run()
        write_provenance()
        return
    if args.stage in ("sample", "all"):
        write_provenance()
        sample_chains(reuse=args.reuse, smoke=False)
    if args.stage in ("analyze", "all"):
        write_provenance()
        analyze(reuse=args.reuse)


if __name__ == "__main__":
    main()
