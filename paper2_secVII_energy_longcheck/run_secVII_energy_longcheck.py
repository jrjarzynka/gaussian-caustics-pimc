#!/usr/bin/env python3
"""Frozen Point 5 Sec. VII long-statistics PI-QMC campaign.

The scientific setup below is copied from
validation_tests/lhap0a_publication_single_seed.py.  Only the predeclared
seeds, n_steps, and burn_in differ.  Output handling occurs after the JIT
kernel returns and therefore consumes no sampler RNG values.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CAMPAIGN_DIR / "results"
VALIDATION_DIR = ROOT / "validation_tests"

for candidate in [ROOT / "code", ROOT / "numerics", ROOT]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from tmd_pimc import HBAR2_OVER_2M0, KB_EV_PER_K
from tmd_pimc.kernels_jit import run_pimc_core_jit_periodic_cell


EXPECTED_CANONICAL_HASHES = {
    "validation_tests/lhap0a_publication_single_seed.py":
        "4451de7d1aaf00c1ec684b1883e6ec24c88003fb4fb9f5581886bbd91561d703",
    "numerics/tmd_pimc/constants.py":
        "2605841057e6c03868505af4e157422992bec0e5d94b9190108f142de8aa419b",
    "numerics/tmd_pimc/kernels_jit.py":
        "d1061cdfea320817d163b1db097b819a7d834b0f22f36d45a3141e2be465eccf",
}

PROVENANCE_FILES = [
    "validation_tests/lhap0a_publication_single_seed.py",
    "numerics/tmd_pimc/constants.py",
    "numerics/tmd_pimc/kernels_jit.py",
    "paper2_secVII_energy_longcheck/run_secVII_energy_longcheck.py",
    "paper2_secVII_energy_longcheck/analyze_secVII_energy_longcheck.py",
    "paper2_secVII_energy_longcheck/RUN_COMMAND.txt",
]

# Frozen scientific settings.
BETA_STAR = 2.70
P = 64
MASS_M0 = 1.0
E0_EV = 0.010
NGRID = 512
N_STEPS = 160000
BURN_IN = 32000
SAMPLE_EVERY = 5
SEEDS = tuple(range(132001, 132017))
GLOBAL_MOVE_PROB = 0.20
AMPLITUDES = np.array([1.00, 0.78, 0.62])
PHASES = np.array([0.00, 0.23, 0.71])
E_STAR_DECLARED = 0.6567185986


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args):
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.rstrip("\n")


def verify_canonical_hashes():
    actual = {
        rel: sha256_file(ROOT / rel) for rel in EXPECTED_CANONICAL_HASHES
    }
    mismatches = {
        rel: {"expected": EXPECTED_CANONICAL_HASHES[rel], "actual": value}
        for rel, value in actual.items()
        if value != EXPECTED_CANONICAL_HASHES[rel]
    }
    if mismatches:
        raise RuntimeError(f"Canonical SHA-256 mismatch: {mismatches}")
    return actual


def build_setup():
    """Build the canonical physical embedding, grid, and proposal scales."""
    sqrt3 = np.sqrt(3.0)
    g1 = np.array([1.0, 0.0])
    g2 = np.array([-0.5, sqrt3 / 2.0])
    g3 = -(g1 + g2)
    _gs = np.array([g1, g2, g3])
    b_recip = np.column_stack([g1, g2])
    a_star = 2.0 * np.pi * np.linalg.inv(b_recip.T)
    duality_error = float(
        np.max(np.abs(b_recip.T @ a_star - 2.0 * np.pi * np.eye(2)))
    )

    lambda_nm2_eV = HBAR2_OVER_2M0 / MASS_M0
    l0_nm = np.sqrt(2.0 * lambda_nm2_eV / E0_EV)
    a_phys = l0_nm * a_star
    a1 = a_phys[:, 0]
    a2 = a_phys[:, 1]
    ainv = np.linalg.inv(a_phys)
    cell_length_1 = float(np.linalg.norm(a1))
    cell_length_2 = float(np.linalg.norm(a2))

    temperature_k = E0_EV / (KB_EV_PER_K * BETA_STAR)
    beta_phys = 1.0 / (KB_EV_PER_K * temperature_k)
    tau = beta_phys / P
    kpf = 1.0 / (4.0 * lambda_nm2_eV * tau)

    with (VALIDATION_DIR / "lha13a0_stationary_points.csv").open() as handle:
        import csv
        stationary = list(csv.DictReader(handle))
    minimum = [row for row in stationary if row["type"] == "minimum"][0]
    s_min = float(minimum["s"])
    t_min = float(minimum["t"])

    def ustar(s, t):
        return (
            -AMPLITUDES[0] * np.cos(2.0 * np.pi * s + PHASES[0])
            -AMPLITUDES[1] * np.cos(2.0 * np.pi * t + PHASES[1])
            -AMPLITUDES[2] * np.cos(-2.0 * np.pi * (s + t) + PHASES[2])
        )

    umin = float(ustar(s_min, t_min))

    def vstar(s, t):
        return ustar(s, t) - umin

    axis = np.arange(NGRID, dtype=float) / NGRID
    s_grid, t_grid = np.meshgrid(axis, axis, indexing="ij")
    vstar_grid = vstar(s_grid, t_grid)
    vgrid_ev = (E0_EV * vstar_grid).astype(np.float64)

    def interp_star(s, t):
        s = np.mod(np.asarray(s), 1.0)
        t = np.mod(np.asarray(t), 1.0)
        fs = s * NGRID
        ft = t * NGRID
        i0 = np.floor(fs).astype(int)
        j0 = np.floor(ft).astype(int)
        ws = fs - i0
        wt = ft - j0
        i1 = (i0 + 1) % NGRID
        j1 = (j0 + 1) % NGRID
        i0 %= NGRID
        j0 %= NGRID
        return (
            (1.0 - ws) * (1.0 - wt) * vstar_grid[i0, j0]
            + ws * (1.0 - wt) * vstar_grid[i1, j0]
            + (1.0 - ws) * wt * vstar_grid[i0, j1]
            + ws * wt * vstar_grid[i1, j1]
        )

    # This independent audit uses its own Generator and cannot affect Numba RNG.
    rng_audit = np.random.default_rng(130211)
    s_test = rng_audit.random(20000)
    t_test = rng_audit.random(20000)
    interp_error = np.abs(interp_star(s_test, t_test) - vstar(s_test, t_test))

    sigma_link_nm = np.sqrt(2.0 * lambda_nm2_eV * tau)
    local_step_nm = 0.50 * sigma_link_nm
    global_step_nm = 0.25 * min(cell_length_1, cell_length_2)
    center_nm = a_phys @ np.array([s_min, t_min])

    with (VALIDATION_DIR / "lha13b0_exact_primitive_finiteP.csv").open() as handle:
        import csv
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows
        if abs(float(row["beta"]) - BETA_STAR) < 1.0e-12
        and int(row["P"]) == P
    ]
    if len(matches) != 1:
        raise RuntimeError("Could not identify unique exact finite-P target")
    exact_finitep = float(matches[0]["V_exact_finiteP"])

    audit = {
        "lattice_duality_error": duality_error,
        "beta_mapping_error": abs(beta_phys * E0_EV - BETA_STAR),
        "spring_mapping_error": abs(kpf * l0_nm**2 - P / (2.0 * BETA_STAR)),
        "potential_mapping_error": abs(tau * E0_EV - BETA_STAR / P),
        "interp_max_error_star": float(np.max(interp_error)),
        "interp_rms_error_star": float(np.sqrt(np.mean(interp_error**2))),
        "exact_finiteP_V": exact_finitep,
        "declared_target_delta": exact_finitep - E_STAR_DECLARED,
    }
    return {
        "ainv": ainv,
        "vgrid_ev": vgrid_ev,
        "vstar": vstar,
        "interp_star": interp_star,
        "center_nm": center_nm,
        "sigma_link_nm": sigma_link_nm,
        "local_step_nm": local_step_nm,
        "global_step_nm": global_step_nm,
        "kpf": kpf,
        "tau": tau,
        "temperature_k": temperature_k,
        "audit": audit,
    }


def structural_preflight(setup):
    audit = setup["audit"]
    checks = {
        "canonical_hashes": True,
        "seed_declaration": SEEDS == tuple(range(132001, 132017)),
        "beta": BETA_STAR == 2.70,
        "beads": P == 64,
        "grid": NGRID == 512,
        "n_steps": N_STEPS == 160000,
        "burn_in": BURN_IN == 32000,
        "sample_every": SAMPLE_EVERY == 5,
        "global_move_probability": GLOBAL_MOVE_PROB == 0.20,
        "lattice_duality": audit["lattice_duality_error"] < 1.0e-12,
        "beta_embedding": audit["beta_mapping_error"] < 1.0e-12,
        "spring_action_embedding": audit["spring_mapping_error"] < 1.0e-12,
        "potential_action_embedding": audit["potential_mapping_error"] < 1.0e-12,
        "periodic_grid_interpolation": audit["interp_max_error_star"] < 1.0e-4,
        "exact_target_rounding": abs(audit["declared_target_delta"]) < 5.0e-11,
    }
    return {name: bool(passed) for name, passed in checks.items()}


def write_provenance(canonical_hashes, setup, checks):
    hashes = {rel: sha256_file(ROOT / rel) for rel in PROVENANCE_FILES}
    provenance = {
        "preflight": "PASS" if all(checks.values()) else "FAIL",
        "preflight_checks": checks,
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "python_version": platform.python_version(),
        "python_version_full": sys.version,
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "sha256": hashes,
        "canonical_hashes_verified": canonical_hashes,
        "scientific_settings": {
            "seeds": list(SEEDS), "beta": BETA_STAR, "P": P,
            "grid": [NGRID, NGRID], "n_steps": N_STEPS,
            "burn_in": BURN_IN, "sample_every": SAMPLE_EVERY,
            "global_move_probability": GLOBAL_MOVE_PROB,
            "staging": False, "adaptive_tuning": False,
        },
        "canonical_audit": setup["audit"],
        "intentional_differences_vs_publication_runner": [
            "Seeds are 132001..132016 inclusive instead of publication seeds.",
            "N_STEPS is 160000 instead of 40000.",
            "BURN_IN is 32000 instead of 8000.",
        ],
        "infrastructure_only_differences": [
            "Loops over the fixed seed set in one campaign invocation.",
            "Uses repository-absolute input and result paths.",
            "Records provenance and per-chain runtime/log files.",
            "Saves already-computed analytic and lookup energy time series, not full bead paths or density histograms.",
            "Defers cross-chain and stationarity statistics to a separate analyzer.",
        ],
        "scientific_difference_count_beyond_declared_three": 0,
    }
    (CAMPAIGN_DIR / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def run_chain(seed, setup):
    # Canonical initialization: the only change is the predeclared seed value.
    rng_init = np.random.default_rng(seed + 991)
    path = (
        setup["center_nm"][None, :]
        + 0.05 * setup["sigma_link_nm"]
        * rng_init.standard_normal((P, 2))
    ).astype(np.float64)

    started = time.perf_counter()
    samples, acc_local, acc_global = run_pimc_core_jit_periodic_cell(
        n_steps=N_STEPS,
        burn_in=BURN_IN,
        sample_every=SAMPLE_EVERY,
        p_beads=P,
        path=path,
        v_grid=setup["vgrid_ev"],
        origin_x=0.0,
        origin_y=0.0,
        ainv00=float(setup["ainv"][0, 0]),
        ainv01=float(setup["ainv"][0, 1]),
        ainv10=float(setup["ainv"][1, 0]),
        ainv11=float(setup["ainv"][1, 1]),
        kpf=float(setup["kpf"]),
        tau=float(setup["tau"]),
        local_step_nm=float(setup["local_step_nm"]),
        global_step_nm=float(setup["global_step_nm"]),
        global_move_prob=float(GLOBAL_MOVE_PROB),
        seed=int(seed),
    )
    runtime_seconds = time.perf_counter() - started

    samples = np.asarray(samples, dtype=float)
    flat = samples.reshape(-1, 2)
    uv = (setup["ainv"] @ flat.T).T
    uv = np.mod(uv, 1.0)
    s_all = uv[:, 0]
    t_all = uv[:, 1]
    exact_ts = setup["vstar"](s_all, t_all).reshape(samples.shape[0], P).mean(axis=1)
    lookup_ts = setup["interp_star"](s_all, t_all).reshape(samples.shape[0], P).mean(axis=1)
    if samples.shape[0] != (N_STEPS - BURN_IN) // SAMPLE_EVERY:
        raise RuntimeError(f"Seed {seed}: unexpected sample count {samples.shape[0]}")
    if not np.all(np.isfinite(exact_ts)):
        raise RuntimeError(f"Seed {seed}: non-finite energy time series")

    raw_path = RESULTS_DIR / f"secVII_energy_longcheck_seed_{seed}.npz"
    np.savez_compressed(
        raw_path,
        Vstar_time_series=exact_ts,
        Vstar_lookup_time_series=lookup_ts,
        seed=seed,
        n_steps=N_STEPS,
        primary_burn_in=BURN_IN,
        sample_every=SAMPLE_EVERY,
        P=P,
        beta_star=BETA_STAR,
        acceptance_local=float(acc_local),
        acceptance_global=float(acc_global),
        local_step_nm=float(setup["local_step_nm"]),
        global_step_nm=float(setup["global_step_nm"]),
        runtime_seconds=runtime_seconds,
    )
    record = {
        "seed": seed,
        "energy_mean": float(np.mean(exact_ts)),
        "lookup_energy_mean": float(np.mean(lookup_ts)),
        "analytic_minus_lookup": float(np.mean(exact_ts) - np.mean(lookup_ts)),
        "acceptance_local": float(acc_local),
        "acceptance_global": float(acc_global),
        "local_step_nm": float(setup["local_step_nm"]),
        "global_step_nm": float(setup["global_step_nm"]),
        "runtime_seconds": runtime_seconds,
        "n_saved_samples": int(exact_ts.size),
        "raw_file": raw_path.name,
    }
    (RESULTS_DIR / f"secVII_energy_longcheck_seed_{seed}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / f"secVII_energy_longcheck_seed_{seed}.log").write_text(
        "\n".join(f"{key}={value}" for key, value in record.items()) + "\n",
        encoding="utf-8",
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight", action="store_true",
        help="verify hashes/imports/configuration and write pre-run provenance only",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    canonical_hashes = verify_canonical_hashes()
    setup = build_setup()
    checks = structural_preflight(setup)
    provenance = write_provenance(canonical_hashes, setup, checks)
    print(f"PREFLIGHT: {provenance['preflight']}")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("Preflight failed; production not started")
    if args.preflight:
        return

    print("Starting complete frozen campaign; all 16 seeds will be retained.", flush=True)
    records = []
    for seed in SEEDS:
        print(f"seed {seed}: START", flush=True)
        record = run_chain(seed, setup)
        records.append(record)
        print(
            f"seed {seed}: COMPLETE E={record['energy_mean']:.12f} "
            f"local={record['acceptance_local']:.6f} "
            f"global={record['acceptance_global']:.6f} "
            f"runtime={record['runtime_seconds']:.3f}s",
            flush=True,
        )
    (RESULTS_DIR / "campaign_completion.json").write_text(
        json.dumps({"status": "COMPLETE", "seeds": list(SEEDS), "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("CAMPAIGN COMPLETE: 16/16 chains", flush=True)


if __name__ == "__main__":
    main()
