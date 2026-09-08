#!/usr/bin/env python3
"""TB-03: finite-P occupation of the Gaussian-invalid region.

Runs the validated two-body periodic staging PI-QMC sampler at P=64/128/256,
evaluates the mass-weighted 4D stability field on PAIRED electron/hole beads
from the same imaginary-time slice, and estimates

    p_inv^(P) = <Theta[zeta(r_e,j, r_h,j)-1]>_P.

Important design choices
------------------------
* same-slice pairing is preserved exactly by flattening samples_e and samples_h
  in the same (frame, bead) order;
* uncertainty is estimated from independent seed/chain means, never from beads;
* the primary zeta field contains the SAME radial wall as the PI-QMC Hamiltonian;
* a wall-off diagnostic is also reported on the same samples to expose any
  dependence of the stability classification on the confining wall;
* raw path arrays are NOT written by default.

The predeclared TB-04 gate used here is:
    |p_inv^(256) - p_inv^(128)| <= 0.02  (finite-P stability), and
    lower 95% chain-bootstrap CI at P=256 > 0.5  -> central-result category.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_tmd_pimc(project_root: Path | None) -> None:
    try:
        import tmd_pimc  # noqa: F401
        return
    except ImportError:
        pass

    candidates: list[Path] = []
    if project_root is not None:
        candidates += [project_root / "numerics", project_root / "code", project_root]
    here = Path(__file__).resolve().parent
    candidates += [here / "numerics", here.parent / "numerics", here / "code", here.parent / "code"]

    for c in candidates:
        if (c / "tmd_pimc").is_dir():
            sys.path.insert(0, str(c))
            import tmd_pimc  # noqa: F401
            return
    raise ImportError(
        "Could not locate tmd_pimc. Use --project-root pointing to the "
        "v1.8b_2D_RK_PIMC_validation project root or install it editable."
    )


def _load_stability_module(path: Path):
    spec = importlib.util.spec_from_file_location("tb03_stability", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import stability module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _vec2(value: Any, default=(0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return tuple(map(float, default))
    a = np.asarray(value, dtype=float).reshape(-1)
    if a.size == 1:
        return (float(a[0]), 0.0)
    if a.size != 2:
        raise ValueError(f"Expected 2-vector, got {value!r}")
    return (float(a[0]), float(a[1]))


def _resolve_amplitudes(cfg: dict[str, Any]) -> tuple[float, float]:
    shared = cfg.get("moire_amplitude_eV", cfg.get("moire_amplitude", None))
    ae = cfg.get("moire_amplitude_e_eV", cfg.get("moire_amplitude_e", shared))
    ah = cfg.get("moire_amplitude_h_eV", cfg.get("moire_amplitude_h", shared))
    if ae is None or ah is None:
        raise KeyError(
            "Config must provide moire_amplitude_eV or both per-carrier amplitudes"
        )
    return float(ae), float(ah)


def _build_interaction(cfg: dict[str, Any]):
    from tmd_pimc.bilayer_keldysh_potential import (
        BilayerKeldyshWallPotential,
        build_bilayer_keldysh_table,
    )

    table = build_bilayer_keldysh_table(
        separation_nm=float(cfg["separation_nm"]),
        screening_length_layer1_nm=float(cfg["screening_length_layer1_nm"]),
        screening_length_layer2_nm=float(cfg["screening_length_layer2_nm"]),
        kappa_environment=float(cfg["kappa_environment"]),
        r_max_nm=float(cfg.get("r_max_nm", 80.0)),
        n_log=int(cfg.get("n_log", 1000)),
        n_linear=int(cfg.get("n_linear", 2000)),
    )
    return BilayerKeldyshWallPotential(
        bilayer=table,
        wall_radius_nm=float(cfg.get("wall_radius_nm", 15.0)),
        wall_height_eV=float(cfg.get("wall_height_eV", 0.08)),
        wall_power=int(cfg.get("wall_power", 8)),
    )


def _build_stability_field(mod, cfg: dict[str, Any], *, include_wall: bool):
    ae, ah = _resolve_amplitudes(cfg)
    period = float(cfg["moire_period_nm"])
    oe = _vec2(cfg.get("origin_e_nm", (0.0, 0.0)))
    oh = _vec2(cfg.get("origin_h_nm", (0.0, 0.0)))

    wall_radius = float(cfg.get("wall_radius_nm", 15.0)) if include_wall else None
    wall_height = float(cfg.get("wall_height_eV", 0.08)) if include_wall else 0.0
    wall_power = int(cfg.get("wall_power", 8))

    return mod.PairStabilityField(
        interaction=mod.BilayerKeldysh(
            separation_nm=float(cfg["separation_nm"]),
            screening_length_layer1_nm=float(cfg["screening_length_layer1_nm"]),
            screening_length_layer2_nm=float(cfg["screening_length_layer2_nm"]),
            kappa_environment=float(cfg["kappa_environment"]),
            wall_radius_nm=wall_radius,
            wall_height_eV=wall_height,
            wall_power=wall_power,
        ),
        landscape_e=mod.MoireLandscape(ae, period, oe),
        landscape_h=mod.MoireLandscape(ah, period, oh),
        mass_e_m0=float(cfg["mass_e_m0"]),
        mass_h_m0=float(cfg["mass_h_m0"]),
    )


def _validate_model_match(sampler, action, field, cfg: dict[str, Any]) -> None:
    rng = np.random.default_rng(92031)
    pts = rng.normal(size=(64, 2)) * float(cfg["moire_period_nm"])

    ve_sampler = np.asarray(sampler._potential_e_for_grid.value(pts), float)
    vh_sampler = np.asarray(sampler._potential_h_for_grid.value(pts), float)
    ve_field = np.asarray(field.landscape_e.value(pts), float)
    vh_field = np.asarray(field.landscape_h.value(pts), float)
    de = float(np.max(np.abs(ve_sampler - ve_field)))
    dh = float(np.max(np.abs(vh_sampler - vh_field)))

    rel = rng.normal(size=(64, 2)) * 5.0
    vint_sampler = np.asarray(action.potential_interaction.value(rel), float)
    vint_field = np.asarray(field.interaction.value(np.linalg.norm(rel, axis=1)), float)
    di = float(np.max(np.abs(vint_sampler - vint_field)))

    print(f"    model-match max |V_e^sampler-V_e^zeta| = {de:.3e} eV")
    print(f"    model-match max |V_h^sampler-V_h^zeta| = {dh:.3e} eV")
    print(f"    model-match max |V_int^sampler-V_int^zeta| = {di:.3e} eV")
    if max(de, dh) > 1e-10:
        raise RuntimeError("Analytic one-body stability landscape does not match sampler landscape")
    if di > 5e-4:
        raise RuntimeError("Stability interaction does not match PI-QMC interaction closely enough")


def _chain_summary(field, field_no_wall, samples_e, samples_h, T: float, chunk: int) -> dict[str, float]:
    se = np.asarray(samples_e, dtype=float)
    sh = np.asarray(samples_h, dtype=float)
    if se.shape != sh.shape or se.ndim != 3 or se.shape[-1] != 2:
        raise ValueError(f"Expected paired arrays (Nframes,P,2), got {se.shape} and {sh.shape}")

    # Flatten identically: element k corresponds to the same (frame, bead j)
    # for electron and hole, so same-slice pairing is preserved exactly.
    re = se.reshape(-1, 2)
    rh = sh.reshape(-1, 2)
    n = re.shape[0]

    z_same = np.empty(n, dtype=np.float32)
    z_nowall = np.empty(n, dtype=np.float32)
    rho = np.empty(n, dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        a = re[lo:hi]
        b = rh[lo:hi]
        z_same[lo:hi] = field.zeta(a, b, T)
        z_nowall[lo:hi] = field_no_wall.zeta(a, b, T)
        rho[lo:hi] = np.linalg.norm(a - b, axis=1)

    inv = z_same >= 1.0
    inv_nw = z_nowall >= 1.0
    return {
        "n_paired_beads": int(n),
        "p_invalid": float(np.mean(inv)),
        "p_invalid_no_wall_diagnostic": float(np.mean(inv_nw)),
        "zeta_median": float(np.median(z_same)),
        "zeta_p95": float(np.percentile(z_same, 95.0)),
        "zeta_max_sampled": float(np.max(z_same)),
        "p_zeta_gt_2": float(np.mean(z_same >= 2.0)),
        "p_zeta_gt_5": float(np.mean(z_same >= 5.0)),
        "rho_mean_nm": float(np.mean(rho)),
        "rho_median_nm": float(np.median(rho)),
        "rho_p05_nm": float(np.percentile(rho, 5.0)),
        "rho_p95_nm": float(np.percentile(rho, 95.0)),
    }


def _bootstrap_chain_means(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    if values.size < 2:
        return mean, sem, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    return mean, sem, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stability-module", type=Path, required=True)
    ap.add_argument("--project-root", type=Path)
    ap.add_argument("--p-values", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=31000)
    ap.add_argument("--n-steps", type=int, default=60000)
    ap.add_argument("--burn-in", type=int, default=15000)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--local-step-nm", type=float, default=0.15)
    ap.add_argument("--global-step-nm", type=float, default=12.0)
    ap.add_argument("--global-move-probability", type=float, default=0.20)
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--output-dir", type=Path, default=Path("results/tb03_pinv_ladder"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_seeds = 1
        args.n_steps = 12000
        args.burn_in = 3000
        args.sample_every = 30

    cfg = json.loads(args.config.read_text())
    T = float(cfg.get("temperature_K", 20.0))
    if abs(T - 20.0) > 1e-9:
        print(f"WARNING: TB-03 was planned at 20 K, but config requests T={T:g} K")
    if float(cfg.get("Fz_eV_per_nm", 0.0)) != 0.0:
        raise ValueError("TB-03 baseline must use Fz=0; use a zero-field config")

    _ensure_tmd_pimc(args.project_root)
    mod = _load_stability_module(args.stability_module)

    from tmd_pimc.potentials import CompositePotential
    from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
    from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT

    interaction = _build_interaction(cfg)
    field = _build_stability_field(mod, cfg, include_wall=True)
    field_no_wall = _build_stability_field(mod, cfg, include_wall=False)
    ae, ah = _resolve_amplitudes(cfg)
    period = float(cfg["moire_period_nm"])
    origin_e = _vec2(cfg.get("origin_e_nm", (0.0, 0.0)))
    origin_h = _vec2(cfg.get("origin_h_nm", (0.0, 0.0)))

    zero_e = CompositePotential(terms=[])
    zero_h = CompositePotential(terms=[])

    stage_lengths = tuple(cfg.get("staging_segment_lengths", [4, 8, 16, 32, 64, 128, 256]))
    stage_moves = int(cfg.get("staging_moves_per_step", 2))
    grid_size = int(cfg.get("periodic_cell_grid_size", 200))

    print("=" * 78)
    print("TB-03 finite-P Gaussian-invalid occupation ladder")
    print("=" * 78)
    print(f"config              : {args.config}")
    print(f"stability module    : {args.stability_module}")
    print(f"T                   : {T:g} K")
    print(f"P ladder            : {args.p_values}")
    print(f"seeds / P           : {args.n_seeds}")
    print(f"steps/burn/sample   : {args.n_steps}/{args.burn_in}/{args.sample_every}")
    print(f"local/global step   : {args.local_step_nm:g}/{args.global_step_nm:g} nm")
    print(f"moire amplitudes    : e={ae:g}, h={ah:g} eV")
    print(f"moire period        : {period:g} nm")
    print(f"origins e/h         : {origin_e} / {origin_h}")
    print(f"wall                : R={cfg.get('wall_radius_nm',15.0)} nm, "
          f"H={cfg.get('wall_height_eV',0.08)} eV, p={cfg.get('wall_power',8)}")
    print("Primary p_inv uses the wall-inclusive zeta field (same Hamiltonian).")
    print("A wall-off zeta classification is saved as a sensitivity diagnostic.\n")

    seed_rows: list[dict[str, Any]] = []

    for P in args.p_values:
        print(f"\n--- P={P} ---")
        action = TwoBodyRingPolymerAction(
            mass_e_m0=float(cfg["mass_e_m0"]),
            mass_h_m0=float(cfg["mass_h_m0"]),
            temperature_K=T,
            n_beads=int(P),
            potential_e=zero_e,
            potential_h=zero_h,
            potential_interaction=interaction,
        )

        for iseed in range(args.n_seeds):
            seed = int(args.seed_base + 1000 * P + iseed)
            sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
                action=action,
                moire_period_nm=period,
                moire_amplitude_eV=None,
                moire_amplitude_e_eV=ae,
                moire_amplitude_h_eV=ah,
                origin_e_nm=origin_e,
                origin_h_nm=origin_h,
                Fz_eV_per_nm=0.0,
                local_step_nm=float(args.local_step_nm),
                global_step_nm=float(args.global_step_nm),
                global_move_probability=float(args.global_move_probability),
                rng_seed=seed,
                staging_segment_lengths=stage_lengths,
                staging_moves_per_step=stage_moves,
                perform_local_sweep=True,
                interaction_table_r_max_nm=float(cfg.get("r_max_nm", 80.0)),
                interaction_table_n_points=int(cfg.get("interaction_table_n_points", 20000)),
                periodic_cell_grid_size=grid_size,
            )

            if iseed == 0:
                _validate_model_match(sampler, action, field, cfg)

            # Start both carriers in the same true moire well. The global move
            # is explicitly lattice-scale so the result does not rely on kinetic pinning.
            center = tuple(cfg.get("tb03_start_center_nm", [0.5 * period, 0.0]))
            t0 = time.time()
            res = sampler.run(
                n_steps=int(args.n_steps),
                burn_in=int(args.burn_in),
                sample_every=int(args.sample_every),
                center_e=center,
                center_h=center,
            )
            elapsed = time.time() - t0

            summ = _chain_summary(
                field, field_no_wall,
                res["samples_e"], res["samples_h"], T, args.chunk_size,
            )
            row = {
                "P": int(P),
                "seed": seed,
                "temperature_K": T,
                "n_frames": int(res["samples_e"].shape[0]),
                **summ,
                "acceptance_local_e": float(res["acceptance_local_e"]),
                "acceptance_local_h": float(res["acceptance_local_h"]),
                "acceptance_staging": float(res["acceptance_staging"]),
                "acceptance_global_joint": float(res["acceptance_global_joint"]),
                "elapsed_s": float(elapsed),
            }
            seed_rows.append(row)
            print(
                f"  seed={seed} p_inv={row['p_invalid']:.6f} "
                f"(no-wall diag {row['p_invalid_no_wall_diagnostic']:.6f}) "
                f"z50={row['zeta_median']:.3f} rho50={row['rho_median_nm']:.3f} nm "
                f"acc_stg={row['acceptance_staging']:.3f} t={elapsed:.1f}s"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "tb03_per_seed.csv", seed_rows)

    aggregate: list[dict[str, Any]] = []
    for P in args.p_values:
        rows = [r for r in seed_rows if r["P"] == P]
        p = np.array([r["p_invalid"] for r in rows])
        pnw = np.array([r["p_invalid_no_wall_diagnostic"] for r in rows])
        mean, sem, lo, hi = _bootstrap_chain_means(p, args.n_bootstrap, args.seed_base + P)
        mnw, semnw, lonw, hinw = _bootstrap_chain_means(pnw, args.n_bootstrap, args.seed_base + P + 1)
        aggregate.append({
            "P": P,
            "n_chains": len(rows),
            "p_invalid": mean,
            "p_invalid_chain_sem": sem,
            "p_invalid_ci95_lo": lo,
            "p_invalid_ci95_hi": hi,
            "p_invalid_no_wall_diagnostic": mnw,
            "p_invalid_no_wall_sem": semnw,
            "p_invalid_no_wall_ci95_lo": lonw,
            "p_invalid_no_wall_ci95_hi": hinw,
            "zeta_median_chain_mean": float(np.mean([r["zeta_median"] for r in rows])),
            "zeta_p95_chain_mean": float(np.mean([r["zeta_p95"] for r in rows])),
            "p_zeta_gt_2_chain_mean": float(np.mean([r["p_zeta_gt_2"] for r in rows])),
            "p_zeta_gt_5_chain_mean": float(np.mean([r["p_zeta_gt_5"] for r in rows])),
            "rho_median_nm_chain_mean": float(np.mean([r["rho_median_nm"] for r in rows])),
            "rho_mean_nm_chain_mean": float(np.mean([r["rho_mean_nm"] for r in rows])),
        })

    byp = {r["P"]: r for r in aggregate}
    for r in aggregate:
        P = r["P"]
        prev = 128 if P == 256 and 128 in byp else (64 if P == 128 and 64 in byp else None)
        r["delta_p_from_prev"] = (
            r["p_invalid"] - byp[prev]["p_invalid"] if prev is not None else float("nan")
        )

    _write_csv(args.output_dir / "tb03_aggregate.csv", aggregate)

    decision = "UNCLASSIFIED"
    stable = None
    if 256 in byp and 128 in byp:
        stable = abs(byp[256]["p_invalid"] - byp[128]["p_invalid"]) <= 0.02
        if stable and np.isfinite(byp[256]["p_invalid_ci95_lo"]) and byp[256]["p_invalid_ci95_lo"] > 0.5:
            decision = "CENTRAL_RESULT"
        elif byp[256]["p_invalid"] < 0.05:
            decision = "APPENDIX_OR_SHORT_EXAMPLE"
        else:
            decision = "FULL_SECTION_NONCENTRAL"

    report = {
        "decision": decision,
        "p_convergence_gate_abs_128_256_le_0p02": stable,
        "tb04_central_threshold": "lower 95% chain-bootstrap CI at P=256 > 0.5",
        "primary_estimator": "wall-inclusive zeta, same finite-P Hamiltonian",
        "sensitivity_estimator": "wall-off zeta evaluated on the same PI-QMC samples",
        "aggregate": aggregate,
    }
    (args.output_dir / "tb03_decision.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print("TB-03 aggregate")
    print("=" * 78)
    for r in aggregate:
        print(
            f"P={r['P']:3d}  p_inv={r['p_invalid']:.6f} "
            f"CI95=[{r['p_invalid_ci95_lo']:.6f},{r['p_invalid_ci95_hi']:.6f}] "
            f"no-wall={r['p_invalid_no_wall_diagnostic']:.6f} "
            f"delta_prev={r['delta_p_from_prev']:+.6f}"
        )
    print(f"\nTB-04 predeclared classification: {decision}")
    print(f"Saved: {args.output_dir / 'tb03_per_seed.csv'}")
    print(f"       {args.output_dir / 'tb03_aggregate.csv'}")
    print(f"       {args.output_dir / 'tb03_decision.json'}")


if __name__ == "__main__":
    main()
