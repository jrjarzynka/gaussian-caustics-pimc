#!/usr/bin/env python3
"""Analyze the frozen Point 5 campaign using independent chain means."""

import csv
import json
import math
from pathlib import Path

import numpy as np


CAMPAIGN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CAMPAIGN_DIR / "results"
SEEDS = tuple(range(132001, 132017))
BURN_INS = (32000, 48000, 64000)
PRIMARY_BURN_IN = 32000
SAMPLE_EVERY = 5
E_STAR = 0.6567185986
E_OLD = 0.66755254
SEM_OLD = 0.00496588
# Two-sided 95% Student-t critical value for the fixed 16-chain design (df=15).
T_CRITICAL_DF15_95 = 2.131449545559323


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chain_summary(values):
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    sem = sd / math.sqrt(n)
    if n != 16:
        raise ValueError("This frozen analysis requires exactly 16 chain means")
    critical = T_CRITICAL_DF15_95
    return {
        "n_chains": n,
        "mean": mean,
        "between_chain_sd": sd,
        "sem": sem,
        "ci95_low": mean - critical * sem,
        "ci95_high": mean + critical * sem,
        "t_star": (mean - E_STAR) / sem,
        "t_critical_df15": critical,
    }


def fmt(value, digits=9):
    return f"{value:.{digits}f}"


def main():
    completion_path = RESULTS_DIR / "campaign_completion.json"
    if not completion_path.exists():
        raise SystemExit("Missing campaign_completion.json; refusing incomplete analysis")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE" or completion.get("seeds") != list(SEEDS):
        raise SystemExit("Campaign completion record does not contain all predeclared seeds")

    data = {}
    metadata = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f"secVII_energy_longcheck_seed_{seed}.npz"
        if not path.exists():
            raise SystemExit(f"Missing raw output for seed {seed}")
        with np.load(path) as archive:
            data[seed] = np.asarray(archive["Vstar_time_series"], dtype=float)
            metadata[seed] = {
                key: float(archive[key])
                for key in (
                    "acceptance_local", "acceptance_global", "local_step_nm",
                    "global_step_nm", "runtime_seconds",
                )
            }
        if data[seed].size != 25600 or not np.all(np.isfinite(data[seed])):
            raise SystemExit(f"Invalid primary time series for seed {seed}")

    per_chain_rows = []
    block_rows = []
    first_means = []
    second_means = []
    for seed in SEEDS:
        ts = data[seed]
        mid = ts.size // 2
        first = float(np.mean(ts[:mid]))
        second = float(np.mean(ts[mid:]))
        first_means.append(first)
        second_means.append(second)
        per_chain_rows.append({
            "seed": seed,
            "energy_mean": float(np.mean(ts)),
            "first_half_mean": first,
            "second_half_mean": second,
            "second_minus_first": second - first,
            **metadata[seed],
            "n_saved_samples": ts.size,
        })
        blocks = np.array_split(ts, 8)
        if any(block.size != 3200 for block in blocks):
            raise RuntimeError("Post-primary-burn series does not divide into 8 equal blocks")
        for index, block in enumerate(blocks, start=1):
            block_rows.append({
                "seed": seed,
                "block": index,
                "start_sweep_inclusive": PRIMARY_BURN_IN + (index - 1) * 3200 * SAMPLE_EVERY,
                "end_sweep_exclusive": PRIMARY_BURN_IN + index * 3200 * SAMPLE_EVERY,
                "n_saved_samples": block.size,
                "energy_mean": float(np.mean(block)),
            })

    write_csv(
        RESULTS_DIR / "secVII_energy_longcheck_per_chain.csv",
        per_chain_rows,
        list(per_chain_rows[0]),
    )
    write_csv(
        RESULTS_DIR / "secVII_energy_longcheck_blocks.csv",
        block_rows,
        list(block_rows[0]),
    )

    sensitivity_rows = []
    summaries = {}
    for burn_in in BURN_INS:
        offset = (burn_in - PRIMARY_BURN_IN) // SAMPLE_EVERY
        means = [float(np.mean(data[seed][offset:])) for seed in SEEDS]
        summary = chain_summary(means)
        summaries[burn_in] = summary
        sensitivity_rows.append({"burn_in": burn_in, **summary})
    write_csv(
        RESULTS_DIR / "secVII_energy_longcheck_burnin_sensitivity.csv",
        sensitivity_rows,
        list(sensitivity_rows[0]),
    )

    first_summary = chain_summary(first_means)
    second_summary = chain_summary(second_means)
    paired_diff = np.asarray(second_means) - np.asarray(first_means)
    paired_mean = float(np.mean(paired_diff))
    paired_sd = float(np.std(paired_diff, ddof=1))
    paired_sem = paired_sd / math.sqrt(len(SEEDS))
    critical = T_CRITICAL_DF15_95
    paired = {
        "mean_second_minus_first": paired_mean,
        "between_chain_sd": paired_sd,
        "sem": paired_sem,
        "ci95_low": paired_mean - critical * paired_sem,
        "ci95_high": paired_mean + critical * paired_sem,
        "t_vs_zero": paired_mean / paired_sem,
    }

    primary = summaries[PRIMARY_BURN_IN]
    sem_comb = math.sqrt(SEM_OLD**2 + primary["sem"]**2)
    old_new = {
        "E_old": E_OLD,
        "SEM_old": SEM_OLD,
        "E_new_minus_E_old": primary["mean"] - E_OLD,
        "SEM_comb": sem_comb,
        "difference_combined_sem_units": (primary["mean"] - E_OLD) / sem_comb,
    }
    analysis_json = {
        "primary": primary,
        "burnin_sensitivity": {str(key): value for key, value in summaries.items()},
        "first_half": first_summary,
        "second_half": second_summary,
        "paired_half_comparison": paired,
        "old_new_compatibility": old_new,
    }
    (RESULTS_DIR / "secVII_energy_longcheck_analysis.json").write_text(
        json.dumps(analysis_json, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Sec. VII Point 5 long-statistics PI-QMC energy check",
        "",
        "Statistical unit: independent chain mean (16 predeclared chains).",
        "",
        "## Per-chain results",
        "",
        "| Seed | Energy mean | Local acc. | Global acc. | Local step (nm) | Global step (nm) | Runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_chain_rows:
        lines.append(
            f"| {row['seed']} | {fmt(row['energy_mean'])} | "
            f"{row['acceptance_local']:.6f} | {row['acceptance_global']:.6f} | "
            f"{row['local_step_nm']:.8f} | {row['global_step_nm']:.8f} | "
            f"{row['runtime_seconds']:.3f} |"
        )
    lines += [
        "", "## Primary result (burn-in 32000)", "",
        f"- Ebar_new: {fmt(primary['mean'], 12)}",
        f"- Between-chain SD: {fmt(primary['between_chain_sd'], 12)}",
        f"- SEM: {fmt(primary['sem'], 12)}",
        f"- 95% Student-t CI (df=15): [{fmt(primary['ci95_low'], 12)}, {fmt(primary['ci95_high'], 12)}]",
        f"- t_star versus {E_STAR:.10f}: {primary['t_star']:+.6f}",
        "", "## Burn-in sensitivity", "",
        "| Burn-in | Mean | SEM | 95% Student-t CI | t_star |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity_rows:
        lines.append(
            f"| {row['burn_in']} | {fmt(row['mean'])} | {fmt(row['sem'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {row['t_star']:+.6f} |"
        )
    lines += [
        "", "## First-half versus second-half stationarity", "",
        "| Seed | First half | Second half | Second - first |",
        "|---:|---:|---:|---:|",
    ]
    for row in per_chain_rows:
        lines.append(
            f"| {row['seed']} | {fmt(row['first_half_mean'])} | "
            f"{fmt(row['second_half_mean'])} | {row['second_minus_first']:+.9f} |"
        )
    lines += [
        "",
        f"Aggregate first-half mean: {fmt(first_summary['mean'], 12)} "
        f"(SEM {fmt(first_summary['sem'], 12)}).",
        f"Aggregate second-half mean: {fmt(second_summary['mean'], 12)} "
        f"(SEM {fmt(second_summary['sem'], 12)}).",
        f"Paired second-minus-first: {paired_mean:+.12f} "
        f"(SEM {paired_sem:.12f}, 95% CI [{paired['ci95_low']:+.12f}, "
        f"{paired['ci95_high']:+.12f}], t={paired['t_vs_zero']:+.6f}).",
        "", "## Old/new compatibility", "",
        f"SEM_comb = {sem_comb:.12f}.",
        f"Ebar_new - E_old = {old_new['E_new_minus_E_old']:+.12f}, "
        f"or {old_new['difference_combined_sem_units']:+.6f} combined-SEM units.",
        "",
        "The deterministic target remains the decisive comparison.",
    ]
    (RESULTS_DIR / "secVII_energy_longcheck_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis_json, indent=2))


if __name__ == "__main__":
    main()
