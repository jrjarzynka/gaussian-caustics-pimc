import csv
import glob
import numpy as np


# ============================================================
# LHA-P0B
#
# Final publication-grade statistical closure.
#
# beta* = 2.70
# P = 64
#
# Independent statistical units:
#     16 independent PI-QMC chains.
#
# Beads are pooled only for density estimation.
# They are NEVER treated as independent observations.
# ============================================================


N_EXPECTED = 16

BOOTSTRAP_REPS = 20000

BOOTSTRAP_SEED = 20260829


# ============================================================
# Helpers
# ============================================================


def L1(A, B):

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    return float(
        np.mean(
            np.abs(A-B)
        )
    )


def normalize_hist(H):

    H = np.asarray(H, dtype=float)

    m = float(
        np.mean(H)
    )

    if m <= 0.0:
        raise RuntimeError(
            "Histogram has zero mean."
        )

    return H/m


# ============================================================
# Read publication energy files
# ============================================================


csv_files = sorted(
    glob.glob(
        "lhap0a_seed_*.csv"
    )
)


hist_files = sorted(
    glob.glob(
        "lhap0a_seed_*_hist.npz"
    )
)


if len(csv_files) != N_EXPECTED:
    raise RuntimeError(
        f"Expected {N_EXPECTED} CSV files, "
        f"found {len(csv_files)}"
    )


if len(hist_files) != N_EXPECTED:
    raise RuntimeError(
        f"Expected {N_EXPECTED} histogram files, "
        f"found {len(hist_files)}"
    )


rows = []


for fn in csv_files:

    with open(fn, newline="") as f:

        row = next(
            csv.DictReader(f)
        )

    rows.append(row)


rows.sort(
    key=lambda r: int(r["seed"])
)


seeds = np.array([
    int(r["seed"])
    for r in rows
])


V = np.array([
    float(r["pimc_V"])
    for r in rows
])


acc_local = np.array([
    float(r["acceptance_local"])
    for r in rows
])


acc_global = np.array([
    float(r["acceptance_global"])
    for r in rows
])


target_P64 = float(
    rows[0]["exact_finiteP_V"]
)


target_cont = float(
    rows[0]["continuum_V"]
)


finiteP_bias_pct = float(
    rows[0]["finiteP_bias_pct"]
)


# ============================================================
# Energy statistics
# ============================================================


V_mean = float(
    np.mean(V)
)


V_sd = float(
    np.std(
        V,
        ddof=1,
    )
)


V_sem = float(
    V_sd/np.sqrt(len(V))
)


V_z = float(
    (V_mean-target_P64)
    / V_sem
)


V_error_pct = float(
    100.0
    * (
        V_mean-target_P64
    )
    / target_P64
)


seed_standardized = (
    (V-V_mean)/V_sd
)


max_seed_standardized = float(
    np.max(
        np.abs(
            seed_standardized
        )
    )
)


max_seed_index = int(
    np.argmax(
        np.abs(
            seed_standardized
        )
    )
)


max_seed = int(
    seeds[
        max_seed_index
    ]
)


# ============================================================
# Leave-one-seed-out audit
# ============================================================


loo_means = []
loo_z = []


for i in range(
    len(V)
):

    keep = np.ones(
        len(V),
        dtype=bool,
    )

    keep[i] = False


    x = V[keep]


    m = float(
        np.mean(x)
    )


    sd = float(
        np.std(
            x,
            ddof=1,
        )
    )


    sem = float(
        sd/np.sqrt(len(x))
    )


    z = float(
        (m-target_P64)
        / sem
    )


    loo_means.append(m)
    loo_z.append(z)


loo_means = np.asarray(
    loo_means
)


loo_z = np.asarray(
    loo_z
)


# ============================================================
# Independent replication:
# previous B2 ensemble if available
# ============================================================


old_replication_available = False

old_mean = np.nan
old_sem = np.nan
old_z = np.nan
new_old_difference = np.nan
new_old_difference_z = np.nan


old_summary = (
    "lha13b2_multiseed_summary.csv"
)


try:

    with open(
        old_summary,
        newline="",
    ) as f:

        old = next(
            csv.DictReader(f)
        )


    old_mean = float(
        old["pimc_mean"]
    )


    old_sem = float(
        old["between_seed_sem"]
    )


    old_z = float(
        old["z_score"]
    )


    new_old_difference = (
        V_mean-old_mean
    )


    new_old_difference_sem = (
        np.sqrt(
            V_sem**2
            + old_sem**2
        )
    )


    new_old_difference_z = (
        new_old_difference
        / new_old_difference_sem
    )


    old_replication_available = True


except FileNotFoundError:

    pass


# ============================================================
# Read histogram files and match seeds
# ============================================================


hist_by_seed = {}


for fn in hist_files:

    d = np.load(fn)


    seed = int(
        d["seed"]
    )


    H_full = np.asarray(
        d["H_full_ts"],
        dtype=float,
    )


    H_first = np.asarray(
        d["H_first_ts"],
        dtype=float,
    )


    H_second = np.asarray(
        d["H_second_ts"],
        dtype=float,
    )


    if not np.array_equal(
        H_full,
        H_first+H_second,
    ):
        raise RuntimeError(
            f"Histogram halves do not "
            f"reconstruct full histogram "
            f"for seed {seed}."
        )


    hist_by_seed[seed] = (
        H_full,
        H_first,
        H_second,
    )


if set(hist_by_seed) != set(seeds):

    raise RuntimeError(
        "CSV and histogram seed sets differ."
    )


H_seed = np.stack([
    hist_by_seed[
        int(seed)
    ][0]
    for seed in seeds
])


H_first_seed = np.stack([
    hist_by_seed[
        int(seed)
    ][1]
    for seed in seeds
])


H_second_seed = np.stack([
    hist_by_seed[
        int(seed)
    ][2]
    for seed in seeds
])


pooled_beads = int(
    np.sum(H_seed)
)


# ============================================================
# Exact + RLH targets
# ============================================================


exact = np.load(
    "lha13b3a_exact_finiteP_registered_density.npz"
)


harmonized = np.load(
    "lha13b3c_harmonized_density.npz"
)


P_exact_P64 = np.asarray(
    exact[
        "P_finiteP_bin_ts"
    ],
    dtype=float,
)


P_exact_cont = np.asarray(
    exact[
        "P_continuum_bin_ts"
    ],
    dtype=float,
)


P_RLH = np.asarray(
    harmonized[
        "P_RLH_bin_ts"
    ],
    dtype=float,
)


if (
    H_seed.shape[1:]
    != P_exact_P64.shape
):
    raise RuntimeError(
        "Histogram and exact-density "
        "shapes differ."
    )


# ============================================================
# Final pooled density
# ============================================================


H_pool = np.sum(
    H_seed,
    axis=0,
)


P_PIMC = normalize_hist(
    H_pool
)


L1_PIMC_P64 = L1(
    P_PIMC,
    P_exact_P64,
)


L1_PIMC_cont = L1(
    P_PIMC,
    P_exact_cont,
)


L1_RLH_P64 = L1(
    P_RLH,
    P_exact_P64,
)


L1_RLH_cont = L1(
    P_RLH,
    P_exact_cont,
)


L1_P64_cont = L1(
    P_exact_P64,
    P_exact_cont,
)


# ============================================================
# Temporal half diagnostic
# ============================================================


P_time_first = normalize_hist(
    np.sum(
        H_first_seed,
        axis=0,
    )
)


P_time_second = normalize_hist(
    np.sum(
        H_second_seed,
        axis=0,
    )
)


L1_time_halves = L1(
    P_time_first,
    P_time_second,
)


# ============================================================
# Independent-chain half diagnostic
#
# Preserve the same convention used previously:
# first half of ordered seeds vs second half.
# ============================================================


half = (
    len(seeds)//2
)


P_seed_first = normalize_hist(
    np.sum(
        H_seed[:half],
        axis=0,
    )
)


P_seed_second = normalize_hist(
    np.sum(
        H_seed[half:],
        axis=0,
    )
)


L1_seed_halves = L1(
    P_seed_first,
    P_seed_second,
)


# Auxiliary odd/even split.
# This is reported but is NOT used as the primary gate.


P_seed_odd = normalize_hist(
    np.sum(
        H_seed[::2],
        axis=0,
    )
)


P_seed_even = normalize_hist(
    np.sum(
        H_seed[1::2],
        axis=0,
    )
)


L1_seed_oddeven = L1(
    P_seed_odd,
    P_seed_even,
)


# ============================================================
# Per-seed density diagnostics
# ============================================================


per_seed_L1 = []


for i, seed in enumerate(
    seeds
):

    P_i = normalize_hist(
        H_seed[i]
    )


    per_seed_L1.append(
        L1(
            P_i,
            P_exact_P64,
        )
    )


per_seed_L1 = np.asarray(
    per_seed_L1
)


# ============================================================
# Independent-chain bootstrap
#
# Resample whole chains with replacement.
# Never resample individual beads.
# ============================================================


rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


boot_V_mean = np.empty(
    BOOTSTRAP_REPS,
    dtype=float,
)


boot_density_L1 = np.empty(
    BOOTSTRAP_REPS,
    dtype=float,
)


nseed = len(seeds)


for b in range(
    BOOTSTRAP_REPS
):

    idx = rng.integers(
        0,
        nseed,
        size=nseed,
    )


    boot_V_mean[b] = float(
        np.mean(
            V[idx]
        )
    )


    H_b = np.sum(
        H_seed[idx],
        axis=0,
    )


    P_b = normalize_hist(
        H_b
    )


    boot_density_L1[b] = L1(
        P_b,
        P_exact_P64,
    )


V_boot_q025, \
V_boot_median, \
V_boot_q975 = np.quantile(
    boot_V_mean,
    [
        0.025,
        0.5,
        0.975,
    ],
)


D_boot_q025, \
D_boot_median, \
D_boot_q975 = np.quantile(
    boot_density_L1,
    [
        0.025,
        0.5,
        0.975,
    ],
)


RLH_minus_boot_upper = float(
    L1_RLH_P64
    - D_boot_q975
)


# ============================================================
# Ratios
# ============================================================


RLH_over_PIMC = float(
    L1_RLH_P64
    / L1_PIMC_P64
)


RLH_over_time = float(
    L1_RLH_P64
    / L1_time_halves
)


RLH_over_seed = float(
    L1_RLH_P64
    / L1_seed_halves
)


PIMC_over_time = float(
    L1_PIMC_P64
    / L1_time_halves
)


PIMC_over_seed = float(
    L1_PIMC_P64
    / L1_seed_halves
)


Trotter_density_over_RLH = float(
    L1_P64_cont
    / L1_RLH_P64
)


# ============================================================
# Report
# ============================================================


print(
    "=== LHA-P0B FINAL PUBLICATION STATISTICS ==="
)

print()


print(
    "=== ENERGY ==="
)

print(
    f"N independent seeds         = "
    f"{len(V)}"
)

print(
    f"exact finite-P target       = "
    f"{target_P64:.12f}"
)

print(
    f"PIMC mean                   = "
    f"{V_mean:.12f}"
)

print(
    f"between-seed SD             = "
    f"{V_sd:.9f}"
)

print(
    f"between-seed SEM            = "
    f"{V_sem:.9f}"
)

print(
    f"error vs exact finite-P     = "
    f"{V_error_pct:+.6f}%"
)

print(
    f"z vs exact finite-P         = "
    f"{V_z:+.6f}"
)

print(
    f"P64 Trotter bias            = "
    f"{finiteP_bias_pct:+.8f}%"
)

print()

print(
    f"bootstrap mean 95% interval = "
    f"[{V_boot_q025:.9f}, "
    f"{V_boot_q975:.9f}]"
)

print()

print(
    f"largest standardized seed   = "
    f"{max_seed} "
    f"({max_seed_standardized:.4f} SD)"
)

print(
    f"leave-one-out z range       = "
    f"{np.min(loo_z):+.4f} .. "
    f"{np.max(loo_z):+.4f}"
)


if old_replication_available:

    print()

    print(
        "=== INDEPENDENT EARLIER B2 REPLICATION ==="
    )

    print(
        f"B2 mean                     = "
        f"{old_mean:.9f}"
    )

    print(
        f"B2 SEM                      = "
        f"{old_sem:.9f}"
    )

    print(
        f"B2 z                        = "
        f"{old_z:+.6f}"
    )

    print(
        f"new - B2 mean difference    = "
        f"{new_old_difference:+.9f}"
    )

    print(
        f"difference / combined SEM   = "
        f"{new_old_difference_z:+.6f}"
    )


print()

print(
    "=== DENSITY ==="
)

print(
    f"pooled bead entries         = "
    f"{pooled_beads}"
)

print()

print(
    f"exact P64-continuum L1      = "
    f"{L1_P64_cont:.9e}"
)

print()

print(
    f"PIMC vs exact P64 L1        = "
    f"{L1_PIMC_P64:.9f}"
)

print(
    f"RLH  vs exact P64 L1        = "
    f"{L1_RLH_P64:.9f}"
)

print()

print(
    f"PIMC vs continuum L1        = "
    f"{L1_PIMC_cont:.9f}"
)

print(
    f"RLH  vs continuum L1        = "
    f"{L1_RLH_cont:.9f}"
)

print()

print(
    "=== INTERNAL SAMPLING SCALE ==="
)

print(
    f"time-half L1                = "
    f"{L1_time_halves:.9f}"
)

print(
    f"seed-half L1                = "
    f"{L1_seed_halves:.9f}"
)

print(
    f"odd/even seed L1 [aux]      = "
    f"{L1_seed_oddeven:.9f}"
)

print()

print(
    f"PIMC / time-half            = "
    f"{PIMC_over_time:.6f}"
)

print(
    f"PIMC / seed-half            = "
    f"{PIMC_over_seed:.6f}"
)

print()

print(
    f"RLH / time-half             = "
    f"{RLH_over_time:.6f}"
)

print(
    f"RLH / seed-half             = "
    f"{RLH_over_seed:.6f}"
)

print(
    f"RLH / PIMC exact-L1         = "
    f"{RLH_over_PIMC:.6f}"
)

print()

print(
    "=== WHOLE-CHAIN BOOTSTRAP ==="
)

print(
    f"PIMC density L1 median      = "
    f"{D_boot_median:.9f}"
)

print(
    f"PIMC density L1 95% range   = "
    f"[{D_boot_q025:.9f}, "
    f"{D_boot_q975:.9f}]"
)

print(
    f"RLH L1 - bootstrap upper    = "
    f"{RLH_minus_boot_upper:+.9f}"
)

print()

print(
    f"per-seed density L1 range   = "
    f"{np.min(per_seed_L1):.9f} .. "
    f"{np.max(per_seed_L1):.9f}"
)


# ============================================================
# Final gates
# ============================================================


checks = {

    "16 INDEPENDENT CHAINS":
        len(V) == 16,

    "LOCAL ACCEPTANCE STABLE":
        bool(
            np.all(
                (acc_local > 0.10)
                &
                (acc_local < 0.85)
            )
        ),

    "GLOBAL ACCEPTANCE STABLE":
        bool(
            np.all(
                (acc_global > 0.02)
                &
                (acc_global < 0.95)
            )
        ),

    "NO >3.5 SD SEED":
        max_seed_standardized
        < 3.5,

    "ENERGY COMPATIBLE |z|<3":
        abs(V_z)
        < 3.0,

    "PIMC BELOW TIME-HALF SCALE":
        L1_PIMC_P64
        < L1_time_halves,

    "PIMC BELOW SEED-HALF SCALE":
        L1_PIMC_P64
        < L1_seed_halves,

    "RLH ABOVE TIME-HALF SCALE":
        L1_RLH_P64
        > L1_time_halves,

    "RLH ABOVE SEED-HALF SCALE":
        L1_RLH_P64
        > L1_seed_halves,

    "RLH ABOVE BOOTSTRAP 97.5%":
        L1_RLH_P64
        > D_boot_q975,

    "TROTTER DENSITY NEGLIGIBLE":
        Trotter_density_over_RLH
        < 0.01,
}


print()

print(
    "=== FINAL STRUCTURAL / STATISTICAL AUDIT ==="
)


for name, passed in checks.items():

    print(
        f"{name:36s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


overall = all(
    checks.values()
)


print()

print(
    "FINAL OVERALL:",
    "PASS"
    if overall
    else "NOT OK"
)


# ============================================================
# Save summary
# ============================================================


with open(
    "lhap0b_final_publication_statistics.csv",
    "w",
    newline="",
) as f:

    w = csv.writer(f)


    w.writerow([
        "n_seeds",
        "pooled_beads",

        "exact_finiteP_V",
        "pimc_V_mean",
        "pimc_V_sd",
        "pimc_V_sem",
        "pimc_V_error_pct",
        "pimc_V_z",

        "V_boot_q025",
        "V_boot_median",
        "V_boot_q975",

        "max_seed_standardized",
        "loo_z_min",
        "loo_z_max",

        "old_B2_available",
        "old_B2_mean",
        "old_B2_sem",
        "old_B2_z",
        "new_old_difference_z",

        "exact_P64_vs_continuum_L1",

        "PIMC_vs_exact_P64_L1",
        "RLH_vs_exact_P64_L1",

        "PIMC_vs_continuum_L1",
        "RLH_vs_continuum_L1",

        "time_half_L1",
        "seed_half_L1",
        "seed_oddeven_L1",

        "PIMC_over_time_half",
        "PIMC_over_seed_half",

        "RLH_over_time_half",
        "RLH_over_seed_half",

        "RLH_over_PIMC_L1",

        "density_boot_q025",
        "density_boot_median",
        "density_boot_q975",

        "RLH_minus_boot_upper",

        "final_overall",
    ])


    w.writerow([
        len(V),
        pooled_beads,

        target_P64,
        V_mean,
        V_sd,
        V_sem,
        V_error_pct,
        V_z,

        V_boot_q025,
        V_boot_median,
        V_boot_q975,

        max_seed_standardized,
        np.min(loo_z),
        np.max(loo_z),

        old_replication_available,
        old_mean,
        old_sem,
        old_z,
        new_old_difference_z,

        L1_P64_cont,

        L1_PIMC_P64,
        L1_RLH_P64,

        L1_PIMC_cont,
        L1_RLH_cont,

        L1_time_halves,
        L1_seed_halves,
        L1_seed_oddeven,

        PIMC_over_time,
        PIMC_over_seed,

        RLH_over_time,
        RLH_over_seed,

        RLH_over_PIMC,

        D_boot_q025,
        D_boot_median,
        D_boot_q975,

        RLH_minus_boot_upper,

        "PASS"
        if overall
        else "NOT OK",
    ])


np.savez_compressed(

    "lhap0b_final_publication_statistics.npz",

    seeds=
        seeds,

    V_seed=
        V,

    P_PIMC_bin_ts=
        P_PIMC,

    P_exact_P64_bin_ts=
        P_exact_P64,

    P_exact_continuum_bin_ts=
        P_exact_cont,

    P_RLH_bin_ts=
        P_RLH,

    P_time_first_bin_ts=
        P_time_first,

    P_time_second_bin_ts=
        P_time_second,

    P_seed_first_bin_ts=
        P_seed_first,

    P_seed_second_bin_ts=
        P_seed_second,

    bootstrap_V_mean=
        boot_V_mean,

    bootstrap_density_L1=
        boot_density_L1,
)


print()

print(
    "saved: "
    "lhap0b_final_publication_statistics.csv"
)

print(
    "saved: "
    "lhap0b_final_publication_statistics.npz"
)
