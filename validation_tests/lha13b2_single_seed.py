import csv
import sys
from pathlib import Path

import numpy as np


# ============================================================
# LHA-13B1
#
# Dimensionless -> physical embedding
# +
# single-seed periodic primitive PI-QMC smoke test
#
# Target:
#   beta* = 2.70
#   P = 64
#
# Exact comparison:
#   LHA-13B0 exact primitive finite-P target.
#
# Physical estimator:
#   physical position marginal = bead marginal,
#   therefore average V over ALL beads.
#
# The centroid is NOT used as the quantum position estimator.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]


# Robust package discovery without modifying repository files.
for candidate in [
    ROOT / "code",
    ROOT / "numerics",
    ROOT,
]:
    if candidate.exists():
        sys.path.insert(
            0,
            str(candidate),
        )


from tmd_pimc import (
    HBAR2_OVER_2M0,
    KB_EV_PER_K,
)

from tmd_pimc.kernels_jit import (
    run_pimc_core_jit_periodic_cell,
)


# ============================================================
# Controlled settings
# ============================================================


BETA_STAR = 2.70

P = 64

MASS_M0 = 1.0

E0_EV = 0.010


NGRID = 512


N_STEPS = 20000
BURN_IN = 4000
SAMPLE_EVERY = 5

SEED = int(__import__("os").environ.get("LHA13_SEED", "130201"))


GLOBAL_MOVE_PROB = 0.20


AMPLITUDES = np.array([
    1.00,
    0.78,
    0.62,
])


PHASES = np.array([
    0.00,
    0.23,
    0.71,
])


# ============================================================
# Dimensionless reciprocal/direct lattice
# ============================================================


sqrt3 = np.sqrt(3.0)


G1 = np.array([
    1.0,
    0.0,
])


G2 = np.array([
    -0.5,
    sqrt3/2.0,
])


G3 = -(G1+G2)


GS = np.array([
    G1,
    G2,
    G3,
])


B_RECIP = np.column_stack([
    G1,
    G2,
])


A_STAR = (
    2.0*np.pi
    * np.linalg.inv(
        B_RECIP.T
    )
)


duality_error = float(
    np.max(
        np.abs(
            B_RECIP.T @ A_STAR
            - 2.0*np.pi*np.eye(2)
        )
    )
)


# ============================================================
# Exact physical embedding
#
# H* = -1/2 nabla_q^2 + V*(q)
#
# r = L0 q
#
# lambda_phys / (E0 L0^2) = 1/2
#
# => L0^2 = 2 lambda_phys / E0
# ============================================================


lambda_nm2_eV = (
    HBAR2_OVER_2M0
    / MASS_M0
)


L0_NM = np.sqrt(
    2.0
    * lambda_nm2_eV
    / E0_EV
)


A_PHYS = (
    L0_NM
    * A_STAR
)


a1 = A_PHYS[:, 0]
a2 = A_PHYS[:, 1]


Ainv = np.linalg.inv(
    A_PHYS
)


cell_length_1 = float(
    np.linalg.norm(a1)
)


cell_length_2 = float(
    np.linalg.norm(a2)
)


cell_area = abs(
    float(
        np.linalg.det(
            A_PHYS
        )
    )
)


# ============================================================
# Temperature mapping
#
# beta* = beta_phys E0
#       = E0 / (kB T)
# ============================================================


T_K = (
    E0_EV
    / (
        KB_EV_PER_K
        * BETA_STAR
    )
)


beta_phys = (
    1.0
    / (
        KB_EV_PER_K
        * T_K
    )
)


tau = (
    beta_phys
    / P
)


beta_star_reconstructed = (
    beta_phys
    * E0_EV
)


# ============================================================
# Primitive-action coefficients
# ============================================================


kpf = (
    1.0
    / (
        4.0
        * lambda_nm2_eV
        * tau
    )
)


# In dimensionless q coordinates:
#
# spring coefficient should be P/(2 beta*)
#
# potential coefficient should be beta*/P.
# ============================================================


spring_coeff_star = (
    kpf
    * L0_NM**2
)


spring_coeff_target = (
    P
    / (
        2.0
        * BETA_STAR
    )
)


potential_coeff_star = (
    tau
    * E0_EV
)


potential_coeff_target = (
    BETA_STAR
    / P
)


spring_mapping_error = abs(
    spring_coeff_star
    - spring_coeff_target
)


potential_mapping_error = abs(
    potential_coeff_star
    - potential_coeff_target
)


beta_mapping_error = abs(
    beta_star_reconstructed
    - BETA_STAR
)


# ============================================================
# Read minimum from A0
# ============================================================


with open(
    "lha13a0_stationary_points.csv",
    newline="",
) as f:

    stationary = list(
        csv.DictReader(f)
    )


minimum = [
    row
    for row in stationary
    if row["type"] == "minimum"
][0]


s_min = float(
    minimum["s"]
)


t_min = float(
    minimum["t"]
)


# ============================================================
# Dimensionless potential
# ============================================================


def Ustar(
    s,
    t,
):

    return (

        - AMPLITUDES[0]
        * np.cos(
            2.0*np.pi*s
            + PHASES[0]
        )

        - AMPLITUDES[1]
        * np.cos(
            2.0*np.pi*t
            + PHASES[1]
        )

        - AMPLITUDES[2]
        * np.cos(
            -2.0*np.pi*(s+t)
            + PHASES[2]
        )
    )


Umin = float(
    Ustar(
        s_min,
        t_min,
    )
)


def Vstar(
    s,
    t,
):

    return (
        Ustar(s, t)
        - Umin
    )


# ============================================================
# Periodic lookup grid used by sampler
#
# IMPORTANT:
# first index = reduced coordinate s
# second index = reduced coordinate t
# ============================================================


axis = (
    np.arange(
        NGRID,
        dtype=float,
    )
    / NGRID
)


S, T = np.meshgrid(
    axis,
    axis,
    indexing="ij",
)


VSTAR_GRID = Vstar(
    S,
    T,
)


VGRID_EV = (
    E0_EV
    * VSTAR_GRID
).astype(
    np.float64
)


# ============================================================
# Vectorized bilinear interpolation matching JIT convention
# ============================================================


def interp_star(
    s,
    t,
):

    s = np.mod(
        np.asarray(s),
        1.0,
    )

    t = np.mod(
        np.asarray(t),
        1.0,
    )


    fs = (
        s*NGRID
    )

    ft = (
        t*NGRID
    )


    i0 = np.floor(
        fs
    ).astype(int)

    j0 = np.floor(
        ft
    ).astype(int)


    ws = (
        fs-i0
    )

    wt = (
        ft-j0
    )


    i1 = (
        i0+1
    ) % NGRID

    j1 = (
        j0+1
    ) % NGRID


    i0 %= NGRID
    j0 %= NGRID


    return (

        (1.0-ws)
        * (1.0-wt)
        * VSTAR_GRID[
            i0,
            j0
        ]

        + ws
        * (1.0-wt)
        * VSTAR_GRID[
            i1,
            j0
        ]

        + (1.0-ws)
        * wt
        * VSTAR_GRID[
            i0,
            j1
        ]

        + ws
        * wt
        * VSTAR_GRID[
            i1,
            j1
        ]
    )


# ============================================================
# Independent interpolation audit
# ============================================================


rng_audit = np.random.default_rng(
    130211
)


s_test = rng_audit.random(
    20000
)


t_test = rng_audit.random(
    20000
)


interp_error = np.abs(
    interp_star(
        s_test,
        t_test,
    )
    -
    Vstar(
        s_test,
        t_test,
    )
)


interp_max_error_star = float(
    np.max(
        interp_error
    )
)


interp_rms_error_star = float(
    np.sqrt(
        np.mean(
            interp_error**2
        )
    )
)


# ============================================================
# Monte Carlo proposal scales
# ============================================================


# Free-link standard deviation:
#
# sigma^2 = 2 lambda tau
# ============================================================


sigma_link_nm = np.sqrt(
    2.0
    * lambda_nm2_eV
    * tau
)


LOCAL_STEP_NM = (
    0.50
    * sigma_link_nm
)


# Same useful scale as the validated LHA-12 periodic bridge:
# roughly one quarter of a primitive lattice vector.
GLOBAL_STEP_NM = (
    0.25
    * min(
        cell_length_1,
        cell_length_2,
    )
)


# ============================================================
# Initial path near exact minimum
# ============================================================


center_nm = (
    A_PHYS
    @ np.array([
        s_min,
        t_min,
    ])
)


rng_init = np.random.default_rng(
    SEED+991
)


path = (
    center_nm[None, :]
    +
    0.05
    * sigma_link_nm
    * rng_init.standard_normal(
        (
            P,
            2,
        )
    )
).astype(
    np.float64
)


# ============================================================
# Exact finite-P target from B0
# ============================================================


with open(
    "lha13b0_exact_primitive_finiteP.csv",
    newline="",
) as f:

    b0_rows = list(
        csv.DictReader(f)
    )


matches = [
    row
    for row in b0_rows
    if (
        abs(
            float(row["beta"])
            - BETA_STAR
        ) < 1.0e-12

        and

        int(row["P"]) == P
    )
]


if len(matches) != 1:

    raise RuntimeError(
        "Could not identify unique exact finite-P target."
    )


target = matches[0]


V_EXACT_FINITEP = float(
    target[
        "V_exact_finiteP"
    ]
)


V_CONTINUUM = float(
    target[
        "V_continuum_BZ"
    ]
)


TROTTER_BIAS_PCT = float(
    target[
        "finiteP_bias_pct"
    ]
)


# ============================================================
# Pre-run report
# ============================================================


print(
    "=== LHA-13B1 "
    "PHYSICAL EMBEDDING + "
    "PERIODIC PI-QMC SMOKE ==="
)

print()

print(
    "=== PHYSICAL EMBEDDING ==="
)

print(
    f"beta*                   = "
    f"{BETA_STAR:.12f}"
)

print(
    f"E0                      = "
    f"{E0_EV:.12f} eV"
)

print(
    f"mass                    = "
    f"{MASS_M0:.8f} m0"
)

print(
    f"L0                      = "
    f"{L0_NM:.12f} nm"
)

print(
    f"T                       = "
    f"{T_K:.9f} K"
)

print(
    f"P                       = "
    f"{P}"
)

print()

print(
    f"|a1|                    = "
    f"{cell_length_1:.9f} nm"
)

print(
    f"|a2|                    = "
    f"{cell_length_2:.9f} nm"
)

print(
    f"cell area               = "
    f"{cell_area:.9f} nm^2"
)

print()

print(
    "=== ACTION MAPPING AUDIT ==="
)

print(
    f"lattice duality error   = "
    f"{duality_error:.6e}"
)

print(
    f"beta* mapping error     = "
    f"{beta_mapping_error:.6e}"
)

print(
    f"spring coefficient      = "
    f"{spring_coeff_star:.12f}"
)

print(
    f"spring target           = "
    f"{spring_coeff_target:.12f}"
)

print(
    f"spring mapping error    = "
    f"{spring_mapping_error:.6e}"
)

print(
    f"potential coefficient   = "
    f"{potential_coeff_star:.12f}"
)

print(
    f"potential target        = "
    f"{potential_coeff_target:.12f}"
)

print(
    f"potential mapping error = "
    f"{potential_mapping_error:.6e}"
)

print()

print(
    "=== PERIODIC LOOKUP AUDIT ==="
)

print(
    f"grid                    = "
    f"{NGRID} x {NGRID}"
)

print(
    f"max interpolation error = "
    f"{interp_max_error_star:.6e} dimensionless"
)

print(
    f"rms interpolation error = "
    f"{interp_rms_error_star:.6e} dimensionless"
)

print()

print(
    "=== SAMPLER SETTINGS ==="
)

print(
    f"sigma_link              = "
    f"{sigma_link_nm:.8f} nm"
)

print(
    f"local step              = "
    f"{LOCAL_STEP_NM:.8f} nm"
)

print(
    f"global step             = "
    f"{GLOBAL_STEP_NM:.8f} nm"
)

print(
    f"global move probability = "
    f"{GLOBAL_MOVE_PROB:.3f}"
)

print(
    f"steps / burn / every    = "
    f"{N_STEPS} / "
    f"{BURN_IN} / "
    f"{SAMPLE_EVERY}"
)

print(
    f"seed                    = "
    f"{SEED}"
)

print()

print(
    "=== EXACT TARGET ==="
)

print(
    f"V continuum BZ          = "
    f"{V_CONTINUUM:.12f}"
)

print(
    f"V exact finite-P        = "
    f"{V_EXACT_FINITEP:.12f}"
)

print(
    f"finite-P bias           = "
    f"{TROTTER_BIAS_PCT:+.8f}%"
)

print()


# ============================================================
# RUN
# ============================================================


samples, acc_local, acc_global = (
    run_pimc_core_jit_periodic_cell(

        n_steps=N_STEPS,

        burn_in=BURN_IN,

        sample_every=SAMPLE_EVERY,

        p_beads=P,

        path=path,

        v_grid=VGRID_EV,

        origin_x=0.0,
        origin_y=0.0,

        ainv00=float(
            Ainv[0, 0]
        ),

        ainv01=float(
            Ainv[0, 1]
        ),

        ainv10=float(
            Ainv[1, 0]
        ),

        ainv11=float(
            Ainv[1, 1]
        ),

        kpf=float(
            kpf
        ),

        tau=float(
            tau
        ),

        local_step_nm=float(
            LOCAL_STEP_NM
        ),

        global_step_nm=float(
            GLOBAL_STEP_NM
        ),

        global_move_prob=float(
            GLOBAL_MOVE_PROB
        ),

        seed=int(
            SEED
        ),
    )
)


samples = np.asarray(
    samples,
    dtype=float,
)


# ============================================================
# Physical single-bead position marginal
#
# Wrap each bead into primary reduced cell only for evaluating
# cell-periodic observables.
# ============================================================


flat = samples.reshape(
    -1,
    2,
)


uv = (
    Ainv
    @ flat.T
).T


uv = np.mod(
    uv,
    1.0,
)


s_all = uv[:, 0]
t_all = uv[:, 1]


# Exact analytic potential estimator.
Vstar_exact_flat = Vstar(
    s_all,
    t_all,
)


# Potential actually sampled by lookup table.
Vstar_grid_flat = interp_star(
    s_all,
    t_all,
)


n_samples = samples.shape[0]


Vstar_exact_ts = (
    Vstar_exact_flat
    .reshape(
        n_samples,
        P,
    )
    .mean(
        axis=1
    )
)


Vstar_grid_ts = (
    Vstar_grid_flat
    .reshape(
        n_samples,
        P,
    )
    .mean(
        axis=1
    )
)


V_PIMC = float(
    np.mean(
        Vstar_exact_ts
    )
)


V_PIMC_GRID = float(
    np.mean(
        Vstar_grid_ts
    )
)


grid_estimator_delta = (
    V_PIMC
    - V_PIMC_GRID
)


V_ERROR_PCT = (
    100.0
    * (
        V_PIMC
        - V_EXACT_FINITEP
    )
    / V_EXACT_FINITEP
)


# ============================================================
# Contiguous block SEM
#
# Blocking is performed on one value per saved sweep:
# average over all beads first, then block in MC time.
#
# Beads are NOT treated as independent measurements.
# ============================================================


def block_sem(
    x,
    n_blocks=20,
):

    x = np.asarray(
        x,
        dtype=float,
    )


    nb = min(
        n_blocks,
        len(x)
    )


    block_len = (
        len(x)//nb
    )


    if (
        nb < 2
        or block_len < 2
    ):

        return (
            np.nan,
            np.nan,
            0,
            0,
        )


    used = (
        nb*block_len
    )


    block_means = (
        x[:used]
        .reshape(
            nb,
            block_len,
        )
        .mean(
            axis=1
        )
    )


    sem = float(
        np.std(
            block_means,
            ddof=1,
        )
        / np.sqrt(nb)
    )


    return (
        sem,
        float(
            np.std(
                block_means,
                ddof=1,
            )
        ),
        nb,
        block_len,
    )


SEM, block_sd, n_blocks, block_len = (
    block_sem(
        Vstar_exact_ts,
        n_blocks=20,
    )
)


if (
    np.isfinite(SEM)
    and SEM > 0.0
):

    z_score = (
        V_PIMC
        - V_EXACT_FINITEP
    ) / SEM

else:

    z_score = np.nan


# ============================================================
# Half-vs-half diagnostic
# ============================================================


mid = (
    len(
        Vstar_exact_ts
    ) // 2
)


V_first = float(
    np.mean(
        Vstar_exact_ts[:mid]
    )
)


V_second = float(
    np.mean(
        Vstar_exact_ts[mid:]
    )
)


half_difference = (
    V_second
    - V_first
)


# ============================================================
# Report
# ============================================================


print(
    "=== PI-QMC SMOKE RESULT ==="
)

print(
    f"samples                 = "
    f"{samples.shape}"
)

print(
    f"acceptance local        = "
    f"{acc_local:.6f}"
)

print(
    f"acceptance global       = "
    f"{acc_global:.6f}"
)

print()

print(
    f"V exact finite-P        = "
    f"{V_EXACT_FINITEP:.12f}"
)

print(
    f"V PI-QMC analytic est.  = "
    f"{V_PIMC:.12f}"
)

print(
    f"V PI-QMC lookup est.    = "
    f"{V_PIMC_GRID:.12f}"
)

print(
    f"analytic-grid estimator "
    f"delta                   = "
    f"{grid_estimator_delta:+.6e}"
)

print()

print(
    f"PI-QMC error            = "
    f"{V_ERROR_PCT:+.6f}%"
)

print(
    f"block SEM               = "
    f"{SEM:.8f}"
)

print(
    f"z vs exact finite-P     = "
    f"{z_score:+.5f}"
)

print()

print(
    f"blocks                  = "
    f"{n_blocks}"
)

print(
    f"samples per block       = "
    f"{block_len}"
)

print(
    f"first-half V            = "
    f"{V_first:.9f}"
)

print(
    f"second-half V           = "
    f"{V_second:.9f}"
)

print(
    f"half difference         = "
    f"{half_difference:+.9f}"
)

print()


# ============================================================
# Structural audit
# ============================================================


checks = {

    "LATTICE DUALITY":
        duality_error
        < 1.0e-12,

    "BETA EMBEDDING":
        beta_mapping_error
        < 1.0e-12,

    "SPRING ACTION EMBEDDING":
        spring_mapping_error
        < 1.0e-12,

    "POTENTIAL ACTION EMBEDDING":
        potential_mapping_error
        < 1.0e-12,

    "PERIODIC GRID INTERPOLATION":
        interp_max_error_star
        < 1.0e-4,

    "FINITE SAMPLES":
        bool(
            np.all(
                np.isfinite(
                    samples
                )
            )
        ),

    "LOCAL ACCEPTANCE HEALTHY":
        (
            acc_local > 0.10
            and acc_local < 0.85
        ),

    "GLOBAL ACCEPTANCE HEALTHY":
        (
            np.isfinite(
                acc_global
            )
            and acc_global > 0.02
            and acc_global < 0.95
        ),

    "GRID/ANALYTIC ESTIMATOR":
        abs(
            grid_estimator_delta
        )
        < 5.0e-4,

    # Smoke criterion only, not publication-level inference.
    "SINGLE-SEED MC COMPATIBILITY":
        (
            np.isfinite(
                z_score
            )
            and abs(
                z_score
            ) < 4.0
        ),
}


print(
    "=== STRUCTURAL AUDIT ==="
)


for name, passed in checks.items():

    print(
        f"{name:32s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


overall = all(
    checks.values()
)


print()

print(
    "STRUCTURAL OVERALL:",
    "PASS"
    if overall
    else "FAIL"
)


print()

print(
    "NOTE:"
)

print(
    "The single-seed z score is only a smoke-test "
    "diagnostic. Publication-level statistical "
    "validation requires the next multiseed checkpoint."
)


# ============================================================
# Save
# ============================================================


summary = {

    "beta_star":
        BETA_STAR,

    "temperature_K":
        T_K,

    "P":
        P,

    "E0_eV":
        E0_EV,

    "L0_nm":
        L0_NM,

    "cell_length_1_nm":
        cell_length_1,

    "cell_length_2_nm":
        cell_length_2,

    "exact_finiteP_V":
        V_EXACT_FINITEP,

    "continuum_V":
        V_CONTINUUM,

    "finiteP_bias_pct":
        TROTTER_BIAS_PCT,

    "pimc_V":
        V_PIMC,

    "pimc_V_lookup":
        V_PIMC_GRID,

    "pimc_error_pct":
        V_ERROR_PCT,

    "block_sem":
        SEM,

    "z_score":
        z_score,

    "acceptance_local":
        acc_local,

    "acceptance_global":
        acc_global,

    "interp_max_error_star":
        interp_max_error_star,

    "interp_rms_error_star":
        interp_rms_error_star,

    "grid_estimator_delta":
        grid_estimator_delta,

    "half_difference":
        half_difference,

    "n_samples":
        n_samples,

    "seed":
        SEED,
}


with open(
    f"lha13b2_seed_{SEED}.csv",
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            summary.keys()
        ),
    )

    w.writeheader()

    w.writerow(
        summary
    )


np.savez_compressed(

    f"lha13b2_seed_{SEED}.npz",

    Vstar_time_series=
        Vstar_exact_ts,

    Vstar_lookup_time_series=
        Vstar_grid_ts,

    beta_star=BETA_STAR,

    temperature_K=T_K,

    P=P,

    exact_finiteP_V=
        V_EXACT_FINITEP,

    acceptance_local=
        acc_local,

    acceptance_global=
        acc_global,
)


print()

print(
    "saved: "
    f"lha13b2_seed_{SEED}.csv"
)

print(
    "saved: "
    f"lha13b2_seed_{SEED}.npz"
)
