"""
Collapse test: can any local scalar predict the pre-caustic RLH error?

Companion analysis for "Before the Caustic".

Question
--------
The manuscript shows that zeta certifies existence but not accuracy.  A
natural follow-up is whether some equally cheap local scalar -- one built
from the same Hessian and the same smearing width the construction already
assigns -- does control the error.  If such a quantity existed, the two
benchmarks would collapse onto a single error curve when plotted against
it, even though they do not collapse against zeta.

Candidates tested (all RLH-weighted averages over the primitive cell, all
computable without any exact quantum reference):

    sigma_a^2 = -F_beta(kappa_a) / beta          smearing width, Eq. (D1)
    e3        = beta |T_aaa| sigma_a^3 / 6       cubic action scale
    e4        = beta |T_aaaa| sigma_a^4 / 24     quartic action scale
    e34       = e3 + e4
    anh       = beta ( T_aaaa sigma^4/8 - beta T_aaa^2 sigma^6/24 )
                                                 leading anharmonic
                                                 free-energy correction

Result
------
None of them collapses the benchmarks.  At matched diagnostic value the
triangular and asymmetric errors still differ by the same factor as at
matched zeta (~1.9 near the caustic), and none reproduces the sign change
at intermediate beta.  This is reported as a negative result: it shows the
accuracy/existence separation is not repaired by a better local scalar.

Inputs (from the validation tree)
---------------------------------
    lha11b3_exact_qm_vs_tensor_rlh.csv            triangular
    lha13a2_tensor_rlh_offstationary_caustic.csv  asymmetric

Author: J. R. Jarzynka
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import caustic_diagnostic as cd

# V0 of the asymmetric benchmark under the convention V_min = 0.
ASYM_V0 = 2.2876040615


def local_diagnostics(pot: cd.FourierPotential2D,
                      beta: float,
                      n: int = 384) -> dict:
    """All candidate local scalars at one inverse temperature."""
    S, T = cd._grid(n)
    r = pot.reduced_to_cartesian(np.stack([S, T], axis=-1))

    kappa, R = np.linalg.eigh(pot.hessian(r))
    e = np.swapaxes(R, -1, -2)

    sigma = np.sqrt(np.clip(
        cd.smearing_variance(kappa, beta, pot.mass, pot.hbar), 0.0, np.inf))

    A = np.asarray(pot.amplitudes)
    u = pot.phase(r)
    proj = np.einsum("...ak,jk->...aj", e, pot.G)
    T3 = np.einsum("...j,...aj->...a", -A * np.sin(u), proj ** 3)
    T4 = np.einsum("...j,...aj->...a", -A * np.cos(u), proj ** 4)

    p = cd.rlh_density(pot, beta, n)
    w = np.where(np.isfinite(p), p, 0.0)
    w = w / w.sum()

    e3 = beta * np.abs(T3) * sigma ** 3 / 6.0
    e4 = beta * np.abs(T4) * sigma ** 4 / 24.0
    anh = beta * (T4 * sigma ** 4 / 8.0
                  - beta * (T3 ** 2) * sigma ** 6 / 24.0)
    zeta = (beta * pot.hbar / np.pi) * np.sqrt(
        np.maximum(0.0, -kappa) / pot.mass)

    wmax = lambda x: float(np.sum(w * np.max(x, axis=-1)))
    wsum = lambda x: float(np.sum(w * np.sum(x, axis=-1)))

    return dict(
        zeta_max=float(np.max(zeta)),
        sigma=wmax(sigma),
        e3=wmax(e3),
        e4=wmax(e4),
        e34=wmax(e3) + wmax(e4),
        anh_signed=wsum(anh),
    )


def build_table(data_dir: Path, n: int = 384) -> pd.DataFrame:
    tri = cd.triangular_benchmark()
    asym = cd.asymmetric_benchmark(V0=ASYM_V0)

    t = pd.read_csv(data_dir / "lha11b3_exact_qm_vs_tensor_rlh.csv")
    a = pd.read_csv(
        data_dir / "lha13a2_tensor_rlh_offstationary_caustic.csv"
    ).dropna(subset=["V_RLH"])

    rows = []
    for _, q in t.iterrows():
        rows.append(dict(bench="triangular", beta=q.beta,
                         err_pct=q.V_error_pct,
                         **local_diagnostics(tri, q.beta, n)))
    for _, q in a.iterrows():
        rows.append(dict(bench="asymmetric", beta=q.beta,
                         err_pct=q.V_error_pct,
                         **local_diagnostics(asym, q.beta, n)))
    return pd.DataFrame(rows)


def collapse_test(df: pd.DataFrame, columns=("zeta_max", "e3", "e34")) -> None:
    """
    Interpolate the asymmetric error onto the triangular diagnostic values.

    A diagnostic that controlled the error would give ratios near unity for
    every row.  Ratios that stay at ~1.9 (and change sign at intermediate
    beta) mean the diagnostic carries no more information than zeta.
    """
    t = df[df.bench == "triangular"]
    a = df[df.bench == "asymmetric"]

    for col in columns:
        print(f"\n-- matched on {col}")
        print(f"{'value':>10} {'err_tri':>9} {'err_asym':>9} {'ratio':>8}")
        for _, row in t.iterrows():
            x = row[col]
            if not (a[col].min() <= x <= a[col].max()):
                continue
            ea = float(np.interp(x, a[col], a.err_pct))
            print(f"{x:10.5f} {row.err_pct:9.3f} {ea:9.3f} "
                  f"{row.err_pct / ea:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="directory holding the two reference CSV files")
    ap.add_argument("--n", type=int, default=384, help="grid resolution")
    ap.add_argument("--out", type=Path, default=Path("collapse_test.csv"))
    args = ap.parse_args()

    df = build_table(args.data_dir, args.n)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:11.5f}"))

    collapse_test(df)

    df.to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
