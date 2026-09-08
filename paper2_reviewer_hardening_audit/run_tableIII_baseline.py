#!/usr/bin/env python3
"""Add the same-beta (unmatched) control to manuscript Table III."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VAL = ROOT / "validation_tests"
sys.path.insert(0, str(VAL))
import caustic_diagnostic as cd  # noqa: E402


BETAS = np.array([0.5, 1.0, 2.0, 2.5])
A = np.array([1.0, 0.78, 0.62])
PHI = np.array([0.0, 0.23, 0.71])
G1 = np.array([1.0, 0.0])
G2 = np.array([-0.5, np.sqrt(3) / 2])
V0 = 2.2876040615


def exact_asymmetric(beta: float, K2max: int = 100, nk: int = 16) -> tuple[float, int]:
    def k2(n1, n2): return n1 * n1 + n2 * n2 - n1 * n2
    R = int(np.ceil(2 * np.sqrt(K2max))) + 2
    basis = [(n1, n2, k2(n1, n2)) for n1 in range(-R, R + 1)
             for n2 in range(-R, R + 1) if k2(n1, n2) <= K2max]
    basis.sort(key=lambda x: (x[2], x[0], x[1]))
    index = {(n1, n2): i for i, (n1, n2, _) in enumerate(basis)}
    nvec = np.array([(n1, n2) for n1, n2, _ in basis])
    Gcart = nvec[:, 0, None] * G1 + nvec[:, 1, None] * G2
    fourier = {
        (0, 0): V0,
        (1, 0): -0.5 * A[0] * np.exp(1j * PHI[0]),
        (-1, 0): -0.5 * A[0] * np.exp(-1j * PHI[0]),
        (0, 1): -0.5 * A[1] * np.exp(1j * PHI[1]),
        (0, -1): -0.5 * A[1] * np.exp(-1j * PHI[1]),
        (-1, -1): -0.5 * A[2] * np.exp(1j * PHI[2]),
        (1, 1): -0.5 * A[2] * np.exp(-1j * PHI[2]),
    }
    Vmat = np.zeros((len(basis), len(basis)), complex)
    for i, (n1, n2, _) in enumerate(basis):
        for (dn1, dn2), coeff in fourier.items():
            j = index.get((n1 - dn1, n2 - dn2))
            if j is not None:
                Vmat[i, j] = coeff
    vals = (np.arange(nk, dtype=float) + 0.5) / nk - 0.5
    spectra = []
    E0 = np.inf
    for ku in vals:
        for kv in vals:
            kg = Gcart + ku * G1 + kv * G2
            H = Vmat.copy()
            H[np.diag_indices_from(H)] += 0.5 * np.sum(kg * kg, axis=1)
            E, C = np.linalg.eigh(H)
            vstate = np.real(np.sum(np.conjugate(C) * (Vmat @ C), axis=0))
            E0 = min(E0, E[0])
            spectra.append((E, vstate))
    Z = 0.0
    Vnum = 0.0
    for E, vstate in spectra:
        w = np.exp(-beta * (E - E0))
        Z += np.sum(w)
        Vnum += np.sum(w * vstate)
    return float(Vnum / Z), len(basis)


def rlh_asymmetric(beta: float, n: int = 512) -> float:
    pot = cd.asymmetric_benchmark(V0=V0)
    S, T = cd._grid(n)
    r = pot.reduced_to_cartesian(np.stack((S, T), axis=-1))
    V = pot.value(r)
    grad = pot.gradient(r)
    kappa, R = np.linalg.eigh(pot.hessian(r))
    Xi = cd.spectral_Xi(kappa, beta, pot.mass, pot.hbar)
    F = cd.spectral_F(kappa, beta, pot.mass, pot.hbar)
    if not np.all(np.isfinite(Xi)):
        raise RuntimeError("RLH undefined at requested beta")
    gmodal = np.einsum("...ia,...i->...a", R, grad)
    logu = 0.5 * np.sum(np.log(Xi), axis=-1) - beta * (V + 0.5 * np.sum(F * gmodal**2, axis=-1))
    u = np.exp(logu - np.max(logu))
    p = u / np.mean(u)
    return float(np.mean(p * V))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f: return list(csv.DictReader(f))


def main() -> None:
    tri = {float(r["beta"]): r for r in load_rows(VAL / "lha11b3_exact_qm_vs_tensor_rlh.csv")}
    asym_existing = {float(r["beta"]): r for r in load_rows(VAL / "lha13a2_tensor_rlh_offstationary_caustic.csv")}
    collapse = load_rows(VAL / "collapse_test.csv")
    tri_diag = [r for r in collapse if r["bench"] == "triangular"]
    asym_diag = [r for r in collapse if r["bench"] == "asymmetric"]

    raw_rows = []
    latex_rows = []
    for beta in BETAS:
        tr = tri[beta]
        if beta in asym_existing:
            ar = asym_existing[beta]
            Va_exact = float(ar["V_exact_BZ"])
            Va_rlh = float(ar["V_RLH"])
        else:
            Va_exact, _ = exact_asymmetric(beta)
            Va_rlh = rlh_asymmetric(beta)
        Vt_exact = float(tr["V_exact"])
        Vt_rlh = float(tr["V_RLH"])
        et = 100 * (Vt_rlh - Vt_exact) / Vt_exact
        ea = 100 * (Va_rlh - Va_exact) / Va_exact
        raw = et / ea
        warning = "opposite signs" if np.sign(et) != np.sign(ea) else ""
        td = next(r for r in tri_diag if float(r["beta"]) == beta)
        matched = {}
        for col in ("zeta_max", "e3", "e34"):
            xx = np.array([float(r[col]) for r in asym_diag])
            yy = np.array([float(r["err_pct"]) for r in asym_diag])
            order = np.argsort(xx)
            den = float(np.interp(float(td[col]), xx[order], yy[order]))
            matched[col] = et / den
        report_matched = {
            key: (np.nan if warning else value) for key, value in matched.items()
        }
        raw_rows.append({
            "beta_triangular": beta,
            "V_exact_triangular": Vt_exact, "V_RLH_triangular": Vt_rlh,
            "signed_relative_error_triangular_pct": et,
            "V_exact_asymmetric": Va_exact, "V_RLH_asymmetric": Va_rlh,
            "signed_relative_error_asymmetric_pct": ea,
            "same_beta_raw_ratio": (np.nan if warning else raw),
            "same_beta_warning": warning,
            "matched_zeta_ratio": report_matched["zeta_max"],
            "matched_epsilon3_ratio": report_matched["e3"],
            "matched_epsilon34_ratio": report_matched["e34"],
        })
        raw_display = "opposite sign" if warning else f"{raw:.2f}"
        matched_display = ["opposite sign" if (beta == 2.0) else f"{matched[x]:.2f}"
                           for x in ("zeta_max", "e3", "e34")]
        latex_rows.append(f"${beta:.1f}$ & {raw_display} & " + " & ".join(matched_display) + r"\\")

    with (HERE / "tableIII_raw_baseline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_rows[0].keys()); w.writeheader(); w.writerows(raw_rows)
    latex = r"""\begin{table}[t]
\caption{Ratio of the triangular to asymmetric signed RLH potential-energy
error.  The same-$\beta$ column is the unmatched control; the remaining
columns interpolate the asymmetric curve to matched diagnostic values.
``Opposite sign'' is reported instead of interpreting a signed ratio as a
collapse factor.}
\label{tab:collapse}
\begin{ruledtabular}
\begin{tabular}{lcccc}
$\beta$ (triangular) & same $\beta$ (no matching) & matched $\zeta$
& matched $\bar{\varepsilon}_3$ & matched $\bar{\varepsilon}_{34}$\\
\colrule
""" + "\n".join(latex_rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
"""
    (HERE / "tableIII_replacement.tex").write_text(latex)
    print(latex)


if __name__ == "__main__":
    main()
