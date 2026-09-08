import csv
import numpy as np


BETA = 2.70
P = 64

NBIN = 16

K2_CONTROL = 81
K2_PROD = 100

NK_CONTROL = 12
NK_PROD = 16


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


sqrt3 = np.sqrt(3.0)


G1 = np.array([
    1.0,
    0.0,
])


G2 = np.array([
    -0.5,
    sqrt3/2.0,
])


# ============================================================
# Energy zero
# ============================================================


with open(
    "lha13a0_stationary_points.csv",
    newline="",
) as f:

    stationary = list(
        csv.DictReader(f)
    )


minimum = [
    r
    for r in stationary
    if r["type"] == "minimum"
][0]


s0 = float(minimum["s"])
t0 = float(minimum["t"])


theta_min = np.array([
    2*np.pi*s0 + PHASES[0],
    2*np.pi*t0 + PHASES[1],
    -2*np.pi*(s0+t0) + PHASES[2],
])


Umin = -float(
    np.sum(
        AMPLITUDES
        * np.cos(theta_min)
    )
)


V0 = -Umin


V_FOURIER = {

    (0, 0):
        V0,

    (1, 0):
        -0.5
        * AMPLITUDES[0]
        * np.exp(1j*PHASES[0]),

    (-1, 0):
        -0.5
        * AMPLITUDES[0]
        * np.exp(-1j*PHASES[0]),

    (0, 1):
        -0.5
        * AMPLITUDES[1]
        * np.exp(1j*PHASES[1]),

    (0, -1):
        -0.5
        * AMPLITUDES[1]
        * np.exp(-1j*PHASES[1]),

    (-1, -1):
        -0.5
        * AMPLITUDES[2]
        * np.exp(1j*PHASES[2]),

    (1, 1):
        -0.5
        * AMPLITUDES[2]
        * np.exp(-1j*PHASES[2]),
}


def K2_of(n1, n2):

    return (
        n1*n1
        + n2*n2
        - n1*n2
    )


def build_basis(K2max):

    R = int(
        np.ceil(
            2*np.sqrt(K2max)
        )
    ) + 2


    basis = []


    for n1 in range(-R, R+1):

        for n2 in range(-R, R+1):

            k2 = K2_of(
                n1,
                n2,
            )


            if k2 <= K2max:

                basis.append(
                    (
                        n1,
                        n2,
                        k2,
                    )
                )


    basis.sort(
        key=lambda x: (
            x[2],
            x[0],
            x[1],
        )
    )


    return basis


def prepare(K2max):

    basis = build_basis(
        K2max
    )


    nvec = np.array([
        [n1, n2]
        for n1, n2, _ in basis
    ], dtype=int)


    index = {
        (n1, n2): i
        for i, (n1, n2, _) in enumerate(basis)
    }


    Gcart = (
        nvec[:, 0, None]*G1[None, :]
        +
        nvec[:, 1, None]*G2[None, :]
    )


    Vmat = np.zeros(
        (
            len(basis),
            len(basis),
        ),
        dtype=complex,
    )


    for i, (n1, n2, _) in enumerate(basis):

        for (dn1, dn2), coeff in V_FOURIER.items():

            j = index.get(
                (
                    n1-dn1,
                    n2-dn2,
                )
            )


            if j is not None:

                Vmat[i, j] = coeff


    return (
        nvec,
        Gcart,
        Vmat,
    )


# ============================================================
# Density matrices:
#
# continuum:
#   rho = exp(-beta H)
#
# finite-P:
#   rho_P = B_tau^P
#
# after BZ averaging and normalization.
# ============================================================


def build_density_matrices(
    K2max,
    NK,
):

    nvec, Gcart, Vmat = prepare(
        K2max
    )


    NB = len(nvec)


    rho_cont_num = np.zeros(
        (
            NB,
            NB,
        ),
        dtype=complex,
    )


    rho_P_num = np.zeros(
        (
            NB,
            NB,
        ),
        dtype=complex,
    )


    Z_cont = 0.0
    Z_P = 0.0


    # V exponential for primitive transfer matrix

    vE, vU = np.linalg.eigh(
        Vmat
    )


    tau = BETA/P


    expVhalf = (
        (
            vU
            * np.exp(
                -0.5*tau*vE
            )[None, :]
        )
        @ vU.conj().T
    )


    kvals = (
        (
            np.arange(
                NK,
                dtype=float,
            )
            + 0.5
        )
        / NK
        - 0.5
    )


    # Safe common scaling for continuum.
    # For finite P we use lambda^P scaled independently
    # per k then accumulate with global log maximum.

    solved_cont = []
    solved_P = []

    E0_global = np.inf
    loglam_global = -np.inf


    for ku in kvals:

        for kv in kvals:

            kcart = (
                ku*G1
                + kv*G2
            )


            KG = (
                Gcart
                + kcart[None, :]
            )


            kinetic = (
                0.5
                * np.sum(
                    KG*KG,
                    axis=1,
                )
            )


            # continuum

            H = Vmat.copy()

            H[
                np.diag_indices_from(H)
            ] += kinetic


            E, C = np.linalg.eigh(
                H
            )


            E0_global = min(
                E0_global,
                float(E[0]),
            )


            solved_cont.append(
                (
                    E,
                    C,
                )
            )


            # finite-P

            expT = np.exp(
                -tau*kinetic
            )


            Btau = (
                expVhalf
                @ (
                    expT[:, None]
                    * expVhalf
                )
            )


            Btau = (
                0.5
                * (
                    Btau
                    + Btau.conj().T
                )
            )


            lam, U = np.linalg.eigh(
                Btau
            )


            if np.min(lam) < -1e-12:

                raise RuntimeError(
                    "Negative transfer eigenvalue."
                )


            lam = np.maximum(
                lam,
                np.finfo(float).tiny,
            )


            loglam = np.log(
                lam
            )


            loglam_global = max(
                loglam_global,
                float(
                    np.max(loglam)
                ),
            )


            solved_P.append(
                (
                    loglam,
                    U,
                )
            )


    for E, C in solved_cont:

        w = np.exp(
            -BETA
            * (
                E-E0_global
            )
        )


        Z_cont += np.sum(w)


        rho_cont_num += (
            (C*w[None, :])
            @ C.conj().T
        )


    for loglam, U in solved_P:

        w = np.exp(
            P
            * (
                loglam-loglam_global
            )
        )


        Z_P += np.sum(w)


        rho_P_num += (
            (U*w[None, :])
            @ U.conj().T
        )


    rho_cont = (
        rho_cont_num/Z_cont
    )


    rho_P = (
        rho_P_num/Z_P
    )


    return (
        nvec,
        Vmat,
        rho_cont,
        rho_P,
    )


# ============================================================
# Fourier coefficients of diagonal density
# ============================================================


def rho_to_fourier(
    nvec,
    rho,
):

    d = (
        nvec[:, None, :]
        -
        nvec[None, :, :]
    )


    d1 = d[..., 0]
    d2 = d[..., 1]


    D = int(
        max(
            np.max(np.abs(d1)),
            np.max(np.abs(d2)),
        )
    )


    coeff = np.zeros(
        (
            2*D+1,
            2*D+1,
        ),
        dtype=complex,
    )


    np.add.at(
        coeff,
        (
            (d2+D).ravel(),
            (d1+D).ravel(),
        ),
        rho.ravel(),
    )


    return (
        coeff,
        D,
    )


# ============================================================
# Exact bin average
#
# Integral of exp(2 pi i n s) over one bin:
#
# sinc(n/N) *
# exp[2 pi i n (i+1/2)/N]
# ============================================================


def bin_integrated_density(
    nvec,
    rho,
    N,
):

    coeff, D = rho_to_fourier(
        nvec,
        rho,
    )


    fft_coeff = np.zeros(
        (
            N,
            N,
        ),
        dtype=complex,
    )


    for j2 in range(
        2*D+1
    ):

        n2 = j2-D


        for j1 in range(
            2*D+1
        ):

            n1 = j1-D


            c = coeff[
                j2,
                j1
            ]


            if abs(c) < 1e-16:
                continue


            factor = (
                np.sinc(
                    n1/N
                )
                *
                np.sinc(
                    n2/N
                )
                *
                np.exp(
                    1j*np.pi
                    * (
                        n1+n2
                    )
                    / N
                )
            )


            fft_coeff[
                n2 % N,
                n1 % N,
            ] += (
                c*factor
            )


    Pbin_complex = (
        np.fft.ifft2(
            fft_coeff
            * N*N
        )
    )


    imag_residual = float(
        np.max(
            np.abs(
                Pbin_complex.imag
            )
        )
    )


    Pbin = (
        Pbin_complex.real
    )


    norm_error = abs(
        float(
            np.mean(Pbin)
        )
        - 1.0
    )


    return (
        Pbin,
        norm_error,
        imag_residual,
    )


def solve(
    K2max,
    NK,
):

    nvec, Vmat, rho_c, rho_P = (
        build_density_matrices(
            K2max,
            NK,
        )
    )


    Pc, nc, ic = bin_integrated_density(
        nvec,
        rho_c,
        NBIN,
    )


    Pp, np_, ip = bin_integrated_density(
        nvec,
        rho_P,
        NBIN,
    )


    Vc = float(
        np.trace(
            rho_c @ Vmat
        ).real
    )


    Vp = float(
        np.trace(
            rho_P @ Vmat
        ).real
    )


    return {
        "Pc":
            Pc,

        "Pp":
            Pp,

        "Vc":
            Vc,

        "Vp":
            Vp,

        "norm_error":
            max(
                nc,
                np_,
            ),

        "imag_residual":
            max(
                ic,
                ip,
            ),

        "min_density":
            min(
                float(
                    np.min(Pc)
                ),
                float(
                    np.min(Pp)
                ),
            ),

        "nbasis":
            len(nvec),
    }


print('BUILD production density matrices K2=100 NK=16', flush=True)
nvec, Vmat, rho_c, rho_P = build_density_matrices(K2_PROD, NK_PROD)
Vc=float(np.trace(rho_c@Vmat).real); Vp=float(np.trace(rho_P@Vmat).real)
print('nbasis',len(nvec),'Vc',Vc,'Vp',Vp, flush=True)
for nb in (8,16,32):
    Pc,nc,ic=bin_integrated_density(nvec,rho_c,nb)
    Pp,np_,ip=bin_integrated_density(nvec,rho_P,nb)
    fn=f'exact_P64_multibin_{nb}.npz'
    np.savez_compressed(fn,P_continuum_bin_ts=Pc,P_finiteP_bin_ts=Pp,nbin=nb,V_continuum=Vc,V_finiteP=Vp,norm_error=max(nc,np_),imag_residual=max(ic,ip),nbasis=len(nvec))
    print('saved',fn,'norm',max(nc,np_),'imag',max(ic,ip),'L1Pcont',float(np.mean(np.abs(Pp-Pc))),flush=True)
