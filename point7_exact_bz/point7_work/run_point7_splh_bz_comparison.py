import csv
import numpy as np

BETAS = [0.50,0.80,1.00,1.30,1.60,1.90,2.10,2.25,2.35,2.42,2.48,2.50,2.54]
NBIN=16
N_VALIDATE=64
N_HI_1=256
N_HI_2=512
sqrt3=np.sqrt(3.0)
G=np.array([[1.0,0.0],[-0.5,sqrt3/2.0],[-0.5,-sqrt3/2.0]])

def splh_density(beta,N,midpoint=False):
    axis=(np.arange(N,dtype=float)+0.5)/N if midpoint else np.arange(N,dtype=float)/N
    S,T=np.meshgrid(axis,axis,indexing='xy')
    theta=np.stack([2*np.pi*S,2*np.pi*T,-2*np.pi*(S+T)],axis=-1)
    c=np.cos(theta); s=np.sin(theta)
    V=3.0-c[...,0]-c[...,1]-c[...,2]
    grad=np.einsum('...a,ai->...i',s,G)
    H=np.einsum('...a,ai,aj->...ij',c,G,G)
    kappa,R=np.linalg.eigh(H)
    Xi=np.empty_like(kappa); F=np.empty_like(kappa)
    eps=1e-9
    pos=kappa>eps; neg=kappa<-eps; zer=~(pos|neg)
    if np.any(pos):
        xi=0.5*beta*np.sqrt(kappa[pos]); Xi[pos]=np.tanh(xi)/xi; F[pos]=(Xi[pos]-1)/kappa[pos]
    if np.any(neg):
        eta=0.5*beta*np.sqrt(-kappa[neg])
        if np.max(eta)>=np.pi/2: raise RuntimeError(f'beta={beta}: caustic crossed')
        Xi[neg]=np.tan(eta)/eta; F[neg]=(Xi[neg]-1)/kappa[neg]
    if np.any(zer):
        k=kappa[zer]; Xi[zer]=1-beta**2*k/12+beta**4*k*k/120; F[zer]=-beta**2/12+beta**4*k/120
    gmodal=np.einsum('...ia,...i->...a',R,grad)
    quadratic=np.sum(F*gmodal*gmodal,axis=-1)
    Phi=V+0.5*quadratic
    logP=0.5*np.sum(np.log(Xi),axis=-1)-beta*Phi
    logP-=np.max(logP); P=np.exp(logP); P/=np.mean(P)
    zeta=beta/np.pi*np.sqrt(np.maximum(0.0,-kappa))
    return P,V,float(np.max(zeta))

def Fourier_bin_average(Pfine,nbin):
    nfine=Pfine.shape[0]
    F=np.fft.fft2(Pfine)/(nfine*nfine)
    freq=(np.fft.fftfreq(nfine)*nfine).astype(int)
    centers=(np.arange(nbin,dtype=float)+0.5)/nbin
    out=np.zeros((nbin,nbin),dtype=complex)
    for i2,n2 in enumerate(freq):
        phase_t=np.exp(2j*np.pi*n2*centers); sinc2=np.sinc(n2/nbin)
        for i1,n1 in enumerate(freq):
            coeff=F[i2,i1]
            if abs(coeff)<1e-15: continue
            phase_s=np.exp(2j*np.pi*n1*centers); sinc1=np.sinc(n1/nbin)
            out += coeff*sinc1*sinc2*phase_t[:,None]*phase_s[None,:]
    imag=float(np.max(np.abs(out.imag))); out=out.real; out/=np.mean(out)
    return out,imag

def block_bin_average(P,nbin):
    N=P.shape[0]
    if N%nbin: raise ValueError('N must be divisible by nbin')
    m=N//nbin
    out=P.reshape(nbin,m,nbin,m).mean(axis=(1,3)); out/=np.mean(out)
    return out

def L1(a,b): return float(np.mean(np.abs(a-b)))

exact=np.load('point7_exact_bz_continuum_13anchors.npz')
betas_exact=np.asarray(exact['betas'],dtype=float)
BZ=np.asarray(exact['densities_BZ'],dtype=float)
rows=[]
for beta in BETAS:
    ie=int(np.argmin(np.abs(betas_exact-beta)))
    if abs(betas_exact[ie]-beta)>1e-12: raise RuntimeError(f'missing beta {beta}')
    P64,V64,zeta=splh_density(beta,N_VALIDATE,midpoint=False)
    V_splh=float(np.mean(P64*V64))
    Pexact16,imag=Fourier_bin_average(BZ[ie],NBIN)
    P256,_,_=splh_density(beta,N_HI_1,midpoint=True)
    P512,_,_=splh_density(beta,N_HI_2,midpoint=True)
    R256=block_bin_average(P256,NBIN); R512=block_bin_average(P512,NBIN)
    binconv=L1(R512,R256)
    l1=L1(R512,Pexact16)
    rows.append([beta,zeta,V_splh,l1,binconv,imag])
    print(f'beta={beta:.2f} zeta={zeta:.12f} V_SPLH={V_splh:.12f} L1_16={l1:.12f} binconv={binconv:.3e} imag={imag:.3e}')

with open('point7_splh_bz_comparison.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['beta','zeta_max','V_SPLH','L1_SPLH_bin16_vs_exact_BZ','SPLH_bin16_L1_256_vs_512','exact_bin_imag_residual']); w.writerows(rows)
np.savez_compressed('point7_splh_bz_comparison.npz', beta=np.array([r[0] for r in rows]), zeta_max=np.array([r[1] for r in rows]), V_SPLH=np.array([r[2] for r in rows]), L1=np.array([r[3] for r in rows]))
print('max binconv',max(r[4] for r in rows)); print('max imag',max(r[5] for r in rows)); print('all zeta<1',all(r[1]<1 for r in rows))
