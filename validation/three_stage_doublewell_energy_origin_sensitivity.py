#!/usr/bin/env python3
"""Energy-origin sensitivity audit for the scalar three-stage double-well benchmark.

Published Ref. [1] convention:
    V_pub(x) = -x^2/2 + g x^4/4 + 1/(16 g)
Minimum-zero convention:
    V_0(x)   = V_pub(x) + 3/(16 g)
             = -x^2/2 + g x^4/4 + 1/(4 g)

Exact, bare, and renormalized normalized densities are invariant under the
constant shift. The final mapped density sqrt(Xi) exp[-beta V Xi] is not,
unless Xi is coordinate independent.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.linalg import eigh_tridiagonal

OUT=Path(__file__).resolve().parent
BETAS=np.array([0.20,0.50,0.80,1.00,1.30,1.60,1.90,2.10,2.30,2.50,2.70,2.85,2.95,3.00,3.05,3.10])
SETTINGS={0.1:(14.0,.004),0.5:(10.0,.0035)}

def Xi(beta,k):
    k=np.asarray(k,float); out=np.empty_like(k); p=k>1e-12; n=k<-1e-12; z=~(p|n)
    x=.5*beta*np.sqrt(k[p]); out[p]=np.tanh(x)/x
    y=.5*beta*np.sqrt(-k[n]); out[n]=np.tan(y)/y
    out[z]=1.; return out

def F(beta,k):
    k=np.asarray(k,float); out=np.empty_like(k); s=np.abs(k)<1e-7
    out[s]=-beta**2/12 + beta**4*k[s]/120 - 17*beta**6*k[s]**2/20160
    q=~s; out[q]=(Xi(beta,k[q])-1)/k[q]; return out

def nlog(lw,x):
    w=np.exp(lw-np.max(lw)); return w/np.trapezoid(w,x)

def l1(a,b,x): return float(np.trapezoid(np.abs(a-b),x))

rows=[]
for g,(L,dx) in SETTINGS.items():
    x=np.arange(-L+dx,L,dx)
    Vp=-.5*x*x+.25*g*x**4+1/(16*g)
    V0=Vp+3/(16*g)
    vp=-x+g*x**3
    k=-1+3*g*x*x
    d=1/dx**2+Vp; o=np.full(len(x)-1,-.5/dx**2)
    E,U=eigh_tridiagonal(d,o,select='i',select_range=(0,499),check_finite=False)
    for beta in BETAS:
        wt=np.exp(-beta*(E-E[0])); wt/=wt.sum()
        Pex=((U*U)@wt)/dx; Pex/=np.trapezoid(Pex,x)
        xi=Xi(beta,k); corr=.5*vp*vp*F(beta,k)
        pos=k>1e-12; neg=k<-1e-12; zero=~(pos|neg); logD=np.empty_like(k)
        a=.5*beta*np.sqrt(k[pos]); logD[pos]=.5*(np.log(2*a)-np.log(np.sinh(2*a)))
        b=.5*beta*np.sqrt(-k[neg]); s=np.sin(2*b)
        if np.any(s<=0): raise RuntimeError('bare stage beyond first caustic')
        logD[neg]=.5*(np.log(2*b)-np.log(s)); logD[zero]=0.
        Pb=nlog(logD-beta*(Vp+corr),x); Pb0=nlog(logD-beta*(V0+corr),x)
        Pr=nlog(.5*np.log(xi)-beta*(Vp+corr),x); Pr0=nlog(.5*np.log(xi)-beta*(V0+corr),x)
        Pm=nlog(.5*np.log(xi)-beta*Vp*xi,x); Pm0=nlog(.5*np.log(xi)-beta*V0*xi,x)
        rows.append({
            'g':g,'beta':float(beta),'zeta_max':float(beta/np.pi),'delta_c':float(3/(16*g)),
            'L1_bare_published_offset':l1(Pb,Pex,x),'L1_bare_minimum_zero':l1(Pb0,Pex,x),
            'max_abs_bare_shift_difference':float(np.max(np.abs(Pb-Pb0))),
            'L1_renorm_published_offset':l1(Pr,Pex,x),'L1_renorm_minimum_zero':l1(Pr0,Pex,x),
            'max_abs_renorm_shift_difference':float(np.max(np.abs(Pr-Pr0))),
            'L1_mapped_published_offset':l1(Pm,Pex,x),'L1_mapped_minimum_zero':l1(Pm0,Pex,x),
            'L1_between_mapped_energy_origins':l1(Pm,Pm0,x),
        })
with open(OUT/'three_stage_doublewell_energy_origin_sensitivity.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('max bare normalized-density shift', max(r['max_abs_bare_shift_difference'] for r in rows))
print('max renorm normalized-density shift', max(r['max_abs_renorm_shift_difference'] for r in rows))
for r in rows:
    if r['beta'] in (1.0,2.5,3.0,3.1): print(r)
