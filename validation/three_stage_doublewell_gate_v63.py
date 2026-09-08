#!/usr/bin/env python3
"""Deterministic 1D gate for bare / renormalized pre-map / final mapped Ref. [1] stages.
Potential and convention: literal Eq. (44) of Ref. [1],
V=-x^2/2+g x^4/4+1/(16g), M=hbar=omega=1.
The additive constant does not set V_min=0: x_min^2=1/g, V_min=-3/(16g),
and the barrier height is 1/(4g). The first Gaussian caustic is beta_c=pi
because min V''=-1 at x=0.
"""
from pathlib import Path
import csv, json
import numpy as np
from scipy.linalg import eigh_tridiagonal
OUT=Path(__file__).resolve().parent
BETAS=np.array([0.20,0.50,0.80,1.00,1.30,1.60,1.90,2.10,2.30,2.50,2.70,2.85,2.95,3.00,3.05,3.10])

def Xi(beta,k):
    k=np.asarray(k,float); out=np.empty_like(k); p=k>1e-12; n=k<-1e-12; z=~(p|n)
    x=.5*beta*np.sqrt(k[p]); out[p]=np.tanh(x)/x
    y=.5*beta*np.sqrt(-k[n]); out[n]=np.tan(y)/y
    out[z]=1.0; return out

def F(beta,k):
    k=np.asarray(k,float); out=np.empty_like(k); s=np.abs(k)<1e-7
    out[s]=-beta**2/12 + beta**4*k[s]/120 - 17*beta**6*k[s]**2/20160
    q=~s; out[q]=(Xi(beta,k[q])-1)/k[q]; return out

def approx(g,beta,x):
    V=-.5*x*x+.25*g*x**4+1/(16*g); vp=-x+g*x**3; k=-1+3*g*x*x; xi=Xi(beta,k); corr=.5*vp*vp*F(beta,k)
    p=k>1e-12; n=k<-1e-12; z=~(p|n); logD=np.empty_like(k)
    a=.5*beta*np.sqrt(k[p]); logD[p]=.5*(np.log(2*a)-np.log(np.sinh(2*a)))
    b=.5*beta*np.sqrt(-k[n]); s=np.sin(2*b)
    if np.any(s<=0): raise ValueError('bare stage beyond first caustic')
    logD[n]=.5*(np.log(2*b)-np.log(s)); logD[z]=0.0
    logs={'bare':logD-beta*(V+corr),'renorm':.5*np.log(xi)-beta*(V+corr),'mapped':.5*np.log(xi)-beta*(V*xi)}
    P={}
    for name,lw in logs.items():
        w=np.exp(lw-lw.max()); P[name]=w/np.trapezoid(w,x)
    return V,P

def exact(g,L,dx,beta,nstates=400):
    x=np.arange(-L+dx,L,dx); V=-.5*x*x+.25*g*x**4+1/(16*g)
    d=1/dx**2+V; o=np.full(len(x)-1,-.5/dx**2)
    E,U=eigh_tridiagonal(d,o,select='i',select_range=(0,nstates-1),check_finite=False)
    w=np.exp(-beta*(E-E[0])); w/=w.sum(); P=((U*U)@w)/dx; P/=np.trapezoid(P,x)
    EV=float(w@((U*U).T@V)); return x,V,P,EV

def l1(a,b,x): return float(np.trapezoid(np.abs(a-b),x))
rows=[]
settings={0.1:(14.0,.004),0.5:(10.0,.0035)}
for g,(L,dx) in settings.items():
    # diagonalize once for all beta values
    x=np.arange(-L+dx,L,dx); V=-.5*x*x+.25*g*x**4+1/(16*g); d=1/dx**2+V; o=np.full(len(x)-1,-.5/dx**2)
    E,U=eigh_tridiagonal(d,o,select='i',select_range=(0,499),check_finite=False); Vn=(U*U).T@V
    for beta in BETAS:
        w=np.exp(-beta*(E-E[0])); w/=w.sum(); Pex=((U*U)@w)/dx; Pex/=np.trapezoid(Pex,x); Eex=float(w@Vn)
        V0,P=approx(g,beta,x)
        r={'g':g,'beta':float(beta),'zeta_max':float(beta/np.pi),'E_exact':Eex}
        for name in ('bare','renorm','mapped'):
            r[f'L1_{name}']=l1(P[name],Pex,x); r[f'E_{name}']=float(np.trapezoid(P[name]*V0,x))
        rows.append(r)
with open(OUT/'three_stage_doublewell_gate_v63.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
# independent coarser-grid convergence at beta 3.10
conv=[]
for g,(L,dx) in {0.1:(12,.006),0.5:(8,.0045)}.items():
    x,V,Pex,Eex=exact(g,L,dx,3.10,300); _,P=approx(g,3.10,x)
    conv.append({'g':g,'L':L,'dx':dx,'E_exact':Eex,**{f'L1_{n}':l1(P[n],Pex,x) for n in ('bare','renorm','mapped')}})
with open(OUT/'three_stage_doublewell_gate_v63_convergence.json','w') as f: json.dump(conv,f,indent=2)
print('beta_c',np.pi)
for r in rows:
    if r['beta'] in (1.0,2.5,3.0,3.1): print(r)
print('coarse',conv)
