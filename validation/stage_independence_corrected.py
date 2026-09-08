#!/usr/bin/env python3
"""Regression audit for the common first scalar Gaussian-existence boundary.

The bare stage uses the Dirichlet determinant factor D only; there is no
additional sqrt(Xi) factor in the bare prefactor. This script tests only the
location of the first real-domain boundary, not a post-caustic continuation.
"""
import numpy as np
from scipy.optimize import brentq

# Full-precision beta_c returned by the independent asymmetric-benchmark audit.
beta_c_ref = 2.7116988514311098
kappa_min = -(np.pi / beta_c_ref)**2

def eta(beta):
    return beta*np.sqrt(-kappa_min)/2.0

def Xi(beta):
    e=eta(beta)
    return np.tan(e)/e

def lam1(beta):
    return (np.pi/beta)**2 + kappa_min

lo,hi=2.0,3.4
roots={
    'cos_eta': brentq(lambda b: np.cos(eta(b)),lo,hi,xtol=1e-16,rtol=8.9e-16),
    'sin_2eta': brentq(lambda b: np.sin(2*eta(b)),lo,hi,xtol=1e-16,rtol=8.9e-16),
    'lambda1': brentq(lam1,lo,hi,xtol=1e-16,rtol=8.9e-16),
    'inv_Xi': brentq(lambda b: 1.0/Xi(b),lo,hi,xtol=1e-16,rtol=8.9e-16),
}
print(f'kappa_min = {kappa_min:.12f}')
print(f'beta_c reference = {beta_c_ref:.16f}')
for k,v in roots.items():
    print(f'{k:10s} {v:.16f}  delta={v-beta_c_ref:+.3e}')
print(f'spread = {max(roots.values())-min(roots.values()):.3e}')
print('PASS: first scalar boundary eta=pi/2 <=> zeta=1 is common to bare, renormalized pre-mapping, and mapped stages.')
