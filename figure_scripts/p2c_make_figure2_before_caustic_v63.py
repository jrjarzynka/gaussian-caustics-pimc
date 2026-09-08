#!/usr/bin/env python3
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ANCH=ROOT/'figure_data/point7_exact_bz_anchors.csv'
CL=ROOT/'figure_data/triangular_classical_baseline.csv'
OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ANCH); dc=pd.read_csv(CL)
assert np.allclose(df.beta,dc.beta)
beta=df.beta.to_numpy(); bc=2.565099660
splh_err=df.SPLH_energy_error_pct.to_numpy(); cl_err=dc.classical_energy_error_pct.to_numpy(); l1=df.SPLH_density_L1_bin16_vs_exact_BZ.to_numpy(); trotter=np.abs(df.P64_Trotter_bias_pct.to_numpy())
abs_splh=np.abs(splh_err)
fig=plt.figure(figsize=(12.6,10.7),constrained_layout=False)
gs=fig.add_gridspec(2,2,height_ratios=[1,0.88],hspace=.32,wspace=.32,left=.08,right=.985,bottom=.08,top=.96)
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,:])
# panel a
ax1.scatter(beta,splh_err,s=70,label='SPLH vs BZ exact')
ax1.scatter(beta,cl_err,s=70,marker='^',label='Classical vs BZ exact')
ax1.axhline(0,lw=1)
ax1.axvline(bc,ls=':',lw=2)
ax1.set_title('(a) Energy accuracy',fontsize=17)
ax1.set_xlabel(r'Inverse temperature $\beta$',fontsize=16); ax1.set_ylabel('Relative energy error (%)',fontsize=16)
ax1.set_xlim(.43,2.69); ax1.set_ylim(-43,23)
ax1.tick_params(labelsize=14); ax1.legend(frameon=False,fontsize=12,loc='lower left')
# highlight beta 2.5 SPLH
j=np.argmin(np.abs(beta-2.5)); ax1.scatter([beta[j]],[splh_err[j]],s=125,edgecolors='k',zorder=5)
ax1.annotate(r'$\beta=2.5$'+'\n'+r'$\zeta_{\max}=0.9746<1$'+'\n'+r'$\Delta V_{\rm SPLH}=+17.40\%$',xy=(beta[j],splh_err[j]),xytext=(1.48,10.5),fontsize=13,arrowprops=dict(arrowstyle='->',lw=1.2))
ax1.annotate(r'$\Delta V_{\rm cl}=-38.50\%$',xy=(beta[j],cl_err[j]),xytext=(1.25,-33.0),fontsize=13,arrowprops=dict(arrowstyle='->',lw=1.2))
ax1.text(bc+.015,19.5,r'$\beta_c$',fontsize=13)
# panel b
ax2.scatter(beta,l1,s=70,label='BZ-exact anchors')
ax2.axvline(bc,ls=':',lw=2)
ax2.set_title('(b) Distribution accuracy',fontsize=17)
ax2.set_xlabel(r'Inverse temperature $\beta$',fontsize=16); ax2.set_ylabel(r'Density error $L^1$',fontsize=16)
ax2.set_xlim(.43,2.69); ax2.set_ylim(0.002,0.14); ax2.tick_params(labelsize=14); ax2.legend(frameon=False,fontsize=12,loc='upper left')
ax2.scatter([beta[j]],[l1[j]],s=125,edgecolors='k',zorder=5)
ax2.annotate(r'$L^1=0.115$'+'\n'+r'while $\zeta_{\max}<1$',xy=(beta[j],l1[j]),xytext=(1.44,.105),fontsize=13,arrowprops=dict(arrowstyle='->',lw=1.2))
ax2.text(bc+.015,.128,r'$\beta_c$',fontsize=13)
# panel c
ax3.scatter(beta,abs_splh,s=65,label=r'$|\Delta V_{\rm SPLH}|$')
ax3.scatter(beta,trotter,s=55,marker='s',label=r'$|\Delta V_{P=64}-\Delta V_{\infty}|$')
ax3.axvline(bc,ls=':',lw=2)
ax3.set_yscale('log'); ax3.set_xlim(.43,2.69); ax3.set_ylim(2.5e-4,35)
ax3.set_title('(c) Approximation error versus primitive Trotter error',fontsize=17)
ax3.set_xlabel(r'Inverse temperature $\beta$',fontsize=16); ax3.set_ylabel('Absolute energy error (%)',fontsize=16)
ax3.tick_params(labelsize=14); ax3.legend(frameon=False,fontsize=13,loc='lower left',ncol=2)
ratio=abs_splh[j]/trotter[j]
ax3.annotate(r'$\beta=2.5:$ '+f'{ratio:.0f}'+r'$\times$ separation',xy=(beta[j],abs_splh[j]),xytext=(1.38,2.8),fontsize=13,arrowprops=dict(arrowstyle='->',lw=1.2))
ax3.text(bc+.015,18,r'$\beta_c$',fontsize=13)
fig.savefig(OUT/'Fig2_before_caustic.pdf',bbox_inches='tight')
fig.savefig(OUT/'Fig2_before_caustic.png',dpi=220,bbox_inches='tight')
plot=pd.DataFrame({'beta':beta,'SPLH_energy_error_pct':splh_err,'classical_energy_error_pct':cl_err,'SPLH_density_L1':l1,'P64_abs_Trotter_bias_pct':trotter})
plot.to_csv(OUT/'Fig2_plot_data_v63.csv',index=False)
print('saved',OUT/'Fig2_before_caustic.pdf','ratio',ratio)
