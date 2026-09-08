from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'point7_exact_bz_anchors.csv'
OUT=ROOT/'publication_figures_point7'
OUT.mkdir(parents=True,exist_ok=True)

with DATA.open(newline='') as f:
    rows=list(csv.DictReader(f))
def col(name): return np.array([float(r[name]) for r in rows])

beta=col('beta')
zeta=col('zeta_max')
V_exact=col('V_exact_BZ')
V_splh=col('V_SPLH')
energy_err=col('SPLH_energy_error_pct')
density_l1=col('SPLH_density_L1_bin16_vs_exact_BZ')
trotter=col('P64_Trotter_bias_pct')

beta_c=np.pi/np.sqrt(1.5)
i_anchor=int(np.argmin(np.abs(beta-2.5)))
if abs(beta[i_anchor]-2.5)>1e-12: raise RuntimeError('beta=2.5 anchor missing')
ratio=abs(energy_err[i_anchor])/abs(trotter[i_anchor])

print('=== POINT 7 FIGURE 2 AUDIT ===')
print(f'anchors = {len(beta)}')
print(f'beta_c = {beta_c:.12f}')
print(f'max anchor beta = {beta.max():.6f}')
print(f'max zeta = {zeta.max():.12f}')
print(f'beta=2.5 error = {energy_err[i_anchor]:+.12f}%')
print(f'beta=2.5 L1 = {density_l1[i_anchor]:.12f}')
print(f'beta=2.5 P64 bias = {trotter[i_anchor]:+.12f}%')
print(f'beta=2.5 ratio = {ratio:.6f}x')
if not np.all(zeta<1): raise RuntimeError('non-precaustic anchor')

plt.rcParams.update({
    'font.size':9.5,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,
    'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
})
fig=plt.figure(figsize=(7.15,6.45))
gs=fig.add_gridspec(2,2,height_ratios=[1.0,0.88],hspace=0.34,wspace=0.32)
ax_a=fig.add_subplot(gs[0,0]); ax_b=fig.add_subplot(gs[0,1]); ax_c=fig.add_subplot(gs[1,:])

# (a) Only genuine BZ-exact anchors; no interpolating/guide curve is plotted.
ax_a.plot(beta,energy_err,marker='o',linestyle='none',markersize=5.0,label='BZ-exact anchors')
ax_a.axhline(0.0,linewidth=0.8,linestyle='-')
ax_a.axvline(beta_c,linewidth=1.15,linestyle=':')
ax_a.plot([beta[i_anchor]],[energy_err[i_anchor]],marker='o',linestyle='none',markersize=7.5,markeredgecolor='black',markeredgewidth=0.8)
ax_a.annotate(r'$\beta=2.5$'+'\n'+r'$\zeta_{\max}=0.9746<1$'+'\n'+r'$\Delta V_{\rm SPLH}=+17.40\%$',xy=(beta[i_anchor],energy_err[i_anchor]),xytext=(1.48,14.2),arrowprops={'arrowstyle':'->','lw':0.8},fontsize=8.2)
ax_a.set_xlabel(r'Inverse temperature $\beta$'); ax_a.set_ylabel(r'SPLH energy error (\%)'); ax_a.set_title('(a) Energy accuracy'); ax_a.legend(frameon=False,loc='upper left')

# (b)
ax_b.plot(beta,density_l1,marker='o',linestyle='none',markersize=5.0,label='BZ-exact anchors')
ax_b.axvline(beta_c,linewidth=1.15,linestyle=':')
ax_b.plot([beta[i_anchor]],[density_l1[i_anchor]],marker='o',linestyle='none',markersize=7.5,markeredgecolor='black',markeredgewidth=0.8)
ax_b.annotate(r'$L^1=0.115$'+'\n'+r'while $\zeta_{\max}<1$',xy=(beta[i_anchor],density_l1[i_anchor]),xytext=(1.44,0.104),arrowprops={'arrowstyle':'->','lw':0.8},fontsize=8.2)
ax_b.set_xlabel(r'Inverse temperature $\beta$'); ax_b.set_ylabel(r'Density error $L^1$'); ax_b.set_title('(b) Distribution accuracy'); ax_b.legend(frameon=False,loc='upper left')

# (c)
abs_splh=np.abs(energy_err); abs_trot=np.abs(trotter)
ax_c.semilogy(beta,abs_splh,marker='o',linestyle='none',markersize=5.0,label=r'$|\Delta V_{\rm SPLH}|$')
ax_c.semilogy(beta,abs_trot,marker='s',linestyle='none',markersize=5.0,label=r'$|\Delta V_{P=64}-\Delta V_{\infty}|$')
ax_c.axvline(beta_c,linewidth=1.15,linestyle=':')
ax_c.annotate(r'$\beta=2.5:$ '+f'{ratio:.0f}'+r'$\times$ separation',xy=(beta[i_anchor],abs_splh[i_anchor]),xytext=(1.38,2.8),arrowprops={'arrowstyle':'->','lw':0.8},fontsize=8.3)
ax_c.set_xlabel(r'Inverse temperature $\beta$'); ax_c.set_ylabel(r'Absolute energy error (\%)'); ax_c.set_title('(c) Approximation error versus primitive Trotter error'); ax_c.legend(frameon=False,ncol=2,loc='lower left')

for ax in [ax_a,ax_b,ax_c]:
    ymin,ymax=ax.get_ylim(); y_text=ymax/1.15 if ax.get_yscale()=='log' else ymax-0.055*(ymax-ymin)
    ax.text(beta_c+0.018,y_text,r'$\beta_c$',ha='left',va='top',fontsize=8.2)
    ax.set_xlim(0.43,beta_c+0.12)

fig.subplots_adjust(top=0.975,bottom=0.09,left=0.10,right=0.985)
pdf=OUT/'Fig2_before_caustic_POINT7.pdf'; png=OUT/'Fig2_before_caustic_POINT7.png'
fig.savefig(pdf,bbox_inches='tight'); fig.savefig(png,dpi=400,bbox_inches='tight'); plt.close(fig)
source_csv=OUT/'Fig2_BZ_anchor_data_POINT7.csv'
with source_csv.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['beta','zeta_max','V_exact_BZ','V_SPLH','SPLH_energy_error_pct','SPLH_density_L1_vs_exact_BZ','P64_Trotter_bias_pct'])
    for i in range(len(beta)): w.writerow([beta[i],zeta[i],V_exact[i],V_splh[i],energy_err[i],density_l1[i],trotter[i]])
print('saved',pdf); print('saved',png); print('saved',source_csv)
