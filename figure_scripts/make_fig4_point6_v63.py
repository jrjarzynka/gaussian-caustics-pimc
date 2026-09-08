from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'figure_data/lhap0b_final_publication_statistics.npz'
SUM=ROOT/'figure_data/point6_l1_noise_floor_summary_variance_corrected.csv'
OUT=ROOT/'figures'; OUT.mkdir(parents=True,exist_ok=True)
npz=np.load(DATA)
P_exact=np.asarray(npz['P_exact_P64_bin_ts'],float); P_pimc=np.asarray(npz['P_PIMC_bin_ts'],float); P_splh=np.asarray(npz['P_RLH_bin_ts'],float); P_cont=np.asarray(npz['P_exact_continuum_bin_ts'],float)
with open(SUM,newline='') as f: rows={int(r['nbin']):r for r in csv.DictReader(f)}
r=rows[16]
L1=lambda A,B: float(np.mean(np.abs(A-B)))
L1_t=L1(P_exact,P_cont); L1_p=float(r['D_PIMC_exactP64']); L1_s=float(r['D_SPLH_exactP64']); lo=float(r['null_q025']); med=float(r['null_median']); hi=float(r['null_q975']); p=float(r['p_null'])
assert abs(L1(P_pimc,P_exact)-L1_p)<1e-12 and abs(L1(P_splh,P_exact)-L1_s)<1e-12
N=P_exact.shape[0]; edges=np.linspace(0,1,N+1); S,T=np.meshgrid(edges,edges); Dp=P_pimc-P_exact; Ds=P_splh-P_exact; rlim=max(np.max(np.abs(Dp)),np.max(np.abs(Ds))); dmax=max(P_exact.max(),P_pimc.max(),P_splh.max())
plt.rcParams.update({'text.usetex':False,'font.family':'serif','font.size':9.4,'axes.labelsize':9.8,'axes.titlesize':10,'xtick.labelsize':8.2,'ytick.labelsize':8.2,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(7.25,5.45)); gs=fig.add_gridspec(2,3,hspace=.38,wspace=.52,width_ratios=[1,1,1.10]); axs=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]]; a,b,c,d,e,f=axs
def mp(ax,arr,title,cmap,vmin,vmax,y=False):
 im=ax.pcolormesh(S,T,arr,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',rasterized=False); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.set_xlabel(r'$s$'); ax.set_ylabel(r'$t$' if y else ''); ax.set_title(title); ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1]); return im
im=mp(a,P_exact,r'(a) Exact $P=64$','viridis',0,dmax,True); mp(b,P_pimc,'(b) PI-QMC','viridis',0,dmax); mp(c,P_splh,'(c) Tensorial SPLH','viridis',0,dmax); div=make_axes_locatable(c); cax=div.append_axes('right',size='4.5%',pad=.06); fig.colorbar(im,cax=cax).set_label('Normalized density')
imr=mp(d,Dp,r'(d) PI-QMC $-$ exact','RdBu_r',-rlim,rlim,True); mp(e,Ds,r'(e) SPLH $-$ exact','RdBu_r',-rlim,rlim); div=make_axes_locatable(e); cax=div.append_axes('right',size='3.0%',pad=.04); cb=fig.colorbar(imr,cax=cax); cb.set_label(r'$P-P_{\rm exact}$',fontsize=7.8,labelpad=2)
x=np.array([0,1,2]); vals=np.array([L1_t,L1_p,L1_s]); f.scatter(x,vals,s=52,zorder=3); f.vlines(1,lo,hi,linewidth=2,zorder=2); f.scatter([1],[med],marker='_',s=110,zorder=4); f.text(.78,hi*1.10,'95% null\nsampling floor',ha='right',va='bottom',fontsize=6.8); f.set_yscale('log'); f.set_xticks(x); f.set_xticklabels(['Trotter\n$P=64$','PI-QMC','SPLH']); f.text(.03,.96,r'Density $L^1$',transform=f.transAxes,ha='left',va='top',fontsize=8); f.set_title('(f) Quantitative hierarchy'); f.grid(axis='y',which='both',linewidth=.35,alpha=.30)
for xi,yi,tx in zip(x,vals,[f'{L1_t:.2e}',f'{L1_p:.4f}',f'{L1_s:.4f}']): f.text(xi,yi*1.32,tx,ha='center',va='bottom',fontsize=7.4)
f.set_ylim(L1_t/3,L1_s*2.7); fig.subplots_adjust(top=.97,bottom=.11,left=.075,right=.965)
fig.savefig(OUT/'Fig4_exact_pimc_SPLH_FINAL.pdf',bbox_inches='tight'); fig.savefig(OUT/'Fig4_exact_pimc_SPLH_FINAL.png',dpi=400,bbox_inches='tight'); plt.close(fig)
print(L1_t,L1_p,L1_s,lo,med,hi,p)
