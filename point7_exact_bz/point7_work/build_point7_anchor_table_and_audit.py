from pathlib import Path
import csv, hashlib, json
import numpy as np

W=Path(__file__).resolve().parent
SRC=W/'source_snapshots'
BETAS=[0.50,0.80,1.00,1.30,1.60,1.90,2.10,2.25,2.35,2.42,2.48,2.50,2.54]

def read_csv(p):
    with open(p,newline='') as f: return list(csv.DictReader(f))

def nearest(rows,beta):
    return min(rows,key=lambda r:abs(float(r['beta'])-beta))

cont=read_csv(W/'point7_exact_bz_continuum_13anchors.csv')
splh=read_csv(W/'point7_splh_bz_comparison.csv')
# individual finite-P rows
p64=[]
for b in BETAS:
    tag=str(b).replace('.','p')
    rows=read_csv(W/f'p7_P64_beta_{tag}.csv')
    assert len(rows)==1
    p64.append(rows[0])

# legacy sources
def source_file(name):
    p=SRC/name
    if p.exists(): return p
    q=SRC/("legacy_"+name)
    if q.exists(): return q
    raise FileNotFoundError(name)

old_cont=read_csv(source_file('lha12c0a_exact_P64_multibeta_converged.csv'))
old_h=read_csv(source_file('lha12c2a_harmonized_density_comparison.csv'))
legacy=[0.5,1.0,1.9,2.5]
legacy_checks=[]

out=[]
for i,b in enumerate(BETAS):
    c=nearest(cont,b); s=nearest(splh,b); p=p64[i]
    assert abs(float(c['beta'])-b)<1e-12 and abs(float(s['beta'])-b)<1e-12 and abs(float(p['beta'])-b)<1e-12
    Vc=float(c['V_matrix_BZ'])
    Vp=float(p['V_finiteP64'])
    Vs=float(s['V_SPLH'])
    err=100*(Vs-Vc)/Vc
    out.append({
        'beta':b,
        'zeta_max':float(s['zeta_max']),
        'V_exact_BZ':Vc,
        'V_finiteP64':Vp,
        'P64_Trotter_bias_pct':float(p['finiteP_bias_pct']),
        'V_SPLH':Vs,
        'SPLH_energy_error_pct':err,
        'SPLH_density_L1_bin16_vs_exact_BZ':float(s['L1_SPLH_bin16_vs_exact_BZ']),
        'BZ_density_basis_L1_K64_to_K81_at_Nk12':float(c['density_L1_K64_vs_K81_at_Nk12']),
        'BZ_density_kgrid_L1_Nk12_to_Nk16_at_K81':float(c['density_L1_Nk12_vs_Nk16_at_K81']),
        'P64_basis_delta_K64_to_K81':float(p['basis_delta_K64_to_K81']),
        'P64_kgrid_delta_N8_to_N12_at_K81':float(p['kgrid_delta_N8_to_N12_at_K81']),
        'SPLH_bin16_L1_256_to_512':float(s['SPLH_bin16_L1_256_vs_512']),
        'exact_bin_imag_residual':float(s['exact_bin_imag_residual']),
    })

with open(W/'point7_exact_bz_anchors.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
np.savez_compressed(W/'point7_exact_bz_anchors.npz', **{k:np.array([r[k] for r in out],float) for k in out[0]})

# legacy checks using the exact legacy public table values
for b in legacy:
    n=next(r for r in out if abs(r['beta']-b)<1e-12)
    oc=nearest(old_cont,b); oh=nearest(old_h,b)
    legacy_checks.append({
      'beta':b,
      'dV_cont':abs(n['V_exact_BZ']-float(oc['V_continuum'])),
      'dV_P64':abs(n['V_finiteP64']-float(oc['V_finiteP64'])),
      'dL1_SPLH':abs(n['SPLH_density_L1_bin16_vs_exact_BZ']-float(oh['L1_RLH_bin16_vs_exact_BZ'])),
      'dV_SPLH':abs(n['V_SPLH']-float(oh['V_RLH_reconstructed'])),
    })

maxes={
 'legacy_dV_cont':max(x['dV_cont'] for x in legacy_checks),
 'legacy_dV_P64':max(x['dV_P64'] for x in legacy_checks),
 'legacy_dL1_SPLH':max(x['dL1_SPLH'] for x in legacy_checks),
 'legacy_dV_SPLH':max(x['dV_SPLH'] for x in legacy_checks),
 'bz_basis_l1':max(r['BZ_density_basis_L1_K64_to_K81_at_Nk12'] for r in out),
 'bz_kgrid_l1':max(r['BZ_density_kgrid_L1_Nk12_to_Nk16_at_K81'] for r in out),
 'p64_basis':max(r['P64_basis_delta_K64_to_K81'] for r in out),
 'p64_kgrid':max(r['P64_kgrid_delta_N8_to_N12_at_K81'] for r in out),
 'splh_binconv':max(r['SPLH_bin16_L1_256_to_512'] for r in out),
 'exact_imag':max(r['exact_bin_imag_residual'] for r in out),
 'zeta_max':max(r['zeta_max'] for r in out),
}
passes={
 'legacy_continuum':maxes['legacy_dV_cont']<=1e-10,
 'legacy_P64':maxes['legacy_dV_P64']<=1e-10,
 'legacy_SPLH_L1':maxes['legacy_dL1_SPLH']<=1e-10,
 'legacy_SPLH_V':maxes['legacy_dV_SPLH']<=1e-10,
 'BZ_basis':maxes['bz_basis_l1']<1e-5,
 'BZ_kgrid':maxes['bz_kgrid_l1']<1e-8,
 'P64_basis':maxes['p64_basis']<1e-6,
 'P64_kgrid':maxes['p64_kgrid']<1e-6,
 'SPLH_bin':maxes['splh_binconv']<1e-4,
 'exact_imag':maxes['exact_imag']<1e-12,
 'all_precaustic':maxes['zeta_max']<1,
}
anchor=next(r for r in out if r['beta']==2.5)
ratio=abs(anchor['SPLH_energy_error_pct'])/abs(anchor['P64_Trotter_bias_pct'])

# provenance hashes
files=[
 SRC/'lha12a1_exact_bz_density_reference.py', SRC/'lha12c0a_exact_P64_multibeta_converged.py', SRC/'lha12c2a_harmonized_density_comparison.py', SRC/'p2c_make_figure2_before_caustic.py',
 W/'POINT7_FIG2_BZ_ANCHORS_PROTOCOL_2026-09-08.md', W/'run_point7_exact_bz_continuum.py', W/'run_point7_exact_P64.py', W/'run_point7_splh_bz_comparison.py', W/'build_point7_anchor_table_and_audit.py'
]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prov={str(p.relative_to(W)):sha(p) for p in files}
with open(W/'point7_source_provenance.json','w') as f: json.dump(prov,f,indent=2)

md=[]
md += ['# Point 7 Figure 2 exact-BZ anchor audit','',f'Overall: **{"PASS" if all(passes.values()) else "FAIL"}**','']
md += ['## Frozen anchors','',', '.join(f'{b:.2f}' for b in BETAS),'']
md += ['## Legacy-anchor reconstruction','', '| beta | dV continuum | dV P64 | dV SPLH | dL1 SPLH |','|---:|---:|---:|---:|---:|']
for x in legacy_checks: md.append(f"| {x['beta']:.2f} | {x['dV_cont']:.3e} | {x['dV_P64']:.3e} | {x['dV_SPLH']:.3e} | {x['dL1_SPLH']:.3e} |")
md += ['','## Maximum convergence diagnostics','']
for k,v in maxes.items(): md.append(f'- {k}: `{v:.12g}`')
md += ['','## Acceptance gates','']
for k,v in passes.items(): md.append(f'- {k}: **{"PASS" if v else "FAIL"}**')
md += ['','## Central beta=2.5 anchor','',f"- zeta_max = `{anchor['zeta_max']:.12f}`",f"- V_exact_BZ = `{anchor['V_exact_BZ']:.12f}`",f"- V_SPLH = `{anchor['V_SPLH']:.12f}`",f"- SPLH energy error = `{anchor['SPLH_energy_error_pct']:+.12f}%` -> `+17.40%`",f"- SPLH density L1 (16x16) = `{anchor['SPLH_density_L1_bin16_vs_exact_BZ']:.12f}`",f"- P64 Trotter bias = `{anchor['P64_Trotter_bias_pct']:+.12f}%`",f"- error-scale separation = `{ratio:.6f}x` -> `{ratio:.0f}x`",'']
(W/'point7_fig2_anchor_audit.md').write_text('\n'.join(md)+'\n')
print('overall',all(passes.values()))
print('maxes',maxes)
print('passes',passes)
print('beta2.5',anchor,'ratio',ratio)
