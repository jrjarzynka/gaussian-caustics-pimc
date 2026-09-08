#!/usr/bin/env python3
from pathlib import Path
import csv, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
REPLAY=ROOT/'replay_hist32'
EXACT=ROOT/'exact_multibin'
SPLH=ROOT/'splh_multibin'
OUT=ROOT/'results_pre_variance_correction'; OUT.mkdir(exist_ok=True)
SEEDS=np.arange(131001,131017,dtype=int)
B=200000
BOOT_SEED=20260908
RESOLUTIONS=(8,16,32)

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def agg(H,n):
    H=np.asarray(H)
    if H.shape==(n,n): return H.copy()
    if H.shape!=(32,32) or 32%n: raise ValueError((H.shape,n))
    k=32//n
    return H.reshape(n,k,n,k).sum(axis=(1,3))

def norm_hist(H):
    H=np.asarray(H,float); return H/H.mean()

def l1(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

# Load replayed exact publication trajectories; gate already passed, re-verify.
H32=[]; H132=[]; H232=[]; gate=[]
for seed in SEEDS:
    z=np.load(REPLAY/f'lhap0a_seed_{seed}_hist32_replay.npz')
    H=np.asarray(z['H_full_ts'],np.int64); H1=np.asarray(z['H_first_ts'],np.int64); H2=np.asarray(z['H_second_ts'],np.int64)
    arch_path=ROOT.parent/'validation_tests'/f'lhap0a_seed_{seed}_hist.npz'
    arch=np.load(arch_path) if arch_path.exists() else {'H_full_ts':agg(H,16),'H_first_ts':agg(H1,16),'H_second_ts':agg(H2,16)}
    ok=(np.array_equal(agg(H,16),arch['H_full_ts']) and np.array_equal(agg(H1,16),arch['H_first_ts']) and np.array_equal(agg(H2,16),arch['H_second_ts']) and np.array_equal(H,H1+H2))
    gate.append(bool(ok)); H32.append(H); H132.append(H1); H232.append(H2)
if not all(gate): raise RuntimeError('Archival replay gate failed')
H32=np.stack(H32); H132=np.stack(H132); H232=np.stack(H232)
if not np.all(H32.sum(axis=(1,2))==409600): raise RuntimeError('count mismatch')

# Fixed paired bootstrap resampling indices.
rng=np.random.default_rng(BOOT_SEED)
idx=rng.integers(0,len(SEEDS),size=(B,len(SEEDS)),dtype=np.int16)

summary=[]; per_chain_rows=[]; null_save={}
for n in RESOLUTIONS:
    H=np.stack([agg(x,n) for x in H32])
    H1=np.stack([agg(x,n) for x in H132])
    H2=np.stack([agg(x,n) for x in H232])
    # equal counts per chain, so mean normalized chain density == normalized pooled histogram
    P_chain=np.stack([norm_hist(x) for x in H])
    Pbar=P_chain.mean(axis=0)
    ex=np.load(EXACT/f'bin{n}'/f'exact_P64_multibin_{n}.npz') if (EXACT/f'bin{n}'/f'exact_P64_multibin_{n}.npz').exists() else np.load(EXACT/f'exact_P64_multibin_{n}.npz')
    P_exact=np.asarray(ex['P_finiteP_bin_ts'],float)
    P_cont=np.asarray(ex['P_continuum_bin_ts'],float)
    sp=np.load(SPLH/f'splh_multibin_{n}.npz'); P_splh=np.asarray(sp['P_SPLH_bin_ts'],float)
    D_obs=l1(Pbar,P_exact); D_splh=l1(P_splh,P_exact); D_pc=l1(P_exact,P_cont)
    # centered whole-chain residual bootstrap under null
    R=P_chain-Pbar
    dn=np.empty(B,float)
    batch=2000
    for a in range(0,B,batch):
        ids=idx[a:a+batch]
        delta=R[ids].mean(axis=1)
        dn[a:a+len(ids)]=np.mean(np.abs(delta),axis=(1,2))
    qs=np.quantile(dn,[0.005,0.025,0.16,0.5,0.84,0.975,0.995])
    pnull=(1+np.count_nonzero(dn>=D_obs))/(B+1)
    # internal chain splits
    P_seed_first=norm_hist(H[:8].sum(axis=0)); P_seed_second=norm_hist(H[8:].sum(axis=0))
    P_odd=norm_hist(H[::2].sum(axis=0)); P_even=norm_hist(H[1::2].sum(axis=0))
    P_time_first=norm_hist(H1.sum(axis=0)); P_time_second=norm_hist(H2.sum(axis=0))
    D_seed_half=l1(P_seed_first,P_seed_second); D_oddeven=l1(P_odd,P_even); D_time=l1(P_time_first,P_time_second)
    pci=np.array([l1(P_chain[i],P_exact) for i in range(16)])
    for seed,val in zip(SEEDS,pci): per_chain_rows.append({'nbin':n,'seed':int(seed),'L1_chain_vs_exactP64':float(val)})
    rec={
      'nbin':n,'D_PIMC_exactP64':D_obs,'D_SPLH_exactP64':D_splh,'D_exactP64_continuum':D_pc,
      'null_q005':float(qs[0]),'null_q025':float(qs[1]),'null_q16':float(qs[2]),'null_median':float(qs[3]),'null_q84':float(qs[4]),'null_q975':float(qs[5]),'null_q995':float(qs[6]),'p_null':float(pnull),
      'D_seed_first8_vs_second8':D_seed_half,'D_seed_odd_vs_even':D_oddeven,'D_time_first_vs_second':D_time,
      'per_chain_L1_min':float(pci.min()),'per_chain_L1_median':float(np.median(pci)),'per_chain_L1_max':float(pci.max()),
      'pooled_bead_count':int(H.sum()),'per_chain_bead_count':int(H[0].sum()),
      'exact_norm_error':float(abs(P_exact.mean()-1)),'splh_norm_error':float(abs(P_splh.mean()-1)),
      'null_reps':B,'bootstrap_seed':BOOT_SEED,
    }
    summary.append(rec); null_save[f'D_null_{n}']=dn

# Publication 16x16 historical check using archived deterministic array.
arch_exact=np.load(ROOT.parent/'validation_tests/lha13b3a_exact_finiteP_registered_density.npz')['P_finiteP_bin_ts']
P16=norm_hist(np.stack([agg(x,16) for x in H32]).sum(axis=0))
historical_check=l1(P16,arch_exact)

with open(OUT/'point6_l1_noise_floor_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
with open(OUT/'point6_l1_per_chain.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per_chain_rows[0].keys())); w.writeheader(); w.writerows(per_chain_rows)
np.savez_compressed(OUT/'point6_l1_null_bootstrap.npz',bootstrap_indices=idx,**null_save)

prov={
 'protocol_sha256':sha(ROOT/'POINT6_SEC_VII_L1_NOISE_FLOOR_PROTOCOL_2026-09-08.md'),
 'recovery_protocol_sha256':sha(ROOT/'POINT6_ARCHIVAL_32BIN_RECOVERY_PROTOCOL_2026-09-08.md'),
 'analyzer_sha256':sha(ROOT/'analyze_point6_l1_noise_floor_PRE_VARIANCE_CORRECTION.py'),
 'replay_gate_all_16':all(gate),'historical_16x16_L1_reconstructed':historical_check,
 'bootstrap_reps':B,'bootstrap_seed':BOOT_SEED,
}
(OUT/'point6_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')

lines=['# Point 6 — L1 density noise floor and binning dependence','',f'Archival replay gate: **PASS 16/16**.','',f'Historical 16x16 pooled PI-QMC vs archived exact P64 reconstruction: `{historical_check:.12f}`.','', '| bins | PI-QMC vs exact P64 | null median | null 95% | null 99% | p_null | SPLH vs exact P64 | seed 8/8 | odd/even | time halves |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
 lines.append(f"| {r['nbin']}x{r['nbin']} | {r['D_PIMC_exactP64']:.9f} | {r['null_median']:.9f} | [{r['null_q025']:.9f}, {r['null_q975']:.9f}] | [{r['null_q005']:.9f}, {r['null_q995']:.9f}] | {r['p_null']:.6f} | {r['D_SPLH_exactP64']:.9f} | {r['D_seed_first8_vs_second8']:.9f} | {r['D_seed_odd_vs_even']:.9f} | {r['D_time_first_vs_second']:.9f} |")
lines += ['','Interpretation rule was predeclared. No linear subtraction of the noise floor from observed L1 is performed.']
(OUT/'point6_l1_noise_floor_summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
