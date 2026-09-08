#!/usr/bin/env python3
"""Final Point-6 whole-chain residual bootstrap with finite-n variance correction.

Uses the archived 32x32 replay histograms and deterministic exact/SPLH multibin
references bundled in this directory. Independent chains are the top-level units.
The centered residuals are multiplied by sqrt(n/(n-1)) before resampling.
"""
from pathlib import Path
import argparse, csv
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS=np.arange(131001,131017,dtype=int)
B=200000; BOOT_SEED=20260908; RES=(8,16,32)
SCALE=np.sqrt(len(SEEDS)/(len(SEEDS)-1))

def agg(H,n):
    H=np.asarray(H)
    if H.shape==(n,n): return H.copy()
    if H.shape!=(32,32) or 32 % n: raise ValueError((H.shape,n))
    k=32//n
    return H.reshape(n,k,n,k).sum(axis=(1,3))

def norm(H):
    H=np.asarray(H,float)
    return H/H.mean()

def l1(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, default=HERE,
                    help='Point-6 directory containing replay_hist32/, exact_multibin/, splh_multibin/.')
    ap.add_argument('--out', type=Path, default=HERE/'results',
                    help='Output directory for corrected final bootstrap products.')
    args=ap.parse_args(); root=args.source_root.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    H32=[]
    for s in SEEDS:
        z=np.load(root/'replay_hist32'/f'lhap0a_seed_{s}_hist32_replay.npz')
        H32.append(z['H_full_ts'])
    H32=np.stack(H32)
    rng=np.random.default_rng(BOOT_SEED)
    idx=rng.integers(0,len(SEEDS),size=(B,len(SEEDS)),dtype=np.int16)
    rows=[]; save={'bootstrap_indices':idx,'residual_scale':np.array(SCALE)}
    for n in RES:
        H=np.stack([agg(x,n) for x in H32])
        P_chain=np.stack([norm(x) for x in H])
        Pbar=P_chain.mean(axis=0)
        ex=np.load(root/'exact_multibin'/f'exact_P64_multibin_{n}.npz')
        sp=np.load(root/'splh_multibin'/f'splh_multibin_{n}.npz')
        Pex=ex['P_finiteP_bin_ts']; Ps=sp['P_SPLH_bin_ts']
        Dobs=l1(Pbar,Pex); Dsplh=l1(Ps,Pex)
        R=(P_chain-Pbar)*SCALE
        dn=np.empty(B,float); batch=2000
        for a in range(0,B,batch):
            ids=idx[a:a+batch]
            delta=R[ids].mean(axis=1)
            dn[a:a+len(ids)]=np.mean(np.abs(delta),axis=(1,2))
        q=np.quantile(dn,[.005,.025,.16,.5,.84,.975,.995])
        p=(1+np.count_nonzero(dn>=Dobs))/(B+1)
        rows.append(dict(nbin=n,D_PIMC_exactP64=Dobs,D_SPLH_exactP64=Dsplh,residual_scale=SCALE,
                         null_q005=q[0],null_q025=q[1],null_q16=q[2],null_median=q[3],null_q84=q[4],
                         null_q975=q[5],null_q995=q[6],p_null=p))
        save[f'D_null_{n}']=dn
    with open(out/'point6_l1_noise_floor_summary_variance_corrected.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out/'point6_l1_null_bootstrap_variance_corrected.npz',**save)
    print(f'residual scale sqrt(16/15) = {SCALE:.12f}')
    for r in rows: print(r)

if __name__=='__main__': main()
