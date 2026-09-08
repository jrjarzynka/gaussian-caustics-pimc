from pathlib import Path
import sys, importlib.util, numpy as np, json, time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'numerics'))
spec=importlib.util.spec_from_file_location('p5', ROOT/'paper2_secVII_energy_longcheck'/'run_secVII_energy_longcheck.py')
p5=importlib.util.module_from_spec(spec); spec.loader.exec_module(p5)
setup=p5.build_setup()
N_STEPS=40000; BURN_IN=8000; SAMPLE_EVERY=5; P=64; GLOBAL_MOVE_PROB=0.20
OUT=ROOT/'point6_replay_hist32'; OUT.mkdir(exist_ok=True)
edges=np.linspace(0.0,1.0,33)
def hist32(samples):
    flat=samples.reshape(-1,2); uv=(setup['ainv']@flat.T).T; uv=np.mod(uv,1.0)
    return np.histogram2d(uv[:,1],uv[:,0],bins=[edges,edges])[0].astype(np.int64)
def agg16(H): return H.reshape(16,2,16,2).sum(axis=(1,3))
records=[]
for seed in range(131001,131017):
    rng_init=np.random.default_rng(seed+991)
    path=(setup['center_nm'][None,:]+0.05*setup['sigma_link_nm']*rng_init.standard_normal((P,2))).astype(np.float64)
    t0=time.perf_counter()
    samples, al, ag=p5.run_pimc_core_jit_periodic_cell(
        n_steps=N_STEPS,burn_in=BURN_IN,sample_every=SAMPLE_EVERY,p_beads=P,path=path,
        v_grid=setup['vgrid_ev'],origin_x=0.0,origin_y=0.0,
        ainv00=float(setup['ainv'][0,0]),ainv01=float(setup['ainv'][0,1]),ainv10=float(setup['ainv'][1,0]),ainv11=float(setup['ainv'][1,1]),
        kpf=float(setup['kpf']),tau=float(setup['tau']),local_step_nm=float(setup['local_step_nm']),global_step_nm=float(setup['global_step_nm']),
        global_move_prob=float(GLOBAL_MOVE_PROB),seed=int(seed))
    runtime=time.perf_counter()-t0
    samples=np.asarray(samples,float); mid=samples.shape[0]//2
    H=hist32(samples); H1=hist32(samples[:mid]); H2=hist32(samples[mid:])
    arch=np.load(ROOT/'validation_tests'/f'lhap0a_seed_{seed}_hist.npz')
    pass_full=np.array_equal(agg16(H),arch['H_full_ts'])
    pass_first=np.array_equal(agg16(H1),arch['H_first_ts'])
    pass_second=np.array_equal(agg16(H2),arch['H_second_ts'])
    pass_halves=np.array_equal(H,H1+H2)
    rec=dict(seed=seed,pass_full=bool(pass_full),pass_first=bool(pass_first),pass_second=bool(pass_second),pass_halves=bool(pass_halves),acceptance_local=float(al),acceptance_global=float(ag),runtime_seconds=runtime,count=int(H.sum()))
    records.append(rec)
    np.savez_compressed(OUT/f'lhap0a_seed_{seed}_hist32_replay.npz',H_full_ts=H,H_first_ts=H1,H_second_ts=H2,density_edges=edges,n_saved_samples=samples.shape[0],n_beads=P,seed=seed,acceptance_local=float(al),acceptance_global=float(ag))
    print(json.dumps(rec),flush=True)
summary={'all_pass':all(r['pass_full'] and r['pass_first'] and r['pass_second'] and r['pass_halves'] for r in records),'records':records}
(OUT/'replay_gate_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('ALL_PASS',summary['all_pass'])
