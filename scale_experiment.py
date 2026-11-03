#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, csv, json, argparse, time
from datetime import datetime
import numpy as np
from qaoa_generator import (
    snake_coords, build_ising, _exact_metrics,
    _run_qaoa_clean, _run_hea_clean, _run_random_clean, _run_sa_clean,
    _setup_logger,
)


def build_problem(rows, cols, problem_seed):
    coords = snake_coords(rows, cols)
    h, J, const, dC, C_target, C_bg = build_ising(coords, seed=problem_seed)
    n = rows * cols
    n_states = 2 ** n
    bits = (np.arange(n_states)[:, None] >> np.arange(n)) & 1
    z = 1 - 2 * bits
    G = const + z @ h + 0.5 * np.einsum("si,ij,sj->s", z, J, z)
    G_star = float(G.min())
    opt_mask = G <= G_star + 1e-9
    return dict(coords=coords, h=h, J=J, const=const, dC=dC,
                C_target=C_target, C_bg=C_bg, G=G, G_star=G_star,
                opt_mask=opt_mask)


def one_run(rows, cols, opt_seed, problem_seed, args, root):
    problem = build_problem(rows, cols, problem_seed)
    run_name = f"{rows}x{cols}_prob{problem_seed}_opt{opt_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    outdir = os.path.join(root, run_name)
    os.makedirs(outdir, exist_ok=False)
    logger = _setup_logger(outdir)
    cfg = vars(args).copy()
    cfg.update(rows=rows, cols=cols, seed=opt_seed, problem_seed=problem_seed)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(outdir, "problem.npz"), **problem)
    logger.info("Scale run: %dx%d | n=%d | problem_seed=%d | optimizer_seed=%d | budget=%dx%d",
                rows, cols, rows*cols, problem_seed, opt_seed, args.generations, args.population)


    local = argparse.Namespace(**vars(args))
    local.seed = opt_seed
    q = _run_qaoa_clean(rows*cols, args.layers, problem["h"], problem["J"], problem["G"],
                         problem["opt_mask"], local, logger, outdir)
    h = _run_hea_clean(rows*cols, args.hea_layers, problem["G"], problem["opt_mask"],
                       local, logger, outdir)
    r = _run_random_clean(rows*cols, args.layers, problem["h"], problem["J"], problem["G"],
                          problem["opt_mask"], local, logger, outdir)
    s = _run_sa_clean(rows*cols, problem["G"], problem["G_star"], local, logger, outdir)

    records=[]
    for name, res in [("QAOA",q),("HEA",h),("Random search",r)]:
        m=_exact_metrics(res["probs"], problem["G"], problem["opt_mask"])
        records.append(dict(rows=rows,cols=cols,n=rows*cols,method=name,seed=opt_seed,
                            problem_seed=problem_seed,parameters=res["n_params"],
                            expG=m["expG"],P_Gstar=m["p_opt"],entropy_bits=m["entropy_bits"],
                            N_eff=m["N_eff"],top_sequence=m["top_sequence"],top_G=m["top_G"],
                            G_star=problem["G_star"],time_sec=res["time_sec"],run_dir=outdir))
    records.append(dict(rows=rows,cols=cols,n=rows*cols,method="SA",seed=opt_seed,
                        problem_seed=problem_seed,parameters=0,expG=np.nan,P_Gstar=np.nan,
                        entropy_bits=np.nan,N_eff=np.nan,top_sequence=s["best_seq"],top_G=s["bestG"],
                        G_star=problem["G_star"],time_sec=s["time_sec"],run_dir=outdir))
    return records


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sizes",default="3x3,3x4,4x4")
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--hea-layers",type=int,default=2)
    ap.add_argument("--generations",type=int,default=200)
    ap.add_argument("--population",type=int,default=40)
    ap.add_argument("--seed",type=int,default=46)
    ap.add_argument("--problem-seed",type=int,default=12345)
    ap.add_argument("--repeats",type=int,default=10)
    ap.add_argument("--backend",choices=["auto","default.qubit","qiskit.aer"],default="auto")
    ap.add_argument("--sample-count",type=int,default=2000)
    ap.add_argument("--sa-reruns",type=int,default=200)
    ap.add_argument("--out-root",default="scale_outputs")
    args=ap.parse_args()
    root=os.path.join(os.path.dirname(os.path.abspath(__file__)),args.out_root)
    os.makedirs(root,exist_ok=True)
    sizes=[tuple(map(int,x.strip().lower().split('x'))) for x in args.sizes.split(',')]
    all_records=[]
    for rows,cols in sizes:
        for k in range(args.repeats):
            opt_seed=args.seed+k
            all_records.extend(one_run(rows,cols,opt_seed,args.problem_seed,args,root))
    path=os.path.join(root,"scale_results.csv")
    fields=list(all_records[0].keys())
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_records)
    # Simple mean +/- std summary; SA distribution metrics stay NaN and are not compared directly.
    groups={}
    for r in all_records: groups.setdefault((r['rows'],r['cols'],r['method']),[]).append(r)
    summary=[]
    for key,items in groups.items():
        def ms(k):
            x=np.array([i[k] for i in items],float); x=x[np.isfinite(x)]
            return (float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.0) if len(x) else (np.nan,np.nan)
        a=ms('expG'); b=ms('P_Gstar'); c=ms('N_eff')
        summary.append(dict(rows=key[0],cols=key[1],method=key[2],n=items[0]['n'],
                            expG_mean=a[0],expG_std=a[1],P_Gstar_mean=b[0],P_Gstar_std=b[1],
                            N_eff_mean=c[0],N_eff_std=c[1],G_star=items[0]['G_star']))
    with open(os.path.join(root,'scale_summary.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print(f"\nScale benchmark finished. Long table: {path}")

if __name__=='__main__': main()
