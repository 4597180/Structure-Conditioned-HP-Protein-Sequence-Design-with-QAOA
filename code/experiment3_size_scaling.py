#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import io
import csv
import json
import time
import math
import argparse
import traceback
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make qaoa_generator.py and evolutionary.py importable from the same folder.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from qaoa_generator import QAOACircuit, snake_coords, build_ising, direct_G, distribution_entropy
from evolutionary import EvolutionaryOptimizer


class Tee:
    """Write simultaneously to console and a file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def save_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_sd_ci(values, confidence=0.95):
    """Mean, sample SD, and approximate t-based CI half-width."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    if len(x) == 1:
        return m, 0.0, 0.0
    sd = float(np.std(x, ddof=1))
    # Avoid a scipy dependency. Critical values for n=10 and common cases
    # are only used for the default 10-seed analysis.
    tcrit_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                   6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
                   10: 2.262, 11: 2.228, 12: 2.201, 15: 2.131,
                   20: 2.093, 25: 2.064, 30: 2.045}
    df = len(x) - 1
    tcrit = tcrit_table.get(df, 1.96)
    half = float(tcrit * sd / math.sqrt(len(x)))
    return m, sd, half


def sequence_from_state(state, n):
    return "".join("H" if ((int(state) >> i) & 1) == 0 else "P" for i in range(n))


def build_state_space(n, h, J, const):
    """Exact G(s) for all 2^n states."""
    n_states = 2 ** n
    states = np.arange(n_states, dtype=np.int64)
    bits = (states[:, None] >> np.arange(n, dtype=np.int64)) & 1
    z = 1 - 2 * bits
    linear = z @ h
    quad = 0.5 * np.einsum("si,ij,sj->s", z, J, z)
    G = const + linear + quad
    return states, G, z


def build_problem(rows, cols, problem_seed, n_background=30):
    coords = snake_coords(rows, cols)
    h, J, const, dC, C_target, C_bg = build_ising(
        coords, n_background=n_background, seed=problem_seed
    )
    n = rows * cols
    states, G_of_state, z_matrix = build_state_space(n, h, J, const)

    # Exact consistency check over all states for these problem sizes.
    if n <= 16:
        exact_err = 0.0
        for s in states:
            exact_err = max(exact_err, abs(
                float(G_of_state[s]) - direct_G(z_matrix[s], dC)
            ))
    else:
        exact_err = float("nan")

    G_star = float(np.min(G_of_state))
    opt_mask = G_of_state <= G_star + 1e-9
    n_opt = int(np.sum(opt_mask))
    G_uniform = float(np.mean(G_of_state))

    # Normalized energy progress: 0 = uniform baseline, 1 = exact optimum.
    # For these instances G_uniform > G_star, so this is well-defined.
    denom = G_uniform - G_star
    uniform_P_Gstar = float(n_opt / (2 ** n))

    problem = {
        "rows": rows,
        "cols": cols,
        "n": n,
        "n_states": int(2 ** n),
        "problem_seed": int(problem_seed),
        "n_background": int(n_background),
        "target_contacts": int(np.sum(C_target) / 2),
        "G_star": G_star,
        "n_opt": n_opt,
        "G_uniform": G_uniform,
        "uniform_P_Gstar": uniform_P_Gstar,
        "normalized_uniform_to_opt_denominator": float(denom),
        "ising_max_error": float(exact_err),
        "coords": coords,
        "h": h,
        "J": J,
        "const": const,
        "dC": dC,
        "C_target": C_target,
        "C_bg": C_bg,
        "states": states,
        "G_of_state": G_of_state,
    }
    return problem


# -----------------------------------------------------------------------------
# QAOA run
# -----------------------------------------------------------------------------

def evaluate_distribution(probs, G_of_state, opt_mask, G_star, G_uniform):
    probs = np.asarray(probs, dtype=float)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if total <= 0:
        raise RuntimeError("QAOA returned a non-positive probability sum")
    probs /= total

    expG = float(np.dot(probs, G_of_state))
    p_opt = float(np.sum(probs[opt_mask]))
    H, Neff = distribution_entropy(probs)
    top_state = int(np.argmax(probs))
    top_p = float(probs[top_state])
    top_G = float(G_of_state[top_state])
    normalized_progress = float(
        (G_uniform - expG) / (G_uniform - G_star)
    ) if abs(G_uniform - G_star) > 1e-15 else float("nan")

    m95 = float(math.log(0.05) / math.log(max(1e-15, 1.0 - p_opt))) if p_opt > 0 and p_opt < 1 else (0.0 if p_opt >= 1 else float("inf"))

    return {
        "expG": expG,
        "P_Gstar": p_opt,
        "entropy_bits": float(H),
        "N_eff": float(Neff),
        "top_state": top_state,
        "top_probability": top_p,
        "top_G": top_G,
        "normalized_progress": normalized_progress,
        "m95_samples": m95,
    }, probs


def run_qaoa_one(problem, rows, cols, seed, args, run_dir):
    n = problem["n"]
    G_of_state = problem["G_of_state"]
    G_star = problem["G_star"]
    G_uniform = problem["G_uniform"]
    opt_mask = G_of_state <= G_star + 1e-9
    p = args.layers
    population = args.population
    generations = args.generations

    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Tee the complete run log.
    log_path = os.path.join(run_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as logf:
        tee = Tee(sys.stdout, logf)
        with redirect_stdout(tee):
            print("=" * 78)
            print("Experiment 3 — QAOA size scaling")
            print("=" * 78)
            print(f"Size: {rows}x{cols} | n={n} | states={2**n:,}")
            print(f"p={p} | parameters={2*p} | population={population} | generations={generations}")
            print(f"problem_seed={args.problem_seed} | optimizer_seed={seed}")
            print(f"backend request={args.backend}")
            print(f"G*={G_star:.10f} | uniform <G>={G_uniform:.10f} | n_opt={problem['n_opt']}")
            print(f"target contacts={problem['target_contacts']} | Ising max error={problem['ising_max_error']:.3e}")

            # Save problem data first.
            np.savez_compressed(
                os.path.join(run_dir, "problem.npz"),
                coords=problem["coords"],
                h=problem["h"],
                J=problem["J"],
                const=problem["const"],
                dC=problem["dC"],
                C_target=problem["C_target"],
                C_bg=problem["C_bg"],
                G_of_state=G_of_state,
                states=problem["states"],
            )
            save_json(os.path.join(run_dir, "problem_summary.json"), {
                k: v for k, v in problem.items() if k not in {
                    "h", "J", "dC", "C_target", "C_bg", "states", "G_of_state"
                }
            })

            config = {
                "experiment": "Experiment 3 — QAOA size scaling",
                "rows": rows,
                "cols": cols,
                "n": n,
                "n_states": int(2 ** n),
                "layers_p": p,
                "n_trainable_parameters": 2 * p,
                "population": population,
                "generations": generations,
                "problem_seed": args.problem_seed,
                "optimizer_seed": seed,
                "n_background": args.n_background,
                "sample_count": args.sample_count,
                "backend": args.backend,
                "mutation_rate": 0.15,
                "crossover_rate": 0.7,
                "mutation_sigma": 0.5,
                "elite_size": max(2, population // 6),
                "tournament_size": 3,
            }
            save_json(os.path.join(run_dir, "config.json"), config)

            np.random.seed(seed)
            circ = QAOACircuit(n, problem["h"], problem["J"], backend=args.backend)
            print(f"QAOA backend actually used: {circ.backend}")

            # Backend-vs-NumPy validation on a random parameter vector.
            backend_check = None
            if circ._pl is not None:
                try:
                    from qaoa_generator import _run_numpy
                    rg = np.random.RandomState(seed + 7001).uniform(0, 2*np.pi, p)
                    rb = np.random.RandomState(seed + 7002).uniform(0, 2*np.pi, p)
                    pp = circ.probs(rg, rb, G_of_state)
                    pn = _run_numpy(rg, rb, G_of_state, n)
                    backend_check = float(np.max(np.abs(pp - pn)))
                    print(f"Backend vs NumPy max probability deviation: {backend_check:.3e}")
                except Exception as e:
                    print(f"Backend validation skipped: {type(e).__name__}: {e}")

            def split_params(params):
                return params[:p], params[p:]

            eval_counter = {"n": 0}

            def probs_fn(params):
                g, b = split_params(params)
                eval_counter["n"] += 1
                return circ.probs(g, b, G_of_state)

            def fitness_fn(params):
                probs = probs_fn(params)
                return float(np.dot(probs, G_of_state))

            bounds = [(0.0, 2*np.pi)] * (2*p)
            opt = EvolutionaryOptimizer(
                n_params=2*p,
                param_bounds=bounds,
                fitness_fn=fitness_fn,
                population_size=population,
                elite_size=max(2, population // 6),
                tournament_size=3,
                mutation_rate=0.15,
                crossover_rate=0.7,
                mutation_sigma=0.5,
                mutation_strategy="gaussian",
            )

            print("\nInitializing population ...")
            opt.initialize_population(seed=seed, verbose=False)

            generation_rows = []
            report_every = max(1, generations // 20)
            t0 = time.time()

            # Generation 0 = initialized population best individual.
            best = opt.best_individual
            m, probs = evaluate_distribution(
                circ.probs(*split_params(best.params), G_of_state),
                G_of_state, opt_mask, G_star, G_uniform
            )
            generation_rows.append({
                "generation": 0,
                "best_fitness": float(best.fitness),
                "expG": m["expG"],
                "P_Gstar": m["P_Gstar"],
                "entropy_bits": m["entropy_bits"],
                "N_eff": m["N_eff"],
                "normalized_progress": m["normalized_progress"],
                "top_state": m["top_state"],
                "top_sequence": sequence_from_state(m["top_state"], n),
                "top_probability": m["top_probability"],
                "top_G": m["top_G"],
                "population_mean_fitness": float(np.mean([x.fitness for x in opt.population])),
                "population_std_fitness": float(np.std([x.fitness for x in opt.population])),
                "mutation_rate": float(opt.mutation_rate),
                "objective_evaluations": int(eval_counter["n"]),
                "elapsed_seconds": 0.0,
            })

            print(f"Gen {0:4d}/{generations} | <G>={m['expG']:.6f} | "
                  f"P(G*)={100*m['P_Gstar']:.3f}% | N_eff={m['N_eff']:.2f} | "
                  f"top={generation_rows[-1]['top_sequence']} | topG={m['top_G']:.6f}")

            for gen in range(1, generations + 1):
                opt.run_generation()
                best = opt.best_individual
                g, b = split_params(best.params)
                probs = circ.probs(g, b, G_of_state)
                eval_counter["n"] += 1
                m, probs = evaluate_distribution(
                    probs, G_of_state, opt_mask, G_star, G_uniform
                )

                pop_f = np.asarray([x.fitness for x in opt.population], dtype=float)
                elapsed = time.time() - t0
                row = {
                    "generation": gen,
                    "best_fitness": float(best.fitness),
                    "expG": m["expG"],
                    "P_Gstar": m["P_Gstar"],
                    "entropy_bits": m["entropy_bits"],
                    "N_eff": m["N_eff"],
                    "normalized_progress": m["normalized_progress"],
                    "top_state": m["top_state"],
                    "top_sequence": sequence_from_state(m["top_state"], n),
                    "top_probability": m["top_probability"],
                    "top_G": m["top_G"],
                    "population_mean_fitness": float(np.mean(pop_f)),
                    "population_std_fitness": float(np.std(pop_f)),
                    "mutation_rate": float(opt.mutation_rate),
                    "objective_evaluations": int(eval_counter["n"]),
                    "elapsed_seconds": float(elapsed),
                }
                generation_rows.append(row)

                if gen % report_every == 0 or gen == generations:
                    print(f"Gen {gen:4d}/{generations} | <G>={m['expG']:.6f} | "
                          f"P(G*)={100*m['P_Gstar']:.3f}% | N_eff={m['N_eff']:.2f} | "
                          f"top={row['top_sequence']} | topG={m['top_G']:.6f} | "
                          f"elapsed={elapsed:.1f}s")

            elapsed_total = time.time() - t0
            best = opt.best_individual
            g, b = split_params(best.params)
            final_probs = circ.probs(g, b, G_of_state)
            eval_counter["n"] += 1
            final_metrics, final_probs = evaluate_distribution(
                final_probs, G_of_state, opt_mask, G_star, G_uniform
            )

            # Sample from the exact final Born distribution. This is a diagnostic,
            # not the primary endpoint.
            sample_rng = np.random.RandomState(seed ^ 0x9E3779B9)
            sample_states = sample_rng.choice(
                len(G_of_state), size=args.sample_count, p=final_probs
            )
            sampled_G = G_of_state[sample_states]
            best_sampled_G = float(np.min(sampled_G))
            sampled_opt_hit_rate = float(np.mean(sampled_G <= G_star + 1e-9))

            # Exact rank table.
            order_by_energy = np.argsort(G_of_state)
            order_by_probability = np.argsort(final_probs)[::-1]
            top20_rows = []
            for rank, s in enumerate(order_by_probability[:20], start=1):
                top20_rows.append({
                    "probability_rank": rank,
                    "state": int(s),
                    "sequence": sequence_from_state(s, n),
                    "G": float(G_of_state[s]),
                    "probability": float(final_probs[s]),
                    "is_global_optimum": bool(opt_mask[s]),
                    "energy_rank": int(np.where(order_by_energy == s)[0][0] + 1),
                })

            # Full state distribution is intentionally exact for all three sizes.
            dist_rows = []
            for s in range(len(G_of_state)):
                dist_rows.append({
                    "state": s,
                    "sequence": sequence_from_state(s, n),
                    "G": float(G_of_state[s]),
                    "probability": float(final_probs[s]),
                    "is_global_optimum": int(opt_mask[s]),
                })

            save_csv(os.path.join(run_dir, "generation_metrics.csv"), generation_rows)
            save_csv(os.path.join(run_dir, "top20_sequences.csv"), top20_rows)
            save_csv(os.path.join(run_dir, "final_state_distribution.csv"), dist_rows,
                     fieldnames=["state", "sequence", "G", "probability", "is_global_optimum"])

            # Energy-resolved probability mass: P(G)=sum_{s:G(s)=G} P(s).
            energy_mass = {}
            for gv, pv in zip(G_of_state, final_probs):
                key = round(float(gv), 10)
                energy_mass[key] = energy_mass.get(key, 0.0) + float(pv)
            energy_rows = [
                {"G": gv, "probability_mass": pv, "is_global_optimum": int(abs(gv - G_star) <= 1e-9)}
                for gv, pv in sorted(energy_mass.items())
            ]
            save_csv(os.path.join(run_dir, "energy_resolved_probability.csv"), energy_rows)

            final_summary = {
                **config,
                "actual_backend": circ.backend,
                "backend_numpy_max_deviation": backend_check,
                "objective_evaluations": int(eval_counter["n"]),
                "runtime_seconds": float(elapsed_total),
                "G_star": G_star,
                "G_uniform": G_uniform,
                "uniform_P_Gstar": problem["uniform_P_Gstar"],
                "n_opt": problem["n_opt"],
                "target_contacts": problem["target_contacts"],
                "problem_seed": problem["problem_seed"],
                "ising_max_error": problem["ising_max_error"],
                **final_metrics,
                "best_sampled_G": best_sampled_G,
                "sampled_optimum_hit_rate": sampled_opt_hit_rate,
                "top_sequence": sequence_from_state(final_metrics["top_state"], n),
                "m95_samples": final_metrics["m95_samples"],
                "relative_energy_gap": float((final_metrics["expG"] - G_star) / abs(G_star)) if abs(G_star) > 1e-15 else float("nan"),
                "absolute_energy_gap": float(final_metrics["expG"] - G_star),
                "qaoa_parameter_vector": best.params,
                "gammas": g,
                "betas": b,
            }
            save_json(os.path.join(run_dir, "summary.json"), final_summary)
            save_csv(os.path.join(run_dir, "final_metrics.csv"), [final_summary])
            np.savez_compressed(
                os.path.join(run_dir, "best_params.npz"),
                params=best.params,
                gammas=g,
                betas=b,
            )

            print("\n" + "-" * 78)
            print("FINAL")
            print("-" * 78)
            print(f"<G>                 = {final_metrics['expG']:.8f}")
            print(f"P(G*)               = {100*final_metrics['P_Gstar']:.4f}%")
            print(f"H                   = {final_metrics['entropy_bits']:.6f} bit")
            print(f"N_eff               = {final_metrics['N_eff']:.4f}")
            print(f"normalized progress = {final_metrics['normalized_progress']:.6f}")
            print(f"relative energy gap = {final_summary['relative_energy_gap']:.6f}")
            print(f"top sequence        = {final_summary['top_sequence']}")
            print(f"top sequence P      = {100*final_metrics['top_probability']:.4f}%")
            print(f"top sequence G      = {final_metrics['top_G']:.8f}")
            print(f"best sampled G      = {best_sampled_G:.8f}")
            print(f"sampled P hit       = {100*sampled_opt_hit_rate:.4f}%")
            print(f"runtime             = {elapsed_total:.2f} s")
            print(f"objective evals     = {eval_counter['n']}")
            print("Top-10 by probability:")
            for r in top20_rows[:10]:
                flag = "*" if r["is_global_optimum"] else " "
                print(f"  {r['probability_rank']:2d}. {r['sequence']}  "
                      f"G={r['G']:.6f}  P={100*r['probability']:.4f}% {flag}")

    return final_summary, generation_rows, top20_rows


# -----------------------------------------------------------------------------
# Aggregation / statistics
# -----------------------------------------------------------------------------

def holm_adjust(pvals):
    """Holm step-down adjustment, preserving input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def paired_wilcoxon_exact_or_normal(x, y):
    """Small dependency-free paired Wilcoxon signed-rank approximation.

    Returns statistic and a two-sided normal-approximation p-value. This is
    explicitly labelled exploratory in the output. For publication, the user
    can verify with scipy.stats.wilcoxon if desired.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    absd = np.abs(d)
    order = np.argsort(absd)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and absd[order[j]] == absd[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    Wplus = float(np.sum(ranks[d > 0]))
    Wminus = float(np.sum(ranks[d < 0]))
    W = min(Wplus, Wminus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2*n + 1) / 24.0
    if var <= 0:
        return W, 1.0
    z = (W - mean) / math.sqrt(var)
    # normal CDF without scipy
    Phi = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    p = 2.0 * (1.0 - Phi)
    return W, float(min(1.0, p))


def aggregate_results(all_summaries):
    sizes = sorted(set((r["rows"], r["cols"]) for r in all_summaries), key=lambda x: x[0]*x[1])
    agg = []
    for rows, cols in sizes:
        items = [r for r in all_summaries if r["rows"] == rows and r["cols"] == cols]
        for key in ["expG", "P_Gstar", "entropy_bits", "N_eff",
                    "normalized_progress", "relative_energy_gap",
                    "absolute_energy_gap", "runtime_seconds",
                    "best_sampled_G", "sampled_optimum_hit_rate",
                    "top_probability", "m95_samples"]:
            vals = [r[key] for r in items]
            m, sd, ci = mean_sd_ci(vals)
            agg_key = key
            row = None
            # build later from dict
            if key == "expG":
                row = {"rows": rows, "cols": cols, "n": items[0]["n"],
                       "n_states": items[0]["n_states"],
                       "seeds": len(items), "params": items[0]["n_trainable_parameters"],
                       "G_star": items[0]["G_star"], "G_uniform": items[0]["G_uniform"],
                       "n_opt": items[0]["n_opt"], "target_contacts": items[0]["target_contacts"],
                       "uniform_P_Gstar": items[0]["uniform_P_Gstar"]}
                agg.append(row)
            agg[-1][f"{agg_key}_mean"] = m
            agg[-1][f"{agg_key}_sd"] = sd
            agg[-1][f"{agg_key}_ci95"] = ci
    return agg


def write_statistics(all_summaries, agg, outdir):
    # Long summary table for the paper/Excel.
    stats_rows = []
    for a in agg:
        stats_rows.append(a)
    save_csv(os.path.join(outdir, "size_summary.csv"), stats_rows)

    # Per-seed compact table.
    seed_rows = []
    for r in all_summaries:
        seed_rows.append({
            "rows": r["rows"], "cols": r["cols"], "n": r["n"],
            "seed": r["optimizer_seed"], "G_star": r["G_star"],
            "G_uniform": r["G_uniform"], "uniform_P_Gstar": r["uniform_P_Gstar"],
            "expG": r["expG"],
            "P_Gstar": r["P_Gstar"], "entropy_bits": r["entropy_bits"],
            "N_eff": r["N_eff"], "normalized_progress": r["normalized_progress"],
            "relative_energy_gap": r["relative_energy_gap"],
            "top_probability": r["top_probability"],
            "top_sequence": r["top_sequence"], "top_G": r["top_G"],
            "runtime_seconds": r["runtime_seconds"],
            "objective_evaluations": r["objective_evaluations"],
            "best_sampled_G": r["best_sampled_G"],
            "sampled_optimum_hit_rate": r["sampled_optimum_hit_rate"],
            "m95_samples": r["m95_samples"],
        })
    save_csv(os.path.join(outdir, "run_summary.csv"), seed_rows)

    # Detailed textual statistical report.
    sizes = sorted(set((r["rows"], r["cols"]) for r in all_summaries), key=lambda x: x[0]*x[1])
    lines = []
    lines.append("Experiment 3 statistical summary")
    lines.append("=" * 72)
    lines.append("Descriptive statistics are primary. Inferential tests are exploratory")
    lines.append("because size changes the underlying Hamiltonian/problem instance.")
    lines.append("")

    for a in agg:
        lines.append(f"{a['rows']}x{a['cols']} (n={a['n']}, states={a['n_states']:,}, seeds={a['seeds']})")
        lines.append(f"  G* = {a['G_star']:.8f}; uniform <G> = {a['G_uniform']:.8f}; n_opt = {a['n_opt']}")
        lines.append(f"  <G>                 = {a['expG_mean']:.8f} ± {a['expG_sd']:.8f}  (95% CI ± {a['expG_ci95']:.8f})")
        lines.append(f"  P(G*)               = {100*a['P_Gstar_mean']:.4f}% ± {100*a['P_Gstar_sd']:.4f}%  (95% CI ± {100*a['P_Gstar_ci95']:.4f}%)")
        lines.append(f"  H                   = {a['entropy_bits_mean']:.6f} ± {a['entropy_bits_sd']:.6f} bit")
        lines.append(f"  N_eff               = {a['N_eff_mean']:.4f} ± {a['N_eff_sd']:.4f}")
        lines.append(f"  normalized progress = {a['normalized_progress_mean']:.6f} ± {a['normalized_progress_sd']:.6f}")
        lines.append(f"  relative gap        = {a['relative_energy_gap_mean']:.6f} ± {a['relative_energy_gap_sd']:.6f}")
        lines.append(f"  runtime              = {a['runtime_seconds_mean']:.2f} ± {a['runtime_seconds_sd']:.2f} s")
        lines.append("")

    # If exactly three sizes are present, perform exploratory paired tests
    # using common optimizer seeds. These are NOT claims of independent
    # identical problems; they are seed-matched sensitivity analyses.
    if len(sizes) >= 2:
        lines.append("Exploratory paired Wilcoxon signed-rank comparisons")
        lines.append("(seed-matched; normal approximation; Holm adjustment across pairwise tests)")
        lines.append("")
        for metric in ["P_Gstar", "expG", "normalized_progress"]:
            pvals = []
            meta = []
            for i in range(len(sizes)):
                for j in range(i+1, len(sizes)):
                    a, b = sizes[i], sizes[j]
                    xa = {r["optimizer_seed"]: r[metric] for r in all_summaries if (r["rows"], r["cols"]) == a}
                    xb = {r["optimizer_seed"]: r[metric] for r in all_summaries if (r["rows"], r["cols"]) == b}
                    common = sorted(set(xa) & set(xb))
                    if not common:
                        continue
                    x = [xa[s] for s in common]
                    y = [xb[s] for s in common]
                    W, p = paired_wilcoxon_exact_or_normal(x, y)
                    pvals.append(p)
                    meta.append((a, b, W, p, len(common)))
            adj = holm_adjust(pvals) if pvals else []
            for k, (a, b, W, p, ncommon) in enumerate(meta):
                lines.append(f"  {metric}: {a[0]}x{a[1]} vs {b[0]}x{b[1]}: "
                             f"W={W:.3f}, p={p:.6g}, Holm-p={adj[k]:.6g}, n={ncommon}")
        lines.append("")
        lines.append("These exploratory p-values should not be presented as evidence of")
        lines.append("statistical superiority. They support sensitivity analysis only.")

    with open(os.path.join(outdir, "statistical_tests.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def plot_scaling(agg, outdir):
    sizes = sorted(agg, key=lambda r: r["n"])
    labels = [f"{r['rows']}×{r['cols']}" for r in sizes]
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(x, [r["expG_mean"] for r in sizes],
                yerr=[r["expG_sd"] for r in sizes], marker="o", linewidth=2, capsize=4)
    ax.plot(x, [r["G_star"] for r in sizes], linestyle="--", linewidth=1.5, label="Exact G*")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("Expected objective ⟨G⟩")
    ax.set_title("Experiment 3A: Expected objective vs problem size")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure6a_expected_objective.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(x, [100*r["P_Gstar_mean"] for r in sizes],
                yerr=[100*r["P_Gstar_sd"] for r in sizes], marker="o", linewidth=2, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("P(G*) (%)")
    ax.set_title("Experiment 3B: Optimal-state probability vs problem size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure6b_PGstar.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(x, [r["normalized_progress_mean"] for r in sizes],
                yerr=[r["normalized_progress_sd"] for r in sizes], marker="o", linewidth=2, capsize=4)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("Normalized progress toward G*")
    ax.set_title("Experiment 3C: Normalized energy progress")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure6c_normalized_progress.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(x, [r["N_eff_mean"] for r in sizes],
                yerr=[r["N_eff_sd"] for r in sizes], marker="o", linewidth=2, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("N_eff = 2^H")
    ax.set_yscale("log")
    ax.set_title("Experiment 3D: Effective sequence count")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure6d_Neff.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Combined journal-ready 2x2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.errorbar(x, [r["expG_mean"] for r in sizes], yerr=[r["expG_sd"] for r in sizes],
                marker="o", linewidth=2, capsize=4)
    ax.plot(x, [r["G_star"] for r in sizes], linestyle="--", linewidth=1.2, label="Exact G*")
    ax.set_ylabel("Expected objective ⟨G⟩"); ax.set_title("(a) Expected objective")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(x, [100*r["P_Gstar_mean"] for r in sizes], yerr=[100*r["P_Gstar_sd"] for r in sizes],
                marker="o", linewidth=2, capsize=4)
    ax.plot(x, [100/(r["n_states"]) for r in sizes], linestyle="--", linewidth=1.2,
            label="Uniform baseline")
    ax.set_ylabel("P(G*) (%)"); ax.set_title("(b) Optimal-state probability")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(x, [r["normalized_progress_mean"] for r in sizes],
                yerr=[r["normalized_progress_sd"] for r in sizes], marker="o", linewidth=2, capsize=4)
    ax.axhline(0, linestyle="--", linewidth=1.0); ax.axhline(1, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Lattice size"); ax.set_ylabel("Normalized progress")
    ax.set_title("(c) Progress from uniform to G*")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(x, [r["N_eff_mean"] for r in sizes], yerr=[r["N_eff_sd"] for r in sizes],
                marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("Lattice size"); ax.set_ylabel("N_eff = 2^H")
    ax.set_title("(d) Distribution concentration")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")

    fig.suptitle("Experiment 3 — QAOA size scaling (p=5, mean ± SD, 10 optimizer seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "Figure6_scaling_metrics.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Runtime / search-space diagnostic.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(x, [r["runtime_seconds_mean"] for r in sizes],
                 yerr=[r["runtime_seconds_sd"] for r in sizes], marker="o", linewidth=2, capsize=4)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Lattice size")
    ax1.set_ylabel("Runtime per run (s)")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3, which="both")
    ax2 = ax1.twinx()
    ax2.plot(x, [r["n_states"] for r in sizes], linestyle="--", marker="s", linewidth=1.5)
    ax2.set_ylabel("Exact state-space size 2^n")
    fig.suptitle("Runtime and exact state-space scaling")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigureS4_runtime_and_search_space.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(all_generations, outdir):
    groups = {}
    for row in all_generations:
        key = (row["rows"], row["cols"])
        groups.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key in sorted(groups, key=lambda k: k[0]*k[1]):
        rows = groups[key]
        gens = sorted(set(int(r["generation"]) for r in rows))
        means, sds = [], []
        for g in gens:
            vals = [r["expG"] for r in rows if int(r["generation"]) == g]
            means.append(float(np.mean(vals))); sds.append(float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0)
        label = f"{key[0]}×{key[1]}"
        ax.plot(gens, means, linewidth=2, label=label)
        ax.fill_between(gens, np.array(means)-np.array(sds), np.array(means)+np.array(sds), alpha=0.12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Expected objective ⟨G⟩")
    ax.set_title("Experiment 3 — QAOA convergence by problem size")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure7_convergence_by_size.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Save aggregated convergence table.
    conv_rows = []
    for key in sorted(groups, key=lambda k: k[0]*k[1]):
        rows = groups[key]
        for g in sorted(set(int(r["generation"]) for r in rows)):
            sub = [r for r in rows if int(r["generation"]) == g]
            conv_rows.append({
                "rows": key[0], "cols": key[1], "n": key[0]*key[1],
                "generation": g,
                "expG_mean": float(np.mean([r["expG"] for r in sub])),
                "expG_sd": float(np.std([r["expG"] for r in sub], ddof=1)) if len(sub)>1 else 0.0,
                "P_Gstar_mean": float(np.mean([r["P_Gstar"] for r in sub])),
                "P_Gstar_sd": float(np.std([r["P_Gstar"] for r in sub], ddof=1)) if len(sub)>1 else 0.0,
                "N_eff_mean": float(np.mean([r["N_eff"] for r in sub])),
                "N_eff_sd": float(np.std([r["N_eff"] for r in sub], ddof=1)) if len(sub)>1 else 0.0,
            })
    save_csv(os.path.join(outdir, "figure_data", "convergence_mean.csv"), conv_rows)


def plot_seed_robustness(all_summaries, outdir):
    sizes = sorted(set((r["rows"], r["cols"]) for r in all_summaries), key=lambda x: x[0]*x[1])
    seeds = sorted(set(r["optimizer_seed"] for r in all_summaries))
    matrix = np.full((len(seeds), len(sizes)), np.nan)
    for i, seed in enumerate(seeds):
        for j, size in enumerate(sizes):
            vals = [r["P_Gstar"] for r in all_summaries
                    if r["optimizer_seed"] == seed and (r["rows"], r["cols"]) == size]
            if vals:
                matrix[i, j] = vals[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix * 100, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([f"{a}×{b}" for a,b in sizes])
    ax.set_yticks(np.arange(len(seeds)))
    ax.set_yticklabels([str(s) for s in seeds])
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("Optimizer seed")
    ax.set_title("Experiment 3 — P(G*) across optimizer seeds")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(G*) (%)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i,j]):
                ax.text(j, i, f"{100*matrix[i,j]:.1f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigureS3_seed_robustness.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_csv(os.path.join(outdir, "figure_data", "seed_robustness.csv"), [
        {"seed": seeds[i], **{f"{a}x{b}_P_Gstar": matrix[i,j] for j,(a,b) in enumerate(sizes)}}
        for i in range(len(seeds))
    ])


def save_problem_summary(all_summaries, outdir):
    rows = []
    seen = set()
    for r in all_summaries:
        key = (r["rows"], r["cols"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "rows": r["rows"], "cols": r["cols"], "n": r["n"],
            "n_states": r["n_states"], "problem_seed": r["problem_seed"],
            "target_contacts": r["target_contacts"], "G_star": r["G_star"],
            "n_opt": r["n_opt"], "G_uniform": r["G_uniform"],
            "uniform_P_Gstar": r["uniform_P_Gstar"],
            "ising_max_error": r["ising_max_error"],
        })
    save_csv(os.path.join(outdir, "problem_summary.csv"), rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Experiment 3 — QAOA size scaling")
    ap.add_argument("--sizes", default="3x3,3x4,4x4",
                    help="comma-separated lattice sizes")
    ap.add_argument("--layers", type=int, default=5,
                    help="QAOA depth p; Experiment 2 selected p=5 for this scaling study")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=46)
    ap.add_argument("--seed-end", type=int, default=55)
    ap.add_argument("--problem-seed", type=int, default=12345)
    ap.add_argument("--n-background", type=int, default=30)
    ap.add_argument("--sample-count", type=int, default=2000)
    ap.add_argument("--backend", choices=["auto", "default.qubit", "qiskit.aer"], default="auto")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="continue other sizes/seeds after a failed run")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    if args.layers < 1:
        raise ValueError("--layers must be >= 1")
    if args.generations < 1 or args.population < 2:
        raise ValueError("generations must be >=1 and population >=2")

    sizes = []
    for item in args.sizes.split(","):
        item = item.strip().lower()
        r, c = item.split("x")
        sizes.append((int(r), int(c)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.abspath(os.path.join(args.out_root, f"experiment3_size_scaling_{timestamp}"))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "figure_data"), exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_end + 1))
    experiment_config = {
        "experiment": "Experiment 3 — QAOA size scaling",
        "timestamp": timestamp,
        "sizes": [f"{r}x{c}" for r,c in sizes],
        "layers_p": args.layers,
        "population": args.population,
        "generations": args.generations,
        "optimizer_seeds": seeds,
        "problem_seed": args.problem_seed,
        "n_background": args.n_background,
        "sample_count": args.sample_count,
        "backend_requested": args.backend,
        "same_budget_all_sizes": True,
        "exact_enumeration_all_sizes": True,
        "primary_endpoint": "P(G*)",
        "secondary_endpoints": ["<G>", "H", "N_eff", "normalized_progress", "relative_energy_gap", "runtime", "top_probability"],
        "note": "QAOA-only controlled size scaling; no quantum-advantage claim.",
    }
    save_json(os.path.join(outdir, "experiment_config.json"), experiment_config)

    print("=" * 78)
    print("Experiment 3 — QAOA size scaling")
    print("=" * 78)
    print(f"sizes={args.sizes}")
    print(f"p={args.layers} | population={args.population} | generations={args.generations}")
    print(f"optimizer seeds={args.seed_start}..{args.seed_end} ({len(seeds)} runs/size)")
    print(f"fixed problem seed={args.problem_seed}")
    print(f"backend={args.backend}")
    print(f"output={outdir}")

    all_summaries = []
    all_generations = []
    failures = []

    for rows, cols in sizes:
        n = rows * cols
        print("\n" + "#" * 78)
        print(f"SIZE {rows}x{cols} | n={n} | exact states={2**n:,}")
        print("#" * 78)

        # Build exactly once per size, then reuse for every optimizer seed.
        problem = build_problem(rows, cols, args.problem_seed, args.n_background)
        size_root = os.path.join(outdir, f"{rows}x{cols}")
        os.makedirs(size_root, exist_ok=True)
        save_json(os.path.join(size_root, "problem_summary.json"), {
            k: v for k,v in problem.items() if k not in {"h","J","dC","C_target","C_bg","states","G_of_state"}
        })
        np.savez_compressed(
            os.path.join(size_root, "problem.npz"),
            coords=problem["coords"], h=problem["h"], J=problem["J"], const=problem["const"],
            dC=problem["dC"], C_target=problem["C_target"], C_bg=problem["C_bg"],
            G_of_state=problem["G_of_state"], states=problem["states"],
        )

        print(f"G*={problem['G_star']:.8f} | n_opt={problem['n_opt']} | "
              f"uniform <G>={problem['G_uniform']:.8f} | contacts={problem['target_contacts']}")
        print(f"Ising exact enumeration check max error={problem['ising_max_error']:.3e}")

        for seed in seeds:
            run_dir = os.path.join(size_root, f"seed_{seed}")
            try:
                summary, gen_rows, top_rows = run_qaoa_one(
                    problem, rows, cols, seed, args, run_dir
                )
                all_summaries.append(summary)
                for gr in gen_rows:
                    all_generations.append({
                        "rows": rows, "cols": cols, "seed": seed, **gr
                    })
            except Exception as e:
                err = {
                    "rows": rows, "cols": cols, "seed": seed,
                    "error_type": type(e).__name__, "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                failures.append(err)
                print(f"\nERROR {rows}x{cols} seed={seed}: {type(e).__name__}: {e}")
                traceback.print_exc()
                save_json(os.path.join(run_dir, "ERROR.json"), err)
                if not args.continue_on_error:
                    raise

    # Aggregate and figures.
    if not all_summaries:
        raise RuntimeError("No successful runs were completed.")

    agg = aggregate_results(all_summaries)
    save_problem_summary(all_summaries, outdir)
    write_statistics(all_summaries, agg, outdir)
    plot_scaling(agg, os.path.join(outdir, "figures"))
    plot_convergence(all_generations, outdir)
    plot_seed_robustness(all_summaries, outdir)

    # Copy figure-data files from root figure_data location if generated there.
    # (plot_convergence / plot_seed_robustness write directly into outdir/figure_data.)

    # Scaling metrics table for plotting / Excel.
    scaling_rows = []
    for a in agg:
        scaling_rows.append({
            "size": f"{a['rows']}x{a['cols']}", "rows": a["rows"], "cols": a["cols"],
            "n": a["n"], "n_states": a["n_states"], "G_star": a["G_star"],
            "G_uniform": a["G_uniform"], "n_opt": a["n_opt"],
            "expG_mean": a["expG_mean"], "expG_sd": a["expG_sd"], "expG_ci95": a["expG_ci95"],
            "P_Gstar_mean": a["P_Gstar_mean"], "P_Gstar_sd": a["P_Gstar_sd"], "P_Gstar_ci95": a["P_Gstar_ci95"],
            "H_mean": a["entropy_bits_mean"], "H_sd": a["entropy_bits_sd"],
            "N_eff_mean": a["N_eff_mean"], "N_eff_sd": a["N_eff_sd"],
            "normalized_progress_mean": a["normalized_progress_mean"],
            "normalized_progress_sd": a["normalized_progress_sd"],
            "relative_energy_gap_mean": a["relative_energy_gap_mean"],
            "relative_energy_gap_sd": a["relative_energy_gap_sd"],
            "runtime_mean_s": a["runtime_seconds_mean"],
            "runtime_sd_s": a["runtime_seconds_sd"],
            "uniform_P_Gstar": a["uniform_P_Gstar"],
            "m95_mean": a["m95_samples_mean"],
            "m95_sd": a["m95_samples_sd"],
        })
    save_csv(os.path.join(outdir, "figure_data", "scaling_metrics.csv"), scaling_rows)

    # Human-readable README/results summary.
    with open(os.path.join(outdir, "README_results.txt"), "w", encoding="utf-8") as f:
        f.write("Experiment 3 — QAOA size scaling\n")
        f.write("=" * 72 + "\n\n")
        f.write("Protocol\n--------\n")
        f.write(f"QAOA p={args.layers}, population={args.population}, generations={args.generations}\n")
        f.write(f"Optimizer seeds={args.seed_start}..{args.seed_end}\n")
        f.write(f"Fixed problem seed={args.problem_seed}\n")
        f.write("Exact enumeration: yes for all requested sizes\n\n")
        f.write("Size summary (mean ± SD)\n-------------------------\n")
        f.write(f"{'Size':<8}{'n':>5}{'States':>10}{'<G>':>16}{'P(G*)':>16}{'H':>14}{'N_eff':>14}{'NormProg':>14}{'m95':>12}{'Runtime(s)':>16}\n")
        for a in agg:
            f.write(f"{a['rows']}x{a['cols']:<4}{a['n']:>5}{a['n_states']:>10}"
                    f"{a['expG_mean']:>8.4f}±{a['expG_sd']:<7.4f}"
                    f"{100*a['P_Gstar_mean']:>8.3f}±{100*a['P_Gstar_sd']:<7.3f}"
                    f"{a['entropy_bits_mean']:>7.3f}±{a['entropy_bits_sd']:<6.3f}"
                    f"{a['N_eff_mean']:>7.2f}±{a['N_eff_sd']:<6.2f}"
                    f"{a['normalized_progress_mean']:>7.3f}±{a['normalized_progress_sd']:<6.3f}"
                    f"{a['m95_samples_mean']:>6.1f}±{a['m95_samples_sd']:<4.1f}"
                    f"{a['runtime_seconds_mean']:>8.1f}±{a['runtime_seconds_sd']:<7.1f}\n")
        if failures:
            f.write("\nFailures\n--------\n")
            for e in failures:
                f.write(f"{e['rows']}x{e['cols']} seed={e['seed']}: {e['error_type']}: {e['error']}\n")
        f.write("\nPrimary endpoint: P(G*). Lower <G> is better. N_eff measures distribution concentration only and does not by itself indicate solution quality.\n")
        f.write("The experiment is intended as controlled size scaling and does not establish quantum advantage.\n")

    save_json(os.path.join(outdir, "final_summary.json"), {
        "experiment_config": experiment_config,
        "successful_runs": len(all_summaries),
        "failed_runs": failures,
        "size_summary": agg,
    })

    print("\n" + "=" * 78)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 78)
    for a in agg:
        print(f"{a['rows']}x{a['cols']}: <G>={a['expG_mean']:.5f}±{a['expG_sd']:.5f} | "
              f"P(G*)={100*a['P_Gstar_mean']:.3f}%±{100*a['P_Gstar_sd']:.3f}% | "
              f"N_eff={a['N_eff_mean']:.2f}±{a['N_eff_sd']:.2f} | "
              f"runtime={a['runtime_seconds_mean']:.1f}±{a['runtime_seconds_sd']:.1f}s")
    print(f"Successful runs: {len(all_summaries)} | Failed runs: {len(failures)}")
    print(f"Results: {outdir}")


if __name__ == "__main__":
    main()
