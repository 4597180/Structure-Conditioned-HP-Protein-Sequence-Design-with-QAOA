#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import time
import math
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evolutionary import EvolutionaryOptimizer
from qaoa_generator import (
    contact_map,
    build_ising,
    direct_G,
    QAOACircuit,
)


# =============================================================================
# Generic IO / JSON helpers
# =============================================================================

def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def save_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def append_text(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class Tee:
    """Write simultaneously to console and log file."""
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


# =============================================================================
# Sequence / distribution metrics
# =============================================================================

def sequence_from_state(state, n):
    # Project convention: bit 0 -> H, bit 1 -> P.
    return "".join(
        "H" if ((int(state) >> i) & 1) == 0 else "P"
        for i in range(n)
    )


def distribution_entropy(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0, 1.0
    H = float(-np.sum(p * np.log2(p)))
    return H, float(2.0 ** H)


def exact_metrics(probs, G_of_state, opt_mask):
    p = np.asarray(probs, dtype=float)
    p = p / float(np.sum(p))
    expG = float(np.dot(p, G_of_state))
    p_opt = float(np.sum(p[opt_mask]))
    H, N_eff = distribution_entropy(p)
    top_state = int(np.argmax(p))
    return {
        "expG": expG,
        "P_Gstar": p_opt,
        "entropy_bits": H,
        "N_eff": N_eff,
        "top_state": top_state,
        "top_sequence": sequence_from_state(top_state, int(round(math.log2(len(p))))),
        "top_probability": float(p[top_state]),
        "top_G": float(G_of_state[top_state]),
    }


def normalized_progress(expG, G_uniform, G_star):
    denom = float(G_uniform - G_star)
    if abs(denom) < 1e-15:
        return float("nan")
    return float((G_uniform - expG) / denom)


def m95_from_pstar(pstar):
    pstar = float(pstar)
    if pstar <= 0:
        return float("inf")
    if pstar >= 1:
        return 1.0
    return float(np.log(0.05) / np.log(1.0 - pstar))


def mean_sd_ci(values, confidence=0.95):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if len(x) == 1:
        return m, 0.0, 0.0
    sd = float(np.std(x, ddof=1))
    # Exact t critical values for common sample sizes; n=10 is the default.
    tcrit_table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 15: 2.131, 20: 2.093, 25: 2.064, 30: 2.045,
    }
    df = len(x) - 1
    tcrit = tcrit_table.get(df, 1.96)
    half = float(tcrit * sd / math.sqrt(len(x)))
    return m, sd, half


# =============================================================================
# Target structure enumeration
# =============================================================================

def grid_adjacency(rows, cols):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    cell_set = set(cells)
    adj = {x: [] for x in cells}
    for x in cells:
        r, c = x
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y = (r + dr, c + dc)
            if y in cell_set:
                adj[x].append(y)
    return adj


def enumerate_unique_target_structures(rows, cols):
    """
    Enumerate all Hamiltonian self-avoiding walks on the finite lattice and
    deduplicate by the residue-indexed contact map.

    Geometric rotations/reflections that preserve the residue-indexed contact
    map collapse automatically because the contact-map signature is the key.
    """
    N = rows * cols
    adj = grid_adjacency(rows, cols)
    cells = list(adj.keys())

    representatives = {}
    total_paths = 0

    def cm_signature(C):
        # Upper triangle is enough and avoids dependence on array metadata.
        tri = C[np.triu_indices(N, k=1)].astype(np.uint8)
        return tri.tobytes()

    def dfs(path, visited):
        nonlocal total_paths
        if len(path) == N:
            total_paths += 1
            coords = np.asarray(path, dtype=np.int32)
            C = contact_map(coords)
            sig = cm_signature(C)
            if sig not in representatives:
                representatives[sig] = coords.copy()
            return

        for nxt in adj[path[-1]]:
            if nxt not in visited:
                dfs(path + [nxt], visited | {nxt})

    for start in cells:
        dfs([start], {start})

    # Stable ordering by contact-map signature.
    items = sorted(representatives.items(), key=lambda kv: kv[0])
    targets = []
    for idx, (sig, coords) in enumerate(items, start=1):
        C = contact_map(coords)
        targets.append({
            "target_index": idx,
            "target_id": f"target_{idx:02d}",
            "coords": coords,
            "contact_map": C,
            "contact_signature": sig.hex(),
            "contact_count": int(np.sum(C) / 2),
        })

    return targets, total_paths


def select_targets(all_targets, target_count, target_seed):
    if target_count is None or target_count <= 0 or target_count >= len(all_targets):
        return list(all_targets)

    rng = np.random.RandomState(target_seed)
    idx = np.sort(rng.choice(len(all_targets), size=target_count, replace=False))
    selected = [all_targets[int(i)] for i in idx]

    # Re-index the selected set for clean target IDs while preserving the
    # original enumeration index as source_target_index.
    out = []
    for new_i, t in enumerate(selected, start=1):
        tt = dict(t)
        tt["source_target_index"] = int(t["target_index"])
        tt["target_index"] = new_i
        tt["target_id"] = f"target_{new_i:02d}"
        out.append(tt)
    return out


# =============================================================================
# Hamiltonian / problem preparation
# =============================================================================

def build_problem(target, rows, cols, problem_seed, n_background=30):
    coords = target["coords"]
    h, J, const, dC, C_target, C_bg = build_ising(
        coords, n_background=n_background, seed=problem_seed
    )

    n = rows * cols
    n_states = 2 ** n
    states = np.arange(n_states)
    bits = (states[:, None] >> np.arange(n)) & 1
    z = 1 - 2 * bits
    G_of_state = const + z @ h + 0.5 * np.einsum("si,ij,sj->s", z, J, z)

    G_star = float(np.min(G_of_state))
    opt_mask = G_of_state <= G_star + 1e-9
    G_uniform = float(np.mean(G_of_state))

    # Full-state direct energy check is cheap for the default 3x3/3x4 sizes.
    check_indices = np.arange(n_states)
    max_err = 0.0
    for s in check_indices:
        max_err = max(max_err, abs(G_of_state[s] - direct_G(z[s], dC)))

    problem_blob = np.concatenate([
        np.asarray(C_target, dtype=np.float64).ravel(),
        np.asarray(C_bg, dtype=np.float64).ravel(),
        np.asarray(dC, dtype=np.float64).ravel(),
        np.asarray(h, dtype=np.float64).ravel(),
        np.asarray(J, dtype=np.float64).ravel(),
        np.asarray([const, G_star], dtype=np.float64),
    ]).tobytes()
    problem_hash = sha256_bytes(problem_blob)

    return {
        "coords": coords,
        "h": h,
        "J": J,
        "const": const,
        "dC": dC,
        "C_target": C_target,
        "C_bg": C_bg,
        "G_of_state": G_of_state,
        "G_star": G_star,
        "opt_mask": opt_mask,
        "G_uniform": G_uniform,
        "n_opt": int(np.sum(opt_mask)),
        "n_states": n_states,
        "n_qubits": n,
        "ising_max_error": float(max_err),
        "problem_hash": problem_hash,
        "n_background_actual": int(C_bg.shape[0] and n_background),
    }


def save_problem_files(target_dir, target, problem, args):
    np.savez(
        target_dir / "problem.npz",
        coords=problem["coords"],
        h=problem["h"],
        J=problem["J"],
        const=problem["const"],
        dC=problem["dC"],
        C_target=problem["C_target"],
        C_bg=problem["C_bg"],
        G_of_state=problem["G_of_state"],
        G_star=problem["G_star"],
        opt_mask=problem["opt_mask"],
        G_uniform=problem["G_uniform"],
    )

    coord_rows = [
        {"residue_index": i + 1, "row": int(x[0]), "col": int(x[1])}
        for i, x in enumerate(problem["coords"])
    ]
    save_csv(target_dir / "target_coords.csv", coord_rows,
             ["residue_index", "row", "col"])

    C = problem["C_target"]
    cm_rows = []
    for i in range(len(C)):
        row = {"residue_index": i + 1}
        for j in range(len(C)):
            row[f"r{j + 1}"] = int(C[i, j])
        cm_rows.append(row)
    save_csv(target_dir / "target_contact_map.csv", cm_rows,
             ["residue_index"] + [f"r{j + 1}" for j in range(len(C))])

    summary = {
        "target_id": target["target_id"],
        "target_index": target["target_index"],
        "source_target_index": target.get("source_target_index", target["target_index"]),
        "rows": args.rows,
        "cols": args.cols,
        "n_qubits": problem["n_qubits"],
        "n_states": problem["n_states"],
        "problem_seed": args.problem_seed,
        "target_seed": args.target_seed,
        "contact_count": target["contact_count"],
        "G_star": problem["G_star"],
        "n_opt": problem["n_opt"],
        "uniform_P_Gstar": problem["n_opt"] / problem["n_states"],
        "G_uniform": problem["G_uniform"],
        "ising_max_error": problem["ising_max_error"],
        "problem_hash": problem["problem_hash"],
        "contact_signature": target["contact_signature"],
        "n_background": args.background_attempts,
    }
    save_json(target_dir / "problem_summary.json", summary)

    with open(target_dir / "problem_summary.txt", "w", encoding="utf-8") as f:
        f.write("Experiment 4 target-specific problem\n")
        f.write("==================================\n")
        f.write(f"target_id: {summary['target_id']}\n")
        f.write(f"lattice: {args.rows}x{args.cols}\n")
        f.write(f"n_qubits: {problem['n_qubits']}\n")
        f.write(f"n_states: {problem['n_states']}\n")
        f.write(f"problem_seed: {args.problem_seed}\n")
        f.write(f"target_seed: {args.target_seed}\n")
        f.write(f"contact_count: {target['contact_count']}\n")
        f.write(f"G*: {problem['G_star']:.10f}\n")
        f.write(f"n_opt: {problem['n_opt']}\n")
        f.write(f"uniform P(G*): {problem['n_opt']/problem['n_states']:.10f}\n")
        f.write(f"uniform <G>: {problem['G_uniform']:.10f}\n")
        f.write(f"Ising max error: {problem['ising_max_error']:.3e}\n")
        f.write(f"problem hash: {problem['problem_hash']}\n")
        f.write(f"contact signature: {target['contact_signature']}\n")


# =============================================================================
# QAOA single run
# =============================================================================

def run_qaoa_one(target_dir, target, problem, args, seed):
    run_dir = target_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"exp4.{target['target_id']}.seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    config = vars(args).copy()
    config.update({
        "target_id": target["target_id"],
        "target_index": target["target_index"],
        "optimizer_seed": seed,
        "problem_hash": problem["problem_hash"],
        "created_at": datetime.now().isoformat(),
        "objective": "structure-conditioned HP sequence design / inverse folding",
    })
    save_json(run_dir / "config.json", config)

    n = problem["n_qubits"]
    p = args.p
    G = problem["G_of_state"]
    opt_mask = problem["opt_mask"]
    G_uniform = problem["G_uniform"]
    G_star = problem["G_star"]

    np.random.seed(seed)
    circ = QAOACircuit(n, problem["h"], problem["J"], backend=args.backend)
    logger.info(
        "target=%s | seed=%d | backend=%s | n=%d | p=%d | parameters=%d",
        target["target_id"], seed, circ.backend, n, p, 2 * p
    )

    def probs_fn(params):
        return circ.probs(params[:p], params[p:], G)

    def fitness_fn(params):
        return float(np.dot(probs_fn(params), G))

    bounds = [(0.0, 2.0 * np.pi)] * (2 * p)
    opt = EvolutionaryOptimizer(
        n_params=2 * p,
        param_bounds=bounds,
        fitness_fn=fitness_fn,
        population_size=args.population,
        elite_size=max(2, args.population // 6),
        tournament_size=3,
        mutation_rate=0.15,
        crossover_rate=0.7,
        mutation_sigma=0.5,
        mutation_strategy="gaussian",
    )
    opt.initialize_population(seed=seed, verbose=False)

    rows = []
    t0 = time.time()

    # Generation 0: initial population has already been evaluated.
    probs = probs_fn(opt.best_individual.params)
    m = exact_metrics(probs, G, opt_mask)
    rows.append({
        "generation": 0,
        "best_fitness": float(opt.best_individual.fitness),
        "expG": m["expG"],
        "P_Gstar": m["P_Gstar"],
        "entropy_bits": m["entropy_bits"],
        "N_eff": m["N_eff"],
        "normalized_progress": normalized_progress(m["expG"], G_uniform, G_star),
        "top_state": m["top_state"],
        "top_sequence": m["top_sequence"],
        "top_probability": m["top_probability"],
        "top_G": m["top_G"],
        "population_mean_fitness": float(np.mean([x.fitness for x in opt.population])),
        "population_std_fitness": float(np.std([x.fitness for x in opt.population])),
        "mutation_rate": float(opt.mutation_rate),
        "objective_evaluations": int(opt.n_evaluations),
        "elapsed_seconds": 0.0,
    })

    report_every = max(1, args.generations // 20)
    for gen in range(1, args.generations + 1):
        opt.run_generation()
        probs = probs_fn(opt.best_individual.params)
        m = exact_metrics(probs, G, opt_mask)
        elapsed = time.time() - t0
        rows.append({
            "generation": gen,
            "best_fitness": float(opt.best_individual.fitness),
            "expG": m["expG"],
            "P_Gstar": m["P_Gstar"],
            "entropy_bits": m["entropy_bits"],
            "N_eff": m["N_eff"],
            "normalized_progress": normalized_progress(m["expG"], G_uniform, G_star),
            "top_state": m["top_state"],
            "top_sequence": m["top_sequence"],
            "top_probability": m["top_probability"],
            "top_G": m["top_G"],
            "population_mean_fitness": float(np.mean([x.fitness for x in opt.population])),
            "population_std_fitness": float(np.std([x.fitness for x in opt.population])),
            "mutation_rate": float(opt.mutation_rate),
            "objective_evaluations": int(opt.n_evaluations),
            "elapsed_seconds": elapsed,
        })
        if gen % report_every == 0 or gen == args.generations:
            logger.info(
                "gen=%d/%d | eval=%d | <G>=%.6f | P(G*)=%.4f | N_eff=%.2f",
                gen, args.generations, opt.n_evaluations,
                m["expG"], m["P_Gstar"], m["N_eff"]
            )

    elapsed = time.time() - t0
    probs = probs_fn(opt.best_individual.params)
    m = exact_metrics(probs, G, opt_mask)
    norm = normalized_progress(m["expG"], G_uniform, G_star)
    m95 = m95_from_pstar(m["P_Gstar"])

    # Exact full state distribution.
    state_rows = []
    for s in range(problem["n_states"]):
        state_rows.append({
            "state": s,
            "sequence": sequence_from_state(s, n),
            "G": float(G[s]),
            "is_global_optimum": int(opt_mask[s]),
            "probability": float(probs[s]),
        })
    save_csv(run_dir / "final_state_distribution.csv", state_rows)

    # Aggregate the exact statevector probabilities by energy level, matching
    # the distribution-level diagnostic used in the earlier experiments.
    energy_mass = defaultdict(float)
    energy_isopt = {}
    for s in range(problem["n_states"]):
        g_key = round(float(G[s]), 12)
        energy_mass[g_key] += float(probs[s])
        energy_isopt[g_key] = bool(energy_isopt.get(g_key, False) or opt_mask[s])
    energy_rows = [
        {"G": g, "probability_mass": mass, "is_global_optimum": int(energy_isopt[g])}
        for g, mass in sorted(energy_mass.items())
    ]
    save_csv(run_dir / "energy_resolved_probability.csv", energy_rows)
    save_csv(run_dir / "generation_metrics.csv", rows)

    np.savez(
        run_dir / "best_params.npz",
        params=np.asarray(opt.best_individual.params, dtype=float),
        best_fitness=float(opt.best_individual.fitness),
    )

    summary = {
        "target_id": target["target_id"],
        "target_index": target["target_index"],
        "optimizer_seed": seed,
        "problem_seed": args.problem_seed,
        "target_seed": args.target_seed,
        "problem_hash": problem["problem_hash"],
        "rows": args.rows,
        "cols": args.cols,
        "n": n,
        "n_states": problem["n_states"],
        "p": p,
        "n_parameters": 2 * p,
        "population": args.population,
        "generations": args.generations,
        "objective_evaluations": int(opt.n_evaluations),
        "backend": circ.backend,
        "G_star": G_star,
        "G_uniform": G_uniform,
        "n_opt": problem["n_opt"],
        "uniform_P_Gstar": problem["n_opt"] / problem["n_states"],
        "expG": m["expG"],
        "P_Gstar": m["P_Gstar"],
        "entropy_bits": m["entropy_bits"],
        "N_eff": m["N_eff"],
        "normalized_progress": norm,
        "relative_energy_gap": float((m["expG"] - G_star) / (G_uniform - G_star)) if abs(G_uniform - G_star) > 1e-15 else np.nan,
        "top_state": m["top_state"],
        "top_sequence": m["top_sequence"],
        "top_probability": m["top_probability"],
        "top_G": m["top_G"],
        "m95_samples": m95,
        "runtime_seconds": elapsed,
        "ising_max_error": problem["ising_max_error"],
        "contact_count": target["contact_count"],
    }
    save_csv(run_dir / "final_metrics.csv", [summary])
    save_json(run_dir / "summary.json", summary)

    logger.info(
        "FINAL | <G>=%.6f | P(G*)=%.6f | N_eff=%.3f | R=%.6f | top=%s | time=%.2fs",
        m["expG"], m["P_Gstar"], m["N_eff"], norm,
        m["top_sequence"], elapsed
    )
    for h in logger.handlers:
        h.flush()
        h.close()
    logger.handlers.clear()

    return summary, rows


# =============================================================================
# Aggregation / statistics
# =============================================================================

def aggregate_convergence(all_histories, args):
    # all_histories: list of (target_id, seed, rows)
    by = defaultdict(list)
    for target_id, seed, rows in all_histories:
        for r in rows:
            by[(target_id, int(r["generation"]))].append(r)

    out = []
    for (target_id, gen), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for metric in ["expG", "P_Gstar", "entropy_bits", "N_eff", "normalized_progress"]:
            vals = [float(r[metric]) for r in rs]
            mean, sd, ci = mean_sd_ci(vals)
            out.append({
                "target_id": target_id,
                "generation": gen,
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "ci95_half": ci,
                "n_runs": len(vals),
            })
    save_csv(Path(args.out_dir) / "convergence_summary.csv", out)

    raw = []
    for target_id, seed, rows in all_histories:
        for r in rows:
            rr = dict(r)
            rr["target_id"] = target_id
            rr["optimizer_seed"] = seed
            raw.append(rr)
    save_csv(Path(args.out_dir) / "convergence_raw.csv", raw)
    return out


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def run_statistics(run_summaries, target_ids, args):
    lines = []
    lines.append("Experiment 4 exploratory target-structure statistics\n")
    lines.append("===================================================\n")
    lines.append("Interpretation note: target structures define different Hamiltonian/problem instances.\n")
    lines.append("Therefore these tests describe target-to-target heterogeneity under the fixed protocol;\n")
    lines.append("they are not evidence that one target structure is universally easier or harder.\n\n")

    by_target = {tid: [] for tid in target_ids}
    for r in run_summaries:
        by_target[r["target_id"]].append(r)

    # Target-level summary and seed-matched matrices.
    metrics = ["P_Gstar", "expG", "N_eff", "normalized_progress"]
    for metric in metrics:
        matrix = np.array([
            [float(next(x[metric] for x in by_target[tid] if x["optimizer_seed"] == seed))
             for tid in target_ids]
            for seed in args.seed_list
        ], dtype=float)

        lines.append(f"Metric: {metric}\n")
        lines.append(f"Shape: seeds x targets = {matrix.shape}\n")

        try:
            from scipy.stats import friedmanchisquare, wilcoxon
            cols = [matrix[:, j] for j in range(matrix.shape[1])]
            if len(target_ids) >= 3:
                stat, p = friedmanchisquare(*cols)
                lines.append(f"Friedman test across targets: statistic={stat:.8g}, p={p:.8g}\n")

            pairs = []
            pvals = []
            for i in range(len(target_ids)):
                for j in range(i + 1, len(target_ids)):
                    try:
                        w, pv = wilcoxon(matrix[:, i], matrix[:, j], zero_method="wilcox", alternative="two-sided")
                        pairs.append((target_ids[i], target_ids[j], float(w), float(pv)))
                        pvals.append(float(pv))
                    except Exception as e:
                        pairs.append((target_ids[i], target_ids[j], np.nan, np.nan))
                        pvals.append(np.nan)
            finite = np.isfinite(pvals)
            adj = np.full(len(pvals), np.nan)
            if np.any(finite):
                adj[finite] = holm_adjust(np.asarray(pvals)[finite])
            lines.append("Paired Wilcoxon signed-rank tests (seed matched; Holm adjusted):\n")
            for k, item in enumerate(pairs):
                a, b, w, pv = item
                lines.append(f"  {a} vs {b}: W={w:.6g}, p={pv:.8g}, Holm-p={adj[k]:.8g}\n")
        except ImportError:
            lines.append("SciPy unavailable; Friedman/Wilcoxon tests were not run.\n")
        lines.append("\n")

    # Spearman correlations between target descriptors and target-level mean outcomes.
    target_rows = {}
    for tid in target_ids:
        vals = by_target[tid]
        target_rows[tid] = {
            "contact_count": float(np.mean([x["contact_count"] for x in vals])),
            "G_star": float(np.mean([x["G_star"] for x in vals])),
            "uniform_P_Gstar": float(np.mean([x["uniform_P_Gstar"] for x in vals])),
            "mean_P_Gstar": float(np.mean([x["P_Gstar"] for x in vals])),
            "mean_expG": float(np.mean([x["expG"] for x in vals])),
            "mean_N_eff": float(np.mean([x["N_eff"] for x in vals])),
            "mean_progress": float(np.mean([x["normalized_progress"] for x in vals])),
        }

    corr_rows = []
    try:
        from scipy.stats import spearmanr
        descriptors = ["contact_count", "G_star", "uniform_P_Gstar"]
        outcomes = ["mean_P_Gstar", "mean_expG", "mean_N_eff", "mean_progress"]
        for d in descriptors:
            for o in outcomes:
                x = [target_rows[tid][d] for tid in target_ids]
                y = [target_rows[tid][o] for tid in target_ids]
                if np.std(x) < 1e-15 or np.std(y) < 1e-15:
                    rho, pv = np.nan, np.nan
                else:
                    rho, pv = spearmanr(x, y)
                    rho, pv = float(rho), float(pv)
                corr_rows.append({"descriptor": d, "outcome": o, "spearman_rho": rho, "p_value": pv})
    except ImportError:
        pass

    save_csv(Path(args.out_dir) / "correlation_summary.csv", corr_rows)
    with open(Path(args.out_dir) / "statistical_tests.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

    return target_rows


def build_target_summary(run_summaries, targets, args):
    by_target = defaultdict(list)
    for r in run_summaries:
        by_target[r["target_id"]].append(r)

    rows = []
    for t in targets:
        vals = by_target[t["target_id"]]
        def ms(metric):
            return mean_sd_ci([x[metric] for x in vals])

        p_m, p_sd, p_ci = ms("P_Gstar")
        e_m, e_sd, e_ci = ms("expG")
        n_m, n_sd, n_ci = ms("N_eff")
        r_m, r_sd, r_ci = ms("normalized_progress")
        rt_m, rt_sd, rt_ci = ms("runtime_seconds")
        m95_m, m95_sd, m95_ci = ms("m95_samples")

        rows.append({
            "target_id": t["target_id"],
            "target_index": t["target_index"],
            "source_target_index": t.get("source_target_index", t["target_index"]),
            "contact_count": t["contact_count"],
            "G_star": vals[0]["G_star"],
            "G_uniform": vals[0]["G_uniform"],
            "n_opt": vals[0]["n_opt"],
            "uniform_P_Gstar": vals[0]["uniform_P_Gstar"],
            "P_Gstar_mean": p_m,
            "P_Gstar_sd": p_sd,
            "P_Gstar_ci95_half": p_ci,
            "expG_mean": e_m,
            "expG_sd": e_sd,
            "expG_ci95_half": e_ci,
            "N_eff_mean": n_m,
            "N_eff_sd": n_sd,
            "N_eff_ci95_half": n_ci,
            "normalized_progress_mean": r_m,
            "normalized_progress_sd": r_sd,
            "normalized_progress_ci95_half": r_ci,
            "m95_mean": m95_m,
            "m95_sd": m95_sd,
            "runtime_seconds_mean": rt_m,
            "runtime_seconds_sd": rt_sd,
            "problem_hash": vals[0]["problem_hash"],
            "contact_signature": t["contact_signature"],
            "n_runs": len(vals),
        })

    save_csv(Path(args.out_dir) / "target_summary.csv", rows)
    return rows


# =============================================================================
# Plots
# =============================================================================

def plot_target_robustness(target_rows, run_summaries, args):
    out = Path(args.out_dir)
    target_ids = [r["target_id"] for r in target_rows]
    x = np.arange(len(target_ids))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [
        ("P_Gstar_mean", "P(G*)", "Probability of exact optimum"),
        ("normalized_progress_mean", "Normalized progress", "Energetic progress"),
        ("N_eff_mean", r"$N_{\mathrm{eff}}$", "Effective sequence count"),
        ("runtime_seconds_mean", "Runtime (s)", "Wall-clock runtime"),
    ]

    for ax, (key, ylabel, title) in zip(axes.flat, panels):
        means = [r[key] for r in target_rows]
        if key == "P_Gstar_mean":
            sds = [r["P_Gstar_sd"] for r in target_rows]
            ax.errorbar(x, np.asarray(means) * 100, yerr=np.asarray(sds) * 100,
                        fmt="o", capsize=4)
            ax.set_ylabel("P(G*) (%)")
        elif key == "normalized_progress_mean":
            sds = [r["normalized_progress_sd"] for r in target_rows]
            ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4)
            ax.set_ylabel(ylabel)
        elif key == "N_eff_mean":
            sds = [r["N_eff_sd"] for r in target_rows]
            ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
        else:
            sds = [r["runtime_seconds_sd"] for r in target_rows]
            ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4)
            ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(target_ids, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Target-structure robustness: {args.rows}×{args.cols}, QAOA p={args.p}",
        fontsize=14
    )
    fig.savefig(out / "Figure8_target_robustness.png", dpi=300)
    plt.close(fig)


def plot_energy_profiles(target_rows, run_summaries, args):
    out = Path(args.out_dir)
    by_target = defaultdict(list)
    for r in run_summaries:
        by_target[r["target_id"]].append(r)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(target_rows))
    exp_means = np.array([r["expG_mean"] for r in target_rows])
    gstars = np.array([r["G_star"] for r in target_rows])
    ax.errorbar(x, exp_means, yerr=[r["expG_sd"] for r in target_rows], fmt="o", capsize=4, label="Final <G>")
    ax.scatter(x, gstars, marker="x", s=70, label="Exact G*")
    ax.set_xticks(x)
    ax.set_xticklabels([r["target_id"] for r in target_rows], rotation=45, ha="right")
    ax.set_ylabel("Objective G")
    ax.set_title("Target-specific objective landscape and final QAOA expectation")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "FigureS4_target_energy_profiles.png", dpi=300)
    plt.close(fig)


def plot_seed_heatmap(run_summaries, target_rows, args):
    out = Path(args.out_dir)
    target_ids = [r["target_id"] for r in target_rows]
    seeds = list(args.seed_list)
    mat = np.array([
        [next(r["P_Gstar"] for r in run_summaries
         if r["target_id"] == tid and r["optimizer_seed"] == seed)
         for tid in target_ids]
        for seed in seeds
    ]) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(target_ids)))
    ax.set_xticklabels(target_ids, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(seeds)))
    ax.set_yticklabels([str(s) for s in seeds])
    ax.set_xlabel("Target structure")
    ax.set_ylabel("Optimizer seed")
    ax.set_title("Seed-level P(G*) across target structures")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(G*) (%)")
    fig.tight_layout()
    fig.savefig(out / "FigureS5_target_seed_heatmap.png", dpi=300)
    plt.close(fig)


# =============================================================================
# README / final summary
# =============================================================================

def write_readme(out_dir, targets, target_rows, run_summaries, args, total_paths):
    p = Path(out_dir) / "README_results.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("Experiment 4 — Target-Structure Robustness\n")
        f.write("===========================================\n\n")
        f.write(f"Lattice: {args.rows}x{args.cols}\n")
        f.write(f"All Hamiltonian paths enumerated: {total_paths}\n")
        f.write(f"Unique contact maps: {len(targets)}\n")
        f.write(f"Targets evaluated: {len(targets)}\n")
        f.write(f"QAOA depth p: {args.p}\n")
        f.write(f"Population x generations: {args.population} x {args.generations}\n")
        f.write(f"Optimizer seeds: {args.seed_list}\n")
        f.write(f"Problem seed: {args.problem_seed}\n")
        f.write(f"Target-selection seed: {args.target_seed}\n")
        f.write(f"Backend: {args.backend}\n")
        f.write(f"Background attempts: {args.background_attempts}\n\n")

        f.write("Important interpretation:\n")
        f.write("Each target defines a different Hamiltonian instance. Target-to-target differences\n")
        f.write("therefore measure sensitivity/heterogeneity of the method across target structures,\n")
        f.write("not a universal ranking of structures.\n\n")

        f.write("Target summary:\n")
        for r in target_rows:
            f.write(
                f"  {r['target_id']}: contacts={r['contact_count']}, G*={r['G_star']:.6f}, "
                f"n_opt={r['n_opt']}, P(G*)={100*r['P_Gstar_mean']:.3f}% +/- {100*r['P_Gstar_sd']:.3f}%, "
                f"<G>={r['expG_mean']:.6f} +/- {r['expG_sd']:.6f}, "
                f"N_eff={r['N_eff_mean']:.3f} +/- {r['N_eff_sd']:.3f}\n"
            )

        f.write("\nFile inventory:\n")
        f.write("  target_catalog.csv                  target coordinates/contact-map catalog\n")
        f.write("  target_summary.csv                  target-level mean/SD summary\n")
        f.write("  problem_summary.csv                 one row per target Hamiltonian/problem\n")
        f.write("  run_summary.csv                     one row per target x optimizer seed\n")
        f.write("  convergence_raw.csv                 all generation-level seed records\n")
        f.write("  convergence_summary.csv             generation-level target x seed aggregation\n")
        f.write("  correlation_summary.csv             exploratory descriptor correlations\n")
        f.write("  validation_summary.txt              reproducibility/integrity checks\n")
        f.write("  statistical_tests.txt               exploratory Friedman/Wilcoxon results\n")
        f.write("  Figure8_target_robustness.png       main manuscript figure candidate\n")
        f.write("  FigureS4_target_energy_profiles.png supporting figure\n")
        f.write("  FigureS5_target_seed_heatmap.png    supporting figure\n")
        f.write("  target_XX/seed_YY/                   complete per-run artifacts\n")


def write_final_summary(out_dir, targets, target_rows, run_summaries, args, total_paths):
    final = {
        "experiment": "Experiment 4 — Target-Structure Robustness",
        "created_at": datetime.now().isoformat(),
        "rows": args.rows,
        "cols": args.cols,
        "all_hamiltonian_paths": total_paths,
        "unique_contact_maps": len(targets),
        "targets_evaluated": len(targets),
        "p": args.p,
        "population": args.population,
        "generations": args.generations,
        "optimizer_seeds": args.seed_list,
        "problem_seed": args.problem_seed,
        "target_seed": args.target_seed,
        "backend": args.backend,
        "background_attempts": args.background_attempts,
        "target_summary": target_rows,
        "n_runs": len(run_summaries),
    }
    save_json(Path(out_dir) / "final_summary.json", final)


def write_validation_summary(out_dir, targets, run_summaries, args, total_paths):
    lines = []
    lines.append("Experiment 4 validation summary\n")
    lines.append("===============================\n")
    lines.append(f"Total Hamiltonian paths enumerated: {total_paths}\n")
    lines.append(f"Unique contact maps found: {len(targets)}\n")
    lines.append(f"Expected runs: {len(targets) * len(args.seed_list)}\n")
    lines.append(f"Completed runs: {len(run_summaries)}\n")
    lines.append(f"Run-count check: {'PASS' if len(run_summaries) == len(targets) * len(args.seed_list) else 'FAIL'}\n")

    hashes_ok = True
    ising_ok = True
    per_target_counts_ok = True
    for t in targets:
        vals = [r for r in run_summaries if r["target_id"] == t["target_id"]]
        hashes = sorted(set(r["problem_hash"] for r in vals))
        if len(hashes) != 1:
            hashes_ok = False
        if len(vals) != len(args.seed_list):
            per_target_counts_ok = False
        if any(float(r["ising_max_error"]) >= 1e-8 for r in vals):
            ising_ok = False
        lines.append(
            f"{t['target_id']}: runs={len(vals)}, unique_problem_hashes={len(hashes)}, "
            f"max_Ising_error={max([float(r['ising_max_error']) for r in vals] or [float('nan')]):.3e}\n"
        )
    lines.append(f"Within-target fixed-problem check: {'PASS' if hashes_ok else 'FAIL'}\n")
    lines.append(f"Ising expansion check (<1e-8): {'PASS' if ising_ok else 'FAIL'}\n")
    lines.append(f"Per-target seed-count check: {'PASS' if per_target_counts_ok else 'FAIL'}\n")
    lines.append("\nNo target was removed based on outcome performance.\n")
    with open(Path(out_dir) / "validation_summary.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)


# =============================================================================
# Main experiment driver
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Experiment 4: QAOA target-structure robustness on HP lattice inverse folding"
    )
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--p", type=int, default=5, help="fixed QAOA depth; default p=5")
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--problem-seed", type=int, default=12345,
                    help="fixed background/Hamiltonian seed")
    ap.add_argument("--target-seed", type=int, default=20260918,
                    help="seed used only when selecting a subset of target contact maps")
    ap.add_argument("--seed-start", type=int, default=46)
    ap.add_argument("--seed-end", type=int, default=55)
    ap.add_argument("--target-count", type=int, default=0,
                    help="number of unique contact maps; 0 means all")
    ap.add_argument("--background-attempts", type=int, default=30)
    ap.add_argument("--backend", choices=["auto", "default.qubit", "qiskit.aer"], default="auto")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--out-dir", default=None,
                    help="fixed output directory; enables resume of completed target/seed runs")
    ap.add_argument("--no-resume", action="store_true",
                    help="rerun completed target/seed folders")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")
    if args.p <= 0:
        raise ValueError("p must be positive")
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be >= seed-start")

    args.seed_list = list(range(args.seed_start, args.seed_end + 1))

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Enumerate targets before creating output so the directory name can record
    # the actual number of unique contact maps.
    all_targets, total_paths = enumerate_unique_target_structures(args.rows, args.cols)
    targets = select_targets(all_targets, args.target_count, args.target_seed)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (
            f"exp4_target_robustness_{args.rows}x{args.cols}_p{args.p}"
            f"_prob{args.problem_seed}_targets{len(targets)}"
            f"_seeds{args.seed_start}-{args.seed_end}_{stamp}"
        )
        out_dir = (Path(__file__).resolve().parent / args.out_root / name).resolve()
        out_dir.mkdir(parents=True, exist_ok=False)

    args.out_dir = str(out_dir)

    # Top-level logger.
    log_path = out_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("exp4")

    logger.info("=" * 72)
    logger.info("Experiment 4 — Target-Structure Robustness")
    logger.info("lattice=%dx%d | total Hamiltonian paths=%d | unique contact maps=%d | selected=%d",
                args.rows, args.cols, total_paths, len(all_targets), len(targets))
    logger.info("p=%d | population=%d | generations=%d | seeds=%s",
                args.p, args.population, args.generations, args.seed_list)
    logger.info("problem_seed=%d | target_seed=%d | backend=%s",
                args.problem_seed, args.target_seed, args.backend)

    # Save complete experiment configuration.
    config = vars(args).copy()
    config.update({
        "created_at": datetime.now().isoformat(),
        "total_hamiltonian_paths": total_paths,
        "all_unique_contact_maps": len(all_targets),
        "selected_target_count": len(targets),
        "selected_target_ids": [t["target_id"] for t in targets],
        "resume": not args.no_resume,
    })
    save_json(out_dir / "experiment_config.json", config)

    # Target catalog contains coordinates and contact-map signature, so the
    # structural test set is completely auditable.
    catalog_rows = []
    for t in targets:
        catalog_rows.append({
            "target_id": t["target_id"],
            "target_index": t["target_index"],
            "source_target_index": t.get("source_target_index", t["target_index"]),
            "contact_count": t["contact_count"],
            "n_residues": args.rows * args.cols,
            "contact_signature": t["contact_signature"],
            "coords_flat": ";".join(f"{int(x[0])},{int(x[1])}" for x in t["coords"]),
        })
    save_csv(out_dir / "target_catalog.csv", catalog_rows)

    # One row per target after problem construction; filled below and written
    # again after all target problems are prepared.
    problem_rows = []

    all_run_summaries = []
    all_histories = []

    # -------------------------------------------------------------------------
    # Target loop
    # -------------------------------------------------------------------------
    for ti, target in enumerate(targets, start=1):
        target_dir = out_dir / target["target_id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("\n[%d/%d] Preparing %s | contacts=%d",
                    ti, len(targets), target["target_id"], target["contact_count"])

        # Build ONE problem per target. It is reused by every optimizer seed.
        problem = build_problem(
            target, args.rows, args.cols,
            args.problem_seed, n_background=args.background_attempts
        )
        save_problem_files(target_dir, target, problem, args)
        problem_rows.append({
            "target_id": target["target_id"],
            "target_index": target["target_index"],
            "contact_count": target["contact_count"],
            "n_qubits": problem["n_qubits"],
            "n_states": problem["n_states"],
            "G_star": problem["G_star"],
            "G_uniform": problem["G_uniform"],
            "n_opt": problem["n_opt"],
            "uniform_P_Gstar": problem["n_opt"] / problem["n_states"],
            "ising_max_error": problem["ising_max_error"],
            "problem_hash": problem["problem_hash"],
            "contact_signature": target["contact_signature"],
        })
        save_csv(out_dir / "problem_summary.csv", problem_rows)
        logger.info(
            "%s | G*=%.6f | n_opt=%d | uniform P(G*)=%.6f | hash=%s",
            target["target_id"], problem["G_star"], problem["n_opt"],
            problem["n_opt"] / problem["n_states"], problem["problem_hash"][:16]
        )

        for seed in args.seed_list:
            run_dir = target_dir / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            if (not args.no_resume) and summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    # Verify the run belongs to this exact problem and seed.
                    if (summary.get("problem_hash") == problem["problem_hash"] and
                            int(summary.get("optimizer_seed")) == int(seed)):
                        logger.info("  %s seed=%d already complete; resume skip", target["target_id"], seed)
                        all_run_summaries.append(summary)
                        hist_path = run_dir / "generation_metrics.csv"
                        if hist_path.exists():
                            hist_rows = []
                            with open(hist_path, "r", encoding="utf-8") as f:
                                hist_rows = list(csv.DictReader(f))
                            # Cast numeric columns needed by aggregation.
                            for rr in hist_rows:
                                for key in ["generation", "objective_evaluations"]:
                                    rr[key] = int(float(rr[key]))
                                for key in ["expG", "P_Gstar", "entropy_bits", "N_eff", "normalized_progress"]:
                                    rr[key] = float(rr[key])
                            all_histories.append((target["target_id"], seed, hist_rows))
                        continue
                except Exception:
                    logger.warning("  Existing summary invalid; rerunning %s seed=%d", target["target_id"], seed)

            logger.info("  Running %s seed=%d ...", target["target_id"], seed)
            summary, hist = run_qaoa_one(target_dir, target, problem, args, seed)
            all_run_summaries.append(summary)
            all_histories.append((target["target_id"], seed, hist))

            # Flush top-level progress to a compact machine-readable file after
            # every completed run so an interruption loses minimal aggregation state.
            save_csv(out_dir / "run_summary.csv", all_run_summaries)

    # -------------------------------------------------------------------------
    # Aggregate outputs
    # -------------------------------------------------------------------------
    save_csv(out_dir / "run_summary.csv", all_run_summaries)
    target_rows = build_target_summary(all_run_summaries, targets, args)
    aggregate_convergence(all_histories, args)
    run_statistics(all_run_summaries, [t["target_id"] for t in targets], args)
    plot_target_robustness(target_rows, all_run_summaries, args)
    plot_energy_profiles(target_rows, all_run_summaries, args)
    plot_seed_heatmap(all_run_summaries, target_rows, args)
    write_validation_summary(out_dir, targets, all_run_summaries, args, total_paths)
    write_readme(out_dir, targets, target_rows, all_run_summaries, args, total_paths)
    write_final_summary(out_dir, targets, target_rows, all_run_summaries, args, total_paths)

    logger.info("\nExperiment 4 complete.")
    logger.info("Output directory: %s", out_dir)
    logger.info("Runs completed: %d", len(all_run_summaries))
    logger.info("Expected runs: %d", len(targets) * len(args.seed_list))
    logger.info("Main figure: %s", out_dir / "Figure8_target_robustness.png")
    print(f"\nRESULT_DIR={out_dir}")


if __name__ == "__main__":
    main()
