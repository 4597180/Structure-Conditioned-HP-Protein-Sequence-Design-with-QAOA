#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import time
import math
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evolutionary import EvolutionaryOptimizer

from qaoa_generator import (
    snake_coords,
    contact_map,
    build_ising,
    direct_G,
    QAOACircuit,
)


# ============================================================
# 1. Basic utilities
# ============================================================

def sequence_from_state(state, n):
    """Project convention: bit=0 -> H, bit=1 -> P."""
    return "".join(
        "H" if ((int(state) >> i) & 1) == 0 else "P"
        for i in range(n)
    )


def distribution_entropy(probs):
    """Shannon entropy (bit) and N_eff = 2^H."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 1e-12]
    if len(p) == 0:
        return 0.0, 1.0
    H = float(-np.sum(p * np.log2(p)))
    return H, float(2.0 ** H)


def exact_metrics(probs, G_of_state, opt_mask):
    """Compute metrics exactly from the full statevector probability."""
    probs = np.asarray(probs, dtype=float)
    s = float(probs.sum())
    if s <= 0:
        raise ValueError("Probability vector has non-positive sum.")
    probs = probs / s

    expG = float(np.dot(probs, G_of_state))
    p_opt = float(probs[opt_mask].sum())
    H, N_eff = distribution_entropy(probs)

    top_state = int(np.argmax(probs))

    return {
        "expG": expG,
        "P_Gstar": p_opt,
        "entropy_bits": H,
        "N_eff": N_eff,
        "top_state": top_state,
        "top_sequence": sequence_from_state(
            top_state, int(round(math.log2(len(probs))))
        ),
        "top_G": float(G_of_state[top_state]),
    }


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_logger(log_path):
    """Output to both the terminal and run.log."""
    logger = logging.getLogger(
        "qaoa_depth_sweep_" + str(log_path)
    )
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def atomic_json_dump(obj, path):
    """Simple atomic JSON save, reducing the chance of empty files caused by interruption."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ============================================================
# 2. Fixed problem: build only once
# ============================================================

def build_problem(rows, cols, problem_seed):
    """
    Fixed target + background ensemble, build:
        h, J, const, Delta C, G(s), G*
    """
    coords = snake_coords(rows, cols)

    h, J, const, dC, C_target, C_bg = build_ising(
        coords,
        seed=problem_seed,
    )

    n = rows * cols
    n_states = 2 ** n

    states = np.arange(n_states, dtype=np.int64)
    bits = (
        (states[:, None] >> np.arange(n, dtype=np.int64))
        & 1
    )
    z = 1 - 2 * bits

    G_of_state = (
        const
        + z @ h
        + 0.5 * np.einsum("si,ij,sj->s", z, J, z)
    )

    # Verify Ising expansion exactly over all states
    check_count = n_states
    max_err = 0.0
    for s in range(check_count):
        err = abs(
            float(G_of_state[s])
            - direct_G(z[s], dC)
        )
        if err > max_err:
            max_err = err

    G_star = float(G_of_state.min())
    opt_mask = G_of_state <= G_star + 1e-9

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
        "n": n,
        "n_states": n_states,
        "ising_max_error": max_err,
    }


def save_problem(problem, path):
    np.savez(
        path,
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
    )


# ============================================================
# 3. Single p + single optimizer seed
# ============================================================

def run_single_qaoa(
    p,
    optimizer_seed,
    problem,
    args,
    run_dir,
):
    """
    Run one:
        fixed problem
        + fixed p
        + one optimizer seed
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path)

    n = problem["n"]
    G_of_state = problem["G_of_state"]
    opt_mask = problem["opt_mask"]

    # --------------------------------------------------------
    # Fix the random stream
    # The current version of evolutionary.py initializes using a local RNG,
    # but some operations may still use the numpy global RNG, so fix both.
    # --------------------------------------------------------
    np.random.seed(optimizer_seed)

    # --------------------------------------------------------
    # QAOA circuit
    # --------------------------------------------------------
    circ = QAOACircuit(
        n,
        problem["h"],
        problem["J"],
        backend=args.backend,
    )

    n_params = 2 * p
    bounds = [(0.0, 2.0 * np.pi)] * n_params

    logger.info(
        "START | p=%d | optimizer_seed=%d | n=%d | params=%d",
        p, optimizer_seed, n, n_params
    )
    logger.info(
        "problem_seed=%d | backend=%s | population=%d | generations=%d",
        args.problem_seed, circ.backend,
        args.population, args.generations
    )

    # --------------------------------------------------------
    # Save run config
    # --------------------------------------------------------
    run_config = {
        "created_at": datetime.now().isoformat(),
        "p": p,
        "optimizer_seed": optimizer_seed,
        "problem_seed": args.problem_seed,
        "rows": args.rows,
        "cols": args.cols,
        "n_qubits": n,
        "n_params": n_params,
        "population": args.population,
        "generations": args.generations,
        "backend_requested": args.backend,
        "backend_actual": circ.backend,
        "mutation_rate": 0.15,
        "crossover_rate": 0.7,
        "mutation_sigma": 0.5,
        "mutation_strategy": "gaussian",
        "objective": "minimize exact QAOA expectation <G>",
        "metric_primary": "P(G*)",
        "metric_secondary": [
            "<G>",
            "entropy_bits",
            "N_eff",
        ],
    }
    atomic_json_dump(run_config, run_dir / "config.json")

    # Keep a copy of the problem so each run is self-contained
    shutil.copy2(
        args.problem_file,
        run_dir / "problem.npz"
    )

    # --------------------------------------------------------
    # QAOA probability / fitness
    # --------------------------------------------------------
    def probs_fn(params):
        params = np.asarray(params, dtype=float)
        gammas = params[:p]
        betas = params[p:]
        probs = circ.probs(
            gammas,
            betas,
            G_of_state,
        )
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum()
        return probs

    def fitness_fn(params):
        probs = probs_fn(params)
        return float(np.dot(probs, G_of_state))

    # --------------------------------------------------------
    # Evolutionary optimizer
    # --------------------------------------------------------
    opt = EvolutionaryOptimizer(
        n_params=n_params,
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

    opt.initialize_population(
        seed=optimizer_seed,
        verbose=False,
    )

    # --------------------------------------------------------
    # generation 0 + 1...generations
    # --------------------------------------------------------
    history = []

    def record_generation(gen):
        probs = probs_fn(
            opt.best_individual.params
        )
        m = exact_metrics(
            probs,
            G_of_state,
            opt_mask,
        )

        optimizer_hist = (
            opt.history[-1]
            if opt.history
            else {}
        )

        row = {
            "p": p,
            "optimizer_seed": optimizer_seed,
            "generation": gen,
            "evaluations": getattr(
                opt, "n_evaluations", np.nan
            ),
            "best_fitness": float(
                opt.best_individual.fitness
            ),
            "expG": m["expG"],
            "P_Gstar": m["P_Gstar"],
            "entropy_bits": m["entropy_bits"],
            "N_eff": m["N_eff"],
            "top_sequence": m["top_sequence"],
            "top_G": m["top_G"],
            "avg_population_fitness": optimizer_hist.get(
                "avg_fitness", np.nan
            ),
            "std_population_fitness": optimizer_hist.get(
                "std_fitness", np.nan
            ),
            "mutation_rate": optimizer_hist.get(
                "mutation_rate",
                getattr(opt, "mutation_rate", 0.15),
            ),
        }

        history.append(row)
        return m, row

    m0, _ = record_generation(0)

    logger.info(
        "gen=0/%d | <G>=%.8f | P(G*)=%.6f | "
        "H=%.4f | N_eff=%.4f | top=%s | top_G=%.6f",
        args.generations,
        m0["expG"],
        m0["P_Gstar"],
        m0["entropy_bits"],
        m0["N_eff"],
        m0["top_sequence"],
        m0["top_G"],
    )

    t0 = time.time()
    report_every = max(1, args.generations // 20)

    for gen in range(1, args.generations + 1):
        opt.run_generation()

        m, _ = record_generation(gen)

        if (
            gen % report_every == 0
            or gen == args.generations
        ):
            logger.info(
                "gen=%d/%d | <G>=%.8f | P(G*)=%.6f | "
                "H=%.4f | N_eff=%.4f | top=%s | top_G=%.6f",
                gen,
                args.generations,
                m["expG"],
                m["P_Gstar"],
                m["entropy_bits"],
                m["N_eff"],
                m["top_sequence"],
                m["top_G"],
            )

    elapsed = time.time() - t0

    # --------------------------------------------------------
    # Final distribution
    # --------------------------------------------------------
    best_params = np.asarray(
        opt.best_individual.params,
        dtype=float,
    ).copy()

    final_probs = probs_fn(best_params)
    final_metrics = exact_metrics(
        final_probs,
        G_of_state,
        opt_mask,
    )

    # --------------------------------------------------------
    # Save generation metrics
    # --------------------------------------------------------
    write_csv(
        run_dir / "generation_metrics.csv",
        history,
    )

    # --------------------------------------------------------
    # Save final state distribution
    # --------------------------------------------------------
    state_rows = []
    for state in range(problem["n_states"]):
        state_rows.append({
            "state": state,
            "sequence": sequence_from_state(
                state, n
            ),
            "G": float(G_of_state[state]),
            "is_global_optimum": int(
                opt_mask[state]
            ),
            "probability": float(
                final_probs[state]
            ),
        })

    write_csv(
        run_dir / "final_state_distribution.csv",
        state_rows,
    )

    # --------------------------------------------------------
    # Save parameters
    # --------------------------------------------------------
    np.savez(
        run_dir / "best_params.npz",
        best_params=best_params,
        gammas=best_params[:p],
        betas=best_params[p:],
    )

    # --------------------------------------------------------
    # final summary
    # --------------------------------------------------------
    summary = {
        "status": "ok",
        "p": p,
        "optimizer_seed": optimizer_seed,
        "problem_seed": args.problem_seed,
        "n_qubits": n,
        "n_params": n_params,
        "population": args.population,
        "generations": args.generations,
        "evaluation_budget_generations_only":
            args.population * args.generations,
        "backend": circ.backend,
        "time_sec": elapsed,
        "G_star": float(problem["G_star"]),
        "n_opt": int(opt_mask.sum()),
        "uniform_expG": float(G_of_state.mean()),
        "final_expG": final_metrics["expG"],
        "final_P_Gstar": final_metrics["P_Gstar"],
        "final_entropy_bits": final_metrics["entropy_bits"],
        "final_N_eff": final_metrics["N_eff"],
        "top_state": final_metrics["top_state"],
        "top_sequence": final_metrics["top_sequence"],
        "top_G": final_metrics["top_G"],
        "best_params": best_params.tolist(),
        "gammas": best_params[:p].tolist(),
        "betas": best_params[p:].tolist(),
        "ising_max_error":
            float(problem["ising_max_error"]),
        "final_distribution_sum":
            float(final_probs.sum()),
    }

    atomic_json_dump(
        summary,
        run_dir / "summary.json"
    )

    logger.info(
        "DONE | p=%d | seed=%d | <G>=%.8f | "
        "P(G*)=%.6f | H=%.4f | N_eff=%.4f | "
        "top=%s | top_G=%.6f | time=%.2fs",
        p,
        optimizer_seed,
        summary["final_expG"],
        summary["final_P_Gstar"],
        summary["final_entropy_bits"],
        summary["final_N_eff"],
        summary["top_sequence"],
        summary["top_G"],
        elapsed,
    )

    return summary


# ============================================================
# 4. Aggregate all p / seed
# ============================================================

def aggregate_results(root_dir, run_records, args, problem):
    root_dir = Path(root_dir)

    ok_records = [
        r for r in run_records
        if r.get("status") == "ok"
    ]

    # --------------------------------------------------------
    # One row per p x seed
    # --------------------------------------------------------
    summary_fields = [
        "p",
        "optimizer_seed",
        "status",
        "problem_seed",
        "n_qubits",
        "n_params",
        "population",
        "generations",
        "G_star",
        "n_opt",
        "uniform_expG",
        "final_expG",
        "final_P_Gstar",
        "final_entropy_bits",
        "final_N_eff",
        "top_sequence",
        "top_G",
        "time_sec",
        "backend",
        "run_dir",
    ]

    rows = []
    for r in run_records:
        row = {
            k: r.get(k, "")
            for k in summary_fields
        }
        rows.append(row)

    rows.sort(
        key=lambda x: (
            int(x["p"]),
            int(x["optimizer_seed"]),
        )
    )

    write_csv(
        root_dir / "depth_summary.csv",
        rows,
        summary_fields,
    )

    # --------------------------------------------------------
    # Aggregate by p: mean / SD / median / min / max
    # --------------------------------------------------------
    by_p = []

    for p in sorted(
        set(int(r["p"]) for r in ok_records)
    ):
        rs = [
            r for r in ok_records
            if int(r["p"]) == p
        ]

        metrics = [
            "final_expG",
            "final_P_Gstar",
            "final_entropy_bits",
            "final_N_eff",
            "time_sec",
        ]

        row = {
            "p": p,
            "n_runs": len(rs),
            "n_params": 2 * p,
        }

        for metric in metrics:
            vals = np.array(
                [float(r[metric]) for r in rs],
                dtype=float,
            )

            row[f"{metric}_mean"] = float(
                np.mean(vals)
            )
            row[f"{metric}_sd"] = float(
                np.std(vals, ddof=1)
            ) if len(vals) > 1 else 0.0
            row[f"{metric}_median"] = float(
                np.median(vals)
            )
            row[f"{metric}_min"] = float(
                np.min(vals)
            )
            row[f"{metric}_max"] = float(
                np.max(vals)
            )

        # How many runs per p found positive probability on G*
        row["runs_with_P_Gstar_gt_0"] = int(
            sum(
                float(r["final_P_Gstar"]) > 1e-12
                for r in rs
            )
        )

        # How many runs per p have their top-probability sequence equal to the global optimum
        row["runs_top_sequence_is_optimal"] = int(
            sum(
                float(r["top_G"])
                <= float(problem["G_star"]) + 1e-9
                for r in rs
            )
        )

        by_p.append(row)

    write_csv(
        root_dir / "depth_summary_by_p.csv",
        by_p,
    )

    # --------------------------------------------------------
    # generation-level aggregation
    # --------------------------------------------------------
    convergence_rows = []

    for r in ok_records:
        run_dir = Path(r["run_dir"])
        metrics_path = run_dir / "generation_metrics.csv"

        if not metrics_path.exists():
            continue

        with metrics_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for rr in reader:
                convergence_rows.append({
                    "p": int(rr["p"]),
                    "optimizer_seed": int(
                        rr["optimizer_seed"]
                    ),
                    "generation": int(
                        rr["generation"]
                    ),
                    "expG": float(rr["expG"]),
                    "P_Gstar": float(
                        rr["P_Gstar"]
                    ),
                    "entropy_bits": float(
                        rr["entropy_bits"]
                    ),
                    "N_eff": float(
                        rr["N_eff"]
                    ),
                })

    write_csv(
        root_dir / "convergence_raw.csv",
        convergence_rows,
    )

    # mean ± SD by p × generation
    convergence_summary = []

    groups = {}
    for r in convergence_rows:
        key = (r["p"], r["generation"])
        groups.setdefault(key, []).append(r)

    for (p, gen) in sorted(groups):
        rs = groups[(p, gen)]

        row = {
            "p": p,
            "generation": gen,
            "n_runs": len(rs),
        }

        for metric in [
            "expG",
            "P_Gstar",
            "entropy_bits",
            "N_eff",
        ]:
            vals = np.array(
                [x[metric] for x in rs],
                dtype=float,
            )

            row[f"{metric}_mean"] = float(
                np.mean(vals)
            )
            row[f"{metric}_sd"] = (
                float(np.std(vals, ddof=1))
                if len(vals) > 1 else 0.0
            )

        convergence_summary.append(row)

    write_csv(
        root_dir / "convergence_summary.csv",
        convergence_summary,
    )

    return by_p, convergence_summary


# ============================================================
# 5. Summary plots
# ============================================================

def plot_depth_metric(
    by_p,
    mean_key,
    sd_key,
    ylabel,
    title,
    output_path,
    lower_is_better=False,
):
    if not by_p:
        return

    pvals = np.array(
        [int(r["p"]) for r in by_p],
        dtype=float,
    )
    means = np.array(
        [float(r[mean_key]) for r in by_p],
        dtype=float,
    )
    sds = np.array(
        [float(r[sd_key]) for r in by_p],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.errorbar(
        pvals,
        means,
        yerr=sds,
        marker="o",
        capsize=4,
        linewidth=1.8,
    )

    ax.set_xlabel(
        "QAOA depth p"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(pvals)
    ax.grid(alpha=0.25)

    if lower_is_better:
        pass

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_convergence_by_depth(
    convergence_summary,
    output_path,
    metric,
    ylabel,
    title,
):
    if not convergence_summary:
        return

    fig, ax = plt.subplots(
        figsize=(8.5, 5.8)
    )

    pvals = sorted(
        set(int(r["p"])
            for r in convergence_summary)
    )

    for p in pvals:
        rs = [
            r for r in convergence_summary
            if int(r["p"]) == p
        ]

        x = np.array(
            [int(r["generation"]) for r in rs]
        )
        y = np.array(
            [float(r[f"{metric}_mean"]) for r in rs]
        )
        sd = np.array(
            [float(r[f"{metric}_sd"]) for r in rs]
        )

        ax.plot(
            x,
            y,
            linewidth=1.5,
            label=f"p={p}",
        )

        # Do not draw all SD shadows by default, to avoid crowding with 9 curves.
        # To show SD, uncomment the block below.
        #
        # ax.fill_between(
        #     x, y-sd, y+sd, alpha=0.08
        # )

    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 6. validation summary
# ============================================================

def write_validation_summary(
    root_dir,
    problem,
    run_records,
    by_p,
    args,
):
    root_dir = Path(root_dir)

    ok = [
        r for r in run_records
        if r.get("status") == "ok"
    ]
    failed = [
        r for r in run_records
        if r.get("status") != "ok"
    ]

    lines = []
    lines.append(
        "QAOA depth sweep validation summary"
    )
    lines.append(
        "=" * 60
    )
    lines.append(
        f"target = {args.rows}x{args.cols}"
    )
    lines.append(
        f"n_qubits = {problem['n']}"
    )
    lines.append(
        f"problem_seed = {args.problem_seed}"
    )
    lines.append(
        f"p range = {args.p_start} ... {args.p_end}"
    )
    lines.append(
        f"optimizer seeds = {args.seed_start} ... {args.seed_end}"
    )
    lines.append(
        f"population = {args.population}"
    )
    lines.append(
        f"generations = {args.generations}"
    )
    lines.append("")
    lines.append(
        f"expected runs = "
        f"{(args.p_end-args.p_start+1) * (args.seed_end-args.seed_start+1)}"
    )
    lines.append(
        f"successful runs = {len(ok)}"
    )
    lines.append(
        f"failed runs = {len(failed)}"
    )
    lines.append("")
    lines.append(
        f"G* = {problem['G_star']:.10f}"
    )
    lines.append(
        f"number of optimal states = "
        f"{int(problem['opt_mask'].sum())}"
    )
    lines.append(
        f"uniform <G> = "
        f"{float(problem['G_of_state'].mean()):.10f}"
    )
    lines.append(
        f"max Ising/direct-G error = "
        f"{problem['ising_max_error']:.3e}"
    )
    lines.append("")

    lines.append(
        "Per-depth summary:"
    )

    for r in by_p:
        lines.append(
            "  "
            f"p={int(r['p']):2d} | "
            f"params={int(r['n_params']):2d} | "
            f"runs={int(r['n_runs']):2d} | "
            f"<G>={float(r['final_expG_mean']):.6f} "
            f"+/- {float(r['final_expG_sd']):.6f} | "
            f"P(G*)={float(r['final_P_Gstar_mean']):.6f} "
            f"+/- {float(r['final_P_Gstar_sd']):.6f} | "
            f"N_eff={float(r['final_N_eff_mean']):.3f} "
            f"+/- {float(r['final_N_eff_sd']):.3f}"
        )

    if failed:
        lines.append("")
        lines.append("Failed runs:")
        for r in failed:
            lines.append(
                f"  p={r.get('p')} "
                f"seed={r.get('optimizer_seed')} "
                f"error={r.get('error', '')}"
            )

    with (
        root_dir / "validation_summary.txt"
    ).open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# 7. Main program
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "QAOA depth sweep p=1..9 "
            "on a fixed HP inverse-folding problem"
        )
    )

    ap.add_argument(
        "--rows",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--cols",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--p-start",
        type=int,
        default=1,
        help="QAOA minimum depth, default=1",
    )
    ap.add_argument(
        "--p-end",
        type=int,
        default=9,
        help="QAOA maximum depth, default=9",
    )

    ap.add_argument(
        "--seed-start",
        type=int,
        default=46,
        help="optimizer seed start, inclusive",
    )
    ap.add_argument(
        "--seed-end",
        type=int,
        default=55,
        help="optimizer seed end, inclusive",
    )

    ap.add_argument(
        "--problem-seed",
        type=int,
        default=12345,
        help="fixed problem/Hamiltonian seed",
    )

    ap.add_argument(
        "--generations",
        type=int,
        default=200,
    )
    ap.add_argument(
        "--population",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--backend",
        choices=[
            "auto",
            "default.qubit",
            "qiskit.aer",
        ],
        default="auto",
    )

    ap.add_argument(
        "--out-root",
        default="outputs",
        help=(
            "root directory used when --out-dir is not given"
        ),
    )

    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "explicit experiment directory; "
            "useful for resume"
        ),
    )

    ap.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "do not skip existing completed p/seed runs"
        ),
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------
    # Argument checks
    # --------------------------------------------------------
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit(
            "--rows and --cols must be positive."
        )

    if args.p_start < 1:
        raise SystemExit(
            "--p-start must be >= 1."
        )

    if args.p_end < args.p_start:
        raise SystemExit(
            "--p-end must be >= --p-start."
        )

    if args.seed_end < args.seed_start:
        raise SystemExit(
            "--seed-end must be >= --seed-start."
        )

    if args.generations < 1:
        raise SystemExit(
            "--generations must be >= 1."
        )

    if args.population < 2:
        raise SystemExit(
            "--population must be >= 2."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    if args.out_dir is not None:
        root_dir = Path(args.out_dir)
        root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        root_dir = (
            SCRIPT_DIR
            / args.out_root
            / (
                f"depth_p{args.p_start}-"
                f"p{args.p_end}_"
                f"{args.rows}x{args.cols}_"
                f"prob{args.problem_seed}_"
                f"seeds{args.seed_start}-"
                f"{args.seed_end}_"
                f"{stamp}"
            )
        )
        root_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    # --------------------------------------------------------
    # Main logger
    # --------------------------------------------------------
    logger = setup_logger(
        root_dir / "experiment.log"
    )

    logger.info("=" * 72)
    logger.info(
        "QAOA DEPTH SWEEP START: p=%d..%d",
        args.p_start,
        args.p_end,
    )
    logger.info(
        "target=%dx%d | problem_seed=%d | "
        "optimizer_seeds=%d..%d",
        args.rows,
        args.cols,
        args.problem_seed,
        args.seed_start,
        args.seed_end,
    )
    logger.info(
        "population=%d | generations=%d | backend=%s",
        args.population,
        args.generations,
        args.backend,
    )
    logger.info(
        "expected total runs=%d",
        (
            (args.p_end - args.p_start + 1)
            * (args.seed_end - args.seed_start + 1)
        ),
    )

    # --------------------------------------------------------
    # Fixed problem: build only once for the entire depth sweep
    # --------------------------------------------------------
    problem = build_problem(
        args.rows,
        args.cols,
        args.problem_seed,
    )

    problem_file = root_dir / "problem.npz"
    save_problem(problem, problem_file)

    # Temporarily attach the absolute path to args for use by the run
    args.problem_file = str(problem_file)

    # Save problem summary
    problem_summary = {
        "rows": args.rows,
        "cols": args.cols,
        "n_qubits": problem["n"],
        "n_states": problem["n_states"],
        "problem_seed": args.problem_seed,
        "G_star": problem["G_star"],
        "n_opt": int(problem["opt_mask"].sum()),
        "uniform_expG": float(
            problem["G_of_state"].mean()
        ),
        "target_contacts": int(
            problem["C_target"].sum() / 2
        ),
        "ising_max_error": problem[
            "ising_max_error"
        ],
    }

    atomic_json_dump(
        problem_summary,
        root_dir / "problem_summary.json",
    )

    # --------------------------------------------------------
    # Experiment config
    # --------------------------------------------------------
    experiment_config = vars(args).copy()
    experiment_config.update({
        "created_at": datetime.now().isoformat(),
        "experiment_type": "QAOA depth sweep",
        "p_values": list(
            range(args.p_start, args.p_end + 1)
        ),
        "optimizer_seeds": list(
            range(args.seed_start, args.seed_end + 1)
        ),
        "fixed_problem_seed": args.problem_seed,
        "objective":
            "structure-conditioned HP sequence design / inverse folding",
        "primary_endpoint":
            "P(G*)",
        "secondary_endpoints": [
            "<G>",
            "entropy_bits",
            "N_eff",
        ],
        "qaoa_parameter_count":
            "2p",
        "evaluation_budget_per_run":
            args.population * args.generations,
        "problem_file": str(problem_file),
    })

    atomic_json_dump(
        experiment_config,
        root_dir / "experiment_config.json",
    )

    # --------------------------------------------------------
    # Loop over p x seed
    # --------------------------------------------------------
    all_records = []

    p_values = range(
        args.p_start,
        args.p_end + 1,
    )
    seed_values = range(
        args.seed_start,
        args.seed_end + 1,
    )

    total_runs = (
        (args.p_end - args.p_start + 1)
        * (args.seed_end - args.seed_start + 1)
    )

    completed_counter = 0
    t_experiment = time.time()

    for p in p_values:
        logger.info("")
        logger.info("-" * 72)
        logger.info(
            "DEPTH p=%d | trainable parameters=2p=%d",
            p,
            2 * p,
        )
        logger.info("-" * 72)

        for seed in seed_values:
            completed_counter += 1

            run_dir = (
                root_dir
                / f"p_{p:02d}"
                / f"seed_{seed}"
            )

            summary_path = (
                run_dir / "summary.json"
            )

            # ------------------------------------------------
            # resume: if a complete summary already exists, read it directly
            # ------------------------------------------------
            if (
                not args.no_resume
                and summary_path.exists()
            ):
                try:
                    with summary_path.open(
                        "r",
                        encoding="utf-8",
                    ) as f:
                        existing = json.load(f)

                    if (
                        existing.get("status") == "ok"
                        and int(existing.get("p", -1)) == p
                        and int(
                            existing.get(
                                "optimizer_seed", -1
                            )
                        ) == seed
                    ):
                        all_records.append(existing)
                        logger.info(
                            "[%d/%d] SKIP existing "
                            "p=%d seed=%d",
                            completed_counter,
                            total_runs,
                            p,
                            seed,
                        )
                        continue

                except Exception as exc:
                    logger.warning(
                        "Existing summary unreadable; "
                        "rerunning p=%d seed=%d: %r",
                        p,
                        seed,
                        exc,
                    )

            logger.info(
                "[%d/%d] RUN p=%d seed=%d",
                completed_counter,
                total_runs,
                p,
                seed,
            )

            t0 = time.time()

            try:
                summary = run_single_qaoa(
                    p=p,
                    optimizer_seed=seed,
                    problem=problem,
                    args=args,
                    run_dir=run_dir,
                )

                summary["run_dir"] = str(
                    run_dir.resolve()
                )
                all_records.append(summary)

                logger.info(
                    "[%d/%d] COMPLETE p=%d seed=%d "
                    "elapsed=%.2fs | "
                    "P(G*)=%.6f | <G>=%.8f",
                    completed_counter,
                    total_runs,
                    p,
                    seed,
                    time.time() - t0,
                    summary["final_P_Gstar"],
                    summary["final_expG"],
                )

            except Exception as exc:
                failed_record = {
                    "status": "failed",
                    "p": p,
                    "optimizer_seed": seed,
                    "problem_seed": args.problem_seed,
                    "n_qubits": problem["n"],
                    "n_params": 2 * p,
                    "error": repr(exc),
                    "run_dir": str(
                        run_dir.resolve()
                    ),
                }

                all_records.append(
                    failed_record
                )

                logger.exception(
                    "[%d/%d] FAILED p=%d seed=%d "
                    "elapsed=%.2fs; continuing",
                    completed_counter,
                    total_runs,
                    p,
                    seed,
                    time.time() - t0,
                )

            # Save the current aggregate immediately after each run completes
            write_csv(
                root_dir / "depth_summary.csv",
                all_records,
            )

    # --------------------------------------------------------
    # Final aggregation
    # --------------------------------------------------------
    logger.info("")
    logger.info("=" * 72)
    logger.info("ALL RUNS FINISHED")
    logger.info(
        "wall-clock time = %.2f min",
        (time.time() - t_experiment) / 60.0,
    )

    by_p, convergence_summary = aggregate_results(
        root_dir,
        all_records,
        args,
        problem,
    )

    # --------------------------------------------------------
    # Plot the main depth-sweep figure
    # --------------------------------------------------------
    plot_depth_metric(
        by_p,
        "final_expG_mean",
        "final_expG_sd",
        "<G>",
        "QAOA depth vs final expected design score",
        root_dir / "depth_vs_expG.png",
        lower_is_better=True,
    )

    plot_depth_metric(
        by_p,
        "final_P_Gstar_mean",
        "final_P_Gstar_sd",
        "P(G*)",
        "QAOA depth vs probability mass on global optimum",
        root_dir / "depth_vs_PGstar.png",
    )

    plot_depth_metric(
        by_p,
        "final_N_eff_mean",
        "final_N_eff_sd",
        r"$N_{\mathrm{eff}} = 2^H$",
        "QAOA depth vs effective sequence number",
        root_dir / "depth_vs_Neff.png",
    )

    # --------------------------------------------------------
    # convergence figure: suggested as a supplementary plot
    # --------------------------------------------------------
    plot_convergence_by_depth(
        convergence_summary,
        root_dir / "convergence_by_depth.png",
        metric="expG",
        ylabel="<G>",
        title="Expected design score convergence across QAOA depths",
    )

    # --------------------------------------------------------
    # validation
    # --------------------------------------------------------
    write_validation_summary(
        root_dir,
        problem,
        all_records,
        by_p,
        args,
    )

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------
    final_summary = {
        "completed_at": datetime.now().isoformat(),
        "root_dir": str(root_dir.resolve()),
        "total_requested_runs": total_runs,
        "successful_runs": int(
            sum(
                r.get("status") == "ok"
                for r in all_records
            )
        ),
        "failed_runs": int(
            sum(
                r.get("status") != "ok"
                for r in all_records
            )
        ),
        "problem": problem_summary,
        "depth_summary_by_p": by_p,
    }

    atomic_json_dump(
        final_summary,
        root_dir / "final_summary.json",
    )

    # --------------------------------------------------------
    # Print final summary to terminal
    # --------------------------------------------------------
    logger.info("")
    logger.info("DEPTH SUMMARY")
    logger.info(
        "%-5s %-8s %-12s %-12s %-12s",
        "p",
        "params",
        "<G> mean",
        "P(G*) mean",
        "N_eff mean",
    )
    logger.info("-" * 60)

    for r in by_p:
        logger.info(
            "%-5d %-8d %-12.6f %-12.6f %-12.3f",
            int(r["p"]),
            int(r["n_params"]),
            float(r["final_expG_mean"]),
            float(r["final_P_Gstar_mean"]),
            float(r["final_N_eff_mean"]),
        )

    logger.info("")
    logger.info(
        "RESULT_DIR=%s",
        root_dir.resolve(),
    )
    logger.info(
        "SUMMARY_CSV=%s",
        (root_dir / "depth_summary.csv").resolve(),
    )
    logger.info(
        "BY_P_CSV=%s",
        (root_dir / "depth_summary_by_p.csv").resolve(),
    )
    logger.info(
        "DONE"
    )

    print("")
    print("RESULT_DIR=" + str(root_dir.resolve()))
    print(
        "SUMMARY_CSV="
        + str(
            (root_dir / "depth_summary.csv").resolve()
        )
    )
    print(
        "BY_P_CSV="
        + str(
            (root_dir / "depth_summary_by_p.csv").resolve()
        )
    )


if __name__ == "__main__":
    main()
