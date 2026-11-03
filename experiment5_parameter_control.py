#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import json
import csv
import time
import argparse
import hashlib
import warnings
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qaoa_generator import (
    ALPHABET, snake_coords, contact_map, build_ising, direct_G,
    QAOACircuit, _bit_reverse_index, _apply_1q, _apply_cnot,
    _RY, _RZ, distribution_entropy,
)
from evolutionary import EvolutionaryOptimizer


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sequence_from_state(s, n):
    return "".join("H" if ((int(s) >> i) & 1) == 0 else "P" for i in range(n))


def exact_metrics(probs, G, opt_mask):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    expG = float(np.sum(probs * G))
    pstar = float(probs[opt_mask].sum())
    H, Neff = distribution_entropy(probs)
    top = int(np.argmax(probs))
    return {
        "expG": expG,
        "P_Gstar": pstar,
        "entropy_bits": float(H),
        "N_eff": float(Neff),
        "top_state": top,
        "top_sequence": _sequence_from_state(top, len(G).bit_length() - 1),
        "top_probability": float(probs[top]),
        "top_G": float(G[top]),
    }


class CompactHEA12:
    """Exactly 12-parameter compact HEA control.

    Parameters 0..8  : one local RY angle per qubit.
    Parameters 9..11 : three shared RZ angles applied to qubit groups
                       (0,1,2), (3,4,5), (6,7,8).
    Then one ring of CNOTs.
    """

    def __init__(self, n, backend="auto"):
        if n != 9:
            raise ValueError("CompactHEA12 is defined for n=9 only.")
        self.n = n
        self.n_params = 12
        self.backend_pref = backend
        self.backend = "numpy"
        self._pl = None
        self._rev = _bit_reverse_index(n)
        self._try_pennylane()

    def _make_device(self, qml):
        if self.backend_pref == "qiskit.aer":
            names = [("qiskit.aer", {"backend": "statevector_simulator"})]
        else:
            names = [("default.qubit", {}),
                     ("qiskit.aer", {"backend": "statevector_simulator"})]
        for name, kwargs in names:
            try:
                return qml.device(name, wires=self.n, **kwargs), name
            except Exception:
                continue
        return None, None

    def _try_pennylane(self):
        try:
            import pennylane as qml
            dev, dev_name = self._make_device(qml)
            if dev is None:
                raise RuntimeError("No PennyLane device available")

            @qml.qnode(dev, interface=None)
            def circuit(params):
                for q in range(self.n):
                    qml.RY(params[q], wires=q)
                groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
                for k, group in enumerate(groups):
                    for q in group:
                        qml.RZ(params[9 + k], wires=q)
                for i in range(self.n):
                    qml.CNOT(wires=[i, (i + 1) % self.n])
                return qml.probs(wires=range(self.n))

            self._pl = circuit
            self.backend = f"pennylane({dev_name})"
        except Exception as e:
            warnings.warn(
                f"PennyLane unavailable ({type(e).__name__}: {e}); "
                "CompactHEA12 will use NumPy statevector simulation."
            )

    def probs(self, params):
        params = np.asarray(params, dtype=float)
        if params.size != 12:
            raise ValueError("CompactHEA12 requires exactly 12 parameters.")
        if self._pl is not None:
            raw = np.asarray(self._pl(params.tolist()), dtype=float)
            return raw[self._rev]
        return self._probs_numpy(params)

    def _probs_numpy(self, params):
        psi = np.zeros(2 ** self.n, dtype=np.complex128)
        psi[0] = 1.0
        for q in range(self.n):
            psi = _apply_1q(psi, _RY(params[q]), q)
        groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        for k, group in enumerate(groups):
            for q in group:
                psi = _apply_1q(psi, _RZ(params[9 + k]), q)
        for i in range(self.n):
            psi = _apply_cnot(psi, i, (i + 1) % self.n)
        return (np.abs(psi) ** 2)[self._rev]


def standard_hea_probs(params, n, layers, backend="auto"):
    """Use the existing standard HEA implementation from qaoa_generator."""
    from qaoa_generator import HardwareEfficientCircuit
    circ = standard_hea_probs._cache.get((n, layers, backend))
    if circ is None:
        circ = HardwareEfficientCircuit(n, layers, backend=backend)
        standard_hea_probs._cache[(n, layers, backend)] = circ
    return circ.probs(np.asarray(params, dtype=float))

standard_hea_probs._cache = {}


def qaoa_probs_factory(n, p, h, J, G, backend="auto"):
    circ = QAOACircuit(n, h, J, backend=backend)
    def probs_fn(params):
        gammas = np.asarray(params[:p], dtype=float)
        betas = np.asarray(params[p:], dtype=float)
        return circ.probs(gammas, betas, G)
    return circ, probs_fn


def run_evolutionary(label, n_params, bounds, probs_fn, G, opt_mask,
                     population, generations, seed, sample_count):
    def fitness(params):
        p = probs_fn(params)
        return float(np.sum(p * G))

    opt = EvolutionaryOptimizer(
        n_params=n_params,
        param_bounds=bounds,
        fitness_fn=fitness,
        population_size=population,
        elite_size=max(2, population // 6),
        tournament_size=3,
        mutation_rate=0.15,
        crossover_rate=0.7,
        mutation_sigma=0.5,
        mutation_strategy="gaussian",
    )

    np.random.seed(seed)
    opt.initialize_population(seed=seed, verbose=False)

    rows = []
    t0 = time.time()
    for gen in range(generations + 1):
        # Record initial population before any generation update.
        if gen > 0:
            opt.run_generation()
        probs = probs_fn(opt.best_individual.params)
        m = exact_metrics(probs, G, opt_mask)
        rows.append({
            "generation": gen,
            "objective_evaluations": opt.n_evaluations,
            "best_fitness": float(opt.best_individual.fitness),
            **m,
        })
    elapsed = time.time() - t0

    final_probs = probs_fn(opt.best_individual.params)
    final_probs = final_probs / final_probs.sum()
    m = exact_metrics(final_probs, G, opt_mask)
    rng = np.random.RandomState(seed ^ 0x9E3779B9)
    samples = rng.choice(len(G), size=sample_count, p=final_probs)
    sampled_vals = G[samples]
    pstar = m["P_Gstar"]
    m95 = (np.log(0.05) / np.log(1 - pstar)) if 0 < pstar < 1 else (1.0 if pstar >= 1 else np.inf)

    return {
        "label": label,
        "n_params": n_params,
        "params": opt.best_individual.params.copy(),
        "probs": final_probs,
        "history": rows,
        "time_sec": elapsed,
        "objective_evaluations": int(opt.n_evaluations),
        "best_sampled_G": float(sampled_vals.min()),
        "sample_hit_rate": float(np.mean(sampled_vals <= G.min() + 1e-9)),
        "samples_for_95pct_hit": float(m95),
        **m,
    }


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
    err = max(abs(G[s] - direct_G(z[s], dC)) for s in range(n_states))
    return {
        "coords": coords, "h": h, "J": J, "const": const,
        "dC": dC, "C_target": C_target, "C_bg": C_bg,
        "G": G, "G_star": G_star, "opt_mask": opt_mask,
        "ising_error": float(err), "n_opt": int(opt_mask.sum()),
    }


def save_problem(outdir, problem, rows, cols, problem_seed):
    np.savez(
        os.path.join(outdir, "problem.npz"),
        coords=problem["coords"], h=problem["h"], J=problem["J"],
        const=problem["const"], dC=problem["dC"],
        C_target=problem["C_target"], C_bg=problem["C_bg"],
        G_of_state=problem["G"], G_star=problem["G_star"],
        opt_mask=problem["opt_mask"],
    )
    with open(os.path.join(outdir, "problem_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Experiment 5 fixed problem\n")
        f.write("==========================\n")
        f.write(f"lattice: {rows}x{cols}\n")
        f.write(f"n_qubits: {rows*cols}\n")
        f.write(f"problem_seed: {problem_seed}\n")
        f.write(f"G_star: {problem['G_star']:.12f}\n")
        f.write(f"n_opt: {problem['n_opt']}\n")
        f.write(f"uniform_P_Gstar: {problem['n_opt']/(2**(rows*cols)):.12f}\n")
        f.write(f"uniform_expG: {problem['G'].mean():.12f}\n")
        f.write(f"target_contacts: {int(problem['C_target'].sum()/2)}\n")
        f.write(f"ising_max_error: {problem['ising_error']:.3e}\n")


def plot_main_summary(outdir, summary_rows):
    labels = [r["condition"] for r in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    metrics = [
        ("P_Gstar", "Final P(G*)", "Probability on global optimum"),
        ("expG", "Final <G>", "Expected objective"),
        ("N_eff", "Final N_eff", "Effective sequence count"),
    ]
    for ax, (key, title, ylabel) in zip(axes, metrics):
        means = np.array([r[f"{key}_mean"] for r in summary_rows], dtype=float)
        sds = np.array([r[f"{key}_sd"] for r in summary_rows], dtype=float)
        ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure9_parameter_control_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_efficiency(outdir, summary_rows):
    x = np.array([r["n_params"] for r in summary_rows], dtype=float)
    y = np.array([r["P_Gstar_mean"] for r in summary_rows], dtype=float)
    yerr = np.array([r["P_Gstar_sd"] for r in summary_rows], dtype=float)
    labels = [r["condition"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final P(G*)")
    ax.set_title("Parameter count versus exact-optimum probability")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FigureS11_parameter_count_vs_PGstar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Experiment 5: parameter-matched and budget-matched QAOA vs HEA")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--qaoa-depth", type=int, default=6)
    ap.add_argument("--problem-seed", type=int, default=12345)
    ap.add_argument("--seed-start", type=int, default=46)
    ap.add_argument("--seed-end", type=int, default=55)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--sample-count", type=int, default=2000)
    ap.add_argument("--backend", choices=["auto", "default.qubit", "qiskit.aer"], default="auto")
    ap.add_argument("--out-root", default="outputs_exp5")
    args = ap.parse_args()

    if args.rows != 3 or args.cols != 3:
        raise ValueError("Experiment 5 is intentionally fixed to 3x3 for the controlled comparison.")
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be >= seed-start")

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_root)
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(base, f"exp5_3x3_p6_prob{args.problem_seed}_seeds{args.seed_start}-{args.seed_end}_{stamp}")
    os.makedirs(batch_dir, exist_ok=True)

    config = vars(args).copy()
    config.update({
        "created_at": datetime.now().isoformat(),
        "objective": "structure-conditioned HP sequence design / inverse folding",
        "conditions": {
            "QAOA-p6": {"parameters": 12, "description": "6 gamma + 6 beta"},
            "Compact-HEA12": {"parameters": 12, "description": "9 local RY + 3 shared RZ + ring CNOT"},
            "HEA-L1": {"parameters": 18, "description": "standard HEA, one layer"},
            "HEA-L2": {"parameters": 36, "description": "standard HEA, two layers"},
        },
        "budget_definition": "same population x generations and same EvolutionaryOptimizer implementation for every condition",
        "total_runs": 4 * (args.seed_end - args.seed_start + 1),
    })
    with open(os.path.join(batch_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    problem = build_problem(args.rows, args.cols, args.problem_seed)
    save_problem(batch_dir, problem, args.rows, args.cols, args.problem_seed)

    G = problem["G"]
    opt_mask = problem["opt_mask"]
    n = args.rows * args.cols
    p = args.qaoa_depth

    # Build circuit objects once and reuse them across all seeds.
    q_circ, q_probs = qaoa_probs_factory(n, p, problem["h"], problem["J"], G, backend=args.backend)
    compact = CompactHEA12(n, backend=args.backend)
    standard1 = None
    standard2 = None

    def compact_probs(params):
        return compact.probs(params)

    def hea1_probs(params):
        return standard_hea_probs(params, n, 1, args.backend)

    def hea2_probs(params):
        return standard_hea_probs(params, n, 2, args.backend)

    conditions = [
        ("QAOA-p6", 12, [(0.0, 2*np.pi)] * 12, q_probs),
        ("Compact-HEA12", 12, [(-np.pi, np.pi)] * 12, compact_probs),
        ("HEA-L1", 18, [(-np.pi, np.pi)] * 18, hea1_probs),
        ("HEA-L2", 36, [(-np.pi, np.pi)] * 36, hea2_probs),
    ]

    # Backend sanity check for the parameter-matched circuit.
    test_params = np.linspace(-0.7, 0.8, 12)
    pl_compact = compact_probs(test_params)
    np_compact = compact._probs_numpy(test_params)
    compact_err = float(np.max(np.abs(pl_compact - np_compact)))

    all_rows = []
    summary_rows = []
    run_counter = 0
    seeds = list(range(args.seed_start, args.seed_end + 1))

    print("=" * 78)
    print(" Experiment 5 — Parameter-matched / budget-matched QAOA vs HEA")
    print("=" * 78)
    print(f"Problem: {args.rows}x{args.cols}, n={n}, problem_seed={args.problem_seed}")
    print(f"G*={problem['G_star']:.6f}, n_opt={problem['n_opt']}, states={2**n}")
    print(f"Budget: population={args.population}, generations={args.generations}")
    print(f"Optimizer seeds: {seeds}")
    print(f"QAOA backend: {q_circ.backend}")
    print(f"Compact-HEA12 backend: {compact.backend}")
    print(f"Compact-HEA12 self-check max probability error: {compact_err:.3e}")
    print("Conditions: QAOA-p6(12), Compact-HEA12(12), HEA-L1(18), HEA-L2(36)")

    with open(os.path.join(batch_dir, "experiment.log"), "w", encoding="utf-8") as log:
        log.write(f"Experiment 5 started {datetime.now().isoformat()}\n")
        log.write(f"problem_seed={args.problem_seed}; seeds={seeds}\n")
        log.write(f"G_star={problem['G_star']}; n_opt={problem['n_opt']}\n")
        log.write(f"compact_backend={compact.backend}; self_check={compact_err:.3e}\n")

    for condition, n_params, bounds, probs_fn in conditions:
        print("\n" + "-" * 78)
        print(f" {condition} | parameters={n_params}")
        print("-" * 78)
        cond_rows = []
        for seed in seeds:
            run_counter += 1
            run_dir = os.path.join(batch_dir, condition.replace("-", "_"), f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            t_run = time.time()
            result = run_evolutionary(
                condition, n_params, bounds, probs_fn, G, opt_mask,
                args.population, args.generations, seed, args.sample_count
            )
            elapsed = time.time() - t_run
            result["time_sec"] = elapsed

            np.savez(
                os.path.join(run_dir, "best_params.npz"),
                best_params=result["params"],
                final_probs=result["probs"],
                G_of_state=G,
                G_star=problem["G_star"],
            )
            _write_csv(os.path.join(run_dir, "generation_metrics.csv"), result["history"])
            final_row = {
                "condition": condition,
                "n_params": n_params,
                "seed": seed,
                "G_star": problem["G_star"],
                "n_opt": problem["n_opt"],
                "uniform_P_Gstar": problem["n_opt"] / len(G),
                "expG": result["expG"],
                "P_Gstar": result["P_Gstar"],
                "entropy_bits": result["entropy_bits"],
                "N_eff": result["N_eff"],
                "top_sequence": result["top_sequence"],
                "top_probability": result["top_probability"],
                "top_G": result["top_G"],
                "best_sampled_G": result["best_sampled_G"],
                "sample_hit_rate": result["sample_hit_rate"],
                "samples_for_95pct_hit": result["samples_for_95pct_hit"],
                "runtime_seconds": result["time_sec"],
                "objective_evaluations": result["objective_evaluations"],
            }
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(final_row, f, indent=2, ensure_ascii=False)
            cond_rows.append(final_row)
            all_rows.append(final_row)
            print(f"  seed {seed}: <G>={result['expG']:.6f} | P(G*)={100*result['P_Gstar']:.2f}% | "
                  f"N_eff={result['N_eff']:.2f} | evals={result['objective_evaluations']} | {result['time_sec']:.1f}s")

        def mean_sd(key):
            vals = np.array([r[key] for r in cond_rows], dtype=float)
            return float(vals.mean()), float(vals.std(ddof=1))

        sm = {
            "condition": condition,
            "n_params": n_params,
            "n_runs": len(cond_rows),
        }
        for key in ["expG", "P_Gstar", "entropy_bits", "N_eff", "runtime_seconds",
                    "objective_evaluations", "samples_for_95pct_hit", "sample_hit_rate"]:
            m, sd = mean_sd(key)
            sm[f"{key}_mean"] = m
            sm[f"{key}_sd"] = sd
        sm["uniform_P_Gstar"] = problem["n_opt"] / len(G)
        sm["P_Gstar_enrichment"] = sm["P_Gstar_mean"] / sm["uniform_P_Gstar"]
        summary_rows.append(sm)
        _write_csv(os.path.join(batch_dir, condition.replace("-", "_"), "seed_summary.csv"), cond_rows)

    _write_csv(os.path.join(batch_dir, "run_summary.csv"), all_rows)
    _write_csv(os.path.join(batch_dir, "condition_summary.csv"), summary_rows)

    with open(os.path.join(batch_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    plot_main_summary(batch_dir, summary_rows)
    plot_parameter_efficiency(batch_dir, summary_rows)

    print("\n" + "=" * 78)
    print(" FINAL SUMMARY")
    print("=" * 78)
    for r in summary_rows:
        print(f"{r['condition']:<16} params={r['n_params']:>2} | "
              f"<G>={r['expG_mean']:.6f}±{r['expG_sd']:.6f} | "
              f"P(G*)={100*r['P_Gstar_mean']:.2f}±{100*r['P_Gstar_sd']:.2f}% | "
              f"N_eff={r['N_eff_mean']:.2f}±{r['N_eff_sd']:.2f} | "
              f"time={r['runtime_seconds_mean']:.1f}s")
    print(f"\nOutput directory:\n{batch_dir}")
    print("RESULT_DIR=" + batch_dir)


if __name__ == "__main__":
    main()
