#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import io
import time
import argparse
import warnings
import json
import csv
import logging
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                      errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import matplotlib.font_manager as fm
for _name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    if any(_name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolutionary import EvolutionaryOptimizer


# ══════════════════════════════════════════════════════════════
#  1. HP model: target structure, contact map, background structures
# ══════════════════════════════════════════════════════════════
ALPHABET = ["H", "P"]          # 2-letter HP model (1 position = 1 qubit)
EPS = np.array([[-1.0, 0.0],   # eps(H,H)=-1 (hydrophobic contact favorable), others 0
                [0.0, 0.0]])


def snake_coords(rows, cols):
    """Snake-like compact conformation: returns array of shape (N, 2) with (row, col) of residue i"""
    coords = []
    for r in range(rows):
        if r % 2 == 0:
            for c in range(cols):
                coords.append((r, c))
        else:
            for c in range(cols - 1, -1, -1):
                coords.append((r, c))
    return np.array(coords, dtype=np.int32)


def contact_map(coords):
    """Contact map: residue pairs with Manhattan distance = 1 and |i-j| > 1 are marked as contacts"""
    N = len(coords)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 2, N):
            d = abs(coords[i, 0] - coords[j, 0]) + abs(coords[i, 1] - coords[j, 1])
            if d == 1:
                C[i, j] = C[j, i] = 1.0
    return C


def random_saw(rows, cols, length, rng):
    """Random self-avoiding walk (used as background / decoy structure)"""
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    for _ in range(200):
        coords = np.zeros((length, 2), dtype=np.int32)
        occupied = {(0, 0)}
        ok = True
        for i in range(1, length):
            rng.shuffle(directions)
            moved = False
            for d in directions:
                p = coords[i - 1] + d
                if 0 <= p[0] < rows and 0 <= p[1] < cols and tuple(p) not in occupied:
                    coords[i] = p
                    occupied.add(tuple(p))
                    moved = True
                    break
            if not moved:
                ok = False
                break
        if ok:
            return coords
    return None


# ══════════════════════════════════════════════════════════════
#  2. Ising mapping: G(S) -> constant + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
# ══════════════════════════════════════════════════════════════
#  HP model: 1 qubit encodes 1 residue, z_i = +1(H) / -1(P)
#    eps(s_i, s_j) = -1*delta_HH = -(1+z_i)(1+z_j)/4
# Hence
#    G(S) = sum_{i<j} DeltaC_ij * eps(s_i,s_j)
#         = constant + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
#  where DeltaC = C_target - <C_background>, and
#    h_i   = -1/4 * sum_{j!=i} DeltaC_ij
#    J_ij  = -1/4 * DeltaC_ij          (i!=j)
#    const = -1/4 * sum_{i<j} DeltaC_ij


def build_ising(coords, n_background=30, seed=12345):
    """Compute Ising coefficients (h, J, const) and DeltaC from target + background structures"""
    N = len(coords)
    rows = int(coords[:, 0].max()) + 1
    cols = int(coords[:, 1].max()) + 1
    C_target = contact_map(coords)

    rng = np.random.RandomState(seed)
    bg_maps = []
    for _ in range(n_background):
        alt = random_saw(rows, cols, N, rng)
        if alt is not None:
            bg_maps.append(contact_map(alt))
    C_bg = np.mean(bg_maps, axis=0) if bg_maps else np.zeros_like(C_target)

    dC = C_target - C_bg                     # effective contact map DeltaC
    h = -0.25 * dC.sum(axis=1)               # h_i = -1/4 sum_j DeltaC_ij (diagonal is 0)
    J = -0.25 * dC                           # J_ij = -1/4 DeltaC_ij
    np.fill_diagonal(J, 0.0)
    const = -0.25 * (dC.sum() / 2.0)         # sum_{i<j}
    return h, J, const, dC, C_target, C_bg


def direct_G(z, dC):
    """Compute G(S) directly from the eps matrix; verify the Ising expansion is consistent (transparent baseline)"""
    # z: (+1=H, -1=P), eps = -1/4 (1+z_i)(1+z_j)
    return float(np.sum([
        dC[i, j] * (-0.25 * (1 + z[i]) * (1 + z[j]))
        for i in range(len(z)) for j in range(i + 1, len(z))
    ]))


# ══════════════════════════════════════════════════════════════
#  3. QAOA circuit (two backends, mathematically equivalent)
# ══════════════════════════════════════════════════════════════
H_GATE = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def _apply_1q(psi, U, q):
    n = int(np.log2(len(psi)))
    psi = psi.reshape([2] * n)
    psi = np.tensordot(U, psi, axes=(1, q))   # (2, remaining axes)
    psi = np.moveaxis(psi, 0, q)              # move back to original position
    return psi.reshape(-1)


def _run_numpy(gammas, betas, G_of_state, n):
    """NumPy exact statevector simulation (equivalent to qiskit.aer statevector)"""
    p = len(gammas)
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[0] = 1.0
    for q in range(n):                        # |0...0> -> |+>^n
        psi = _apply_1q(psi, H_GATE, q)

    for g, b in zip(gammas, betas):
        # cost layer e^{-i gamma H_C}: diagonal phase e^{-i gamma G(s)} (energy encoded in phase)
        psi = psi * np.exp(-1j * g * G_of_state)
        # mixing layer e^{-i beta H_B} = product_q RX(2 beta)
        th = 2.0 * b
        RX = np.array([[np.cos(th / 2), -1j * np.sin(th / 2)],
                       [-1j * np.sin(th / 2), np.cos(th / 2)]])
        for q in range(n):
            psi = _apply_1q(psi, RX, q)
    return np.abs(psi) ** 2


def _bit_reverse_index(n):
    """Bit-reversal index for n bits: PennyLane qml.probs (first wire = MSB) -> align with our LSB convention"""
    n_states = 2 ** n
    idx = np.arange(n_states)
    rev = np.zeros(n_states, dtype=int)
    tmp = idx.copy()
    for _ in range(n):
        rev = (rev << 1) | (tmp & 1)
        tmp >>= 1
    return rev


class QAOACircuit:
    """QAOA circuit (PennyLane implementation, device chain qiskit.aer -> default.qubit -> NumPy fallback)"""

    def __init__(self, n, h, J, backend="auto"):
        self.n = n
        self.h = h
        self.J = J
        self.backend_pref = backend   # "auto" / "default.qubit" / "qiskit.aer"
        self.backend = "numpy"
        self._pl = None
        self._rev = _bit_reverse_index(n)
        self._try_pennylane()

    def _make_device(self, qml):
        """By default use PennyLane's built-in default.qubit (exact, fast, no qiskit translation overhead);
        only when --backend qiskit.aer is explicitly set do we switch back to qiskit's statevector backend."""
        if self.backend_pref == "qiskit.aer":
            names = [("qiskit.aer", {"backend": "statevector_simulator"})]
        else:  # both "auto" and "default.qubit" prefer default.qubit
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
                raise RuntimeError("no available PennyLane device")
            n, h, J = self.n, self.h, self.J

            @qml.qnode(dev, interface=None)
            def circuit(params):
                gammas = params[:len(params) // 2]
                betas = params[len(params) // 2:]
                for q in range(n):
                    qml.Hadamard(wires=q)
                for g, b in zip(gammas, betas):
                    # -- cost layer e^{-i gamma H_C}: energy encoded as RZ + IsingZZ gates --
                    for i in range(n):
                        if abs(h[i]) > 1e-12:
                            qml.RZ(2 * g * h[i], wires=i)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if abs(J[i, j]) > 1e-12:
                                qml.IsingZZ(2 * g * J[i, j], wires=[i, j])
                    # -- mixing layer e^{-i beta H_B}: RX gates --
                    for q in range(n):
                        qml.RX(2 * b, wires=q)
                return qml.probs(wires=range(n))

            self._pl = circuit
            self.backend = f"pennylane({dev_name})"
        except Exception as e:
            warnings.warn(f"PennyLane unavailable ({type(e).__name__}: {e}),"
                          f" falling back to NumPy statevector simulation.")

    def probs(self, gammas, betas, G_of_state):
        """Return probability distribution aligned to the LSB convention (PennyLane first wire = MSB, needs bit-reversal alignment)"""
        if self._pl is not None:
            params = np.concatenate([gammas, betas]).tolist()
            raw = np.asarray(self._pl(params), dtype=float)
            return raw[self._rev]
        return _run_numpy(gammas, betas, G_of_state, self.n)


# ══════════════════════════════════════════════════════════════
#  3b. Hardware-efficient ansatz (your original quantum_generator.py circuit, used for comparison)
# ══════════════════════════════════════════════════════════════
def _RY(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _RZ(t):
    return np.array([[np.exp(-1j * t / 2), 0.0],
                     [0.0, np.exp(1j * t / 2)]])


def _apply_cnot(psi, control, target):
    """Apply CNOT on the statevector (flips target when control=1)"""
    n = int(np.log2(len(psi)))
    psi = psi.reshape([2] * n)
    order = [control, target] + [q for q in range(n) if q not in (control, target)]
    psi = np.transpose(psi, order).reshape(4, -1)
    psi[[0, 1, 2, 3], :] = psi[[0, 1, 3, 2], :]      # |10> <-> |11>
    psi = psi.reshape([2, 2] + [2] * (n - 2))
    psi = np.transpose(psi, np.argsort(order))
    return psi.reshape(-1)


def _run_numpy_hea(params, n, n_layers, rev):
    """NumPy statevector simulation of hardware-efficient ansatz (RY/RZ + ring CNOT)"""
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[0] = 1.0
    idx = 0
    for _ in range(n_layers):
        for q in range(n):
            psi = _apply_1q(psi, _RY(params[idx]), q); idx += 1
            psi = _apply_1q(psi, _RZ(params[idx]), q); idx += 1
        for i in range(n):
            psi = _apply_cnot(psi, i, (i + 1) % n)
    return (np.abs(psi) ** 2)[rev]


class HardwareEfficientCircuit:
    """Hardware-efficient ansatz: RY/RZ rotation layers + ring CNOT entanglement (PennyLane implementation)"""

    def __init__(self, n, n_layers=2, backend="auto"):
        self.n = n
        self.n_layers = n_layers
        self.n_params = n * 2 * n_layers
        self.backend_pref = backend   # "auto" / "default.qubit" / "qiskit.aer"
        self.backend = "numpy"
        self._pl = None
        self._rev = _bit_reverse_index(n)
        self._try_pennylane()

    def _make_device(self, qml):
        if self.backend_pref == "qiskit.aer":
            names = [("qiskit.aer", {"backend": "statevector_simulator"})]
        else:  # both "auto" and "default.qubit" prefer default.qubit
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
                raise RuntimeError("no available PennyLane device")
            n, L = self.n, self.n_layers

            @qml.qnode(dev, interface=None)
            def circuit(params):
                idx = 0
                for _ in range(L):
                    for q in range(n):
                        qml.RY(params[idx], wires=q); idx += 1
                        qml.RZ(params[idx], wires=q); idx += 1
                    for i in range(n):
                        qml.CNOT(wires=[i, (i + 1) % n])
                return qml.probs(wires=range(n))

            self._pl = circuit
            self.backend = f"pennylane({dev_name})"
        except Exception as e:
            warnings.warn(f"PennyLane unavailable ({type(e).__name__}: {e}),"
                          f" hardware-efficient ansatz falls back to NumPy.")

    def probs(self, params):
        """Return LSB-aligned probability distribution (consistent with G_of_state)"""
        if self._pl is not None:
            raw = np.asarray(self._pl(params.tolist()), dtype=float)
            return raw[self._rev]
        return _run_numpy_hea(params, self.n, self.n_layers, self._rev)


# ══════════════════════════════════════════════════════════════
#  3c. Training / random search helpers (reused by comparison mode)
# ══════════════════════════════════════════════════════════════
def train_evolutionary(fitness_fn, probs_fn, n_params, bounds, population,
                       generations, seed, G_of_state, n_samples=2000,
                       label="optimize"):
    """(mu+lambda)-ES optimization, returns (optimizer, expected <G> curve, sampled best-G curve)"""
    np.random.seed(seed)   # fix random stream per training run for reproducibility with same seed
    opt = EvolutionaryOptimizer(
        n_params=n_params, param_bounds=bounds, fitness_fn=fitness_fn,
        population_size=population, elite_size=max(2, population // 6),
        tournament_size=3, mutation_rate=0.15, crossover_rate=0.7,
        mutation_sigma=0.5, mutation_strategy="gaussian",
    )
    opt.initialize_population(seed=seed, verbose=False)
    loss_curve = []     # expected <G>
    bestG_curve = []    # best G(s) obtained by sampling n_samples from the current best distribution
    samp_rng = np.random.RandomState(seed ^ 0x9E3779B9)
    report = max(1, generations // 20)   # print progress every 5% to avoid flooding output
    print(f"    [{label}] starting {generations} generations of training ...", flush=True)
    for gi in range(generations):
        opt.run_generation()
        loss_curve.append(opt.best_individual.fitness)
        probs = probs_fn(opt.best_individual.params)
        probs = probs / probs.sum()
        samples = samp_rng.choice(len(probs), size=n_samples, p=probs)
        bestG_curve.append(float(G_of_state[samples].min()))
        if (gi + 1) % report == 0 or gi == generations - 1:
            print(f"    [{label}] gen {gi + 1}/{generations}  "
                  f"<G>={loss_curve[-1]:.4f}  bestG={bestG_curve[-1]:.4f}",
                  flush=True)
    return opt, loss_curve, bestG_curve


def random_search(fitness_fn, probs_fn, n_params, bounds, population, generations,
                  seed, G_of_state, n_samples=2000, label="random search"):
    """Random search baseline: randomly sample parameters with the same budget, returns (expected <G> curve, sampled best-G curve, best params)"""
    rng = np.random.RandomState(seed)
    samp_rng = np.random.RandomState(seed ^ 0x9E3779B9)
    lows = np.array([lo for lo, hi in bounds])
    highs = np.array([hi for lo, hi in bounds])
    best = float("inf")
    best_params = None
    loss_curve = []
    bestG_curve = []
    report = max(1, generations // 20)
    print(f"    [{label}] starting {generations} generations of random sampling ...", flush=True)
    for gi in range(generations):
        for _ in range(population):
            p = rng.uniform(lows, highs)
            f = fitness_fn(p)
            if f < best:
                best = f
                best_params = p
        loss_curve.append(best)
        probs = probs_fn(best_params)
        probs = probs / probs.sum()
        samples = samp_rng.choice(len(probs), size=n_samples, p=probs)
        bestG_curve.append(float(G_of_state[samples].min()))
        if (gi + 1) % report == 0 or gi == generations - 1:
            print(f"    [{label}] gen {gi + 1}/{generations}  "
                  f"<G>={loss_curve[-1]:.4f}  bestG={bestG_curve[-1]:.4f}",
                  flush=True)
    return loss_curve, bestG_curve, best_params


def simulated_annealing(G_of_state, n, generations, population, seed,
                        T0=2.0, T1=0.05):
    """Simulated annealing / Metropolis: minimize G(s) over sequence space (classical baseline).
    Total evaluation steps = generations x population, returns (best-G curve, best sequence, best G)."""
    rng = np.random.RandomState(seed)
    n_states = 2 ** n
    s = int(rng.randint(0, n_states))
    Gs = float(G_of_state[s])
    best_s, best_G = s, Gs
    curve = []
    total = generations * population
    for step in range(total):
        T = T0 * (T1 / T0) ** (step / max(1, total - 1))   # geometric cooling
        s_new = s ^ (1 << int(rng.randint(0, n)))          # flip one bit
        G_new = float(G_of_state[s_new])
        dG = G_new - Gs
        if dG <= 0 or rng.random() < np.exp(-dG / T):
            s, Gs = s_new, G_new
        if Gs < best_G:
            best_G, best_s = Gs, s
        if (step + 1) % population == 0:
            curve.append(best_G)
    best_seq = "".join("H" if (best_s >> i) & 1 == 0 else "P" for i in range(n))
    return curve, best_seq, best_G


def distribution_entropy(probs):
    """Shannon entropy (bits) and effective sequence count N_eff = 2^H"""
    p = probs[probs > 1e-12]
    H = float(-np.sum(p * np.log2(p)))
    return H, float(2 ** H)


def sa_rerun_freq(G_of_state, n, generations, population, base_seed, n_rerun=500):
    """Re-run simulated annealing n_rerun times (different seeds) and count the frequency of each optimal sequence"""
    from collections import Counter
    counter = Counter()
    for k in range(n_rerun):
        _, best_seq, _ = simulated_annealing(
            G_of_state, n, generations, population, base_seed + k)
        counter[best_seq] += 1
    return counter


def run_compare(n, p, h, J, G_of_state, G_star, opt_mask, args, outdir):
    """Under the same seeds + equal evolutionary budget, compare QAOA / hardware-efficient ansatz / random search / simulated annealing"""
    print("\n" + "=" * 68)
    print("  Comparison mode: QAOA vs hardware-efficient ansatz vs random search vs simulated annealing")
    print("=" * 68)

    G0 = float(G_of_state.mean())
    n_samples = 2000

    # -- 1) QAOA --
    qaoa_circ = QAOACircuit(n, h, J, backend=args.backend)
    print(f"  [QAOA]         backend {qaoa_circ.backend}, parameters 2p = {2 * p}")

    def qaoa_probs(params):
        return qaoa_circ.probs(params[:p], params[p:], G_of_state)

    def qaoa_fitness(params):
        probs = qaoa_probs(params)
        return float(np.sum(probs * G_of_state))

    qaoa_bounds = [(0.0, 2 * np.pi)] * (2 * p)
    t0 = time.time()
    qaoa_opt, qaoa_loss, qaoa_bestG = train_evolutionary(
        qaoa_fitness, qaoa_probs, 2 * p, qaoa_bounds, args.population,
        args.generations, args.seed, G_of_state, n_samples, label="QAOA")
    qaoa_time = time.time() - t0

    # -- 2) hardware-efficient ansatz --
    hea_circ = HardwareEfficientCircuit(n, n_layers=args.hea_layers,
                                        backend=args.backend)
    print(f"  [HEA]         backend {hea_circ.backend}, parameters {hea_circ.n_params}")

    def hea_probs(params):
        return hea_circ.probs(params)

    def hea_fitness(params):
        probs = hea_probs(params)
        return float(np.sum(probs * G_of_state))

    hea_bounds = [(-np.pi, np.pi)] * hea_circ.n_params
    t0 = time.time()
    hea_opt, hea_loss, hea_bestG = train_evolutionary(
        hea_fitness, hea_probs, hea_circ.n_params, hea_bounds,
        args.population, args.generations, args.seed, G_of_state, n_samples,
        label="HEA")
    hea_time = time.time() - t0

    # -- 3) random search baseline (random sampling in QAOA parameter space) --
    t0 = time.time()
    rs_loss, rs_bestG, rs_best_params = random_search(
        qaoa_fitness, qaoa_probs, 2 * p, qaoa_bounds, args.population,
        args.generations, args.seed, G_of_state, n_samples, label="Random search")
    rs_time = time.time() - t0

    # -- 4) simulated annealing (classical baseline) --
    t0 = time.time()
    sa_bestG, sa_best_seq, sa_best = simulated_annealing(
        G_of_state, n, args.generations, args.population, args.seed)
    sa_time = time.time() - t0

    # -- summary --
    pq = qaoa_probs(qaoa_opt.best_individual.params)
    ph = hea_probs(hea_opt.best_individual.params)
    prs = qaoa_probs(rs_best_params)

    expect = [float(np.sum(pq * G_of_state)), float(np.sum(ph * G_of_state)),
              float(np.sum(prs * G_of_state))]          # <G> of quantum methods
    accs = [float(pq[opt_mask].sum()), float(ph[opt_mask].sum()),
            float(prs[opt_mask].sum())]                 # P(G*) of quantum methods
    bestG_final = [qaoa_bestG[-1], hea_bestG[-1], rs_bestG[-1], sa_best]
    times = [qaoa_time, hea_time, rs_time, sa_time]

    # distribution diversity: Shannon entropy / effective sequence count (QAOA 'distribution' vs SA 'point')
    ent_q = distribution_entropy(pq)
    ent_h = distribution_entropy(ph)
    ent_r = distribution_entropy(prs)
    ents = [ent_q, ent_h, ent_r, (0.0, 1.0)]   # SA is a delta distribution: H=0, N_eff=1

    labels = ["QAOA", "Hardware-efficient ansatz", "Random search", "Simulated annealing"]
    colors = ["#2c7bb6", "#fd8d3c", "#969696", "#1b9e77"]

    print("\n  " + "-" * 70)
    print(f"  {'Method':<16} {'final <G>':>10} {'best G(s)':>10} {'P(G*)':>9} {'time':>8}")
    print("  " + "-" * 70)
    for i, lab in enumerate(labels):
        exp = f"{expect[i]:.4f}" if i < 3 else "        -"
        acc = f"{accs[i] * 100:.2f}%" if i < 3 else "       -"
        print(f"  {lab:<16} {exp:>10} {bestG_final[i]:>10.4f} {acc:>9} {times[i]:>7.1f}s")
    print("  " + "-" * 70)
    print(f"  reference: uniform superposition <G>0 = {G0:.4f}   global optimum G* = {G_star:.4f}")
    print(f"  'best G(s)' = minimum over {n_samples} samples from the final distribution of the quantum method; SA = optimum found by annealing")

    top_state = int(np.argmax(pq))
    top_seq = "".join("H" if (top_state >> i) & 1 == 0 else "P" for i in range(n))
    print(f"\n  Most likely QAOA sequence: {top_seq}  (G={G_of_state[top_state]:.3f})")
    print(f"  Simulated annealing best sequence: {sa_best_seq}  (G={sa_best:.3f})")

    print("\n  Distribution diversity (Shannon entropy / effective sequence count N_eff):")
    print(f"    QAOA        H = {ent_q[0]:.2f} bit,  N_eff = {ent_q[1]:.1f}")
    print(f"    HEA            H = {ent_h[0]:.2f} bit,  N_eff = {ent_h[1]:.1f}")
    print(f"    Random search  H = {ent_r[0]:.2f} bit,  N_eff = {ent_r[1]:.1f}")
    print(f"    Simulated annealing  N_eff = 1  (single point / delta distribution)")

    # rerun SA several times -> sequence frequencies (for the right panel of the sequence spectrum)
    sa_counter = sa_rerun_freq(G_of_state, n, args.generations, args.population,
                               args.seed)

    # -- plotting --
    plot_comparison(outdir, [qaoa_bestG, hea_bestG, rs_bestG, sa_bestG],
                    bestG_final, labels, colors, G_star)
    plot_sequence_spectrum(outdir, pq, G_of_state, n, sa_counter, G_star)

    # -- saving --
    np.savez(
        os.path.join(outdir, "comparison.npz"),
        qaoa_loss=np.array(qaoa_loss), hea_loss=np.array(hea_loss),
        rs_loss=np.array(rs_loss),
        qaoa_bestG=np.array(qaoa_bestG), hea_bestG=np.array(hea_bestG),
        rs_bestG=np.array(rs_bestG), sa_bestG=np.array(sa_bestG),
        finals_bestG=np.array(bestG_final), expect=np.array(expect),
        accs=np.array(accs), times=np.array(times),
        G_star=G_star, G0=G0,
        entropy=np.array([e[0] for e in ents]),
        n_eff=np.array([e[1] for e in ents]),
        qaoa_best_params=qaoa_opt.best_individual.params,
        hea_best_params=hea_opt.best_individual.params,
        rs_best_params=rs_best_params, sa_best_seq=sa_best_seq,
        qaoa_final_probs=pq, hea_final_probs=ph, rs_final_probs=prs,
        G_of_state=G_of_state,
    )
    with open(os.path.join(outdir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Comparison: QAOA vs hardware-efficient ansatz vs random search vs simulated annealing\n")
        f.write("=" * 48 + "\n")
        f.write(f"lattice {args.rows}x{args.cols}, n={n}, p={p} layers, hea layers={args.hea_layers}\n")
        f.write(f"evolution budget: {args.generations} generations x {args.population} individuals; {n_samples} samples\n\n")
        f.write(f"{'Method':<16} {'final <G>':>10} {'best G(s)':>10} {'P(G*)':>9} {'time':>8}\n")
        for i, lab in enumerate(labels):
            exp = f"{expect[i]:.4f}" if i < 3 else "-"
            acc = f"{accs[i] * 100:.2f}%" if i < 3 else "-"
            f.write(f"{lab:<16} {exp:>10} {bestG_final[i]:>10.4f} {acc:>9} {times[i]:>7.1f}s\n")
        f.write(f"\nreference <G>0 = {G0:.4f}, G* = {G_star:.4f}\n")
        f.write(f"\nQAOA most likely sequence: {top_seq} (G={G_of_state[top_state]:.3f})\n")
        f.write(f"simulated annealing best sequence: {sa_best_seq} (G={sa_best:.3f})\n")
        f.write(f"\nDistribution diversity (Shannon entropy / N_eff):\n")
        f.write(f"  QAOA: H={ent_q[0]:.2f} bit, N_eff={ent_q[1]:.1f}\n")
        f.write(f"  HEA: H={ent_h[0]:.2f} bit, N_eff={ent_h[1]:.1f}\n")
        f.write(f"  Random search: H={ent_r[0]:.2f} bit, N_eff={ent_r[1]:.1f}\n")
        f.write(f"  Simulated annealing: N_eff=1 (single point)\n")
        total_reruns = sum(sa_counter.values())
        f.write(f"\nSA rerun sequence frequency Top-3 over {total_reruns} runs:\n")
        for seq, cnt in sa_counter.most_common(3):
            f.write(f"  {seq}: {cnt}/{total_reruns} = {cnt / total_reruns * 100:.1f}%\n")

    print(f"\n  Comparison results and figures saved to: {outdir}/")
    print(f"    - comparison.png                 best G(s) curve + bar comparison")
    print(f"    - sequence_spectrum.png          QAOA distribution spectrum vs SA single-point spectrum")
    print(f"    - comparison.npz / comparison_summary.txt\n")


# ══════════════════════════════════════════════════════════════
#  4. Main workflow
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="QAOA quantum generator for protein design")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--layers", type=int, default=6, help="QAOA depth p")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--mode", choices=["qaoa", "compare"], default="qaoa",
                    help="qaoa=run only QAOA; compare=compare QAOA / hardware-efficient ansatz / random search / simulated annealing under the same seed")
    ap.add_argument("--hea-layers", type=int, default=2,
                    help="number of layers for the hardware-efficient ansatz in comparison mode")
    ap.add_argument("--backend", choices=["auto", "default.qubit", "qiskit.aer"],
                    default="auto",
                    help="quantum backend: auto=prefer default.qubit (fast / no qiskit translation overhead);"
                         "qiskit.aer=force qiskit statevector")
    args = ap.parse_args()

    # force line buffering: under non-tty (VSCode "Run" button / redirect / pipe) print output is
    # not swallowed by block-buffering, so progress is shown immediately.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    np.random.seed(args.seed)      # fix global random seed for reproducibility

    N = args.rows * args.cols
    n = N                                   # HP model: 1 qubit / position
    p = args.layers
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 68)
    print("  QAOA quantum generator -- energy function encoded directly in the circuit")
    print("=" * 68)
    print(f"  lattice: {args.rows}x{args.cols}  (N={N}, qubits n={n})")
    print(f"  QAOA depth p = {p}  ->  trainable parameters 2p = {2 * p}  (gamma_1..gamma_p, beta_1..beta_p)")
    print(f"  alphabet: {ALPHABET}  (HP model, eps(H,H)={EPS[0,0]:.0f})")

    # -- target structure and Ising coefficients --
    coords = snake_coords(args.rows, args.cols)
    h, J, const, dC, C_target, C_bg = build_ising(coords, seed=args.seed)
    n_contacts = int(np.sum(C_target) / 2)
    print(f"  target structure contacts: {n_contacts}")
    print(f"  effective contact map DeltaC = C_target - <C_background> computed")
    print(f"  Ising: constant={const:.4f},  h nonzero terms={int(np.sum(np.abs(h) > 1e-12))}, "
          f"J nonzero terms={int(np.sum(np.abs(J) > 1e-12)) / 2:.0f}")

    # -- energy G(s) of all 2^n states (used both for phase and as objective / brute-force search) --
    n_states = 2 ** n
    bits = (np.arange(n_states)[:, None] >> np.arange(n)) & 1
    z_matrix = 1 - 2 * bits                     # +1(H) / -1(P)
    linear = z_matrix @ h
    quad = 0.5 * np.einsum("si,ij,sj->s", z_matrix, J, z_matrix)
    G_of_state = const + linear + quad          # G(s) of each state

    # verify Ising expansion == direct eps computation
    sample = np.random.RandomState(0).choice(n_states, size=256, replace=False)
    err = max(abs(G_of_state[s] - direct_G(z_matrix[s], dC)) for s in sample)
    print(f"  [self-check] Ising expansion vs direct eps computation max error = {err:.2e}  "
          f"{'[OK] consistent' if err < 1e-8 else '[X] inconsistent!'}")

    G_star = float(G_of_state.min())            # global optimum G*
    opt_mask = G_of_state <= G_star + 1e-9
    n_opt = int(opt_mask.sum())
    print(f"  global optimum G* = {G_star:.4f}   (there are {n_opt} optimal-sequence states)")
    print(f"  expected <G>_0 under uniform superposition = {float(G_of_state.mean()):.4f}  (starting point)")

    # -- circuit and objective function --
    circ = QAOACircuit(n, h, J, backend=args.backend)
    print(f"  quantum backend: {circ.backend}")

    # self-check: PennyLane circuit vs NumPy exact reference (only run when PennyLane is available)
    if circ._pl is not None:
        _rg = np.random.RandomState(7).uniform(0, 2 * np.pi, p)
        _rb = np.random.RandomState(8).uniform(0, 2 * np.pi, p)
        _p_pl = circ.probs(_rg, _rb, G_of_state)
        _p_np = _run_numpy(_rg, _rb, G_of_state, n)
        _d = float(np.max(np.abs(_p_pl - _p_np)))
        print(f"  [self-check] PennyLane vs NumPy reference max probability deviation = {_d:.2e}  "
              f"{'[OK] consistent' if _d < 1e-6 else '[X] inconsistent!'}")

    def split(params):
        g = params[:p]
        b = params[p:]
        return g, b

    def fitness_fn(params):
        g, b = split(params)
        probs = circ.probs(g, b, G_of_state)
        return float(np.sum(probs * G_of_state))   # <G> = sum P(s) G(s)

    def accuracy(probs):
        return float(probs[opt_mask].sum())        # fraction of probability mass on the global optimum

    param_bounds = [(0.0, 2 * np.pi)] * (2 * p)    # gamma, beta both in [0, 2*pi]

    if args.mode == "compare":
        run_compare(n, p, h, J, G_of_state, G_star, opt_mask, args, outdir)
        return

    opt = EvolutionaryOptimizer(
        n_params=2 * p, param_bounds=param_bounds, fitness_fn=fitness_fn,
        population_size=args.population, elite_size=max(2, args.population // 6),
        tournament_size=3, mutation_rate=0.15, crossover_rate=0.7,
        mutation_sigma=0.5, mutation_strategy="gaussian",
    )

    # -- training loop (record loss / accuracy per generation) --
    print("\n  Starting evolutionary training...\n")
    print(f"  {'Gen':>4} | {'<G> loss':>10} | {'accuracy P(G*)':>12} | "
          f"{'best sequence':>12} | {'seq G':>8}")
    print("  " + "-" * 66)

    opt.initialize_population(seed=args.seed, verbose=True)
    hist_loss, hist_acc, hist_bestG = [], [], []

    t0 = time.time()
    for gen in range(args.generations):
        opt.run_generation()
        best = opt.best_individual
        g, b = split(best.params)
        probs = circ.probs(g, b, G_of_state)
        loss = float(np.sum(probs * G_of_state))
        acc = accuracy(probs)

        # most likely sequence (argmax probability) and its G
        top_state = int(np.argmax(probs))
        top_seq = "".join("H" if (top_state >> i) & 1 == 0 else "P"
                          for i in range(n))
        top_G = float(G_of_state[top_state])

        hist_loss.append(loss)
        hist_acc.append(acc)
        hist_bestG.append(top_G)

        if gen % 10 == 0 or gen == args.generations - 1:
            print(f"  {gen + 1:4d} | {loss:10.4f} | {acc * 100:11.2f}% | "
                  f"{top_seq:>12} | {top_G:8.2f}")

    elapsed = time.time() - t0
    print(f"\n  Training complete, time {elapsed:.1f}s\n")

    # -- final results --
    g, b = split(opt.best_individual.params)
    probs = circ.probs(g, b, G_of_state)

    # aggregate: state -> sequence
    seq_of_state = ["".join("H" if (s >> i) & 1 == 0 else "P" for i in range(n))
                    for s in range(n_states)]
    seq_prob = {}
    seq_G = {}
    for s in range(n_states):
        sq = seq_of_state[s]
        seq_prob[sq] = seq_prob.get(sq, 0.0) + probs[s]
        seq_G[sq] = float(G_of_state[s])
    ranked = sorted(seq_G.items(), key=lambda kv: kv[1])   # sort by G ascending

    # sample 2000 sequences to see what sequences are actually obtained
    samples = np.random.choice(n_states, size=2000, p=probs)
    sampled_G = [float(G_of_state[s]) for s in samples]
    best_sampled_G = float(np.min(sampled_G))
    hit_rate = float(np.mean(np.array(sampled_G) <= G_star + 1e-9))

    print("=" * 68)
    print("  Final results")
    print("=" * 68)
    print(f"  final <G> (expected design score)  = {float(np.sum(probs * G_of_state)):.4f}"
          f"   (start {float(G_of_state.mean()):.4f}, global optimum {G_star:.4f})")
    print(f"  accuracy P(G*)             = {accuracy(probs) * 100:.2f}%")
    print(f"  fraction of samples hitting the global optimum    = {hit_rate * 100:.2f}%")
    print(f"  best sampled G(S)       = {best_sampled_G:.4f}  (global optimum {G_star:.4f})")
    print(f"\n  Global-optimum sequences (G*={G_star:.4f}):")
    for sq, gv in ranked:
        if gv > G_star + 1e-9:
            break
        print(f"    {sq}   P={seq_prob[sq] * 100:6.2f}%")
    print(f"\n  Top-10 low-energy sequences (sorted by G):")
    shown = 0
    for sq, gv in ranked:
        print(f"    {sq}   G={gv:.3f}   P={seq_prob[sq] * 100:6.2f}%")
        shown += 1
        if shown >= 10:
            break

    # -- save results npz --
    np.savez(
        os.path.join(outdir, "results.npz"),
        best_params=opt.best_individual.params,
        gammas=g, betas=b,
        best_fitness=float(np.sum(probs * G_of_state)),
        accuracy=accuracy(probs),
        G_star=G_star,
        G_of_state=G_of_state,
        h=h, J=J, const=const, dC=dC, C_target=C_target, C_bg=C_bg,
        coords=coords,
        hist_loss=np.array(hist_loss),
        hist_acc=np.array(hist_acc),
        hist_bestG=np.array(hist_bestG),
        final_probs=probs,
    )

    # -- plotting --
    plot_training(outdir, hist_loss, hist_acc, hist_bestG, G_star)
    plot_distribution(outdir, ranked, seq_prob, seq_G, G_star, n=10)
    plot_structure(outdir, coords, args.rows, args.cols, ranked, seq_prob, C_target)

    # -- text summary --
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("QAOA quantum generator -- result summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"lattice {args.rows}x{args.cols}, N={N}, n={n} qubits, p={p} layers\n")
        f.write(f"backend: {circ.backend}\n")
        f.write(f"global optimum G* = {G_star:.4f}\n")
        f.write(f"final <G> = {float(np.sum(probs * G_of_state)):.4f}\n")
        f.write(f"accuracy P(G*) = {accuracy(probs) * 100:.2f}%\n")
        f.write(f"sample hit rate = {hit_rate * 100:.2f}%\n\n")
        f.write("Optimized parameters (gamma, beta):\n")
        f.write(f"  gamma = {np.round(g, 4).tolist()}\n")
        f.write(f"  beta = {np.round(b, 4).tolist()}\n\n")
        f.write("Top sequences (G | probability):\n")
        for sq, gv in ranked[:10]:
            f.write(f"  {sq}  G={gv:.3f}  P={seq_prob[sq] * 100:.2f}%\n")

    print(f"\n  Results and figures saved to: {outdir}/")
    print(f"    - training_curves.png    training curves (loss/accuracy)")
    print(f"    - probability_distribution.png  final sequence distribution")
    print(f"    - target_structure.png   target structure + best sequence")
    print(f"    - results.npz / summary.txt\n")


# ══════════════════════════════════════════════════════════════
#  5. Plotting
# ══════════════════════════════════════════════════════════════
def plot_training(outdir, loss, acc, bestG, G_star):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    gen = np.arange(1, len(loss) + 1)

    axes[0].plot(gen, loss, color="#2c7bb6", lw=2, label="<G> loss")
    axes[0].axhline(G_star, color="#d7191c", ls="--", lw=1.2,
                    label=f"Global optimum G*={G_star:.2f}")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("<G> = sum P(s) G(s)   (lower is better)")
    axes[0].set_title("QAOA loss curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(gen, np.array(acc) * 100, color="#2ca02c", lw=2,
                 label="P(G*) accuracy")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("P(G*) (%)")
    axes[1].set_title("Accuracy: probability mass on the global optimum")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "training_curves.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distribution(outdir, ranked, seq_prob, seq_G, G_star, n=10):
    seqs = [sq for sq, _ in ranked[:n]]
    gs = [seq_G[sq] for sq in seqs]
    ps = [seq_prob[sq] * 100 for sq in seqs]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d7191c" if g <= G_star + 1e-9 else "#2c7bb6" for g in gs]
    y = np.arange(len(seqs))[::-1]
    ax.barh(y, ps, color=colors, edgecolor="#333", linewidth=0.8)
    for yi, (sq, g, p) in enumerate(zip(seqs, gs, ps)):
        ax.text(p + 0.5, y[yi], f"  {sq}  G={g:.2f}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Probability P(sequence) (%)")
    ax.set_title(f"Final sequence distribution (red = global optimum G*={G_star:.2f})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "probability_distribution.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_structure(outdir, coords, rows, cols, ranked, seq_prob, C_target):
    best_seq, best_G = ranked[0]
    inv = {tuple(c): i for i, c in enumerate(coords)}

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # draw the lattice grid
    for r in range(rows):
        for c in range(cols):
            i = inv.get((r, c), -1)
            if i < 0:
                continue
            aa = best_seq[i]
            fc = "#2c7bb6" if aa == "H" else "#f7f7f7"
            ax.add_patch(plt.Rectangle((c - 0.5, rows - 1 - r - 0.5), 1, 1,
                                       facecolor=fc, edgecolor="#555", lw=1.2))
            ax.text(c, rows - 1 - r, f"{i}\n{aa}", ha="center", va="center",
                    fontsize=11,
                    color="white" if aa == "H" else "#333",
                    fontweight="bold")

    # draw contact edges
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if C_target[i, j] > 0.5:
                r1, c1 = coords[i]
                r2, c2 = coords[j]
                x1, y1 = c1, rows - 1 - r1
                x2, y2 = c2, rows - 1 - r2
                ax.plot([x1, x2], [y1, y2], ls="--", color="#d7191c", lw=1.5, zorder=5)

    ax.set_xlim(-1, cols)
    ax.set_ylim(-1, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Target structure + best sequence {best_seq}\nG={best_G:.2f} "
                 f"(blue=H hydrophobic core, white=P polar, red dashed=contacts)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "target_structure.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison(outdir, bestG_curves, finals_bestG, labels, colors, G_star):
    """Comparison plot: left = curve of best G(s) found, right = bar comparison of final best G(s) (including simulated annealing)"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gen = np.arange(1, len(bestG_curves[0]) + 1)

    for c, lab, col in zip(bestG_curves, labels, colors):
        axes[0].plot(gen, c, lw=2, color=col, label=lab)
    axes[0].axhline(G_star, color="black", ls="--", lw=1.2, label=f"G* = {G_star:.2f}")
    axes[0].set_xlabel("Evaluation budget (unit = population evals)")
    axes[0].set_ylabel("Best G(s) found   (lower is better)")
    axes[0].set_title("Best design score vs budget (same seed)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    x = np.arange(len(finals_bestG))
    axes[1].bar(x, finals_bestG, color=colors, edgecolor="#333", linewidth=0.8)
    axes[1].axhline(G_star, color="black", ls="--", lw=1.2, label=f"G* = {G_star:.2f}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Final best G(s)")
    axes[1].set_title("Final best G(s) comparison")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    for xi, fv in zip(x, finals_bestG):
        y = fv - 0.05 if fv < 0 else fv + 0.05
        axes[1].text(xi, y, f"{fv:.2f}", ha="center",
                     va="top" if fv < 0 else "bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sequence_spectrum(outdir, qaoa_probs, G_of_state, n, sa_counter, G_star,
                           n_top=12):
    """Sequence spectrum comparison: left = top sequences of the QAOA distribution, right = sequence frequencies from SA reruns (distribution vs point)"""
    n_states = 2 ** n
    seq_of_state = ["".join("H" if (s >> i) & 1 == 0 else "P" for i in range(n))
                    for s in range(n_states)]

    # QAOA top sequences (descending by probability)
    order = np.argsort(qaoa_probs)[::-1][:n_top]
    q_seqs = [seq_of_state[s] for s in order]
    q_probs = [qaoa_probs[s] * 100 for s in order]

    # SA sequence frequencies (descending by count)
    total_reruns = sum(sa_counter.values())
    sa_items = sa_counter.most_common(n_top)
    s_seqs = [s for s, _ in sa_items]
    s_freq = [c / total_reruns * 100 for _, c in sa_items]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    y = np.arange(len(q_seqs))[::-1]
    axes[0].barh(y, q_probs, color="#2c7bb6", edgecolor="#333", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(q_seqs, fontsize=10)
    axes[0].set_xlabel("Probability P(sequence) (%)")
    axes[0].set_title("QAOA: a distribution over many sequences")
    axes[0].grid(axis="x", alpha=0.3)

    y2 = np.arange(len(s_seqs))[::-1]
    axes[1].barh(y2, s_freq, color="#1b9e77", edgecolor="#333", linewidth=0.8)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(s_seqs, fontsize=10)
    axes[1].set_xlabel(f"Frequency over {total_reruns} reruns (%)")
    axes[1].set_title("Simulated annealing: a single point (delta)")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_xlim(0, 105)

    fig.suptitle(f"Sequence spectrum: distribution vs point (G* = {G_star:.2f})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "sequence_spectrum.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)



# ══════════════════════════════════════════════════════════════
#  6. JCIM clean benchmark workflow
# ══════════════════════════════════════════════════════════════

def _setup_logger(outdir):
    """Write to both terminal and run.log; each experiment uses an independent directory."""
    logger = logging.getLogger("jcim_qaoa")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(outdir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_run_dir(base, args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{args.rows}x{args.cols}_p{args.layers}_"
            f"prob{args.problem_seed}_opt{args.seed}_{stamp}")
    outdir = os.path.join(base, name)
    os.makedirs(outdir, exist_ok=False)
    return outdir


def _sequence_from_state(s, n):
    # project convention: bit=0 -> H, bit=1 -> P.
    return "".join("H" if ((s >> i) & 1) == 0 else "P" for i in range(n))


def _exact_metrics(probs, G_of_state, opt_mask):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    expG = float(np.dot(probs, G_of_state))
    p_opt = float(probs[opt_mask].sum())
    H, Neff = distribution_entropy(probs)
    top_state = int(np.argmax(probs))
    return {
        "expG": expG,
        "p_opt": p_opt,
        "entropy_bits": H,
        "N_eff": Neff,
        "top_state": top_state,
        "top_sequence": _sequence_from_state(top_state, int(np.log2(len(probs)))),
        "top_G": float(G_of_state[top_state]),
    }


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _evaluate_training_run(label, optimizer, probs_fn, G_of_state, opt_mask,
                           generations, logger, csv_path):
    """Training + exact per-generation recording; no longer randomly sample 2000 times per generation."""
    history = []
    fieldnames = ["generation", "best_fitness", "expG", "p_opt",
                  "entropy_bits", "N_eff", "top_sequence", "top_G",
                  "avg_population_fitness", "std_population_fitness",
                  "mutation_rate"]

    def record(gen):
        probs = probs_fn(optimizer.best_individual.params)
        m = _exact_metrics(probs, G_of_state, opt_mask)
        hist = optimizer.history[-1] if optimizer.history else {}
        row = {
            "generation": gen,
            "best_fitness": float(optimizer.best_individual.fitness),
            "expG": m["expG"], "p_opt": m["p_opt"],
            "entropy_bits": m["entropy_bits"], "N_eff": m["N_eff"],
            "top_sequence": m["top_sequence"], "top_G": m["top_G"],
            "avg_population_fitness": hist.get("avg_fitness", np.nan),
            "std_population_fitness": hist.get("std_fitness", np.nan),
            "mutation_rate": hist.get("mutation_rate", optimizer.mutation_rate),
        }
        history.append(row)
        return m, row

    m0, row0 = record(0)
    logger.info("[%s] gen 0/%d | <G>=%.6f | P(G*)=%.4f | N_eff=%.2f | top=%s",
                 label, generations, m0["expG"], m0["p_opt"], m0["N_eff"], m0["top_sequence"])

    report = max(1, generations // 20)
    for gen in range(1, generations + 1):
        optimizer.run_generation()
        m, _ = record(gen)
        if gen % report == 0 or gen == generations:
            logger.info("[%s] gen %d/%d | <G>=%.6f | P(G*)=%.4f | N_eff=%.2f | top=%s",
                        label, gen, generations, m["expG"], m["p_opt"],
                        m["N_eff"], m["top_sequence"])

    _write_csv(csv_path, history, fieldnames)
    return history


def _run_qaoa_clean(n, p, h, J, G_of_state, opt_mask, args, logger, outdir, label="QAOA"):
    circ = QAOACircuit(n, h, J, backend=args.backend)
    logger.info("[%s] backend=%s | parameters=%d | p=%d", label, circ.backend, 2*p, p)

    def probs_fn(params):
        return circ.probs(params[:p], params[p:], G_of_state)
    def fitness_fn(params):
        return float(np.dot(probs_fn(params), G_of_state))

    bounds = [(0.0, 2*np.pi)] * (2*p)
    opt = EvolutionaryOptimizer(
        n_params=2*p, param_bounds=bounds, fitness_fn=fitness_fn,
        population_size=args.population,
        elite_size=max(2, args.population//6), tournament_size=3,
        mutation_rate=0.15, crossover_rate=0.7,
        mutation_sigma=0.5, mutation_strategy="gaussian")
    opt.initialize_population(seed=args.seed, verbose=False)
    t0 = time.time()
    hist = _evaluate_training_run(label, opt, probs_fn, G_of_state, opt_mask,
                                  args.generations, logger,
                                  os.path.join(outdir, f"{label.lower()}_generation_metrics.csv"))
    elapsed = time.time() - t0
    probs = probs_fn(opt.best_individual.params)
    return {
        "label": label, "optimizer": opt, "circuit": circ, "probs": probs,
        "history": hist, "time_sec": elapsed, "n_params": 2*p,
        "params": opt.best_individual.params.copy()
    }


def _run_hea_clean(n, L, G_of_state, opt_mask, args, logger, outdir):
    circ = HardwareEfficientCircuit(n, n_layers=L, backend=args.backend)
    logger.info("[HEA] backend=%s | parameters=%d | layers=%d", circ.backend, circ.n_params, L)

    def probs_fn(params): return circ.probs(params)
    def fitness_fn(params): return float(np.dot(probs_fn(params), G_of_state))
    bounds = [(-np.pi, np.pi)] * circ.n_params
    opt = EvolutionaryOptimizer(
        n_params=circ.n_params, param_bounds=bounds, fitness_fn=fitness_fn,
        population_size=args.population,
        elite_size=max(2, args.population//6), tournament_size=3,
        mutation_rate=0.15, crossover_rate=0.7,
        mutation_sigma=0.5, mutation_strategy="gaussian")
    opt.initialize_population(seed=args.seed, verbose=False)
    t0 = time.time()
    hist = _evaluate_training_run("HEA", opt, probs_fn, G_of_state, opt_mask,
                                  args.generations, logger, os.path.join(outdir, "hea_generation_metrics.csv"))
    elapsed = time.time() - t0
    probs = probs_fn(opt.best_individual.params)
    return {"label":"HEA", "optimizer":opt, "circuit":circ, "probs":probs,
            "history":hist, "time_sec":elapsed, "n_params":circ.n_params,
            "params":opt.best_individual.params.copy()}


def _run_random_clean(n, p, h, J, G_of_state, opt_mask, args, logger, outdir):
    # Use exactly the same parameter space, population x generations evaluation budget as QAOA.
    qcirc = QAOACircuit(n, h, J, backend=args.backend)
    def probs_fn(params): return qcirc.probs(params[:p], params[p:], G_of_state)
    def fitness_fn(params): return float(np.dot(probs_fn(params), G_of_state))
    lows = np.zeros(2*p); highs = np.full(2*p, 2*np.pi)
    rng = np.random.RandomState(args.seed)
    best = None; best_f = float("inf")
    rows=[]; eval_count=0; t0=time.time(); report=max(1,args.generations//20)
    for gen in range(args.generations+1):
        if gen == 0:
            # initial random parameters, as the first batch of budget=population evaluations.
            pass
        else:
            pass
        # to keep the same total number of evaluations as evolutionary methods: evaluate population new parameters per generation.
        batch = rng.uniform(lows, highs, size=(args.population, 2*p))
        for par in batch:
            f=fitness_fn(par); eval_count += 1
            if f < best_f: best_f=f; best=par.copy()
        probs=probs_fn(best); m=_exact_metrics(probs,G_of_state,opt_mask)
        rows.append({"generation":gen,"evaluations":eval_count,"best_fitness":best_f,
                     "expG":m["expG"],"p_opt":m["p_opt"],"entropy_bits":m["entropy_bits"],
                     "N_eff":m["N_eff"],"top_sequence":m["top_sequence"],"top_G":m["top_G"]})
        if gen % report==0 or gen==args.generations:
            logger.info("[Random search] batch %d/%d | eval=%d | <G>=%.6f | P(G*)=%.4f | N_eff=%.2f",
                        gen,args.generations,eval_count,m["expG"],m["p_opt"],m["N_eff"])
    _write_csv(os.path.join(outdir,"random_generation_metrics.csv"),rows,list(rows[0].keys()))
    return {"label":"Random search","probs":probs,"history":rows,"time_sec":time.time()-t0,
            "n_params":2*p,"params":best}


def _run_sa_clean(n, G_of_state, G_star, args, logger, outdir):
    # each SA run has total steps = population x generations, comparable to the parameter evaluations of quantum optimization.
    t0=time.time()
    curve, best_seq, best_G=simulated_annealing(G_of_state,n,args.generations,args.population,args.seed)
    elapsed=time.time()-t0
    rows=[]
    for i,g in enumerate(curve,1):
        rows.append({"generation":i,"evaluations":i*args.population,"bestG":g})
    _write_csv(os.path.join(outdir,"sa_generation_metrics.csv"),rows,["generation","evaluations","bestG"])
    logger.info("[SA] steps=%d | bestG=%.6f | best_sequence=%s | time=%.2fs",
                args.generations*args.population,best_G,best_seq,elapsed)
    # independent reruns are used for the classical endpoint distribution; not part of the single optimization curve.
    counter=sa_rerun_freq(G_of_state,n,args.generations,args.population,args.seed,args.sa_reruns)
    total=sum(counter.values())
    freq_rows=[{"sequence":s,"count":c,"frequency":c/total} for s,c in counter.most_common()]
    _write_csv(os.path.join(outdir,"sa_rerun_sequence_frequency.csv"),freq_rows,["sequence","count","frequency"])
    return {"label":"SA","bestG":best_G,"best_seq":best_seq,"curve":curve,
            "time_sec":elapsed,"n_params":0,"counter":counter}


def _save_final_distributions(outdir, results, G_of_state, G_star, opt_mask, args):
    n_qubits=int(np.log2(len(results["QAOA"]["probs"]))); n_states=2**n_qubits
    states=np.arange(n_states)
    rows=[]
    for s in states:
        seq=_sequence_from_state(int(s),n_qubits)
        g=float(G_of_state[s]); isopt=bool(opt_mask[s])
        rows.append({"state":int(s),"sequence":seq,"G":g,"is_global_optimum":int(isopt),
                     "P_QAOA":float(results["QAOA"]["probs"][s]),
                     "P_HEA":float(results["HEA"]["probs"][s]),
                     "P_Random":float(results["Random search"]["probs"][s])})
    fields=list(rows[0].keys())
    _write_csv(os.path.join(outdir,"final_state_distribution.csv"),rows,fields)

    summary=[]
    for name in ["QAOA","HEA","Random search"]:
        m=_exact_metrics(results[name]["probs"],G_of_state,opt_mask)
        summary.append({"method":name,"parameters":results[name]["n_params"],
                        "expG":m["expG"],"P_Gstar":m["p_opt"],
                        "entropy_bits":m["entropy_bits"],"N_eff":m["N_eff"],
                        "top_sequence":m["top_sequence"],"top_G":m["top_G"],
                        "best_sampled_G":np.nan,"sample_hit_rate":np.nan,
                        "expected_hits":np.nan,"samples_for_95pct_hit":np.nan,
                        "time_sec":results[name]["time_sec"]})
    sa=results["SA"]
    rng=np.random.RandomState(args.seed ^ 0xA5A5A5A5)
    sample_count=args.sample_count
    q=results["QAOA"]["probs"]
    h=results["HEA"]["probs"]
    r=results["Random search"]["probs"]
    for name,probs in [("QAOA",q),("HEA",h),("Random search",r)]:
        ss=rng.choice(n_states,size=sample_count,p=probs)
        vals=G_of_state[ss]
        hit=float(np.mean(vals<=G_star+1e-9))
        for row in summary:
            if row["method"]==name:
                row["best_sampled_G"]=float(vals.min()); row["sample_hit_rate"]=hit
                pstar=max(float(row["P_Gstar"]), 0.0)
                row["expected_hits"]=sample_count*pstar
                row["samples_for_95pct_hit"]= (np.log(0.05)/np.log(1-pstar) if 0 < pstar < 1 else (1.0 if pstar >= 1 else np.inf))
    summary.append({"method":"SA","parameters":0,"expG":np.nan,"P_Gstar":np.nan,
                    "entropy_bits":np.nan,"N_eff":np.nan,"top_sequence":sa["best_seq"],
                    "top_G":sa["bestG"],"best_sampled_G":np.nan,"sample_hit_rate":np.nan,
                    "expected_hits":np.nan,"samples_for_95pct_hit":np.nan,
                    "time_sec":sa["time_sec"]})
    _write_csv(os.path.join(outdir,"final_metrics.csv"),summary,list(summary[0].keys()))
    return summary


def _run_clean(args, base, outdir=None):
    """Run one JCIM clean benchmark; args.seed is the optimizer seed."""
    os.makedirs(base, exist_ok=True)
    if outdir is None:
        outdir=_make_run_dir(base,args)
    else:
        os.makedirs(outdir, exist_ok=False)
    logger=_setup_logger(outdir)
    # EvolutionaryOptimizer initialization uses a local RNG, but its crossover/mutation still uses
    # the NumPy global RNG; here we explicitly reset it at the start of each optimizer seed, to ensure
    # seed=46 in a batch produces exactly the same random stream as running --seed 46 alone.
    np.random.seed(args.seed)
    config=vars(args).copy(); config["created_at"]=datetime.now().isoformat()
    config["objective"]="structure-conditioned HP sequence design / inverse folding"
    with open(os.path.join(outdir,"config.json"),"w",encoding="utf-8") as f:
        json.dump(config,f,indent=2,ensure_ascii=False)
    logger.info("JCIM clean benchmark started")
    logger.info("target=%dx%d | n=%d | p=%d | HEA layers=%d | generations=%d | population=%d",
                args.rows,args.cols,args.rows*args.cols,args.layers,args.hea_layers,args.generations,args.population)
    logger.info("problem_seed=%d (fixed) | optimizer_seed=%d | backend=%s",
                args.problem_seed,args.seed,args.backend)

    coords=snake_coords(args.rows,args.cols)
    h,J,const,dC,C_target,C_bg=build_ising(coords,seed=args.problem_seed)
    n=args.rows*args.cols; n_states=2**n
    bits=(np.arange(n_states)[:,None]>>np.arange(n))&1; z=1-2*bits
    G_of_state=const+z@h+0.5*np.einsum("si,ij,sj->s",z,J,z)
    G_star=float(G_of_state.min()); opt_mask=G_of_state<=G_star+1e-9
    err=max(abs(G_of_state[s]-direct_G(z[s],dC)) for s in range(min(n_states,256)))
    logger.info("problem check: G*=%.6f | n_opt=%d | <G>uniform=%.6f | Ising error=%.2e",
                G_star,int(opt_mask.sum()),float(G_of_state.mean()),err)
    logger.info("contacts=%d | nonzero h=%d | nonzero J=%d",
                int(C_target.sum()/2),int(np.sum(np.abs(h)>1e-12)),int(np.sum(np.abs(J)>1e-12)/2))
    np.savez(os.path.join(outdir,"problem.npz"),coords=coords,h=h,J=J,const=const,dC=dC,
             C_target=C_target,C_bg=C_bg,G_of_state=G_of_state,G_star=G_star,opt_mask=opt_mask)

    with open(os.path.join(outdir,"problem_summary.txt"),"w",encoding="utf-8") as f:
        f.write("JCIM clean benchmark problem\n============================\n")
        f.write(f"target lattice: {args.rows}x{args.cols}\nn_qubits: {n}\n")
        f.write(f"problem_seed: {args.problem_seed}\noptimizer_seed: {args.seed}\nG*: {G_star:.8f}\n")
        f.write(f"number of global-optimal sequences: {int(opt_mask.sum())}\n")
        f.write(f"uniform <G>: {float(G_of_state.mean()):.8f}\n")
        f.write(f"target contacts: {int(C_target.sum()/2)}\n")

    results={}
    results["QAOA"]=_run_qaoa_clean(n,args.layers,h,J,G_of_state,opt_mask,args,logger,outdir)
    results["HEA"]=_run_hea_clean(n,args.hea_layers,G_of_state,opt_mask,args,logger,outdir)
    results["Random search"]=_run_random_clean(n,args.layers,h,J,G_of_state,opt_mask,args,logger,outdir)
    results["SA"]=_run_sa_clean(n,G_of_state,G_star,args,logger,outdir)
    summary=_save_final_distributions(outdir,results,G_of_state,G_star,opt_mask,args)

    with open(os.path.join(outdir,"summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    logger.info("FINAL")
    for row in summary:
        if row["method"]=="SA":
            logger.info("%s | bestG=%.6f | seq=%s | time=%.2fs",
                        row["method"],row["top_G"],row["top_sequence"],row["time_sec"])
        else:
            logger.info("%s | params=%d | <G>=%.6f | P(G*)=%.4f | N_eff=%.2f | top=%s | sample_hit=%.4f",
                        row["method"],row["parameters"],row["expG"],row["P_Gstar"],
                        row["N_eff"],row["top_sequence"],row["sample_hit_rate"])
    logger.info("saved run directory: %s",outdir)
    print("\nRESULT_DIR="+outdir)
    return outdir, summary


def _make_batch_logger(batch_dir):
    """Batch-level logger: records the start, completion, failure, and elapsed time of each seed."""
    logger = logging.getLogger("jcim_qaoa_batch")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(batch_dir, "batch.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _write_batch_summary(batch_dir, rows):
    """Save batch_summary.csv/json; each row corresponds to one optimizer seed."""
    csv_path = os.path.join(batch_dir, "batch_summary.csv")
    fields = [
        "seed", "status", "run_dir", "error", "G_star", "n_opt",
        "QAOA_expG", "QAOA_P_Gstar", "QAOA_N_eff", "QAOA_top_sequence",
        "HEA_expG", "HEA_P_Gstar", "HEA_N_eff", "HEA_top_sequence",
        "Random_expG", "Random_P_Gstar", "Random_N_eff", "Random_top_sequence",
        "SA_bestG", "SA_best_sequence"
    ]
    _write_csv(csv_path, rows, fields)
    with open(os.path.join(batch_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main_clean():
    ap=argparse.ArgumentParser(
        description="JCIM clean benchmark: single seed or a sequential optimizer-seed batch")
    ap.add_argument("--rows",type=int,default=3)
    ap.add_argument("--cols",type=int,default=3)
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--hea-layers",type=int,default=2)
    ap.add_argument("--generations",type=int,default=200)
    ap.add_argument("--population",type=int,default=40)
    ap.add_argument("--seed",type=int,default=46,
                    help="single optimizer seed; ignored when --seed-start is used")
    ap.add_argument("--seed-start",type=int,default=None,
                    help="batch mode: first optimizer seed (inclusive)")
    ap.add_argument("--seed-end",type=int,default=None,
                    help="batch mode: last optimizer seed (inclusive)")
    ap.add_argument("--problem-seed",type=int,default=12345,
                    help="fixed seed for target/background Hamiltonian")
    ap.add_argument("--backend",choices=["auto","default.qubit","qiskit.aer"],default="auto")
    ap.add_argument("--sample-count",type=int,default=2000,
                    help="final-only Monte Carlo sample count")
    ap.add_argument("--sa-reruns",type=int,default=200,
                    help="independent SA reruns for classical endpoint distribution")
    ap.add_argument("--out-root",default="outputs")
    args=ap.parse_args()
    try: sys.stdout.reconfigure(line_buffering=True)
    except Exception: pass

    if (args.seed_start is None) != (args.seed_end is None):
        ap.error("--seed-start and --seed-end must be supplied together")
    if args.seed_start is not None and args.seed_end < args.seed_start:
        ap.error("--seed-end must be >= --seed-start")

    base=os.path.join(os.path.dirname(os.path.abspath(__file__)),args.out_root)
    os.makedirs(base,exist_ok=True)

    # single seed: keep original behavior; batch: fix problem_seed, only change optimizer seed.
    if args.seed_start is None:
        _run_clean(args, base)
        return

    seeds=list(range(args.seed_start,args.seed_end+1))
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir=os.path.join(
        base, f"{args.rows}x{args.cols}_p{args.layers}_prob{args.problem_seed}_batch_{stamp}")
    os.makedirs(batch_dir, exist_ok=False)
    logger=_make_batch_logger(batch_dir)

    # save batch config; seed is a range, do not treat any single seed as part of the problem definition.
    batch_config=vars(args).copy()
    batch_config.update({
        "created_at":datetime.now().isoformat(),
        "mode":"batch",
        "optimizer_seeds":seeds,
        "problem_seed_fixed":args.problem_seed,
        "objective":"structure-conditioned HP sequence design / inverse folding",
    })
    with open(os.path.join(batch_dir,"batch_config.json"),"w",encoding="utf-8") as f:
        json.dump(batch_config,f,indent=2,ensure_ascii=False)

    logger.info("BATCH START | %d optimizer seeds: %s",len(seeds),seeds)
    logger.info("FIXED problem_seed=%d | target=%dx%d | p=%d | HEA layers=%d | generations=%d | population=%d",
                args.problem_seed,args.rows,args.cols,args.layers,args.hea_layers,
                args.generations,args.population)

    rows=[]
    for idx,seed in enumerate(seeds,1):
        run_args=argparse.Namespace(**vars(args))
        run_args.seed=seed
        seed_dir=os.path.join(batch_dir,f"seed_{seed}")
        t0=time.time()
        logger.info("[%d/%d] START optimizer seed=%d",idx,len(seeds),seed)
        try:
            run_dir,summary=_run_clean(run_args,batch_dir,outdir=seed_dir)
            row={"seed":seed,"status":"ok","run_dir":run_dir,"error":"",
                 "G_star":np.nan,"n_opt":np.nan}
            by_method={r["method"]:r for r in summary}
            if by_method:
                # G* is the same for the same fixed problem; reading it from run problem.npz is most reliable.
                with np.load(os.path.join(run_dir,"problem.npz")) as dat:
                    row["G_star"]=float(dat["G_star"])
                    row["n_opt"]=int(dat["opt_mask"].sum())
            for prefix,name in [("QAOA","QAOA"),("HEA","HEA"),("Random","Random search")]:
                r=by_method[name]
                row[f"{prefix}_expG"]=r["expG"]
                row[f"{prefix}_P_Gstar"]=r["P_Gstar"]
                row[f"{prefix}_N_eff"]=r["N_eff"]
                row[f"{prefix}_top_sequence"]=r["top_sequence"]
            row["SA_bestG"]=by_method["SA"]["top_G"]
            row["SA_best_sequence"]=by_method["SA"]["top_sequence"]
            rows.append(row)
            logger.info("[%d/%d] DONE seed=%d | elapsed=%.2fs | run_dir=%s",
                        idx,len(seeds),seed,time.time()-t0,run_dir)
        except Exception as exc:
            row={"seed":seed,"status":"failed","run_dir":"","error":repr(exc),
                 "G_star":np.nan,"n_opt":np.nan,
                 "QAOA_expG":np.nan,"QAOA_P_Gstar":np.nan,"QAOA_N_eff":np.nan,"QAOA_top_sequence":"",
                 "HEA_expG":np.nan,"HEA_P_Gstar":np.nan,"HEA_N_eff":np.nan,"HEA_top_sequence":"",
                 "Random_expG":np.nan,"Random_P_Gstar":np.nan,"Random_N_eff":np.nan,"Random_top_sequence":"",
                 "SA_bestG":np.nan,"SA_best_sequence":""}
            rows.append(row)
            logger.exception("[%d/%d] FAILED seed=%d after %.2fs; continuing to next seed",
                             idx,len(seeds),seed,time.time()-t0)
        finally:
            _write_batch_summary(batch_dir,rows)

    ok=sum(r["status"]=="ok" for r in rows)
    failed=len(rows)-ok
    logger.info("BATCH END | completed=%d/%d | failed=%d | summary=%s",
                ok,len(seeds),failed,os.path.join(batch_dir,"batch_summary.csv"))
    print("\nBATCH_RESULT_DIR="+batch_dir)
    print("BATCH_SUMMARY="+os.path.join(batch_dir,"batch_summary.csv"))

if __name__ == "__main__":
    main_clean()
