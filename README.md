# SCHPSD-QAOA — Codebase Guide

**Structure-conditioned HP sequence design with QAOA**

A self-contained research codebase for **structure-conditioned protein sequence
design** (inverse folding) using the **Quantum Approximate Optimization
Algorithm (QAOA)** as a quantum proposal distribution, benchmarked against a
hardware-efficient ansatz (HEA), random search, and simulated annealing (SA).

Everything in this repository is written in English (code comments, docstrings,
console output, report files).

**Repository:** https://github.com/4597180/Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA

---

## Table of Contents

1. [What the project does](#1-what-the-project-does)
2. [Scientific background](#2-scientific-background)
3. [File-by-file overview](#3-file-by-file-overview)
4. [Installation](#4-installation)
5. [How to run each script](#5-how-to-run-each-script)
6. [Output directory structure](#6-output-directory-structure)
7. [Metrics reference](#7-metrics-reference)
8. [Using this repository with Git](#8-using-this-repository-with-git)
9. [Reproducibility notes](#9-reproducibility-notes)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What the project does

The pipeline is:

```
Target lattice structure (rows x cols)
        |
        v
Random self-avoiding background walks  --->  mean background contact map C_bg
        |
        v
Effective contact map  DeltaC = C_target - C_bg
        |
        v
Ising expansion  G(S) = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
        |
        v
QAOA circuit (cost layer e^{-i gamma H_C}  +  mixer layer e^{-i beta H_B})
        |
        v
Exact statevector probabilities over all 2^n sequences
        |
        v
(mu + lambda)-ES evolutionary optimizer  ->  minimize <G>
        |
        v
Metrics: <G>, P(G*), Shannon entropy H, effective sequence count N_eff
```

The central scientific claim being tested: **QAOA produces a probability
distribution over sequences rather than a single point**, and the depth `p` of
the QAOA ansatz controls how well that distribution concentrates on the
globally optimal sequence(s).

---

## 2. Scientific background

### The HP model

The protein is modelled as a chain of `H` (hydrophobic) and `P` (polar)
residues placed on an `rows x cols` square lattice in a snake (compact)
conformation. The contact energy is given by a 2x2 interaction matrix `EPS`:

| Pair | Energy | Meaning |
|------|--------|---------|
| H-H | -1 | Hydrophobic contact is favourable |
| H-P | 0 | Neutral |
| P-P | 0 | Neutral |

### Background / decoy structures

To avoid over-fitting a single arbitrary target, the code generates `n_background`
(random default 30) random **self-avoiding walks** on the same lattice and
averages their contact maps. The quantity that actually drives the optimization
is the *effective* contact map:

```
DeltaC = C_target - mean(C_background)
```

This is what makes the task "structure-conditioned" rather than simply
"fold a fixed lattice".

### Ising encoding

With one qubit per residue and `z_i = +1` (H) / `-1` (P):

```
eps(s_i, s_j) = -1 * delta_HH = -(1 + z_i)(1 + z_j) / 4

G(S) = sum_{i<j} DeltaC_ij * eps(s_i, s_j)
     = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j

h_i   = -1/4 * sum_{j != i} DeltaC_ij
J_ij  = -1/4 * DeltaC_ij            (i != j)
const = -1/4 * sum_{i<j} DeltaC_ij
```

`qaoa_generator.direct_G()` provides a completely transparent reference
implementation that evaluates `G(S)` straight from the `EPS` matrix. Every
experiment cross-checks the Ising expansion against this reference and reports
the maximum deviation (expected to be at machine precision, ~1e-16).

### Bit-ordering convention

**Project-wide convention: `bit = 0` -> `H`, `bit = 1` -> `P`**, and qubit `i`
carries bit `i` of the state index (LSB-first). PennyLane's `qml.probs`
orders states MSB-first, so `_bit_reverse_index()` re-indexes the returned
probability vector back into LSB alignment. Every downstream computation
(`G_of_state`, `opt_mask`, sequence strings) uses this same convention.

---

## 3. File-by-file overview

| File | Lines | Role |
|------|-------|------|
| `qaoa_generator.py` | 1508 | **Core library + CLI.** HP model, Ising construction, QAOA circuit, HEA circuit, optimizers, baselines, plotting. |
| `evolutionary.py` | 194 | **Core library.** The `(mu + lambda)` evolution strategy optimizer. |
| `depth_experiment.py` | 1621 | **Experiment 1.** QAOA depth sweep `p = 1..9` on a fixed problem. |
| `experiment3_size_scaling.py` | 1116 | **Experiment 3.** Controlled scaling of lattice size at fixed QAOA depth. |
| `experiment4_target_robustness.py` | 1235 | **Experiment 4.** Robustness across many distinct target structures. |
| `experiment5_parameter_control.py` | 515 | **Experiment 5.** Parameter-matched / budget-matched QAOA vs HEA comparison. |
| `scale_experiment.py` | 111 | **Benchmark driver.** QAOA / HEA / random search / SA at several lattice sizes. |

### 3.1 `qaoa_generator.py` — core library and reference CLI

The backbone of the project. Everything else imports from it.

**Sections**

| Section | Contents |
|---------|----------|
| 1. HP model | `ALPHABET`, `EPS`, `snake_coords()`, `contact_map()`, `random_saw()` |
| 2. Ising | `build_ising()`, `direct_G()` |
| 3. QAOA circuit | `QAOACircuit`, `_run_numpy()`, `_bit_reverse_index()` |
| 3b. HEA ansatz | `HardwareEfficientCircuit`, `_RY()`, `_RZ()`, `_apply_cnot()` |
| 3c. Optimization helpers | `train_evolutionary()`, `random_search()`, `simulated_annealing()` |
| 4. Main workflow | `main()`, `run_compare()` |
| 5. Plotting | `plot_training()`, `plot_distribution()`, `plot_structure()`, `plot_comparison()`, `plot_sequence_spectrum()` |
| 6. Clean benchmark | `main_clean()`, `_run_clean()`, `_run_qaoa_clean()`, `_run_hea_clean()`, `_run_random_clean()`, `_run_sa_clean()`, `_save_final_distributions()` |

**Public API used by the other scripts**

```python
from qaoa_generator import (
    snake_coords,      # (rows, cols) -> array of shape (N, 2)
    contact_map,       # coords -> binary contact matrix
    build_ising,       # coords -> (h, J, const, dC, C_target, C_bg)
    direct_G,          # (z, dC) -> G(S), the transparent reference
    QAOACircuit,       # (n, h, J, backend) -> .probs(gammas, betas, G_of_state)
    HardwareEfficientCircuit,  # (n, n_layers, backend) -> .probs(params)
    distribution_entropy,      # probs -> (H_bits, N_eff)
)

# Shared helpers reused by experiment3 / experiment4 / scale_experiment
from qaoa_generator import (
    _exact_metrics, _run_qaoa_clean, _run_hea_clean,
    _run_random_clean, _run_sa_clean, _setup_logger,
)
```

**Two entry points.** The module defines both `main()` (an earlier
single-shot workflow with a `--mode` flag) and `main_clean()` (the
"JCIM clean benchmark"). The `if __name__ == "__main__"` block at the bottom
of the file calls **`main_clean()`**, so that is the CLI you get. `main()` is
still importable and callable from Python if you want the older
`--mode qaoa` / `--mode compare` workflow, but it is **not** reachable from the
command line.

**Backends.** `QAOACircuit` tries, in order: PennyLane `default.qubit` ->
PennyLane `qiskit.aer` -> a pure-NumPy exact statevector simulator. The
`--backend` flag can force `qiskit.aer`. The fallback is mathematically
equivalent, so results are reproducible with or without PennyLane installed.

### 3.2 `evolutionary.py` — the optimizer

A single self-contained class, `EvolutionaryOptimizer`, implementing a
`(mu + lambda)` evolution strategy that **minimizes** a scalar fitness.

- `Individual` dataclass: `params`, `fitness`, `generation`, `id`.
- Selection: tournament (`tournament_size=3`).
- Crossover: two-point / "cut and splice" crossover.
- Mutation: Gaussian (default) or uniform resampling, with per-parameter rate.
- Elitism: the best `elite_size` individuals survive unchanged.
- Adaptive mutation rate: boosted when the population fitness collapses to a
  single value, decayed otherwise.
- `history` records `generation`, `best_fitness`, `avg_fitness`, `std_fitness`,
  `mutation_rate` per generation.
- `get_diversity()` reports the mean distance of individuals from the
  population centroid.

The optimizer is fully RNG-driven through a local `np.random.RandomState`, so a
given `seed` reproduces a run exactly.

### 3.3 `depth_experiment.py` — Experiment 1: depth sweep

Sweeps the QAOA depth `p` from `--p-start` to `--p-end` (default 1..9) against
a **single fixed problem** (one Hamiltonian, one target structure, one
`problem_seed`), repeating each depth across several optimizer seeds.

Key behaviours:

- The problem (Ising coefficients, `G_of_state`, `G*`, `opt_mask`) is built
  **once** and reused for every run.
- Each `(p, seed)` combination gets its own self-contained directory containing
  a copy of `problem.npz`, its own `config.json`, and its own `run.log`.
- **Resume support**: an existing `summary.json` with `status == "ok"` and
  matching `p` / `optimizer_seed` is skipped. Use `--no-resume` to force
  re-execution, or `--out-dir` to point at a specific directory.
- Per-generation metrics are recorded exactly (no random sampling per
  generation).
- Produces aggregated CSVs, error-bar plots, convergence plots, and a
  `validation_summary.txt`.

### 3.4 `experiment3_size_scaling.py` — Experiment 3: size scaling

Runs QAOA at a **fixed depth** across several lattice sizes (default
`3x3,3x4,4x4`), with an identical optimization budget
(`population x generations`) for every size, and every size fully enumerated
(`2^n` states, so exact optimum `G*` is known).

- Supports `--continue-on-error` so one failed run does not abort the sweep.
- Computes size-normalized progress metrics.
- Includes dependency-free statistical helpers: `holm_adjust()` and
  `paired_wilcoxon_exact_or_normal()` (a small-sample normal approximation,
  explicitly labelled exploratory in the output).
- Emits manuscript-style figures `Figure6a..6d`, `Figure6_scaling_metrics`,
  `Figure7_convergence_by_size`, plus supporting `FigureS3` / `FigureS4`.

### 3.5 `experiment4_target_robustness.py` — Experiment 4: target robustness

Asks: *does the QAOA pipeline work for many different target structures, not
just one?*

- `enumerate_unique_target_structures()` enumerates all Hamiltonian
  self-avoiding walks on the lattice and **de-duplicates them by their
  residue-indexed contact-map signature** (so symmetric variants collapse
  automatically).
- `--target-count` selects how many targets to use (`0` = all).
- Each target becomes a separate problem instance with its own Hamiltonian, and
  each is run across the optimizer seed range.
- Produces `target_catalog.csv`, `target_summary.csv`, `problem_summary.csv`,
  `run_summary.csv`, `convergence_raw.csv`, `convergence_summary.csv`,
  `correlation_summary.csv`, `statistical_tests.txt`,
  `validation_summary.txt`, and three figures
  (`Figure8_target_robustness`, `FigureS4_target_energy_profiles`,
  `FigureS5_target_seed_heatmap`).
- Uses SciPy's `friedmanchisquare` and `wilcoxon` when SciPy is available;
  gracefully degrades when it is not.

### 3.6 `experiment5_parameter_control.py` — Experiment 5: parameter control

Compares QAOA against hardware-efficient ansätze under **matched parameter
counts and matched optimization budgets**. Hard-locked to a `3x3` lattice.

Four conditions:

| Condition | Parameters | Description |
|-----------|-----------|-------------|
| `QAOA-p6` | 12 | 6 gamma + 6 beta |
| `Compact-HEA12` | 12 | 9 local RY + 3 shared RZ + ring CNOT |
| `HEA-L1` | 18 | standard HEA, one layer |
| `HEA-L2` | 36 | standard HEA, two layers |

`CompactHEA12` is a locally defined class that mirrors the PennyLane circuit
exactly with a NumPy statevector implementation, and the script performs a
**self-check** comparing the two backends before the experiment starts (the
maximum probability deviation is printed and logged).

It also draws `sample_count` sequences from the final distribution to estimate
the practical hit rate and the number of samples needed for a 95% hit.

### 3.7 `scale_experiment.py` — multi-method benchmark driver

A compact driver that, for each lattice size and repeat, runs four methods
under matched budgets and writes a long-format table:

- `QAOA` (`--layers` deep)
- `HEA` (`--hea-layers` deep)
- `Random search` (same parameter space, same budget)
- `SA` (classical simulated annealing over sequence space)

SA produces a point solution, so its distribution metrics (`expG`, `P_Gstar`,
`N_eff`) stay `NaN` by design — they are intentionally not comparable.

---

## 4. Installation

### Requirements

```
python >= 3.9
numpy        (required)
matplotlib  (required, for all figures)
pennylane   (optional -- enables the PennyLane backend; NumPy fallback otherwise)
scipy       (optional -- enables Friedman / Wilcoxon tests in experiment4)
qiskit-aer  (optional -- only for --backend qiskit.aer)
```

### Install

```bash
pip install -r requirements.txt
```

Or install the pieces individually:

```bash
pip install numpy matplotlib
pip install pennylane          # optional, recommended
pip install scipy              # optional, recommended for experiment4
```

> **Note:** the code runs correctly with **only** `numpy` and `matplotlib`.
> Without PennyLane it prints a warning and falls back to the built-in NumPy
> exact statevector simulator, which produces mathematically equivalent
> probabilities.

### Clone and verify

```bash
git clone https://github.com/4597180/Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA.git
cd Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA
pip install -r requirements.txt

# Quick end-to-end sanity check (runs in a few seconds)
python qaoa_generator.py --rows 3 --cols 3 --layers 2 --hea-layers 1 \
    --generations 2 --population 6 --seed 46
```

The run should print a `FINAL` block listing `QAOA`, `HEA`, `Random search`
and `SA` rows, and a `RESULT_DIR=` line pointing at the generated output
folder.

### Put the scripts on the path (optional)

All scripts insert their own directory into `sys.path`, so they can be launched
from anywhere:

```bash
cd Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA
python qaoa_generator.py --help
```

---

## 5. How to run each script

All commands are run from the project root:

```bash
cd Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA
```

---

### 5.1 `qaoa_generator.py` — clean benchmark (QAOA / HEA / random / SA)

The command-line entry point is `main_clean()`, which runs **all four methods
under a matched budget** in a single directory: QAOA, the hardware-efficient
ansatz, random search over the same parameter space, and simulated annealing
over sequence space.

**Single seed:**

```bash
python qaoa_generator.py --rows 3 --cols 3 --layers 6 --hea-layers 2 --generations 200 --population 40 --seed 46
```

**Batch over a seed range** (fixes `problem_seed`, varies only the optimizer
seed — the correct design for measuring optimizer variance):

```bash
python qaoa_generator.py --rows 3 --cols 3 --layers 6 --seed-start 46 --seed-end 55 --generations 200 --population 40
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--rows` / `--cols` | 3 / 3 | Lattice dimensions (`n = rows * cols` qubits) |
| `--layers` | 6 | QAOA depth `p` (2p trainable parameters) |
| `--hea-layers` | 2 | Depth of the hardware-efficient ansatz |
| `--generations` | 200 | Evolutionary generations |
| `--population` | 40 | Individuals per generation |
| `--seed` | 46 | Single optimizer seed (ignored when `--seed-start` is used) |
| `--seed-start` / `--seed-end` | `None` | Batch mode: optimizer seed range (must be given together) |
| `--problem-seed` | 12345 | Fixed seed for the target/background Hamiltonian |
| `--backend` | `auto` | `auto` / `default.qubit` / `qiskit.aer` |
| `--sample-count` | 2000 | Final-only Monte Carlo sample count |
| `--sa-reruns` | 200 | Independent SA reruns for the classical endpoint distribution |
| `--out-root` | `outputs` | Output root directory |

**Output.**

Single-seed mode creates `outputs/<rows>x<cols>_p<layers>_prob<problem_seed>_opt<seed>_<timestamp>/`
containing `run.log`, `config.json`, `problem.npz`, `problem_summary.txt`,
`summary.json`, `final_metrics.csv`, `final_state_distribution.csv`,
`qaoa_generation_metrics.csv`, `hea_generation_metrics.csv`,
`random_generation_metrics.csv`, `sa_generation_metrics.csv`, and
`sa_rerun_sequence_frequency.csv`.

Batch mode creates `outputs/<rows>x<cols>_p<layers>_prob<problem_seed>_batch_<timestamp>/`
containing `batch.log`, `batch_config.json`, `batch_summary.csv`,
`batch_summary.json`, and one `seed_<n>/` subdirectory per seed with the
per-run files listed above.

> **Note:** the older `main()` workflow (with `--mode qaoa` / `--mode compare`
> and the `training_curves.png` / `comparison.png` figures) is still present in
> the module but is **not** wired to the command line. If you need it, call
> `main()` explicitly from Python.

---

### 5.2 `depth_experiment.py` — Experiment 1, depth sweep

```bash
python depth_experiment.py --rows 3 --cols 3 --p-start 1 --p-end 9 --seed-start 46 --seed-end 55 --generations 200 --population 40
```

Quick smoke test (one depth, one seed, few generations):

```bash
python depth_experiment.py --p-start 1 --p-end 2 --seed-start 46 --seed-end 47 --generations 5 --population 10 --out-dir outputs/smoke
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--rows` / `--cols` | 3 / 3 | Lattice dimensions |
| `--p-start` / `--p-end` | 1 / 9 | QAOA depth range (inclusive) |
| `--seed-start` / `--seed-end` | 46 / 55 | Optimizer seed range (inclusive) |
| `--problem-seed` | 12345 | Seed of the fixed problem / Hamiltonian |
| `--generations` | 200 | Generations per run |
| `--population` | 40 | Population size per run |
| `--backend` | `auto` | Quantum backend |
| `--out-root` | `outputs` | Root directory for timestamped result folders |
| `--out-dir` | `None` | Explicit output directory (enables resume) |
| `--no-resume` | off | Re-run even if a valid `summary.json` exists |

Total runs = `(p_end - p_start + 1) * (seed_end - seed_start + 1)`
(default `9 * 10 = 90`).

**Resume example** — the same command with `--out-dir` pointing at an existing
folder will skip every completed `(p, seed)` pair:

```bash
python depth_experiment.py --out-dir outputs/depth_p1-p9_3x3_prob12345_seeds46-55
```

---

### 5.3 `experiment3_size_scaling.py` — Experiment 3, size scaling

```bash
python experiment3_size_scaling.py --sizes 3x3,3x4,4x4 --layers 5 --generations 200 --population 40 --seed-start 46 --seed-end 55
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sizes` | `3x3,3x4,4x4` | Comma-separated lattice sizes |
| `--layers` | 5 | Fixed QAOA depth `p` for every size |
| `--generations` | 200 | Generations per run |
| `--population` | 40 | Population size per run |
| `--seed-start` / `--seed-end` | 46 / 55 | Optimizer seed range |
| `--problem-seed` | 12345 | Problem seed |
| `--n-background` | 30 | Number of random background walks |
| `--sample-count` | 2000 | Samples drawn from the final distribution |
| `--backend` | `auto` | Quantum backend |
| `--out-root` | `outputs` | Output root directory |
| `--continue-on-error` | off | Keep going when a single run fails |

**Output:** `outputs/experiment3_size_scaling_<timestamp>/` (relative to your
current working directory, plus `figures/` and `figure_data/` subfolders).

> **Note:** `experiment3_size_scaling.py` resolves `--out-root` against the
> **current working directory**, unlike the other scripts which resolve it
> against the script's own directory. Run it from the project root, or pass an
> absolute path.

---

### 5.4 `experiment4_target_robustness.py` — Experiment 4, target robustness

```bash
python experiment4_target_robustness.py --rows 3 --cols 3 --p 5 --target-count 8 --seed-start 46 --seed-end 55 --generations 200 --population 40
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--rows` / `--cols` | 3 / 3 | Lattice dimensions |
| `--p` | 5 | Fixed QAOA depth |
| `--population` | 40 | Population size |
| `--generations` | 200 | Generations per run |
| `--problem-seed` | 12345 | Background-walk seed |
| `--target-seed` | 20260918 | Seed used to select targets |
| `--seed-start` / `--seed-end` | 46 / 55 | Optimizer seed range |
| `--target-count` | 0 | Number of targets to use (`0` = all unique targets) |
| `--background-attempts` | 30 | Background walks per problem |
| `--backend` | `auto` | Quantum backend |
| `--out-root` | `outputs` | Output root directory |
| `--out-dir` | `None` | Explicit output directory (enables resume) |
| `--no-resume` | off | Re-run existing completed runs |

**Output:** `outputs/exp4_target_robustness_<rows>x<cols>_p<p>_prob<seed>_targets<n>_seeds<a>-<b>_<timestamp>/`,
with one subdirectory per target, each containing one `seed_<n>/` subdirectory
per optimizer seed. When `--out-dir` is given, that directory is used as-is.

---

### 5.5 `experiment5_parameter_control.py` — Experiment 5, parameter control

```bash
python experiment5_parameter_control.py --seed-start 46 --seed-end 55 --generations 200 --population 40 --sample-count 2000
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--rows` / `--cols` | 3 / 3 | Must stay `3x3` (hard-locked for a controlled comparison) |
| `--qaoa-depth` | 6 | QAOA depth; the QAOA condition uses `2p = 12` parameters |
| `--problem-seed` | 12345 | Problem seed |
| `--seed-start` / `--seed-end` | 46 / 55 | Optimizer seed range |
| `--generations` | 200 | Generations per run (identical for every condition) |
| `--population` | 40 | Population size (identical for every condition) |
| `--sample-count` | 2000 | Samples drawn from the final distribution |
| `--backend` | `auto` | Quantum backend |
| `--out-root` | `outputs_exp5` | Output root directory |

**Output:** `outputs_exp5/exp5_3x3_p6_prob<seed>_seeds<a>-<b>_<timestamp>/`,
with a subdirectory per condition (`QAOA_p6`, `Compact_HEA12`, `HEA_L1`,
`HEA_L2`), plus `Figure9_parameter_control_summary.png` and
`FigureS11_parameter_count_vs_PGstar.png`.

> **Note:** the folder name is hard-coded to `exp5_3x3_p6_...` regardless of
> the `--qaoa-depth` you pass, because the experiment is designed around the
> fixed 12-parameter `QAOA-p6` condition.

---

### 5.6 `scale_experiment.py` — multi-method size benchmark

```bash
python scale_experiment.py --sizes 3x3,3x4,4x4 --layers 6 --hea-layers 2 --repeats 10 --generations 200 --population 40
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sizes` | `3x3,3x4,4x4` | Comma-separated lattice sizes |
| `--layers` | 6 | QAOA depth |
| `--hea-layers` | 2 | HEA depth |
| `--generations` | 200 | Generations per run |
| `--population` | 40 | Population size |
| `--seed` | 46 | Base optimizer seed (repeats use `seed + k`) |
| `--problem-seed` | 12345 | Problem seed |
| `--repeats` | 10 | Repeats per size |
| `--sa-reruns` | 200 | Simulated-annealing reruns for sequence-frequency statistics |
| `--sample-count` | 2000 | Samples drawn from the final distribution |
| `--backend` | `auto` | Quantum backend |
| `--out-root` | `scale_outputs` | Output root directory |

**Output:** `scale_outputs/scale_results.csv` (long table) and
`scale_outputs/scale_summary.csv` (mean +/- SD per size and method).

---

## 6. Output directory structure

### `depth_experiment.py`

```
outputs/depth_p1-p9_3x3_prob12345_seeds46-55_<timestamp>/
|-- experiment.log                 # experiment-level log
|-- experiment_config.json         # full resolved configuration
|-- problem_summary.json           # target/Hamiltonian summary
|-- problem.npz                    # the fixed problem (h, J, dC, G_of_state, opt_mask, ...)
|-- depth_summary.csv              # one row per (p, seed)
|-- depth_summary_by_p.csv         # mean/SD/median/min/max aggregated per p
|-- convergence_raw.csv            # every generation of every run
|-- convergence_summary.csv        # mean +/- SD per (p, generation)
|-- validation_summary.txt         # integrity checks and failed-run report
|-- final_summary.json             # machine-readable final rollup
|-- depth_vs_expG.png
|-- depth_vs_PGstar.png
|-- depth_vs_Neff.png
|-- convergence_by_depth.png
`-- p_01/
    |-- seed_46/
    |   |-- run.log
    |   |-- config.json
    |   |-- problem.npz            # copy, so each run is self-contained
    |   |-- generation_metrics.csv
    |   |-- final_state_distribution.csv
    |   |-- best_params.npz
    |   `-- summary.json
    `-- seed_47/ ...
```

### `experiment4_target_robustness.py`

```
outputs/<batch_name>/
|-- experiment.log
|-- experiment_config.json
|-- target_catalog.csv             # target coordinates / contact-map catalog
|-- problem_summary.csv            # one row per target Hamiltonian
|-- run_summary.csv                # one row per (target, seed)
|-- target_summary.csv             # target-level mean/SD
|-- convergence_raw.csv
|-- convergence_summary.csv
|-- correlation_summary.csv
|-- statistical_tests.txt
|-- validation_summary.txt
|-- final_summary.json
|-- Figure8_target_robustness.png
|-- FigureS4_target_energy_profiles.png
|-- FigureS5_target_seed_heatmap.png
`-- <target_id>/
    |-- target_coords.csv
    |-- target_contact_map.csv
    |-- problem_summary.json / .txt
    `-- seed_46/
        |-- run.log
        |-- config.json
        |-- generation_metrics.csv
        |-- final_state_distribution.csv
        |-- energy_resolved_probability.csv
        |-- final_metrics.csv
        `-- summary.json
```

---

## 7. Metrics reference

| Metric | Symbol | Meaning | Better when |
|--------|--------|---------|-------------|
| Expected design score | `<G>` | `sum_s p(s) * G(s)` over all `2^n` states | Lower |
| Probability of global optimum | `P(G*)` | Total probability mass on states with `G(s) = G*` | Higher |
| Shannon entropy | `H` | `-sum p log2 p`, in bits | (diversity descriptor) |
| Effective sequence count | `N_eff` | `2^H` | (diversity descriptor) |
| Normalized progress | — | `(<G> - <G>_uniform) / (G* - <G>_uniform)` | Higher |
| Top sequence | — | The `argmax` probability state, rendered as an `H`/`P` string | — |
| Uniform expectation | `<G>_uniform` | `mean(G_of_state)` — the starting point of the optimization | baseline |

`N_eff = 2^H` is the key "distribution vs point" descriptor: a point solution
(SA) has `N_eff = 1`, while a broad QAOA distribution has a much larger value.

The exact global optimum `G*` is known because the state space is small enough
to enumerate exhaustively (up to `4x4 = 16` qubits = 65 536 states).

---

## 8. Using this repository with Git

The repository is a plain git repo with no submodules and no vendored
dependencies.

### Clone

```bash
git clone https://github.com/4597180/Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA.git
cd Structure-Conditioned-HP-Protein-Sequence-Design-with-QAOA
pip install -r requirements.txt
```

### Branch layout

`main` holds the released state. For experimental work, branch off rather than
committing directly:

```bash
git switch -c feature/<topic>
# ... work ...
git add -A
git commit -m "Describe the change"
git push -u origin feature/<topic>
```

Then open a Pull Request on GitHub.

### What is deliberately not tracked

`.gitignore` excludes every experiment output directory (`outputs/`,
`outputs_exp5/`, `scale_outputs/`) along with `__pycache__/`, virtualenvs and
editor settings. This keeps the repository small, since experiment results are
regenerable from the scripts given the seeds recorded in each run's
`config.json`.

If you want to publish a specific set of results alongside the code, add a
narrow exception rather than removing the ignore rule wholesale:

```gitignore
# ...in .gitignore...
!outputs/published_figures/
```

### Reproducing a published result

Every run directory stores the exact configuration it used in `config.json`
(including `problem_seed`, optimizer seed, population, generations and
backend). To reproduce a run, pass those values back on the command line:

```bash
python depth_experiment.py --problem-seed 12345 --p-start 1 --p-end 9 \
    --seed-start 46 --seed-end 55 --generations 200 --population 40
```

---

## 9. Reproducibility notes

- **Problem seed vs optimizer seed.** `--problem-seed` fixes the *Hamiltonian*
  (target structure plus background ensemble). `--seed` / `--seed-start` /
  `--seed-end` fix the *optimization* random stream. Experiments 1, 3, 4 and 5
  hold the problem seed constant and vary the optimizer seed, which is the
  correct design for measuring optimizer variance.
- **Dual RNG fixing.** `depth_experiment.py` and `experiment5` set both
  `np.random.seed(seed)` and the optimizer's internal `RandomState`, because
  some optimizer operations still touch the NumPy global RNG.
- **Exact evaluation.** Every objective evaluation uses the full `2^n`
  statevector, so `<G>` and `P(G*)` are exact — there is no shot noise.
- **Backend equivalence.** PennyLane and the NumPy fallback produce the same
  probabilities up to floating-point precision; the self-checks printed at
  startup quantify this.
- **Interruption safety.** JSON writes go through a temporary file and an
  atomic replace, so an interrupted run never leaves a truncated `summary.json`.
  `depth_experiment.py` and `experiment4` support resuming.

---

## 10. Troubleshooting

**`PennyLane unavailable ... falling back to NumPy statevector simulation`**
This is a warning, not an error. Install PennyLane (`pip install pennylane`)
to use the hardware-accelerated backend, or ignore it — the NumPy path gives
identical results.

**`ModuleNotFoundError: No module named 'scipy'`**
Only affects the Friedman / Wilcoxon tests in `experiment4_target_robustness.py`.
The script degrades gracefully and notes the missing tests in
`statistical_tests.txt`. Install with `pip install scipy` to enable them.

**`--backend qiskit.aer` fails**
Requires `qiskit` and `qiskit-aer` to be installed. Use `--backend auto` or
`default.qubit` otherwise.

**Runs take too long**
Reduce the budget. The number of statevector evaluations scales as
`population x generations x 2^n`. For example:

```bash
python depth_experiment.py --p-start 1 --p-end 3 --seed-start 46 --seed-end 48 --generations 50 --population 20
```

**`FileExistsError` when using `--out-dir`**
`depth_experiment.py` creates the directory with `exist_ok=False` only when it
generates a timestamped name. If you pass an existing `--out-dir`, it is
reused and existing completed runs are skipped (this is the intended resume
behaviour).

**Chinese characters still appear in console output**
All source files have been fully translated. If you are running an *older
copy* of the scripts, or a cached `.pyc`, re-download the files and clear
`__pycache__`.

---

## License / Attribution

Provided as-is for research use. The methodology follows standard QAOA and
Ising-mapping formulations; cite the original QAOA paper (Farhi, Goldstone &
Gutmann, 2014) and the HP model (Tavis, 1989) as appropriate.
