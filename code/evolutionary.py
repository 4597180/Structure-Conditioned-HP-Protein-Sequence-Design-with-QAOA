"""
Evolutionary optimizer: a (mu + lambda)-ES evolution strategy.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class Individual:
    """An individual in the evolutionary population."""
    params: np.ndarray          # Circuit parameters
    fitness: float = float("inf")  # Fitness value (lower is better)
    generation: int = 0
    id: int = 0


class EvolutionaryOptimizer:
    """Evolutionary optimizer - a (mu + lambda) evolution strategy (minimizes fitness)."""

    def __init__(
        self,
        n_params: int,
        param_bounds: List[Tuple[float, float]],
        fitness_fn: Callable[[np.ndarray], float],
        population_size: int = 40,
        elite_size: int = 6,
        tournament_size: int = 3,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        mutation_sigma: float = 0.3,
        mutation_strategy: str = "gaussian",
    ):
        self.n_params = n_params
        self.param_bounds = param_bounds
        self.fitness_fn = fitness_fn

        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_sigma = mutation_sigma
        self.mutation_strategy = mutation_strategy

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self.history: List[Dict] = []

        self._success_count = 0
        self._total_mutations = 0
        self.rng = np.random.RandomState(0)
        self.n_evaluations = 0

    def initialize_population(self, seed: int = 46, verbose: bool = True):
        self.rng = np.random.RandomState(seed)
        self.population = []
        if verbose:
            print(f"  Initializing population ({self.population_size} individuals) ...")
        for i in range(self.population_size):
            params = np.array([
                self.rng.uniform(low, high) for low, high in self.param_bounds
            ])
            self.population.append(Individual(params=params, id=i))
        self._evaluate_population()
        self._update_best()
        if verbose:
            print(f"  Initialization done (initial best <G> = {self.best_individual.fitness:.4f})")
        return self.population

    def run_generation(self):
        self.population.sort(key=lambda ind: ind.fitness)
        elites = [
            Individual(params=ind.params.copy(), fitness=ind.fitness,
                       generation=ind.generation, id=ind.id)
            for ind in self.population[:self.elite_size]
        ]

        new_population = elites
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            if self.rng.random_sample() < self.crossover_rate:
                child1_params, child2_params = self._crossover(
                    parent1.params, parent2.params)
            else:
                child1_params = parent1.params.copy()
                child2_params = parent2.params.copy()
            child1_params = self._mutate(child1_params)
            child2_params = self._mutate(child2_params)
            child1_params = self._clip_params(child1_params)
            child2_params = self._clip_params(child2_params)
            new_population.append(Individual(params=child1_params,
                                             generation=self.generation + 1))
            if len(new_population) < self.population_size:
                new_population.append(Individual(params=child2_params,
                                                 generation=self.generation + 1))

        self.population = new_population[:self.population_size]
        self._evaluate_population()
        self._update_best()
        self.generation += 1
        self._adapt_mutation()
        self._record_history()
        return self.population

    def _tournament_select(self) -> Individual:
        candidates = self.rng.choice(
            self.population,
            size=min(self.tournament_size, len(self.population)),
            replace=False,
        )
        return min(candidates, key=lambda ind: ind.fitness)

    def _crossover(self, params1, params2):
        n = len(params1)
        cut1 = self.rng.randint(0, n)
        cut2 = self.rng.randint(cut1, n + 1)
        child1 = np.concatenate([params1[:cut1], params2[cut1:cut2], params1[cut2:]])
        child2 = np.concatenate([params2[:cut1], params1[cut1:cut2], params2[cut2:]])
        return child1, child2

    def _mutate(self, params):
        mutated = params.copy()
        n_params = len(params)
        if self.mutation_strategy == "gaussian":
            mask = self.rng.random_sample(n_params) < self.mutation_rate
            noise = self.rng.randn(n_params) * self.mutation_sigma
            mutated = mutated + mask * noise
            self._total_mutations += mask.sum()
        elif self.mutation_strategy == "uniform":
            for i in range(n_params):
                if self.rng.random_sample() < self.mutation_rate:
                    low, high = self.param_bounds[i]
                    mutated[i] = self.rng.uniform(low, high)
        return mutated

    def _clip_params(self, params):
        for i, (low, high) in enumerate(self.param_bounds):
            params[i] = np.clip(params[i], low, high)
        return params

    def _evaluate_population(self):
        unevaluated = [ind for ind in self.population if np.isinf(ind.fitness)]
        total = len(unevaluated)
        for i, ind in enumerate(unevaluated):
            ind.fitness = self.fitness_fn(ind.params)
            self.n_evaluations += 1
            # Raise the print threshold so per-generation output (10/40, ...) does not flood the console.
            if total > 5000 and (i + 1) % 1000 == 0:
                print(f"{i+1}/{total}...", end="", flush=True)

    def _update_best(self):
        current_best = min(self.population, key=lambda ind: ind.fitness)
        if (self.best_individual is None or
                current_best.fitness < self.best_individual.fitness):
            self.best_individual = Individual(
                params=current_best.params.copy(),
                fitness=current_best.fitness,
                generation=self.generation,
            )

    def _avg_fitness(self):
        return np.mean([ind.fitness for ind in self.population])

    def _success_rate(self):
        if self._total_mutations == 0:
            return 0.5
        return self._success_count / self._total_mutations

    def _adapt_mutation(self):
        fitnesses = np.array([ind.fitness for ind in self.population])
        if np.std(fitnesses) < 1e-6:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
        else:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.999)

    def _record_history(self):
        fitnesses = [ind.fitness for ind in self.population]
        self.history.append({
            "generation": self.generation,
            "best_fitness": self.best_individual.fitness,
            "avg_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "mutation_rate": self.mutation_rate,
        })

    def get_diversity(self):
        params_array = np.array([ind.params for ind in self.population])
        centroid = np.mean(params_array, axis=0)
        return float(np.mean([np.linalg.norm(p - centroid) for p in params_array]))
