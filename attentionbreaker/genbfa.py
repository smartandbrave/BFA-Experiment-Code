"""Genetic reduction of a sensitive projection's candidate weight set."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch

from analysis import flip_sign_bit


@dataclass
class Solution:
    indices: tuple[int, ...]
    loss: float = float("-inf")
    fitness: float = float("-inf")


def _mutate(indices: tuple[int, ...], rng: random.Random, deletion_rate: float) -> tuple[int, ...]:
    retained = tuple(index for index in indices if rng.random() >= deletion_rate)
    return retained or (rng.choice(indices),)


def _crossover(
    best: tuple[int, ...], parent: tuple[int, ...], rng: random.Random
) -> tuple[int, ...]:
    universe = sorted(set(best) | set(parent))
    best_set, parent_set = set(best), set(parent)
    child = tuple(
        index
        for index in universe
        if ((index in best_set) if rng.random() < 0.5 else (index in parent_set))
    )
    return child or (rng.choice(universe),)


def _tournament(population: list[Solution], rng: random.Random) -> Solution:
    left, right = rng.sample(population, 2)
    return left if left.fitness >= right.fitness else right


def optimize(
    parameter: torch.nn.Parameter,
    candidate_indices: list[int],
    evaluate_loss: Callable[[], float],
    loss_threshold: float,
    initial_random_solutions: int = 50,
    population_size: int = 40,
    max_generations: int = 700,
    patience: int = 200,
    seed: int = 42,
) -> tuple[Solution, list[dict]]:
    if not candidate_indices:
        raise ValueError("The GenBFA candidate pool is empty.")
    rng = random.Random(seed)
    full = tuple(dict.fromkeys(int(index) for index in candidate_indices))
    population = [Solution(full)]
    population.extend(
        Solution(_mutate(full, rng, rng.uniform(0.01, 0.09)))
        for _ in range(initial_random_solutions)
    )
    flat = parameter.data.reshape(-1)
    history: list[dict] = []
    best_successful: Solution | None = None
    best_evaluated: Solution | None = None
    stalled = 0

    def evaluate(solution: Solution) -> None:
        device_indices = torch.tensor(solution.indices, device=flat.device, dtype=torch.long)
        original = flat[device_indices].clone()
        try:
            flat[device_indices] = flip_sign_bit(original)
            solution.loss = float(evaluate_loss())
        finally:
            flat[device_indices] = original
        sign = 1.0 if solution.loss >= loss_threshold else -1.0
        solution.fitness = sign * solution.loss / len(solution.indices)

    for generation in range(max_generations):
        for solution in population:
            evaluate(solution)
        population.sort(key=lambda solution: solution.fitness, reverse=True)
        if best_evaluated is None or population[0].fitness > best_evaluated.fitness:
            best_evaluated = Solution(
                population[0].indices, population[0].loss, population[0].fitness
            )
        successful = [solution for solution in population if solution.loss >= loss_threshold]
        generation_best = min(successful, key=lambda solution: len(solution.indices)) if successful else population[0]

        improved = (
            generation_best.loss >= loss_threshold
            and (best_successful is None or len(generation_best.indices) < len(best_successful.indices))
        )
        if improved:
            best_successful = Solution(generation_best.indices, generation_best.loss, generation_best.fitness)
            stalled = 0
        else:
            stalled += 1
        history.append(
            {
                "generation": generation,
                "candidate_count": len(generation_best.indices),
                "loss": generation_best.loss,
                "fitness": generation_best.fitness,
            }
        )
        if stalled >= patience or (best_successful and len(best_successful.indices) == 1):
            break

        elite = best_successful or generation_best
        next_population = [Solution(elite.indices)]
        next_population.append(Solution(_mutate(elite.indices, rng, rng.uniform(0.01, 0.09))))
        while len(next_population) < population_size:
            parent = _tournament(population, rng)
            child = _crossover(elite.indices, parent.indices, rng)
            child = _mutate(child, rng, rng.uniform(0.01, 0.09))
            next_population.append(Solution(child))
        population = next_population

    final = best_successful or best_evaluated
    if final is None:
        raise RuntimeError("GenBFA did not evaluate any candidate solution.")
    return final, history

