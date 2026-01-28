from __future__ import annotations

import numpy as np


def crossover(a: np.ndarray, b: np.ndarray, prob: float, rng: np.random.Generator):
    # swap random rows/cols with prob
    out_a = a.copy()
    out_b = b.copy()
    H, W = a.shape
    for i in range(H):
        if rng.random() < prob:
            out_a[i, :], out_b[i, :] = out_b[i, :].copy(), out_a[i, :].copy()
    for j in range(W):
        if rng.random() < prob:
            out_a[:, j], out_b[:, j] = out_b[:, j].copy(), out_a[:, j].copy()
    return out_a, out_b


def mutate(a: np.ndarray, prob: float, rng: np.random.Generator, vmin=0, vmax=10):
    out = a.copy()
    H, W = a.shape
    for i in range(H):
        for j in range(W):
            if rng.random() < prob:
                delta = rng.integers(-1, 2)
                if delta == 0:
                    delta = 1
                out[i, j] = np.clip(out[i, j] + delta, vmin, vmax)
    return out


def evolve_population(pop, fitness, rng, swap_prob=0.2, mutate_prob=0.2, vmin=0, vmax=10):
    # selection: top half
    idx = np.argsort(fitness)[::-1]
    pop = [pop[i] for i in idx]
    keep = pop[: len(pop)//2]
    children = []
    while len(children) + len(keep) < len(pop):
        a, b = rng.choice(keep, size=2, replace=False)
        ca, cb = crossover(a, b, swap_prob, rng)
        ca = mutate(ca, mutate_prob, rng, vmin, vmax)
        cb = mutate(cb, mutate_prob, rng, vmin, vmax)
        children.extend([ca, cb])
    return keep + children[: len(pop) - len(keep)]
