import math
import random
from typing import List, Optional, Tuple, Sequence

import numpy as np


def distance_matrix(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix for `points`.

    `points` is a sequence of (x, y) pairs.
    """
    n = len(points)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(n):
            x2, y2 = points[j]
            d[i, j] = math.hypot(x1 - x2, y1 - y2)
    return d


def tour_length(tour: List[int], dist: np.ndarray) -> float:
    """Return total length of `tour` using distance matrix `dist`.
    The tour is assumed to be cyclic.
    """
    n = len(tour)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        s += float(dist[tour[i], tour[(i + 1) % n]])
    return s


def nearest_neighbor(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda x: float(dist[last, x]))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour


def two_opt(tour: List[int], dist: np.ndarray) -> List[int]:
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j - 1], tour[j % n]
                delta = float(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])
                if delta < -1e-9:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
        if improved:
            continue
    return tour


def held_karp(dist: np.ndarray) -> Tuple[float, List[int]]:
    n = dist.shape[0]
    if n == 0:
        return 0.0, []

    start = 0
    N = 1 << n
    dp: List[dict] = [dict() for _ in range(N)]
    dp[1 << start][start] = (0.0, -1)

    for mask in range(N):
        if not (mask & (1 << start)):
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            if j not in dp[mask]:
                continue
            cost_j, _ = dp[mask][j]
            for k in range(n):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                new_cost = cost_j + dist[j, k]
                prev = dp[new_mask].get(k)
                if prev is None or new_cost < prev[0]:
                    dp[new_mask][k] = (new_cost, j)

    full = (1 << n) - 1
    best_cost = float("inf")
    last = -1
    for j in range(n):
        if j == start:
            continue
        if j in dp[full]:
            cost_j, _ = dp[full][j]
            total = cost_j + dist[j, start]
            if total < best_cost:
                best_cost = total
                last = j

    if last == -1:
        return 0.0, [start]

    tour = []
    mask = full
    cur = last
    while cur != -1:
        tour.append(cur)
        _, prev = dp[mask][cur]
        mask ^= 1 << cur
        cur = prev
    tour.reverse()
    return float(best_cost), tour


def simulated_annealing(
    dist: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    T0: float = 1.0,
    alpha: float = 0.995,
    steps: int = 10000,
) -> List[int]:
    n = dist.shape[0]
    if n == 0:
        return []
    if n < 2:
        return [i for i in range(n)]
    if initial_tour is None:
        tour = nearest_neighbor(dist)
    else:
        tour = initial_tour[:]
    best = tour[:]
    best_cost = tour_length(best, dist)
    T = T0
    for step in range(steps):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        new = tour[:]
        new[i:j + 1] = reversed(new[i:j + 1])
        new_cost = tour_length(new, dist)
        cur_cost = tour_length(tour, dist)
        if new_cost < cur_cost or random.random() < math.exp(
            (cur_cost - new_cost) / max(1e-9, T)
        ):
            tour = new
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best = tour[:]
        T *= alpha
        if T < 1e-6:
            break
    return best
