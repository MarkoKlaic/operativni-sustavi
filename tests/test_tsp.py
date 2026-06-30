import unittest
import itertools
import random

import numpy as np

import tsp
import utils


def brute_force_tsp(dist: np.ndarray):
    n = dist.shape[0]
    best = None
    best_tour = None
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        length = 0.0
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            length += float(dist[a, b])
        if best is None or length < best:
            best = length
            best_tour = list(tour)
    return best, best_tour


class TestTSPCore(unittest.TestCase):
    def test_distance_matrix_and_length(self) -> None:
        pts = [(0.0, 0.0), (3.0, 4.0), (3.0, 0.0)]
        d = tsp.distance_matrix(pts)
        self.assertAlmostEqual(d[0, 1], 5.0)
        self.assertAlmostEqual(d[0, 2], 3.0)
        tour = [0, 2, 1]
        length = tsp.tour_length(tour, d)
        self.assertAlmostEqual(length, 3.0 + 4.0 + 5.0)

    def test_nearest_neighbor(self) -> None:
        pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        d = tsp.distance_matrix(pts)
        tour = tsp.nearest_neighbor(d, start=0)
        self.assertEqual(set(tour), {0, 1, 2})

    def test_two_opt_improves(self) -> None:
        pts = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        d = tsp.distance_matrix(pts)
        tour = [0, 2, 1, 3]
        before = tsp.tour_length(tour, d)
        after = tsp.tour_length(tsp.two_opt(tour, d), d)
        self.assertLessEqual(after, before)

    def test_held_karp_small(self) -> None:
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        d = tsp.distance_matrix(pts)
        cost, tour = tsp.held_karp(d)
        self.assertAlmostEqual(cost, tsp.tour_length(tour, d))
        self.assertEqual(set(tour), {0, 1, 2, 3})

    def test_utils_distance_doctest(self) -> None:
        self.assertEqual(utils.distance_between((0, 0), (3, 4)), 5.0)


class TestTSPExtra(unittest.TestCase):
    def test_edge_cases_empty_and_small(self):
        d0 = tsp.distance_matrix([])
        self.assertEqual(d0.size, 0)
        self.assertEqual(tsp.tour_length([], d0), 0.0)

        pts1 = [(0.0, 0.0)]
        d1 = tsp.distance_matrix(pts1)
        self.assertEqual(tsp.nearest_neighbor(d1), [0])
        self.assertEqual(tsp.simulated_annealing(d1), [0])

        pts2 = [(0, 0), (1, 0)]
        d2 = tsp.distance_matrix(pts2)
        self.assertEqual(set(tsp.nearest_neighbor(d2)), {0, 1})

    def test_held_karp_vs_bruteforce_small(self):
        random.seed(1)
        pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(7)]
        d = tsp.distance_matrix(pts)
        bf_cost, bf_tour = brute_force_tsp(d)
        hk_cost, hk_tour = tsp.held_karp(d)
        self.assertAlmostEqual(bf_cost, hk_cost, places=6)
        self.assertEqual(set(hk_tour), set(range(len(pts))))

    def test_solvers_return_valid_permutation(self):
        random.seed(2)
        pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(8)]
        d = tsp.distance_matrix(pts)
        for fn in (tsp.nearest_neighbor, tsp.simulated_annealing, tsp.held_karp):
            if fn is tsp.held_karp:
                cost, tour = fn(d)
            else:
                tour = fn(d)
            self.assertEqual(set(tour), set(range(len(pts))))

    def test_no_duplicate_start(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        d = tsp.distance_matrix(pts)
        _, tour = tsp.held_karp(d)
        self.assertFalse(len(tour) >= 2 and tour[0] == tour[1])


if __name__ == "__main__":
    unittest.main()
