"""Kiem thu tinh dung dan:  python -m pytest -q   (hoac: python tests/test_algorithms.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hrp.algorithms import (order_fhrp, order_hrp, recursive_bisection, hrp, fhrp,  # noqa: E402
                            ivp, min_variance)
from hrp.data import load_returns, synthetic_corr_cov  # noqa: E402
from hrp.exact import adjacent_cost, brute_force_orders, held_karp_path  # noqa: E402
from hrp.algorithms import corr_distance  # noqa: E402

R = load_returns()
COV, CORR = R.cov().values, R.corr().values


def test_weights_valid():
    for f in (hrp, fhrp):
        w = f(COV, CORR)
        assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12


def test_orders_are_permutations():
    n = COV.shape[0]
    assert sorted(order_hrp(CORR)) == list(range(n))
    assert sorted(order_fhrp(CORR)) == list(range(n))


def test_theorem1_even_invariance():
    o = order_hrp(CORR)                      # N = 30 (chan)
    h = len(o) // 2
    w1, w2 = recursive_bisection(COV, o), recursive_bisection(COV, o[h:] + o[:h])
    assert np.abs(w1 - w2).max() < 1e-12


def test_corollary2_odd_not_invariant():
    o = order_hrp(CORR)[:29]                 # N = 29 (le)
    h = len(o) // 2
    w1, w2 = recursive_bisection(COV, o), recursive_bisection(COV, o[h:] + o[:h])
    assert np.abs(w1 - w2).max() > 1e-6


def test_table4_small_n():
    rng = np.random.default_rng(1)
    expected = {3: 3, 4: 3, 5: 30, 6: 90}
    for n, e in expected.items():
        _, cov = synthetic_corr_cov(n, 0.0, 0.6, 0.004, rng)
        assert brute_force_orders(cov)[1] == e


def test_held_karp_is_optimal_small():
    import itertools
    D = corr_distance(CORR[:7, :7])
    best = min(adjacent_cost(p, D) for p in itertools.permutations(range(7)))
    assert abs(held_karp_path(D)[0] - best) < 1e-12


def test_ivp_and_minvar():
    assert abs(ivp(COV).sum() - 1) < 1e-12
    w = min_variance(COV)
    assert w @ COV @ w <= ivp(COV) @ COV @ ivp(COV) + 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
