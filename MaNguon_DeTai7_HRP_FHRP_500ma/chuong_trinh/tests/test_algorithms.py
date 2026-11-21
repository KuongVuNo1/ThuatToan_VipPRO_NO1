"""Kiem thu tinh dung dan:  python -m pytest -q   (hoac: python tests/test_algorithms.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hrp.algorithms import (order_fhrp, order_hrp, recursive_bisection, hrp, fhrp,  # noqa: E402
                            ivp, min_variance)
from hrp.data import load_returns, synthetic_corr_cov, vn30_codes, window  # noqa: E402
from hrp.exact import adjacent_cost, brute_force_orders, held_karp_path  # noqa: E402
from hrp.algorithms import corr_distance  # noqa: E402

R_ALL = load_returns()                       # 500 ma, 517 tuan (co o trong truoc ngay niem yet)
R = window(R_ALL, 52, vn30_codes())          # ro VN30, 52 tuan cuoi, du lieu day du
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
    o = order_hrp(CORR)                      # N chan
    h = len(o) // 2
    w1, w2 = recursive_bisection(COV, o), recursive_bisection(COV, o[h:] + o[:h])
    assert np.abs(w1 - w2).max() < 1e-12


def test_corollary2_odd_not_invariant():
    o = order_hrp(CORR)[: len(CORR) - 1]     # N le
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


def test_backtest_dynamic_universe():
    """Ma niem yet muon (o trong dau chuoi) chi duoc dua vao khi du F tuan du lieu."""
    from hrp.backtest import rolling_backtest
    from hrp.algorithms import PORTFOLIO_METHODS
    r = R_ALL[vn30_codes()].iloc[-200:].copy()
    r = r.loc[:, r.notna().all().values]
    k = r.shape[1]
    r.iloc[:80, :5] = np.nan                       # 5 ma "niem yet" sau tuan thu 80
    n_log = []
    port, _, _ = rolling_backtest(r, {"HRP": PORTFOLIO_METHODS["HRP"]}, F=52, B=4, log_n=n_log)
    assert n_log[0][1] == k - 5 and n_log[-1][1] == k and port["HRP"].notna().all()


def test_data_500():
    """Bo du lieu chinh: 500 ma, du 30 ma VN30, khong trung ma."""
    assert R_ALL.shape[1] == 500 and R_ALL.columns.is_unique
    assert set(vn30_codes()) <= set(R_ALL.columns) and len(vn30_codes()) == 30


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
