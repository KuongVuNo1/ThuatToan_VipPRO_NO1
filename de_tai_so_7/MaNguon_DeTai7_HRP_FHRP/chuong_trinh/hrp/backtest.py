"""Kiem dinh cuon (rolling backtest) theo thiet ke muc 4.3 cua bai bao.

Cua so uoc luong F tuan, giu danh muc B tuan, truot B tuan, lap lai den het mau.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def rolling_backtest(returns: pd.DataFrame, methods: dict, F: int = 52, B: int = 4,
                     seed: int = 0):
    """Tra ve (loi suat log tuan ngoai mau cua tung danh muc, thoi gian xay dung TB, vong quay TB)."""
    rng = np.random.default_rng(seed)
    out = {m: [] for m in methods}
    build_time = {m: [] for m in methods}
    turnover = {m: [] for m in methods}
    prev = {m: None for m in methods}
    index = []
    T = len(returns)
    for start in range(F, T, B):
        est = returns.iloc[start - F:start]
        hold = returns.iloc[start:start + B]
        cov, corr = est.cov().values, est.corr().values
        simple = np.expm1(hold.values)
        for m, f in methods.items():
            t0 = time.perf_counter()
            w = f(cov, corr, rng)
            build_time[m].append(time.perf_counter() - t0)
            out[m].append(np.log1p(simple @ w))
            if prev[m] is not None:
                turnover[m].append(0.5 * np.abs(w - prev[m]).sum())
            prev[m] = w
        index.append(hold.index)
    port = pd.DataFrame({m: np.concatenate(v) for m, v in out.items()},
                        index=np.concatenate(index))
    return (port,
            {m: float(np.mean(v)) for m, v in build_time.items()},
            {m: float(np.mean(v)) if v else 0.0 for m, v in turnover.items()})


def performance(log_ret: pd.Series) -> dict:
    """Chi so ngoai mau tu chuoi loi suat log tuan."""
    simple = np.expm1(log_ret)
    wealth = (1.0 + simple).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    years = len(log_ret) / 52.0
    ann_ret = wealth.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = simple.std(ddof=1) * np.sqrt(52.0)
    return {
        "PhuongSaiTuan_x1e4": 1e4 * simple.var(ddof=1),
        "DoLechChuanNam_%": 100.0 * ann_vol,
        "LoiSuatNam_%": 100.0 * ann_ret,
        "Sharpe(rf=0)": ann_ret / ann_vol,
        "SutGiamToiDa_%": 100.0 * dd.min(),
    }
