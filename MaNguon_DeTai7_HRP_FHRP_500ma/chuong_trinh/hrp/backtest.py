"""Kiem dinh cuon (rolling backtest) theo thiet ke muc 4.3 cua bai bao.

Cua so uoc luong F tuan, giu danh muc B tuan, truot B tuan, lap lai den het mau.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def rolling_backtest(returns: pd.DataFrame, methods: dict, F: int = 52, B: int = 4,
                     seed: int = 0, min_assets: int = 10, log_n: list | None = None):
    """Tra ve (loi suat log tuan ngoai mau cua tung danh muc, thoi gian xay dung TB, vong quay TB).

    Tap ma thay doi theo thoi gian: o moi lan tai can bang chi dung cac ma CO DU F tuan du lieu
    trong cua so uoc luong (ma niem yet muon duoc dua vao khi du lich su). Neu trong B tuan nam
    giu mot ma thieu du lieu, loi suat tuan do coi bang 0 (giu nguyen gia).
    log_n: neu truyen list, ghi lai (ngay tai can bang, so ma n) o moi lan tai can bang.
    """
    rng = np.random.default_rng(seed)
    out = {m: [] for m in methods}
    build_time = {m: [] for m in methods}
    turnover = {m: [] for m in methods}
    prev = {m: None for m in methods}
    index = []
    T = len(returns)
    for start in range(F, T, B):
        est = returns.iloc[start - F:start]
        v = est.values
        full = ~np.isnan(v).any(axis=0)
        sd_ok = np.zeros(v.shape[1], dtype=bool)
        sd_ok[full] = v[:, full].std(axis=0) > 0
        cols = est.columns[full & sd_ok]
        if len(cols) < min_assets:
            continue
        est = est[cols]
        hold = returns.iloc[start:start + B][cols].fillna(0.0)
        cov = np.cov(est.values, rowvar=False)          # nhanh hon pandas .cov() (khong co NaN)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        simple = np.expm1(hold.values)
        if log_n is not None:
            log_n.append((returns.index[start], len(cols)))
        for m, f in methods.items():
            t0 = time.perf_counter()
            w = f(cov, corr, rng)
            build_time[m].append(time.perf_counter() - t0)
            out[m].append(np.log1p(simple @ w))
            w_full = pd.Series(w, index=cols)
            if prev[m] is not None:
                a, b = w_full.align(prev[m], fill_value=0.0)
                turnover[m].append(0.5 * float(np.abs(a - b).sum()))
            prev[m] = w_full
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
