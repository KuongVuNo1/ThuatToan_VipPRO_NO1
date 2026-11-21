"""Demo ngan gon (chay < 10 giay): danh muc HRP va FHRP cho ro VN30 va cho toan bo 500 ma
(52 tuan gan nhat cua bo du lieu 500 co phieu HOSE + HNX).

    python demo_nhanh.py
"""
import time

import numpy as np
import pandas as pd

from hrp.algorithms import ivp, order_fhrp, order_hrp, recursive_bisection
from hrp.data import load_returns, vn30_codes, window

R = load_returns()
print(f"Du lieu: {R.shape[1]} ma, {len(R)} tuan loi suat ({R.index[0].date()} -> {R.index[-1].date()})\n")

for ten, cols in (("RO VN30", vn30_codes()), ("TOAN BO 500 MA", None)):
    r = window(R, 52, cols)
    cov, corr = r.cov().values, r.corr().values
    names = list(r.columns)
    t0 = time.perf_counter(); o_h = order_hrp(corr); t_h = time.perf_counter() - t0
    t0 = time.perf_counter(); o_f = order_fhrp(corr); t_f = time.perf_counter() - t0
    print(f"===== {ten}: {len(names)} ma du 52 tuan du lieu =====")
    print("Thu tu HRP  (10 ma dau):", " ".join(names[i] for i in o_h[:10]))
    print("Thu tu FHRP (10 ma dau):", " ".join(names[i] for i in o_f[:10]))
    print(f"Thoi gian sap thu tu: HRP {1e3*t_h:.3f} ms | FHRP {1e3*t_f:.3f} ms")
    w = pd.DataFrame({"HRP": recursive_bisection(cov, o_h), "FHRP": recursive_bisection(cov, o_f),
                      "IVP": ivp(cov)}, index=names)
    print("10 ma ty trong lon nhat theo HRP (%):")
    print((100 * w).round(2).sort_values("HRP", ascending=False).head(10).to_string())
    for m in w:
        v = w[m].values @ cov @ w[m].values
        print(f"{m:5s}: do lech chuan nam trong mau = {100*np.sqrt(52*v):.2f}% | "
              f"so ma hieu dung = {1/np.sum(w[m].values**2):.1f}")
    print()
