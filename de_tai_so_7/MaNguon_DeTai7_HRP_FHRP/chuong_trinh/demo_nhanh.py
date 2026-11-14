"""Demo ngan gon (chay < 5 giay): tinh danh muc HRP va FHRP cho 30 ma VN30.

    python demo_nhanh.py
"""
import time

import numpy as np
import pandas as pd

from hrp.algorithms import order_fhrp, order_hrp, recursive_bisection, ivp
from hrp.data import load_returns

r = load_returns()
cov, corr = r.cov().values, r.corr().values
names = list(r.columns)
print(f"Du lieu: {r.shape[1]} ma, {len(r)} tuan loi suat ({r.index[0].date()} -> {r.index[-1].date()})\n")

t0 = time.perf_counter(); o_hrp = order_hrp(corr); t_h = time.perf_counter() - t0
t0 = time.perf_counter(); o_fhrp = order_fhrp(corr); t_f = time.perf_counter() - t0
print("Thu tu HRP  (phan cum lien ket don):", " ".join(names[i] for i in o_hrp))
print("Thu tu FHRP (xep hang tuong quan)  :", " ".join(names[i] for i in o_fhrp))
print(f"Thoi gian sap thu tu: HRP {1e3*t_h:.3f} ms | FHRP {1e3*t_f:.3f} ms\n")

w = pd.DataFrame({"HRP": recursive_bisection(cov, o_hrp),
                  "FHRP": recursive_bisection(cov, o_fhrp),
                  "IVP": ivp(cov)}, index=names)
print("Trong so (%), sap theo HRP:")
print((100 * w).round(2).sort_values("HRP", ascending=False).to_string())
for m in w:
    v = w[m].values @ cov @ w[m].values
    print(f"{m:5s}: do lech chuan nam trong mau = {100*np.sqrt(52*v):.2f}%")
