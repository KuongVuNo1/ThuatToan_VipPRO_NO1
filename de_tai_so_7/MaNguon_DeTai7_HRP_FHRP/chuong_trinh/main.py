"""Chay toan bo thuc nghiem cua de tai so 7 va ghi ket qua vao thu muc ket_qua/.

    python main.py            # day du (khoang 10-20 phut tuy may)
    python main.py --nhanh    # ban rut gon de kiem tra (khoang 1-2 phut)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hrp import plots
from hrp.algorithms import PORTFOLIO_METHODS
from hrp.data import load_prices, load_returns
from hrp.experiments import (tn1_kiem_chung_ly_thuyet, tn2_thoi_gian, tn3_phuong_sai_tong_hop,
                             tn4_vn30_cuon, tn5_vn30_tap_con, tn6_chat_luong_thu_tu,
                             vn30_synthetic_params)

OUT = Path(__file__).resolve().parent / "ket_qua"


def main(nhanh: bool = False):
    OUT.mkdir(exist_ok=True)
    t_all = time.perf_counter()
    prices, r = load_prices(), load_returns()
    info = {"python": platform.python_version(), "may": platform.processor() or platform.machine(),
            "he_dieu_hanh": platform.platform(), "so_tuan_loi_suat": len(r),
            "tu_ngay": str(r.index[0].date()), "den_ngay": str(r.index[-1].date()),
            "so_ma": r.shape[1], "tham_so_tong_hop": vn30_synthetic_params(r), "che_do_nhanh": nhanh}
    print("Du lieu:", info)

    print("TN1 - kiem chung ly thuyet ...")
    inv, cnt = tn1_kiem_chung_ly_thuyet(r, n_max=7 if nhanh else 8, n_datasets=2 if nhanh else 5)
    print("TN2 - thoi gian ...")
    t2 = tn2_thoi_gian(r, sizes=(10, 50, 100, 200, 400, 800) if nhanh
                       else (10, 50, 100, 200, 400, 800, 1000, 2000, 3000))
    print("TN3 - phuong sai tong hop ...")
    t3 = tn3_phuong_sai_tong_hop(r, sizes=(10, 50, 100, 200) if nhanh else (10, 50, 100, 200, 400, 800),
                                 reps=5 if nhanh else 20)
    print("TN4 - VN30 kiem dinh cuon ...")
    port, perf, mc = tn4_vn30_cuon(r, F=52, B=4, n_random=20 if nhanh else 200)
    port2, perf2, mc2 = tn4_vn30_cuon(r, F=104, B=13, n_random=20 if nhanh else 200)
    print("TN5 - VN30 tap con ...")
    t5 = tn5_vn30_tap_con(r, draws=5 if nhanh else 30)
    print("TN6 - chat luong thu tu so voi nghiem dung ...")
    hk, hk_sum, bf = tn6_chat_luong_thu_tu(r, draws_hk=5 if nhanh else 30,
                                           n_bf=7 if nhanh else 8, draws_bf=3 if nhanh else 10)

    cov, corr = r.cov().values, r.corr().values
    W = pd.DataFrame({m: f(cov, corr, np.random.default_rng(0))
                      for m, f in PORTFOLIO_METHODS.items()}, index=r.columns)

    mc_sum = pd.DataFrame({
        "F52_B4": [mc.mean(), mc.quantile(0.05), mc.quantile(0.95),
                   100 * (mc < perf.loc["HRP", "DoLechChuanNam_%"]).mean(),
                   100 * (mc < perf.loc["FHRP", "DoLechChuanNam_%"]).mean()],
        "F104_B13": [mc2.mean(), mc2.quantile(0.05), mc2.quantile(0.95),
                     100 * (mc2 < perf2.loc["HRP", "DoLechChuanNam_%"]).mean(),
                     100 * (mc2 < perf2.loc["FHRP", "DoLechChuanNam_%"]).mean()]},
        index=["DLC_TB_%", "DLC_phan_vi_5%", "DLC_phan_vi_95%",
               "%_lan_Random_tot_hon_HRP", "%_lan_Random_tot_hon_FHRP"])

    tables = {"TN1_BatBien": inv, "TN1_SoNghiem": cnt, "TN2_ThoiGian": t2,
              "TN3_PhuongSaiTongHop": t3, "TN4_VN30_F52_B4": perf, "TN4_VN30_F104_B13": perf2,
              "TN4_MonteCarlo": mc_sum, "TN5_TapCon": t5, "TN6_HeldKarp": hk_sum,
              "TN6_DuyetToanBo": bf.describe().T, "TrongSo_ToanMau": W}
    with pd.ExcelWriter(OUT / "ket_qua_thuc_nghiem.xlsx") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31])
        prices.to_excel(xw, sheet_name="DuLieu_GiaTuan")
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")

    plots.dendrogram_plot(r, OUT)
    plots.timing_plot(t2, OUT)
    plots.relvar_plot(t3, OUT)
    plots.wealth_plot(port, OUT)
    plots.montecarlo_plot(mc, perf, OUT)
    plots.weights_plot(W, OUT)
    plots.ordering_plot(hk_sum, OUT)

    info["tong_thoi_gian_chay_giay"] = round(time.perf_counter() - t_all, 1)
    (OUT / "thong_tin_lan_chay.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    pd.set_option("display.width", 200)
    for name, df in tables.items():
        print(f"\n===== {name} =====\n{df.round(4).to_string()}")
    print(f"\nXong sau {info['tong_thoi_gian_chay_giay']} giay. Ket qua trong: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhanh", action="store_true", help="che do rut gon de kiem tra nhanh")
    main(**vars(ap.parse_args()))
