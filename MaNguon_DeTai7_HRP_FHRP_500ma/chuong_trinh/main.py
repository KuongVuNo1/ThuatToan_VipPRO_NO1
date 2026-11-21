"""Chay toan bo thuc nghiem cua de tai so 7 tren bo du lieu 500 co phieu HOSE + HNX.

    python main.py            # day du (khoang 30-60 phut tuy may)
    python main.py --nhanh    # ban rut gon de kiem tra (vai phut)

Du lieu: data/hose_hnx_500_weekly_adjclose.csv (tao bang: python tai_du_lieu_hose_hnx.py
roi python chon_500_ma.py). Ket qua ghi vao thu muc ket_qua/.
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
from hrp.data import load_meta, load_prices, load_returns, vn30_codes, window
from hrp.experiments import (synthetic_params, tn1_kiem_chung_ly_thuyet, tn2_thoi_gian,
                             tn2_thoi_gian_that, tn3_phuong_sai_tong_hop, tn4_kiem_dinh_cuon,
                             tn5_tap_con, tn6_chat_luong_thu_tu)

OUT = Path(__file__).resolve().parent / "ket_qua"


def mc_summary(mc, perf):
    return [mc.mean(), mc.quantile(0.05), mc.quantile(0.95),
            100 * (mc < perf.loc["HRP", "DoLechChuanNam_%"]).mean(),
            100 * (mc < perf.loc["FHRP", "DoLechChuanNam_%"]).mean()]


def main(nhanh: bool = False):
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*"):
        if f.is_file():
            f.unlink()
    t_all = time.perf_counter()
    prices, r, meta, vn30 = load_prices(), load_returns(), load_meta(), vn30_codes()
    w52 = window(r, 52)
    c = w52.corr().values
    off = c[np.triu_indices_from(c, 1)]
    info = {"python": platform.python_version(), "may": platform.processor() or platform.machine(),
            "he_dieu_hanh": platform.platform(), "so_ma": r.shape[1], "so_tuan_loi_suat": len(r),
            "tu_ngay": str(r.index[0].date()), "den_ngay": str(r.index[-1].date()),
            "so_ma_du_lieu_day_du_10_nam": int(r.notna().all().sum()),
            "so_ma_trong_cua_so_52_tuan_cuoi": int(w52.shape[1]),
            "tuong_quan_52_tuan_cuoi": {"trung_binh": float(off.mean()), "nho_nhat": float(off.min()),
                                        "lon_nhat": float(off.max())},
            "tham_so_tong_hop": synthetic_params(r), "che_do_nhanh": nhanh}
    print("Du lieu:", json.dumps(info, ensure_ascii=False, default=str))

    print("TN1 - kiem chung ly thuyet ...")
    inv, cnt = tn1_kiem_chung_ly_thuyet(r, n_max=7 if nhanh else 8, n_datasets=2 if nhanh else 5)
    print("TN2 - thoi gian ...")
    t2 = tn2_thoi_gian(r, sizes=(10, 50, 100, 200, 400, 800) if nhanh
                       else (10, 50, 100, 200, 400, 800, 1000, 2000, 3000))
    t2r = tn2_thoi_gian_that(r, repeat=3 if nhanh else 7)
    print("TN3 - phuong sai tong hop ...")
    t3 = tn3_phuong_sai_tong_hop(r, sizes=(10, 50, 100, 200) if nhanh else (10, 50, 100, 200, 400, 800),
                                 reps=5 if nhanh else 20)
    print("TN4 - kiem dinh cuon 500 ma ...")
    port, perf, mc, n_ma = tn4_kiem_dinh_cuon(r, F=52, B=4, n_random=5 if nhanh else 100,
                                              with_minvar=not nhanh)
    port2, perf2, mc2, n_ma2 = tn4_kiem_dinh_cuon(r, F=104, B=13, n_random=5 if nhanh else 100,
                                                  with_minvar=not nhanh)
    print("TN4b - kiem dinh cuon rieng ro VN30 ...")
    port3, perf3, mc3, _ = tn4_kiem_dinh_cuon(r[vn30], F=52, B=4, n_random=5 if nhanh else 100,
                                              with_minvar=not nhanh)
    print("TN5 - tap con ...")
    t5 = tn5_tap_con(r, vn30, sizes=(50, 100, 500) if nhanh else (50, 100, 200, 300, 400, 500),
                     draws=2 if nhanh else 10)
    print("TN6 - chat luong thu tu so voi nghiem dung ...")
    hk, hk_sum, bf = tn6_chat_luong_thu_tu(r, draws_hk=5 if nhanh else 30,
                                           n_bf=7 if nhanh else 8, draws_bf=3 if nhanh else 10)

    cov, corr = w52.cov().values, w52.corr().values
    W = pd.DataFrame({m: f(cov, corr, np.random.default_rng(0))
                      for m, f in PORTFOLIO_METHODS.items()}, index=w52.columns)
    W_sum = pd.DataFrame({"So_ma_hieu_dung": 1 / (W ** 2).sum(),
                          "Ty_trong_lon_nhat_%": 100 * W.max(),
                          "Ma_ty_trong_lon_nhat": W.idxmax(),
                          "So_ma_ty_trong_0": (W < 1e-6).sum(),
                          "Ty_trong_10_ma_lon_nhat_%": 100 * W.apply(lambda s: s.nlargest(10).sum())})

    mc_sum = pd.DataFrame({"500ma_F52_B4": mc_summary(mc, perf), "500ma_F104_B13": mc_summary(mc2, perf2),
                           "VN30_F52_B4": mc_summary(mc3, perf3)},
                          index=["DLC_TB_%", "DLC_phan_vi_5%", "DLC_phan_vi_95%",
                                 "%_lan_Random_tot_hon_HRP", "%_lan_Random_tot_hon_FHRP"])
    so_ma = pd.DataFrame({"F52_B4": n_ma.describe(), "F104_B13": n_ma2.describe()})

    tables = {"TN1_BatBien": inv, "TN1_SoNghiem": cnt, "TN2_ThoiGian_TongHop": t2,
              "TN2_ThoiGian_DuLieuThat": t2r, "TN3_PhuongSaiTongHop": t3,
              "TN4_500ma_F52_B4": perf, "TN4_500ma_F104_B13": perf2, "TN4_VN30_F52_B4": perf3,
              "TN4_MonteCarlo": mc_sum, "TN4_SoMaTheoThoiGian": so_ma, "TN5_TapCon": t5,
              "TN6_HeldKarp": hk_sum, "TN6_DuyetToanBo": bf.describe().T,
              "TrongSo_TomTat": W_sum, "TrongSo_52TuanCuoi": W}
    with pd.ExcelWriter(OUT / "ket_qua_thuc_nghiem.xlsx") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31])
        meta.to_excel(xw, sheet_name="DanhMuc_500_ma")
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")
    n_ma.to_csv(OUT / "TN4_SoMa_F52_B4_chi_tiet.csv", encoding="utf-8-sig")

    plots.dendrogram_plot(window(r, 52, vn30), OUT,
                          "Cây phân cụm liên kết đơn của 30 mã VN30 (52 tuần đến 13/11/2026)")
    plots.timing_plot(t2, OUT, real=t2r)
    plots.relvar_plot(t3, OUT)
    plots.wealth_plot(port, n_ma, OUT)
    plots.montecarlo_plot(mc, perf, OUT)
    plots.weights_plot(W, OUT)
    plots.ordering_plot(hk_sum, OUT)

    info["tong_thoi_gian_chay_giay"] = round(time.perf_counter() - t_all, 1)
    (OUT / "thong_tin_lan_chay.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    pd.set_option("display.width", 220)
    for name, df in tables.items():
        if name != "TrongSo_52TuanCuoi":
            print(f"\n===== {name} =====\n{df.round(4).to_string()}")
    print(f"\nXong sau {info['tong_thoi_gian_chay_giay']} giay. Ket qua trong: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhanh", action="store_true", help="che do rut gon de kiem tra nhanh")
    main(**vars(ap.parse_args()))
