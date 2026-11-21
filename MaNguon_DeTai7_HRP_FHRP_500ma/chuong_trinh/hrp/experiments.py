"""Sau thuc nghiem cua bao cao (muc 4).

Du lieu chinh: 500 co phieu HOSE + HNX (VN30 + 470 ma thanh khoan cao nhat), 518 tuan 11/2016-11/2026.

TN1  Kiem chung ly thuyet: Dinh ly 1 (giao hoan nua danh sach) va Bang 4 (so nghiem phan biet)
TN2  Thoi gian chay theo n: du lieu tong hop (n = 10..3000) va ma tran that (n = 30..500)
TN3  Phuong sai tuong doi so voi HRP tren du lieu tong hop (Monte Carlo)
TN4  Kiem dinh cuon 10 nam tren 500 ma (F=52,B=4 nhu bai bao; F=104,B=13) + Monte Carlo Random HRP
TN5  Tap con N ma: VN30 va N = 50..500 ma lay ngau nhien
TN6  Chat luong thu tu so voi nghiem dung (Held-Karp, duyet n!)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .algorithms import (PORTFOLIO_METHODS, corr_distance, order_fhrp, order_hrp,
                         order_random, recursive_bisection, ivp)
from .backtest import performance, rolling_backtest
from .data import synthetic_corr_cov, window
from .exact import adjacent_cost, brute_force_orders, held_karp_path


def synthetic_params(returns: pd.DataFrame, F: int = 52) -> dict:
    """Tham so du lieu tong hop lay tu F tuan cuoi cua bo 500 ma (bai bao lay tu S&P 500):
    cmin, cmax = phan vi 5% va 95% cua cac he so tuong quan; vmax = phan vi 95% cua phuong sai
    tuan (dung phan vi thay cho gia tri lon nhat de tranh vai ma bien dong bat thuong chi phoi)."""
    returns = window(returns, F)
    c = returns.corr().values
    off = c[np.triu_indices_from(c, 1)]
    return {"cmin": float(np.percentile(off, 5)), "cmax": float(np.percentile(off, 95)),
            "vmax": float(np.percentile(returns.var().values, 95))}


# ------------------------------------------------------------------ TN1
def tn1_kiem_chung_ly_thuyet(returns: pd.DataFrame, n_max: int = 8, n_datasets: int = 5,
                             seed: int = 1, F: int = 52):
    w = window(returns, F)
    w = w.iloc[:, : 2 * (w.shape[1] // 2)]            # so ma chan
    cov, corr = w.cov().values, w.corr().values
    o = order_hrp(corr)
    rows = []
    for n in (len(o), len(o) - 1):          # N chan va N - 1 le
        q = o[:n]
        h = n // 2
        w1 = recursive_bisection(cov, q)
        w2 = recursive_bisection(cov, q[h:] + q[:h])
        rows.append({"So_tai_san_N": n, "Chan/Le": "chan" if n % 2 == 0 else "le",
                     "Chenh_lech_trong_so_lon_nhat": float(np.abs(w1 - w2).max()),
                     "Ket_luan": "bat bien (Dinh ly 1)" if np.abs(w1 - w2).max() < 1e-12
                     else "khong bat bien (He qua 2)"})
    inv = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    p = synthetic_params(returns)
    paper = {3: 3, 4: 3, 5: 30, 6: 90, 7: 315, 8: 315, 9: 11340, 10: 113400}
    cnt = []
    for n in range(3, n_max + 1):
        counts, secs = [], []
        for _ in range(n_datasets):
            _, cv = synthetic_corr_cov(n, p["cmin"], p["cmax"], p["vmax"], rng)
            t0 = time.perf_counter()
            _, d = brute_force_orders(cv)
            secs.append(time.perf_counter() - t0)
            counts.append(d)
        cnt.append({"N": n, "So_hoan_vi_N!": int(np.prod(range(1, n + 1))),
                    "Nghiem_phan_biet_(nhom)": f"{min(counts)}" if min(counts) == max(counts)
                    else f"{min(counts)}-{max(counts)}",
                    "Nghiem_phan_biet_(bai_bao)": paper[n],
                    "Thoi_gian_duyet_TB_giay": float(np.mean(secs))})
    return inv, pd.DataFrame(cnt)


# ------------------------------------------------------------------ TN2
def _time(f, *a, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        f(*a)
        best = min(best, time.perf_counter() - t0)
    return best


def tn2_thoi_gian(returns: pd.DataFrame, sizes=(10, 50, 100, 200, 400, 800, 1000, 2000, 3000),
                  seed: int = 2):
    rng = np.random.default_rng(seed)
    p = synthetic_params(returns)
    rows = []
    for n in sizes:
        corr, cov = synthetic_corr_cov(n, p["cmin"], p["cmax"], p["vmax"], rng, psd=False)
        o_h, o_f = order_hrp(corr), order_fhrp(corr)
        rows.append({
            "n": n,
            "SapThuTu_HRP_s": _time(order_hrp, corr),
            "SapThuTu_FHRP_s": _time(order_fhrp, corr),
            "ChiaDoi_s": _time(recursive_bisection, cov, o_h),
            "ToanBo_HRP_s": _time(lambda: recursive_bisection(cov, order_hrp(corr))),
            "ToanBo_FHRP_s": _time(lambda: recursive_bisection(cov, order_fhrp(corr))),
            "ToanBo_Random_s": _time(lambda: recursive_bisection(cov, order_random(n, rng))),
            "IVP_s": _time(ivp, cov),
        })
        del o_h, o_f
    df = pd.DataFrame(rows).set_index("n")
    df["TangToc_SapThuTu_(HRP/FHRP)"] = df["SapThuTu_HRP_s"] / df["SapThuTu_FHRP_s"]
    df["TangToc_ToanBo_(HRP/FHRP)"] = df["ToanBo_HRP_s"] / df["ToanBo_FHRP_s"]
    return df


def tn2_thoi_gian_that(returns: pd.DataFrame, sizes=(30, 50, 100, 200, 300, 400, 500), F: int = 52,
                       repeat: int = 7):
    """Thoi gian tren ma tran hiep phuong sai/tuong quan THAT cua F tuan cuoi (n ma dau tien
    theo thu tu cot); n lon nhat = so ma co du du lieu trong cua so."""
    last = window(returns, F)
    ns = sorted({min(n, last.shape[1]) for n in sizes})
    rows = []
    for n in ns:
        sub = last.iloc[:, :n]
        cov, corr = sub.cov().values, sub.corr().values
        o_h = order_hrp(corr)
        rows.append({"n": n,
                     "SapThuTu_HRP_s": _time(order_hrp, corr, repeat=repeat),
                     "SapThuTu_FHRP_s": _time(order_fhrp, corr, repeat=repeat),
                     "ChiaDoi_s": _time(recursive_bisection, cov, o_h, repeat=repeat),
                     "ToanBo_HRP_s": _time(lambda: recursive_bisection(cov, order_hrp(corr)), repeat=repeat),
                     "ToanBo_FHRP_s": _time(lambda: recursive_bisection(cov, order_fhrp(corr)), repeat=repeat)})
    df = pd.DataFrame(rows).set_index("n")
    df["TangToc_SapThuTu_(HRP/FHRP)"] = df["SapThuTu_HRP_s"] / df["SapThuTu_FHRP_s"]
    df["TangToc_ToanBo_(HRP/FHRP)"] = df["ToanBo_HRP_s"] / df["ToanBo_FHRP_s"]
    return df


# ------------------------------------------------------------------ TN3
def tn3_phuong_sai_tong_hop(returns: pd.DataFrame, sizes=(10, 50, 100, 200, 400, 800),
                            reps: int = 20, seed: int = 3):
    rng = np.random.default_rng(seed)
    p = synthetic_params(returns)
    rows = []
    for n in sizes:
        rel = {"FHRP": [], "RandomHRP": [], "IVP": []}
        for _ in range(reps):
            corr, cov = synthetic_corr_cov(n, p["cmin"], p["cmax"], p["vmax"], rng)
            var = lambda w: float(w @ cov @ w)
            v_h = var(recursive_bisection(cov, order_hrp(corr)))
            rel["FHRP"].append(var(recursive_bisection(cov, order_fhrp(corr))) / v_h)
            rel["RandomHRP"].append(var(recursive_bisection(cov, order_random(n, rng))) / v_h)
            rel["IVP"].append(var(ivp(cov)) / v_h)
        rows.append({"n": n, **{f"{k}/HRP_TB": float(np.mean(v)) for k, v in rel.items()},
                     **{f"{k}/HRP_DLC": float(np.std(v)) for k, v in rel.items()}})
    return pd.DataFrame(rows).set_index("n")


# ------------------------------------------------------------------ TN4
def tn4_kiem_dinh_cuon(returns: pd.DataFrame, F: int = 52, B: int = 4, n_random: int = 100,
                       seed: int = 4, with_minvar: bool = True):
    """Kiem dinh cuon; tap ma tang dan theo thoi gian (ma moi duoc dua vao khi du F tuan).
    Tra ve (loi suat ngoai mau, bang hieu qua, Monte Carlo Random HRP, so ma theo thoi gian)."""
    methods = {m: f for m, f in PORTFOLIO_METHODS.items() if with_minvar or m != "MinVar"}
    n_log: list = []
    port, t_build, turn = rolling_backtest(returns, methods, F=F, B=B, seed=seed, log_n=n_log)
    perf = pd.DataFrame({m: performance(port[m]) for m in port}).T
    perf["ThoiGianXayDung_ms"] = pd.Series(t_build) * 1e3
    perf["VongQuayTB_%"] = pd.Series(turn) * 100
    # Monte Carlo cho Random HRP: lap lai voi nhieu hat giong
    vols = []
    for s in range(n_random):
        pr, _, _ = rolling_backtest(returns, {"RandomHRP": PORTFOLIO_METHODS["RandomHRP"]},
                                    F=F, B=B, seed=1000 + s)
        vols.append(performance(pr["RandomHRP"])["DoLechChuanNam_%"])
    mc = pd.Series(vols, name="DoLechChuanNam_%_RandomHRP")
    n_ma = pd.Series(dict(n_log), name="so_ma")
    return port, perf, mc, n_ma


# ------------------------------------------------------------------ TN5
def tn5_tap_con(returns: pd.DataFrame, vn30: list, sizes=(50, 100, 200, 300, 400, 500),
                draws: int = 10, F: int = 52, B: int = 4, seed: int = 5):
    """Phuong sai ngoai mau tuong doi so voi HRP: dong dau la ro VN30 (co dinh), cac dong sau la
    N ma lay ngau nhien trong 500 ma (N = 500: toan bo, 1 lan)."""
    rng = np.random.default_rng(seed)
    methods = {k: PORTFOLIO_METHODS[k] for k in ("HRP", "FHRP", "RandomHRP", "IVP")}
    rows = []
    for n in ["VN30", *sizes]:
        rel = {"FHRP": [], "RandomHRP": [], "IVP": []}
        reps = 1 if n == "VN30" or n >= returns.shape[1] else draws
        for d in range(reps):
            cols = vn30 if n == "VN30" else rng.choice(returns.columns, size=n, replace=False)
            port, _, _ = rolling_backtest(returns[cols], methods, F=F, B=B, seed=d, min_assets=10)
            v = np.expm1(port).var()
            for k in rel:
                rel[k].append(v[k] / v["HRP"])
        rows.append({"N": n, "So_lan_lay_mau": len(rel["FHRP"]),
                     **{f"{k}/HRP_TB": float(np.mean(v)) for k, v in rel.items()},
                     **{f"{k}<HRP_%": 100 * float(np.mean(np.array(v) < 1)) for k, v in rel.items()}})
    return pd.DataFrame(rows).set_index("N")


# ------------------------------------------------------------------ TN6
def tn6_chat_luong_thu_tu(returns: pd.DataFrame, n_hk: int = 12, draws_hk: int = 30,
                          n_bf: int = 8, draws_bf: int = 10, seed: int = 6, F: int = 52):
    rng = np.random.default_rng(seed)
    returns = window(returns, F)
    corr_all, cov_all = returns.corr(), returns.cov()
    rows = []
    for _ in range(draws_hk):
        cols = list(rng.choice(returns.columns, size=n_hk, replace=False))
        c = corr_all.loc[cols, cols].values
        D = corr_distance(c)
        t0 = time.perf_counter()
        opt, _ = held_karp_path(D)
        t_hk = time.perf_counter() - t0
        rand = np.mean([adjacent_cost(order_random(n_hk, rng), D) for _ in range(200)])
        rows.append({"Toi_uu_HeldKarp": opt, "HRP": adjacent_cost(order_hrp(c), D),
                     "FHRP": adjacent_cost(order_fhrp(c), D), "Ngau_nhien_TB": rand,
                     "ThoiGian_HeldKarp_s": t_hk})
    hk = pd.DataFrame(rows)
    hk_sum = pd.DataFrame({
        "TongKhoangCach_TB": hk[["Toi_uu_HeldKarp", "HRP", "FHRP", "Ngau_nhien_TB"]].mean(),
        "TySo_so_voi_ToiUu": (hk[["Toi_uu_HeldKarp", "HRP", "FHRP", "Ngau_nhien_TB"]]
                              .div(hk["Toi_uu_HeldKarp"], axis=0).mean()),
    })
    hk_sum.loc["ThoiGian_HeldKarp_TB_s", "TongKhoangCach_TB"] = hk["ThoiGian_HeldKarp_s"].mean()

    rows = []
    for _ in range(draws_bf):
        cols = list(rng.choice(returns.columns, size=n_bf, replace=False))
        cv, c = cov_all.loc[cols, cols].values, corr_all.loc[cols, cols].values
        variances, _ = brute_force_orders(cv)
        v = lambda o: float(recursive_bisection(cv, o) @ cv @ recursive_bisection(cv, o))
        vh, vf = v(order_hrp(c)), v(order_fhrp(c))
        rows.append({"PS_tot_nhat/HRP": variances.min() / vh,
                     "PS_FHRP/HRP": vf / vh,
                     "PhanVi_HRP_%": 100 * float(np.mean(variances < vh - 1e-15)),
                     "PhanVi_FHRP_%": 100 * float(np.mean(variances < vf - 1e-15)),
                     "PS_trung_vi/HRP": float(np.median(variances)) / vh})
    bf = pd.DataFrame(rows)
    return hk, hk_sum, bf


