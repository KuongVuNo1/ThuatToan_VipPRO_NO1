"""
HRP va FHRP cho ro VN30 - de tai 7 (chia de tri)
Can cu: Salas-Molina & Nin (2026), "Fast hierarchical risk parity methods for
portfolio selection", Annals of Operations Research (CC BY 4.0).

Chay:  python hrp_vn30.py
Can:   numpy, pandas, scipy, matplotlib, openpyxl
Dau vao: vn30_weekly_adjclose_2021_2026.csv (gia dong cua dieu chinh theo tuan, nghin VND)
"""
import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "vn30_weekly_adjclose_2021_2026.csv"
TRAIN_END = "2024-12-31"          # mau trong: 26/03/2021 - 27/12/2024
COLORS = {"HRP": "#2a78d6", "FHRP": "#eb6834", "IVP": "#1baf7a",
          "EW": "#eda100", "MinVar": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# ---------------------------------------------------------------- du lieu
def load_returns(path=DATA):
    p = pd.read_csv(path, parse_dates=["week_end"], index_col="week_end")
    p = p.dropna()                     # bo 11 tuan dau (SSB niem yet 24/03/2021)
    r = np.log(p).diff().dropna()      # loi suat log theo tuan
    return p, r


# ---------------------------------------------------------------- HRP (Lopez de Prado 2016)
def corr_distance(corr):
    """d_ij = sqrt((1 - rho_ij) / 2): 0 khi tuong quan hoan hao, 1 khi nguoc dau."""
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def cluster_order(corr):
    """Buoc 1 + 2: phan cum lien ket don, roi tua cheo hoa = thu tu la cua cay. O(n^2)."""
    link = linkage(squareform(corr_distance(corr), checks=False), method="single")
    return list(leaves_list(link)), link


def ivp(cov):
    """Trong so nghich dao phuong sai."""
    w = 1.0 / np.diag(cov)
    return w / w.sum()


def cluster_var(cov, idx):
    """Phuong sai cua mot cum khi ben trong cum chia theo IVP."""
    sub = cov[np.ix_(idx, idx)]
    w = ivp(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov, order):
    """Buoc 3 - CHIA DE TRI: chia doi danh sach, chia von theo nghich dao phuong sai
    cua tung nua, de quy cho den khi moi nhom con 1 ma."""
    w = pd.Series(1.0, index=order)
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) <= 1:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        v1, v2 = cluster_var(cov, left), cluster_var(cov, right)
        alpha = 1.0 - v1 / (v1 + v2)           # nua rui ro thap nhan nhieu von hon
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack += [left, right]
    return w.sort_index().values


def hrp(cov, corr):
    order, _ = cluster_order(corr)
    return recursive_bisection(cov, order)


def fhrp(cov, corr):
    """FHRP: thay phan cum bang xep hang theo tuong quan lon nhat ngoai duong cheo.
    Sap xep O(n log n). Chieu sap xep (tang dan) la lua chon cua nguoi soan."""
    c = corr.copy()
    np.fill_diagonal(c, -np.inf)
    score = c.max(axis=1)
    order = list(np.argsort(score))
    return recursive_bisection(cov, order)


# ---------------------------------------------------------------- doi chung
def min_variance(cov):
    """Markowitz phuong sai nho nhat, chi mua (w >= 0, tong = 1)."""
    n = cov.shape[0]
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n),
                   jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


def equal_weight(cov):
    n = cov.shape[0]
    return np.full(n, 1 / n)


# ---------------------------------------------------------------- danh gia
def metrics(rets):
    """Chi so ngoai mau tu chuoi loi suat log tuan cua danh muc."""
    simple = np.expm1(rets)
    wealth = (1 + simple).cumprod()
    dd = wealth / wealth.cummax() - 1
    ann_ret = wealth.iloc[-1] ** (52 / len(rets)) - 1
    ann_vol = simple.std() * np.sqrt(52)
    return {"LoiSuatNam_%": 100 * ann_ret, "DoLechChuanNam_%": 100 * ann_vol,
            "Sharpe(rf=0)": ann_ret / ann_vol, "SutGiamToiDa_%": 100 * dd.min()}


def _timed(f, *a):
    t0 = time.perf_counter(); f(*a); return time.perf_counter() - t0


def main():
    prices, r = load_returns()
    tickers = list(r.columns)
    train, test = r.loc[:TRAIN_END], r.loc[TRAIN_END:].iloc[1:]
    cov, corr = train.cov().values, train.corr().values
    print(f"Mau trong: {train.index[0].date()} - {train.index[-1].date()} ({len(train)} tuan)")
    print(f"Mau ngoai: {test.index[0].date()} - {test.index[-1].date()} ({len(test)} tuan)")

    methods = {"HRP": hrp, "FHRP": fhrp, "IVP": lambda c, _: ivp(c),
               "EW": lambda c, _: equal_weight(c), "MinVar": lambda c, _: min_variance(c)}
    W = pd.DataFrame({m: f(cov, corr) for m, f in methods.items()}, index=tickers)

    # ---- 1. kiem dinh tinh: uoc luong tren mau trong, giu nguyen tren mau ngoai
    simple_test = np.expm1(test)
    port = {m: np.log1p(simple_test.values @ W[m].values) for m in W}
    port = pd.DataFrame(port, index=test.index)
    M = pd.DataFrame({m: metrics(port[m]) for m in W}).T
    M["SoMaHieuDung(1/sum w^2)"] = 1 / (W ** 2).sum()
    M["TyTrongLonNhat_%"] = 100 * W.max()

    # ---- 2. kiem dinh cuon: cua so 104 tuan, tai can bang moi 13 tuan
    roll = {m: [] for m in methods}
    idx = []
    for start in range(104, len(r) - 1, 13):
        est = r.iloc[start - 104:start]
        hold = np.expm1(r.iloc[start:start + 13])
        cv, cr = est.cov().values, est.corr().values
        for m, f in methods.items():
            roll[m].append(np.log1p(hold.values @ f(cv, cr)))
        idx.append(hold.index)
    roll = pd.DataFrame({m: np.concatenate(v) for m, v in roll.items()},
                        index=np.concatenate(idx))
    MR = pd.DataFrame({m: metrics(roll[m]) for m in roll}).T

    # ---- 3. do thoi gian buoc SAP THU TU (phan khac nhau giua hai phuong phap):
    #      HRP = phan cum lien ket don O(n^2); FHRP = xep hang + sap xep O(n log n)
    #      (buoc chia doi de quy dung chung nen khong dua vao phep so sanh nay)
    def order_hrp(cr):
        return cluster_order(cr)[0]

    def order_fhrp(cr):
        c = cr.copy(); np.fill_diagonal(c, -np.inf)
        return list(np.argsort(c.max(axis=1)))

    rng = np.random.default_rng(7)
    timing = []
    for n in [100, 250, 500, 1000, 2000, 4000]:
        X = rng.standard_normal((n + 60, n))
        cr = np.corrcoef(X, rowvar=False)
        for name, f in [("HRP", order_hrp), ("FHRP", order_fhrp)]:
            best = min(_timed(f, cr) for _ in range(3))
            timing.append({"so_ma": n, "phuong_phap": name, "giay": best})
    T = pd.DataFrame(timing).pivot(index="so_ma", columns="phuong_phap", values="giay")

    # ---------------------------------------------------------------- xuat ket qua
    W.round(4).to_csv("ket_qua_trong_so.csv")
    M.round(3).to_csv("ket_qua_ngoai_mau_tinh.csv")
    MR.round(3).to_csv("ket_qua_ngoai_mau_cuon.csv")
    T.to_csv("ket_qua_thoi_gian.csv")
    with pd.ExcelWriter("HRP_VN30_ket_qua.xlsx") as xw:
        prices.to_excel(xw, sheet_name="GiaTuan")
        r.round(6).to_excel(xw, sheet_name="LoiSuatLogTuan")
        W.round(4).to_excel(xw, sheet_name="TrongSo")
        M.round(3).to_excel(xw, sheet_name="NgoaiMau_Tinh")
        MR.round(3).to_excel(xw, sheet_name="NgoaiMau_Cuon")
        T.to_excel(xw, sheet_name="ThoiGian")
    print("\nTrong so (%):\n", (100 * W).round(1).sort_values("HRP", ascending=False).to_string())
    print("\nNgoai mau - tinh:\n", M.round(2).to_string())
    print("\nNgoai mau - cuon:\n", MR.round(2).to_string())
    print("\nThoi gian (giay):\n", T.round(4).to_string())

    charts(train, W, port, T, tickers)


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(train, W, port, T, tickers):
    plt.rcParams["font.family"] = "DejaVu Sans"
    # 1. cay phan cum
    _, link = cluster_order(train.corr().values)
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#fcfcfb")
    dendrogram(link, labels=tickers, ax=ax, color_threshold=0,
               above_threshold_color=COLORS["HRP"], leaf_font_size=9)
    style(ax, "Cây phân cụm VN30 theo khoảng cách tương quan (mẫu trong 2021–2024)")
    ax.set_ylabel("Khoảng cách √((1−ρ)/2)", color=INK2)
    fig.tight_layout(); fig.savefig("bieu_do_1_cay_phan_cum.png", dpi=160); plt.close(fig)

    # 2. trong so - HRP so voi MinVar (2 dai, khong chong)
    order = W["HRP"].sort_values(ascending=False).index
    x = np.arange(len(order)); bw = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#fcfcfb")
    for k, m in enumerate(["HRP", "MinVar"]):
        ax.bar(x + (k - 0.5) * bw, 100 * W.loc[order, m], bw - 0.04,
               color=COLORS[m], label=m)
    ax.set_xticks(x, order, rotation=90)
    ax.set_ylabel("Tỷ trọng (%)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2)
    style(ax, "HRP rải vốn cho cả 30 mã; Markowitz MinVar dồn vào ít mã")
    fig.tight_layout(); fig.savefig("bieu_do_2_trong_so.png", dpi=160); plt.close(fig)

    # 3. tai san tich luy ngoai mau
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#fcfcfb")
    for m in ["HRP", "FHRP", "IVP", "EW", "MinVar"]:
        wealth = 100 * np.exp(port[m].cumsum())
        ax.plot(wealth.index, wealth.values, color=COLORS[m], linewidth=2,
                label=f"{m} ({wealth.values[-1]:.0f})")
    ax.set_ylabel("Giá trị (gốc = 100)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, ncol=5, loc="upper left")
    style(ax, "Giá trị danh mục ngoài mẫu, 01/2025 – 10/2026 (trọng số cố định từ mẫu trong)")
    fig.tight_layout(); fig.savefig("bieu_do_3_ngoai_mau.png", dpi=160); plt.close(fig)

    # 4. thoi gian chay
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#fcfcfb")
    for m in ["HRP", "FHRP"]:
        ax.plot(T.index, T[m], color=COLORS[m], linewidth=2, marker="o", markersize=6, label=m)
        ax.annotate(m, (T.index[-1], T[m].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=9, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Số mã n (thang log)", color=INK2)
    ax.set_ylabel("Thời gian (giây, thang log)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2)
    style(ax, "Bước sắp thứ tự: FHRP (xếp hạng + sắp xếp) nhanh hơn HRP (phân cụm) 5–7 lần")
    fig.tight_layout(); fig.savefig("bieu_do_4_thoi_gian.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
