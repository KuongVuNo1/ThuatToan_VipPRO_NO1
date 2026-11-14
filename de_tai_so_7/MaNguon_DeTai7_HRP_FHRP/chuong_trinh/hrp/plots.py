"""Bieu do cho bao cao (bang mau phan loai da kiem tra kha nang phan biet mau)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from .algorithms import corr_distance

COLORS = {"HRP": "#2a78d6", "FHRP": "#eb6834", "RandomHRP": "#1baf7a", "IVP": "#eda100",
          "EW": "#e87ba4", "MinVar": "#6250d6"}
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200, facecolor=BG)
    plt.close(fig)


def dendrogram_plot(returns, out: Path):
    corr = returns.corr().values
    link = linkage(squareform(corr_distance(corr), checks=False), method="single")
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    dendrogram(link, labels=list(returns.columns), ax=ax, color_threshold=0,
               above_threshold_color=COLORS["HRP"], leaf_font_size=8.5)
    ax.set_ylabel("Khoảng cách √(0,5(1−ρ))", color=INK2)
    _style(ax, "Cây phân cụm liên kết đơn của 30 mã VN30 (04/2021 – 10/2026)")
    _save(fig, out, "hinh_1_cay_phan_cum.png")


def timing_plot(t, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), facecolor=BG)
    ax = axes[0]
    for col, lab, m in [("SapThuTu_HRP_s", "HRP – phân cụm", "HRP"),
                        ("SapThuTu_FHRP_s", "FHRP – xếp hạng + sắp xếp", "FHRP")]:
        ax.plot(t.index, t[col], color=COLORS[m], lw=2, marker="o", ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Số tài sản n (thang log)", color=INK2)
    ax.set_ylabel("Giây (thang log)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    _style(ax, "(a) Bước sắp thứ tự")
    ax = axes[1]
    for col, lab, m in [("ToanBo_HRP_s", "HRP", "HRP"), ("ToanBo_FHRP_s", "FHRP", "FHRP"),
                        ("ToanBo_Random_s", "Random HRP", "RandomHRP"), ("IVP_s", "IVP", "IVP")]:
        ax.plot(t.index, t[col], color=COLORS[m], lw=2, marker="o", ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Số tài sản n (thang log)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    _style(ax, "(b) Toàn bộ thuật toán")
    _save(fig, out, "hinh_2_thoi_gian.png")


def relvar_plot(r3, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=BG)
    ax.axhline(1.0, color=COLORS["HRP"], lw=2, ls="--", label="HRP (mốc = 1)")
    for k, lab in [("FHRP", "FHRP"), ("RandomHRP", "Random HRP")]:
        ax.errorbar(r3.index, r3[f"{k}/HRP_TB"], yerr=r3[f"{k}/HRP_DLC"], color=COLORS[k],
                    lw=2, marker="o", ms=6, capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Số tài sản n (thang log)", color=INK2)
    ax.set_ylabel("Phương sai / phương sai HRP", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    _style(ax, "Phương sai danh mục tương đối so với HRP – dữ liệu tổng hợp")
    _save(fig, out, "hinh_3_phuong_sai_tong_hop.png")


def wealth_plot(port, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    for m in ["HRP", "FHRP", "RandomHRP", "IVP", "EW", "MinVar"]:
        w = 100 * np.exp(port[m].cumsum())
        ax.plot(w.index, w.values, color=COLORS[m], lw=1.8, label=f"{m} ({w.values[-1]:.0f})")
    ax.set_ylabel("Giá trị danh mục (gốc = 100)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, fontsize=9, loc="upper left")
    _style(ax, "Giá trị danh mục ngoài mẫu trên VN30 (cửa sổ 52 tuần, tái cân bằng 4 tuần)")
    _save(fig, out, "hinh_4_gia_tri_ngoai_mau.png")


def montecarlo_plot(mc, perf, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=BG)
    ax.hist(mc.values, bins=25, color=COLORS["RandomHRP"], edgecolor="white",
            label=f"Random HRP ({len(mc)} lần mô phỏng)")
    for m, ls in [("HRP", "-"), ("FHRP", "--")]:
        v = perf.loc[m, "DoLechChuanNam_%"]
        ax.axvline(v, color=COLORS[m], lw=2.2, ls=ls, label=f"{m} = {v:.2f}%")
    ax.set_xlabel("Độ lệch chuẩn năm ngoài mẫu (%)", color=INK2)
    ax.set_ylabel("Số lần", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    _style(ax, "Rủi ro ngoài mẫu: HRP và FHRP so với phân phối của Random HRP")
    _save(fig, out, "hinh_5_monte_carlo.png")


def weights_plot(W, out: Path):
    order = W["HRP"].sort_values(ascending=False).index
    x = np.arange(len(order)); bw = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 4.2), facecolor=BG)
    for k, m in enumerate(["HRP", "FHRP", "MinVar"]):
        ax.bar(x + (k - 1) * bw, 100 * W.loc[order, m], bw - 0.03, color=COLORS[m], label=m)
    ax.set_xticks(x, order, rotation=90)
    ax.set_ylabel("Tỷ trọng (%)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2)
    _style(ax, "Trọng số danh mục trên toàn mẫu VN30: HRP, FHRP và Markowitz MinVar")
    _save(fig, out, "hinh_6_trong_so.png")


def ordering_plot(hk_sum, out: Path):
    rows = ["Toi_uu_HeldKarp", "HRP", "FHRP", "Ngau_nhien_TB"]
    labels = ["Tối ưu\n(Held–Karp)", "HRP", "FHRP", "Ngẫu nhiên"]
    cols = ["#52514e", COLORS["HRP"], COLORS["FHRP"], COLORS["RandomHRP"]]
    vals = hk_sum.loc[rows, "TySo_so_voi_ToiUu"].values
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=BG)
    bars = ax.bar(labels, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center",
                color=INK2, fontsize=9)
    ax.set_ylim(0.9, max(vals) * 1.08)
    ax.set_ylabel("Tổng khoảng cách kề / tối ưu", color=INK2)
    _style(ax, "Chất lượng thứ tự so với nghiệm tối ưu (n = 12, 30 lần lấy mẫu)")
    _save(fig, out, "hinh_7_chat_luong_thu_tu.png")
