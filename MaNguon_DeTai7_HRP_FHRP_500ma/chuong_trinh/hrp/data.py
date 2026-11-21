"""Doc du lieu chinh (500 co phieu HOSE + HNX, 10 nam) va sinh du lieu tong hop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "hose_hnx_500_weekly_adjclose.csv"    # tao bang: python chon_500_ma.py
META_FILE = DATA_DIR / "hose_hnx_500_danh_muc_ma.csv"
VN30_FILE = DATA_DIR / "vn30_hien_hanh.csv"


def load_prices(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Gia dong cua dieu chinh theo tuan (nghin dong), cot = ma, dong = ngay thu Sau cuoi tuan.
    Giu o trong (NaN) truoc ngay co du lieu dau tien cua tung ma; cac tuan tam ngung giao dich
    ben trong lich su duoc lap bang gia tuan truoc (forward-fill, loi suat tuan do = 0)."""
    p = pd.read_csv(path, parse_dates=["week_end"], index_col="week_end").sort_index()
    return p.ffill()


def load_returns(path: Path | str = DATA_FILE, max_abs: float = 1.0) -> pd.DataFrame:
    """Loi suat log tuan r_t = ln(P_t / P_{t-1}); NaN = ma chua co du lieu o tuan do.
    Bo cac gia tri |r| > max_abs (bien dong > 170%/tuan - thuong la loi dieu chinh gia)."""
    r = np.log(load_prices(path)).diff().iloc[1:]
    return r.mask(r.abs() > max_abs)


def load_meta(path: Path | str = META_FILE) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", index_col="ma")


def vn30_codes(path: Path | str = VN30_FILE) -> list[str]:
    return list(pd.read_csv(path)["ma"])


def window(returns: pd.DataFrame, F: int = 52, cols=None) -> pd.DataFrame:
    """F tuan cuoi cung, chi giu cac ma co du lieu day du va phuong sai > 0 trong cua so.
    Dung cho cac phan tich 'tai mot thoi diem' (TN1, TN2 du lieu that, TN6, trong so)."""
    w = returns.iloc[-F:] if cols is None else returns[cols].iloc[-F:]
    return w.loc[:, w.notna().all().values & (w.std() > 0).values]


def nearest_psd_corr(c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Dua ma tran ve dang tuong quan nua xac dinh duong (cat tri rieng am)."""
    vals, vecs = np.linalg.eigh((c + c.T) / 2)
    c2 = (vecs * np.clip(vals, eps, None)) @ vecs.T
    d = np.sqrt(np.diag(c2))
    c2 = c2 / np.outer(d, d)
    np.fill_diagonal(c2, 1.0)
    return c2


def synthetic_corr_cov(n: int, cmin: float, cmax: float, vmax: float,
                       rng: np.random.Generator, psd: bool = True):
    """Sinh du lieu tong hop theo muc 4.3 cua bai bao.

    - rho_ij ~ U(cmin, cmax), duong cheo = 1
    - phuong sai sigma_i^2 ~ U(0.1*vmax, vmax);  V = D C D
    - psd=True: hieu chinh ve ma tran nua xac dinh duong de phuong sai danh muc co nghia
      (bai bao khong neu buoc nay - day la gia dinh cua nhom).
    """
    u = rng.uniform(cmin, cmax, size=(n, n))
    c = np.triu(u, 1)
    c = c + c.T
    np.fill_diagonal(c, 1.0)
    if psd:
        c = nearest_psd_corr(c)
    sd = np.sqrt(rng.uniform(0.1 * vmax, vmax, size=n))
    cov = c * np.outer(sd, sd)
    return c, cov
