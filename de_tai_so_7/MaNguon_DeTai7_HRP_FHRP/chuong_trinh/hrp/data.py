"""Doc du lieu VN30 va sinh du lieu tong hop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "vn30_weekly_adjclose_2021_2026.csv"


def load_prices(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Gia dong cua dieu chinh theo tuan (nghin VND), bo cac tuan co o trong."""
    p = pd.read_csv(path, parse_dates=["week_end"], index_col="week_end")
    return p.dropna()


def load_returns(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Loi suat log theo tuan r_t = ln(P_t / P_{t-1})."""
    p = load_prices(path)
    return np.log(p).diff().dropna()


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
