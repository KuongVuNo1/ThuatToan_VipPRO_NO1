"""Cac thuat toan phan bo danh muc.

Huong giai 1 - HRP (Lopez de Prado, 2016):
    (1) phan cum phan cap lien ket don  -> tham lam, tuong duong Kruskal/MST, O(n^2)
    (2) tua cheo hoa (thu tu la cua cay) -> O(n)
    (3) chia doi de quy (CHIA DE TRI)    -> O(n^2) phep tinh dang toan phuong
Huong giai 2 - FHRP (Salas-Molina & Nin, 2026):
    thay (1)+(2) bang xep hang theo tuong quan lon nhat -> sap xep O(n log n)
    (gia su ma tran tuong quan da co san, nhu bai bao).
Doi chung: Random HRP (thu tu ngau nhien), IVP, EW, MinVar (Markowitz).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# --------------------------------------------------------------------- tien ich
def corr_distance(corr: np.ndarray) -> np.ndarray:
    """Khoang cach tuong quan d_ij = sqrt(0.5 * (1 - rho_ij)), nam trong [0, 1]."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def ivp(cov: np.ndarray) -> np.ndarray:
    """Danh muc nghich dao phuong sai: w_i = (1/sigma_i^2) / sum_j (1/sigma_j^2)."""
    inv = 1.0 / np.diag(cov)
    return inv / inv.sum()


def equal_weight(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Chieu Euclid len don hinh {w >= 0, sum w = 1} (Duchi va cs., 2008), O(n log n)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0.0)


def _min_variance_fista(cov: np.ndarray, iters: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """Gradient chieu tang toc (FISTA) cho QP loi min w'Vw tren don hinh - nhanh voi n lon."""
    n = cov.shape[0]
    L = 2.0 * np.linalg.eigvalsh(cov)[-1]
    w = y = np.full(n, 1.0 / n)
    t = 1.0
    for _ in range(iters):
        w_new = _project_simplex(y - (2.0 * cov @ y) / L)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def min_variance(cov: np.ndarray) -> np.ndarray:
    """Markowitz phuong sai nho nhat, chi mua: min w'Vw, sum w = 1, w >= 0 (QP loi).
    n <= 60: SLSQP; n > 60: FISTA (SLSQP qua cham khi n ~ 400)."""
    n = cov.shape[0]
    if n > 60:
        return _min_variance_fista(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n),
                   jac=lambda w: 2.0 * cov @ w, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


# --------------------------------------------------------------------- buoc sap thu tu
def order_hrp(corr: np.ndarray) -> list[int]:
    """Buoc 1 + 2 cua HRP.

    Phan cum lien ket don tren khoang cach tuong quan, roi lay thu tu la cua cay
    phan cap (chinh la ket qua cua buoc tua cheo hoa). Lien ket don chinh la
    thuat toan Kruskal dung som de lay cay khung nho nhat (Kleinberg-Tardos 4.7).
    """
    dist = squareform(corr_distance(corr), checks=False)
    link = linkage(dist, method="single")
    return [int(i) for i in leaves_list(link)]


def order_fhrp(corr: np.ndarray) -> list[int]:
    """Buoc sap thu tu cua FHRP - dung theo muc 4.1 cua bai bao.

    1. Voi moi tai san i: v_i = max_{j != i} rho_ij, ghi nho doi tac j(i).
    2. Sap xep v tang dan.                                  -> O(n log n)
    3. Duyet tu phan tu cuoi (tuong quan cao nhat) ve dau:
       them i neu chua co, them doi tac j(i) neu chua co.    -> O(n)
    """
    c = np.array(corr, dtype=float, copy=True)
    np.fill_diagonal(c, -np.inf)
    partner = np.argmax(c, axis=1)          # doi tac tuong quan manh nhat
    v = c[np.arange(len(c)), partner]       # v_i = max_{j != i} rho_ij
    sorted_idx = np.argsort(v, kind="mergesort")   # tang dan
    in_list = np.zeros(len(c), dtype=bool)
    order: list[int] = []
    for k in range(len(sorted_idx) - 1, -1, -1):
        i = int(sorted_idx[k])
        j = int(partner[i])
        for a in (i, j):
            if not in_list[a]:
                in_list[a] = True
                order.append(a)
        if len(order) == len(c):
            break
    return order


def order_random(n: int, rng: np.random.Generator) -> list[int]:
    """Doi chung ngau nhien hoa: thu tu tai san la mot hoan vi ngau nhien."""
    return [int(i) for i in rng.permutation(n)]


# --------------------------------------------------------------------- chia de tri
def _cluster_var(cov: np.ndarray, idx: np.ndarray) -> float:
    """Phuong sai cum (cong thuc (2)-(3) cua bai bao): w~' V w~ voi w~ = IVP trong cum."""
    sub = cov[np.ix_(idx, idx)]
    w = 1.0 / np.diag(sub)
    w /= w.sum()
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Buoc 3 - CHIA DE TRI (muc 2.3 cua bai bao).

    CHIA: cat danh sach co thu tu thanh hai nua, |L1| = int(|L|/2).
    TRI : phuong sai cum V1, V2; he so alpha = 1 - V1 / (V1 + V2).
    HOP : nhan trong so nua trai voi alpha, nua phai voi (1 - alpha), de quy tiep.
    Ket qua: trong so khong am, tong bang 1, tra ve theo chi so tai san goc.
    """
    n = len(order)
    w = np.ones(cov.shape[0])
    stack = [np.asarray(order, dtype=int)]
    while stack:
        items = stack.pop()
        if len(items) <= 1:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        v1, v2 = _cluster_var(cov, left), _cluster_var(cov, right)
        alpha = 1.0 - v1 / (v1 + v2)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack.append(left)
        stack.append(right)
    mask = np.zeros(cov.shape[0], dtype=bool)
    mask[np.asarray(order, dtype=int)] = True
    w[~mask] = 0.0
    return w / w.sum() if n else w


# --------------------------------------------------------------------- danh muc hoan chinh
def hrp(cov: np.ndarray, corr: np.ndarray, rng=None) -> np.ndarray:
    return recursive_bisection(cov, order_hrp(corr))


def fhrp(cov: np.ndarray, corr: np.ndarray, rng=None) -> np.ndarray:
    return recursive_bisection(cov, order_fhrp(corr))


def random_hrp(cov: np.ndarray, corr: np.ndarray, rng=None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    return recursive_bisection(cov, order_random(cov.shape[0], rng))


PORTFOLIO_METHODS = {
    "HRP": hrp,
    "FHRP": fhrp,
    "RandomHRP": random_hrp,
    "IVP": lambda cov, corr, rng=None: ivp(cov),
    "EW": lambda cov, corr, rng=None: equal_weight(cov),
    "MinVar": lambda cov, corr, rng=None: min_variance(cov),
}
