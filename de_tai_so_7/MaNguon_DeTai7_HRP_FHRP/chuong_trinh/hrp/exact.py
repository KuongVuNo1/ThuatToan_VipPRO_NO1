"""Thuat toan dung (exact) de doi chieu chat luong voi HRP/FHRP tren bai toan nho.

1. held_karp_path : sap thu tu toi uu (seriation) = duong Hamilton ngan nhat
                    tren ma tran khoang cach tuong quan. Quy hoach dong
                    Held-Karp, O(n^2 2^n) thoi gian, O(n 2^n) bo nho.
2. brute_force_orders : duyet toan bo n! hoan vi, ghi nhan phuong sai danh
                    muc sau chia doi de quy -> phan vi cua HRP/FHRP trong
                    khong gian tim kiem; dem so nghiem phan biet (Bang 4 bai bao).
"""
from __future__ import annotations

import itertools

import numpy as np

from .algorithms import recursive_bisection


def adjacent_cost(order, dist: np.ndarray) -> float:
    """Tong khoang cach giua cac tai san ke nhau trong thu tu (cang nho cang tot)."""
    o = list(order)
    return float(sum(dist[o[k], o[k + 1]] for k in range(len(o) - 1)))


def held_karp_path(dist: np.ndarray) -> tuple[float, list[int]]:
    """Duong Hamilton (khong co dinh diem dau/cuoi) co tong khoang cach nho nhat.

    dp[S, j] = chi phi nho nhat cua duong di qua dung tap dinh S va ket thuc tai j.
    dp[S, j] = min_{i in S\\{j}} dp[S\\{j}, i] + d(i, j)       (cong thuc truy hoi)
    """
    n = dist.shape[0]
    full = 1 << n
    INF = np.inf
    dp = np.full((full, n), INF)
    parent = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = 0.0
    # duyet tap theo so phan tu tang dan (thu tu tu nhien cua mask da dam bao S\{j} < S)
    for mask in range(1, full):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        members = [i for i in range(n) if mask >> i & 1]
        for j in range(n):
            if mask >> j & 1:
                continue
            nm = mask | (1 << j)
            cand = row[members] + dist[members, j]
            k = int(np.argmin(cand))
            if cand[k] < dp[nm, j]:
                dp[nm, j] = cand[k]
                parent[nm, j] = members[k]
    last = int(np.argmin(dp[full - 1]))
    best = float(dp[full - 1, last])
    path, mask, j = [], full - 1, last
    while j != -1:
        path.append(j)
        pj = int(parent[mask, j])
        mask ^= 1 << j
        j = pj
    return best, path[::-1]


def brute_force_orders(cov: np.ndarray, decimals: int = 10):
    """Duyet n! thu tu: tra ve (mang phuong sai danh muc, so nghiem phan biet).

    Chi dung cho n <= 9 (9! = 362.880 hoan vi).
    """
    n = cov.shape[0]
    variances = []
    distinct = set()
    for perm in itertools.permutations(range(n)):
        w = recursive_bisection(cov, list(perm))
        variances.append(float(w @ cov @ w))
        distinct.add(tuple(np.round(w, decimals)))
    return np.array(variances), len(distinct)
