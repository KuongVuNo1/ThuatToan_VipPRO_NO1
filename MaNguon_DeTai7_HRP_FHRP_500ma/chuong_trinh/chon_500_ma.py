"""Dung bo du lieu chinh cua de tai: 500 co phieu HOSE + HNX = VN30 + 470 ma thanh khoan cao nhat.

    python chon_500_ma.py                    # doc data/raw_vndirect/*.json (do tai_du_lieu_hose_hnx.py tai ve)

Dau vao (thu muc data/):
  raw_vndirect/<MA>.json        gia ngay 10 nam cua cac ma gia >= 3 nghin dong (668 ma, tai ngay 14/11/2026)
  vu_tru_hose_hnx_<ngay>.csv    708 ma co phieu co giao dich phien 13/11/2026 tren HOSE/HNX va san cua tung ma
  vn30_hien_hanh.csv            30 ma thuoc ro VN30 (API VNDirect, truy van 15/11/2026)

Quy tac chon (ghi vao bao cao, muc 4.1):
  1. Lay du 30 ma VN30 hien hanh.
  2. Cac ma con lai: loai ma co < 20 phien co giao dich trong 52 tuan gan nhat,
     xep theo gia tri giao dich (GTGD) binh quan phien 52 tuan giam dan, lay 470 ma dau.
  3. Gia tuan = gia dong cua DA DIEU CHINH (adClose) phien cuoi moi tuan (tuan ket thuc thu Sau).

Dau ra (thu muc data/):
  hose_hnx_500_weekly_adjclose.csv     gia tuan (nghin dong), o trong truoc ngay co du lieu dau tien
  hose_hnx_500_danh_muc_ma.csv         STT, ma, san, thuoc VN30, gia cuoi, GTGD, hang thanh khoan, ngay dau...
  hose_hnx_500_kiem_tra_chat_luong.txt bao cao kiem tra du lieu
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw_vndirect"
N_TONG, MIN_PHIEN = 500, 20


def week_end(d: str) -> str:
    x = dt.date.fromisoformat(d)
    return (x + dt.timedelta(days=(4 - x.weekday()) % 7)).isoformat()


def doc_csv(path: Path, key: str = "ma") -> dict:
    with path.open(encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def main(raw: Path = RAW, out: Path = DATA, n_tong: int = N_TONG):
    vu_tru = sorted(out.glob("vu_tru_hose_hnx_*.csv"))[-1]          # file moi nhat
    san = {k: v["san"] for k, v in doc_csv(vu_tru).items()}
    vn30 = list(doc_csv(out / "vn30_hien_hanh.csv"))
    files = {f.stem: f for f in raw.glob("*.json")}
    session = max(json.loads(f.read_text(encoding="utf-8"))["end"] for f in list(files.values())[:5])
    one_year = (dt.date.fromisoformat(session) - dt.timedelta(days=364)).isoformat()

    info, weekly = {}, {}
    for c, f in sorted(files.items()):
        rows = [r for r in json.loads(f.read_text(encoding="utf-8"))["rows"] if r.get("adClose") and r["adClose"] > 0]
        if not rows:
            continue
        ly = [r for r in rows if r["date"] >= one_year]
        vals = [r["value"] if r.get("value") else (r["close"] or 0) * 1000 * (r.get("volume") or 0) for r in ly]
        w = {}
        for r in rows:
            w[week_end(r["date"])] = r["adClose"]
        weekly[c] = w
        info[c] = {"ma": c, "san": san.get(c, ""), "thuoc_VN30": "x" if c in vn30 else "",
                   "gia_cuoi_nghin_dong": rows[-1]["close"], "ngay_gia_cuoi": rows[-1]["date"],
                   "gtgd_tb_phien_52tuan_ty_dong": round(sum(vals) / max(len(vals), 1) / 1e9, 3),
                   "so_phien_co_gd_52tuan": sum(1 for r in ly if (r.get("volume") or 0) > 0),
                   "ngay_du_lieu_dau": rows[0]["date"], "so_tuan_du_lieu": len(w)}

    thieu_vn30 = [c for c in vn30 if c not in info]
    if thieu_vn30:
        raise SystemExit(f"Thieu du lieu cua ma VN30: {thieu_vn30}")
    xep = sorted(info.values(), key=lambda x: -x["gtgd_tb_phien_52tuan_ty_dong"])
    for k, x in enumerate(xep, 1):
        x["hang_thanh_khoan"] = k
    loai = [x["ma"] for x in xep if x["so_phien_co_gd_52tuan"] < MIN_PHIEN and x["ma"] not in vn30]
    con_lai = [x for x in xep if x["ma"] not in vn30 and x["ma"] not in loai][: n_tong - len(vn30)]
    chon = [info[c] for c in vn30] + con_lai
    chon.sort(key=lambda x: (x["thuoc_VN30"] != "x", x["hang_thanh_khoan"]))
    for k, x in enumerate(chon, 1):
        x["stt"] = k

    codes = sorted(x["ma"] for x in chon)
    weeks = sorted({w for c in codes for w in weekly[c]})
    with (out / f"hose_hnx_{n_tong}_weekly_adjclose.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["week_end"] + codes)
        for wk in weeks:
            wr.writerow([wk] + [weekly[c].get(wk, "") for c in codes])
    cols = ["stt", "ma", "san", "thuoc_VN30", "hang_thanh_khoan", "gtgd_tb_phien_52tuan_ty_dong",
            "so_phien_co_gd_52tuan", "gia_cuoi_nghin_dong", "ngay_gia_cuoi", "ngay_du_lieu_dau", "so_tuan_du_lieu"]
    with (out / f"hose_hnx_{n_tong}_danh_muc_ma.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows({k: x[k] for k in cols} for x in chon)

    jumps = []
    for c in codes:
        ws = sorted(weekly[c].items())
        for (w0, p0), (w1, p1) in zip(ws, ws[1:]):
            if abs(p1 / p0 - 1) > 0.6:
                jumps.append(f"   {c} {w0}->{w1}: {p0:.2f}->{p1:.2f} ({100 * (p1 / p0 - 1):+.0f}%)")
    first = weeks[0]
    lines = [
        f"Phien giao dich gan nhat: {session}; so tuan: {len(weeks)} ({weeks[0]} -> {weeks[-1]})",
        f"Vu tru: {len(san)} co phieu HOSE+HNX giao dich phien gan nhat; co du lieu gia >= 3 nghin dong: {len(info)}",
        f"Loai do < {MIN_PHIEN} phien co giao dich trong 52 tuan: {len(loai)} ({' '.join(loai)})",
        f"Chon: {len(chon)} ma = {len(vn30)} ma VN30 + {len(con_lai)} ma thanh khoan cao nhat",
        f"  HOSE {sum(x['san'] == 'HOSE' for x in chon)} ma, HNX {sum(x['san'] == 'HNX' for x in chon)} ma",
        f"  Ma co du lieu tu tuan dau ({first}): {sum(1 for c in codes if min(weekly[c]) == first)}",
        f"  Hang thanh khoan thap nhat trong VN30: {max(info[c]['hang_thanh_khoan'] for c in vn30)}",
        f"  GTGD binh quan phien cua ma thu {len(chon)}: {con_lai[-1]['gtgd_tb_phien_52tuan_ty_dong']} ty dong ({con_lai[-1]['ma']})",
        f"Bien dong tuan > 60% (gioi han 5 phien tran san HNX ~61%): {len(jumps)}", *jumps,
    ]
    (out / f"hose_hnx_{n_tong}_kiem_tra_chat_luong.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
