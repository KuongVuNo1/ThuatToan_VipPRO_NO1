"""Tai du lieu cho de tai 7: co phieu HOSE + HNX gia >= 3 nghin dong, gia ngay 10 nam -> bo 500 ma.

CACH CHAY (chi can Python >= 3.9, KHONG can cai them thu vien):

    python tai_du_lieu_hose_hnx.py --kiem-tra      # 1) thu ket noi API, in cau truc du lieu (~10 giay)
    python tai_du_lieu_hose_hnx.py                 # 2) tai that (~10-25 phut), sau do tu goi chon_500_ma.py
    python tai_du_lieu_hose_hnx.py --tiep-tuc      #    neu bi ngat giua chung: chay lai, dung cache da tai

Tham so: --so-ma 500  --gia-toi-thieu 3  --nam 10  --luong 4  --lam-moi (bo cache)

NGUON: API gia cong khai cua VNDirect (finfo). Chuong trinh tu thu 2 ten mien
(api-finfo.vndirect.com.vn, finfo-api.vndirect.com.vn) va dung ten mien nao tra loi duoc.

QUY TAC (ghi vao bao cao):
  B1. Xac dinh PHIEN GIAO DICH GAN NHAT cua thi truong.
  B2. Vu tru = moi ma CO PHIEU co giao dich trong phien do tren HOSE hoac HNX (chi ma 3 ky tu,
      loai ETF/chung quyen/chung chi quy); luu vao data/vu_tru_hose_hnx_<ngay>.csv.
  B3. Loc gia dong cua phien gan nhat >= 3 nghin dong; tai lich su 10 nam cua cac ma nay.
  B4. Lay ro VN30 hien hanh tu API (stocks?q=indexCode:VN30) -> data/vn30_hien_hanh.csv.
  B5. Goi chon_500_ma.py: 30 ma VN30 + 470 ma co GTGD binh quan 52 tuan cao nhat
      (loai ma < 20 phien co giao dich trong 52 tuan); gop gia dong cua dieu chinh theo tuan.

KET QUA (thu muc data/):
  raw_vndirect/<MA>.json               du lieu ngay tho cua tung ma (cache, chay lai khong phai tai lai)
  vu_tru_hose_hnx_<ngay>.csv, vn30_hien_hanh.csv
  hose_hnx_500_weekly_adjclose.csv, hose_hnx_500_danh_muc_ma.csv, hose_hnx_500_kiem_tra_chat_luong.txt
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOSTS = ["https://api-finfo.vndirect.com.vn/v4", "https://finfo-api.vndirect.com.vn/v4"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://dstock.vndirect.com.vn/",
}
FLOORS = ("HOSE", "HNX")
PAGE = 9990                      # kich thuoc trang lon nhat thuong dung voi API nay
DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw_vndirect"
API: str | None = None           # ten mien dung duoc, xac dinh khi chay


# ------------------------------------------------------------------ goi API
def _get(url: str, retries: int = 5, timeout: int = 60) -> dict:
    err = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e
            if e.code in (400, 404):            # loi cu phap truy van: thu lai vo ich
                raise
        except Exception as e:  # noqa: BLE001  (mat mang, het gio, JSON hong...)
            err = e
        time.sleep(min(30, 2 ** k))
    raise RuntimeError(f"Khong goi duoc {url}: {err}")


def api(path: str, **params) -> dict:
    """Goi API tren ten mien dang dung; lan dau tu do ten mien hoat dong."""
    global API
    qs = urllib.parse.urlencode(params, safe=":~,")
    hosts = [API] if API else HOSTS
    last = None
    for h in hosts:
        try:
            out = _get(f"{h}/{path}?{qs}", retries=5 if API else 2)
            API = h
            return out
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"Khong ten mien nao tra loi cho /{path}: {last}")


def api_all(path: str, **params) -> list[dict]:
    """Lay du moi trang (phong khi so ban ghi > PAGE)."""
    rows, page = [], 1
    while True:
        js = api(path, size=PAGE, page=page, **params)
        data = js.get("data") or []
        rows += data
        total_pages = js.get("totalPages") or 1
        if page >= total_pages or not data:
            return rows
        page += 1


def g(d: dict, *names, default=None):
    """Lay truong theo nhieu ten co the (khong phan biet hoa thuong)."""
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] is not None:
            return low[n.lower()]
    return default


# ------------------------------------------------------------------ cac buoc
def latest_session(today: dt.date) -> str:
    """B1: ngay giao dich gan nhat = ngay lon nhat co gia cua cac ma tru cot."""
    start = (today - dt.timedelta(days=20)).isoformat()
    best = ""
    for code in ("VNM", "FPT", "HPG", "SHB", "PVS"):
        rows = api("stock_prices", sort="date:desc", q=f"code:{code}~date:gte:{start}", size=5, page=1).get("data") or []
        if rows:
            best = max(best, str(g(rows[0], "date"))[:10])
    if not best:
        raise RuntimeError("Khong xac dinh duoc phien giao dich gan nhat")
    return best


def stock_types() -> dict[str, str]:
    """Bang ma -> loai chung khoan (neu API stocks co); dung de loai ETF/quy. Loi thi tra rong."""
    try:
        rows = api_all("stocks", q="floor:HOSE,HNX")
    except Exception as e:  # noqa: BLE001
        print(f"  (bo qua danh sach /stocks: {e})")
        return {}
    return {str(g(r, "code")).upper(): str(g(r, "type", default="")).upper() for r in rows if g(r, "code")}


def universe(session: str) -> list[dict]:
    """B2: ma co phieu co giao dich trong phien gan nhat tren HOSE/HNX."""
    rows = []
    for fl in FLOORS:
        part = api_all("stock_prices", q=f"floor:{fl}~date:{session}")
        if not part:      # mot so phien ban API khong loc theo floor -> lay ca ngay roi loc tay
            part = [r for r in api_all("stock_prices", q=f"date:{session}") if str(g(r, "floor", default="")).upper() == fl]
        rows += part
    types = stock_types()
    out, seen = [], set()
    for r in rows:
        code = str(g(r, "code", default="")).upper()
        fl = str(g(r, "floor", default="")).upper()
        if len(code) != 3 or not code.isalnum() or fl not in FLOORS or code in seen:
            continue
        if types and types.get(code, "STOCK") not in ("STOCK", ""):
            continue
        seen.add(code)
        out.append({"code": code, "floor": fl, "close": g(r, "close"), "date": str(g(r, "date"))[:10]})
    return out


def fetch_prices(code: str, start: str, end: str, refresh: bool) -> list[dict]:
    """Lich su ngay cua mot ma (co cache)."""
    f = RAW / f"{code}.json"
    if f.exists() and not refresh:
        cached = json.loads(f.read_text(encoding="utf-8"))
        if cached.get("start", "9") <= start and cached.get("end", "") >= end:
            return cached["rows"]
    raw = api_all("stock_prices", sort="date", q=f"code:{code}~date:gte:{start}~date:lte:{end}")
    rows = []
    for d in raw:
        rows.append({"date": str(g(d, "date"))[:10],
                     "close": g(d, "close"),
                     "adClose": g(d, "adClose", "adjClose", "adjustedClose"),
                     "volume": g(d, "nmVolume", "volume", default=0) or 0,
                     "value": g(d, "nmValue", "value", default=None)})
    rows = sorted({r["date"]: r for r in rows}.values(), key=lambda r: r["date"])   # bo trung ngay
    f.write_text(json.dumps({"start": start, "end": end, "rows": rows}), encoding="utf-8")
    return rows


def week_end(d: str) -> str:
    x = dt.date.fromisoformat(d)
    return (x + dt.timedelta(days=(4 - x.weekday()) % 7)).isoformat()


# ------------------------------------------------------------------ chuong trinh chinh
def kiem_tra():
    print("Thu ket noi API VNDirect ...")
    s = latest_session(dt.date.today())
    print(f"  Ten mien dung duoc : {API}")
    print(f"  Phien gan nhat     : {s}")
    r = api("stock_prices", sort="date:desc", q="code:FPT", size=2, page=1)
    print("  Cac truong cua stock_prices:", sorted((r.get("data") or [{}])[0].keys()))
    u = universe(s)
    print(f"  So ma co phieu HOSE+HNX giao dich phien {s}: {len(u)}  (vi du: {[x['code'] for x in u[:8]]})")
    print("OK - co the chay tai that.")


def main(so_ma: int, gia_toi_thieu: float, nam: int, luong: int, lam_moi: bool):
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    today = dt.date.today()
    session = latest_session(today)
    s_date = dt.date.fromisoformat(session)
    start = s_date.replace(year=s_date.year - nam).isoformat()
    one_year = (s_date - dt.timedelta(days=364)).isoformat()
    print(f"Ten mien API: {API} | Phien gan nhat: {session} | Lay du lieu tu {start}")

    uni = universe(session)
    units_k = sorted(x["close"] for x in uni if x["close"])[len(uni) // 2] if uni else 0
    scale = 1000.0 if units_k > 1000 else 1.0                 # phong khi API tra gia theo dong
    elig = [x for x in uni if x["close"] and x["close"] / scale >= gia_toi_thieu]
    print(f"B2 vu tru: {len(uni)} ma co phieu | B3 gia >= {gia_toi_thieu}: {len(elig)} ma")

    info, series, loi = [], {}, []
    with ThreadPoolExecutor(max_workers=luong) as ex:
        futs = {ex.submit(fetch_prices, x["code"], start, session, lam_moi): x for x in elig}
        for i, fu in enumerate(as_completed(futs), 1):
            x = futs[fu]
            try:
                rows = fu.result()
            except Exception as e:  # noqa: BLE001
                loi.append(f"{x['code']}: {e}")
                continue
            if i % 50 == 0 or i == len(futs):
                print(f"  da tai {i}/{len(futs)} ma ({time.time() - t0:.0f} giay)")
            rows = [r for r in rows if r["adClose"] and r["adClose"] > 0]
            if not rows:
                loi.append(f"{x['code']}: khong co adClose")
                continue
            last_year = [r for r in rows if r["date"] >= one_year]
            vals = [(r["value"] if r["value"] else (r["close"] or 0) * scale * 1000 * (r["volume"] or 0))
                    for r in last_year]
            traded = sum(1 for r in last_year if (r["volume"] or 0) > 0)
            weekly = {}
            for r in rows:
                weekly[week_end(r["date"])] = r["adClose"] / scale
            series[x["code"]] = weekly
            info.append({"ma": x["code"], "san": x["floor"], "gia_cuoi_nghin_dong": round(x["close"] / scale, 3),
                         "ngay_gia_cuoi": x["date"], "gtgd_tb_phien_52tuan_ty_dong": round(sum(vals) / max(len(vals), 1) / 1e9, 3),
                         "so_phien_co_gd_52tuan": traded, "ngay_dau_tien": rows[0]["date"], "so_tuan_du_lieu": len(weekly)})

    # ---- luu vu tru (ma + san) va ro VN30 hien hanh, roi dung bo 500 ma
    with (DATA / f"vu_tru_hose_hnx_{session.replace('-', '')}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ma", "san"])
        w.writerows(sorted((x["code"], x["floor"]) for x in uni))
    vn30 = [str(g(r, "code")).upper() for r in api_all("stocks", q="indexCode:VN30")]
    if len(vn30) != 30:
        print(f"CANH BAO: API tra ve {len(vn30)} ma VN30 (khong phai 30) - kiem tra lai data/vn30_hien_hanh.csv")
    with (DATA / "vn30_hien_hanh.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ma", "nguon", "ngay_truy_van"])
        w.writerows((c, "API VNDirect /v4/stocks?q=indexCode:VN30", today.isoformat()) for c in vn30)
    print(f"Tai xong {len(series)} ma (loi {len(loi)}): {', '.join(loi) if loi else 'khong'}")
    import chon_500_ma
    chon_500_ma.main(n_tong=so_ma)
    print(f"Xong sau {time.time() - t0:.0f} giay. Ket qua trong {DATA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so-ma", type=int, default=500, help="tong so ma (gom 30 ma VN30)")
    ap.add_argument("--gia-toi-thieu", type=float, default=3.0, help="nghin dong (mac dinh 3)")
    ap.add_argument("--nam", type=int, default=10)
    ap.add_argument("--luong", type=int, default=4, help="so luong tai song song")
    ap.add_argument("--lam-moi", action="store_true", help="bo qua cache, tai lai tat ca")
    ap.add_argument("--tiep-tuc", action="store_true", help="(mac dinh da dung cache) giu cho tuong thich")
    ap.add_argument("--kiem-tra", action="store_true", help="chi thu ket noi va in cau truc du lieu")
    a = ap.parse_args()
    try:
        if a.kiem_tra:
            kiem_tra()
        else:
            main(a.so_ma, a.gia_toi_thieu, a.nam, a.luong, a.lam_moi)
    except KeyboardInterrupt:
        sys.exit("Da dung. Chay lai de tiep tuc (du lieu da tai duoc giu trong data/raw_vndirect/).")
