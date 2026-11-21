# Chương trình – Đề tài số 7: HRP và FHRP trên 500 cổ phiếu HOSE + HNX

Học phần Thuật toán Nâng cao – Trường Đại học Thủy Lợi.

Bài báo căn cứ: Salas-Molina, F. & Nin, J. (2026). *Fast hierarchical risk parity methods for portfolio selection*. Annals of Operations Research. DOI: 10.1007/s10479-026-07149-2 (CC BY 4.0).

## 1. Cài đặt và chạy

```
pip install -r requirements.txt
python demo_nhanh.py               # demo < 10 giây: thứ tự và tỷ trọng HRP/FHRP cho rổ VN30 và cho 500 mã
python tests/test_algorithms.py    # 9 kiểm thử tính đúng đắn (hoặc: python -m pytest -q)
python main.py --nhanh             # chạy rút gọn toàn bộ thực nghiệm (~3 phút)
python main.py                     # chạy đầy đủ (~30–60 phút tùy máy) -> thư mục ket_qua/
```

Yêu cầu Python ≥ 3.10. Bộ dữ liệu 500 mã đã có sẵn trong `data/`; chỉ cần tải lại khi muốn cập nhật.

## 2. Dữ liệu

| Tệp | Nội dung |
| --- | --- |
| `data/hose_hnx_500_weekly_adjclose.csv` | **Dữ liệu chính.** Giá đóng cửa điều chỉnh theo tuần (nghìn đồng) của 500 mã, 518 tuần 18/11/2016 – 13/11/2026; ô trống = chưa có dữ liệu (mã niêm yết muộn) |
| `data/hose_hnx_500_danh_muc_ma.csv` | STT, mã, sàn, thuộc VN30, hạng thanh khoản, giá trị giao dịch bình quân phiên 52 tuần, ngày có dữ liệu đầu tiên… |
| `data/hose_hnx_500_kiem_tra_chat_luong.txt` | Báo cáo kiểm tra dữ liệu: số mã ở từng bước lọc, các tuần biến động > 60% |
| `data/vn30_hien_hanh.csv` | 30 mã rổ VN30 (API VNDirect, truy vấn 15/11/2026) |
| `data/vu_tru_hose_hnx_20261113.csv` | 708 cổ phiếu HOSE + HNX có giao dịch phiên 13/11/2026 và sàn của từng mã |
| `data/raw_vndirect/` | Dữ liệu ngày thô của 668 mã giá ≥ 3 nghìn đồng (không đưa vào gói nộp vì dung lượng ~150 MB) |

**Quy tắc chọn 500 mã** (`chon_500_ma.py`): 30 mã VN30 hiện hành + 470 mã có giá trị giao dịch bình quân phiên 52 tuần cao nhất trong số các mã HOSE/HNX có giao dịch phiên gần nhất, giá ≥ 3 nghìn đồng và có ít nhất 20 phiên giao dịch trong 52 tuần.

**Cập nhật dữ liệu** (cần Internet tới API VNDirect):
```
python tai_du_lieu_hose_hnx.py --kiem-tra    # thử kết nối (~10 giây)
python tai_du_lieu_hose_hnx.py               # tải ~10–25 phút, có cache, rồi tự gọi chon_500_ma.py
```

## 3. Cấu trúc mã nguồn

| Tệp | Nội dung |
| --- | --- |
| `hrp/algorithms.py` | Hướng giải 1 (HRP: phân cụm liên kết đơn + chia đôi đệ quy), hướng giải 2 (FHRP: xếp hạng tương quan + sắp xếp), Random HRP, IVP, EW, MinVar |
| `hrp/exact.py` | Thuật toán đúng để đối chiếu: quy hoạch động Held–Karp O(n²2ⁿ) và duyệt toàn bộ n! hoán vị |
| `hrp/backtest.py` | Kiểm định cuốn: cửa sổ F tuần, tái cân bằng mỗi B tuần; tập mã thay đổi theo thời gian |
| `hrp/experiments.py` | Sáu thực nghiệm TN1–TN6 |
| `hrp/plots.py` | 7 hình dùng trong báo cáo |
| `hrp/data.py` | Đọc dữ liệu, cửa sổ 52 tuần cuối, sinh dữ liệu tổng hợp |

## 4. Sáu thực nghiệm

| Mã | Nội dung |
| --- | --- |
| TN1 | Kiểm chứng Định lý 1 và Hệ quả 2 (N chẵn/lẻ trên dữ liệu thật) và Bảng 4 của bài báo (số nghiệm phân biệt, n = 3…8) |
| TN2 | Thời gian chạy theo n: dữ liệu tổng hợp n = 10…3.000 và ma trận thật n = 30…493 |
| TN3 | Phương sai tương đối so với HRP trên dữ liệu tổng hợp (Monte Carlo, 20 lần/kích thước) |
| TN4 | Kiểm định cuốn 10 năm trên 500 mã (F = 52, B = 4 như bài báo; F = 104, B = 13), riêng rổ VN30; Monte Carlo 100 lần cho Random HRP |
| TN5 | Tập con: rổ VN30 và N = 50…500 mã lấy ngẫu nhiên (10 lần) |
| TN6 | Chất lượng thứ tự so với nghiệm tối ưu Held–Karp (n = 12) và duyệt toàn bộ 8! thứ tự (n = 8) |

## 5. Ghi chú

- Đơn vị giá: nghìn đồng. Mọi lấy mẫu dùng hạt giống cố định nên chạy lại cho cùng kết quả (trừ cột thời gian).
- Các phân tích "tại một thời điểm" (TN1, TN2 dữ liệu thật, TN6, tỷ trọng) dùng 52 tuần cuối và các mã có đủ dữ liệu trong cửa sổ đó (493/500 mã).
- Danh sách 500 mã chọn theo thời điểm 13/11/2026 nên kiểm định 10 năm có thiên lệch sống sót; mã ít giao dịch có giá "đứng yên" làm phương sai đo được thấp hơn thực tế.
