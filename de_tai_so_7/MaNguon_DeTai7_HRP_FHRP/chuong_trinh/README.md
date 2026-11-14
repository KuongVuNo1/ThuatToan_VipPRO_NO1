# Chương trình demo – Đề tài số 7: HRP và FHRP cho danh mục VN30

Học phần Thuật toán Nâng cao (CSE586) – Lớp Cao học CHK33 – Trường Đại học Thủy Lợi.

Bài báo căn cứ: Salas-Molina, F. & Nin, J. (2026). *Fast hierarchical risk parity methods for portfolio selection*. Annals of Operations Research. DOI: 10.1007/s10479-026-07149-2 (CC BY 4.0).

## 1. Cài đặt và chạy

```
pip install -r requirements.txt
python demo_nhanh.py           # demo < 5 giây: thứ tự và trọng số HRP/FHRP cho VN30
python tests/test_algorithms.py  # 7 kiểm thử tính đúng đắn (hoặc: python -m pytest -q)
python main.py --nhanh         # chạy rút gọn toàn bộ thực nghiệm (~20 giây)
python main.py                 # chạy đầy đủ (~3 phút trên CPU 2 nhân) -> thư mục ket_qua/
```

Yêu cầu Python ≥ 3.10.

## 2. Cấu trúc

| Tệp | Nội dung |
| --- | --- |
| `hrp/algorithms.py` | Hướng giải 1 (HRP: phân cụm liên kết đơn + chia đôi đệ quy), hướng giải 2 (FHRP: xếp hạng tương quan + sắp xếp), Random HRP, IVP, EW, MinVar |
| `hrp/exact.py` | Thuật toán đúng để đối chiếu: quy hoạch động Held–Karp (thứ tự tối ưu, O(n²2ⁿ)) và duyệt toàn bộ n! hoán vị |
| `hrp/backtest.py` | Kiểm định cuốn: cửa sổ ước lượng F tuần, tái cân bằng mỗi B tuần |
| `hrp/experiments.py` | 6 thực nghiệm TN1–TN6 của báo cáo |
| `hrp/plots.py` | 7 hình dùng trong báo cáo |
| `hrp/data.py` | Đọc dữ liệu VN30, sinh dữ liệu tổng hợp |
| `data/vn30_weekly_adjclose_2021_2026.csv` | Giá đóng cửa điều chỉnh theo tuần của 30 mã, 08/01/2021–16/10/2026 (nguồn: API giá công khai VNDirect, tải 18/10/2026) |
| `ket_qua/` | Kết quả: `ket_qua_thuc_nghiem.xlsx`, các tệp CSV, hình PNG, `thong_tin_lan_chay.json` |

## 3. Sáu thực nghiệm

| Mã | Nội dung |
| --- | --- |
| TN1 | Kiểm chứng Định lý 1 (bất biến khi đổi chỗ hai nửa danh sách) và Bảng 4 (số nghiệm phân biệt, n = 3…8) |
| TN2 | Thời gian chạy theo n = 10…3.000 trên dữ liệu tổng hợp |
| TN3 | Phương sai tương đối so với HRP trên dữ liệu tổng hợp (Monte Carlo, 20 lần/kích thước) |
| TN4 | Kiểm định cuốn VN30 (F = 52, B = 4 như bài báo; F = 104, B = 13), Monte Carlo 200 lần cho Random HRP |
| TN5 | VN30 với tập con N = 10…30 mã (30 lần lấy mẫu) |
| TN6 | Chất lượng thứ tự so với nghiệm tối ưu Held–Karp (n = 12) và duyệt toàn bộ 8! thứ tự (n = 8) |

## 4. Ghi chú

- Mọi thực nghiệm dùng hạt giống ngẫu nhiên cố định, nên chạy lại cho cùng kết quả (trừ cột thời gian, phụ thuộc máy).
- Dữ liệu tổng hợp lấy tham số từ chính VN30: tương quan thuộc khoảng phân vị 5%–95% của tương quan thực, phương sai tối đa bằng phương sai tuần lớn nhất. Ma trận được hiệu chỉnh về nửa xác định dương khi tính phương sai (bài báo không nêu bước này).
