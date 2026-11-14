# Bộ dữ liệu VN30 cho đề tài 7 – HRP/FHRP (chia để trị)

Bài báo căn cứ: Salas-Molina, F. & Nin, J. (2026). *Fast hierarchical risk parity methods for portfolio selection*. Annals of Operations Research (Q1, SJR 2025). PDF mở, giấy phép CC BY 4.0: https://link.springer.com/content/pdf/10.1007/s10479-026-07149-2.pdf

## 1. Tệp dữ liệu chính

`vn30_weekly_adjclose_2021_2026.csv`

| Thuộc tính | Giá trị |
| --- | --- |
| Nội dung | Giá đóng cửa đã điều chỉnh (adClose) theo tuần của 30 mã, đơn vị nghìn VND |
| Giai đoạn | 08/01/2021 – 16/10/2026, 299 tuần |
| Cột | `week_end` = ngày giao dịch cuối cùng của tuần; 30 cột còn lại là mã cổ phiếu |
| Nguồn | API dữ liệu giá công khai của VNDirect (`api-finfo.vndirect.com.vn/v4/stock_prices`), tải ngày 18/10/2026 |
| Điều chỉnh | Dùng trường `adClose`, giá đã điều chỉnh theo chia tách, cổ tức cổ phiếu (theo quy ước của VNDirect) |
| Ô trống | Chỉ SSB trống 11 tuần đầu (niêm yết HOSE ngày 24/03/2021) |
| Tổng kiểm | 8.959 ô có giá trị, tổng = 330.300,76 (đã đối chiếu với dữ liệu gốc) |

**Danh sách 30 mã:** ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, MBB, MSN, MWG, PLX, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.

Lưu ý: đây là danh sách 30 mã vốn hóa lớn, thường có mặt trong VN30 giai đoạn 2021–2026. Thành phần VN30 thay đổi mỗi kỳ xét định kỳ (tháng 1 và tháng 7), nên danh sách này **chưa đối chiếu** với kỳ gần nhất của HOSE.

**Vì sao dùng dữ liệu tuần:** bài báo dùng lợi suất tuần của 530 cổ phiếu S&P 500 (2021–2024). Bộ này dùng cùng tần suất để so sánh được.

## 2. Kiểm tra chất lượng

- Không có ô trống ngoài 11 tuần đầu của SSB.
- Biến động một tuần lớn nhất: SHB +39% (tuần 02/04/2021), SSI −26% (tuần 17/06/2022). Vẫn nằm trong giới hạn biên độ HOSE: 5 phiên tăng trần liên tiếp là 1,07⁵ ≈ +40%. Nên đối chiếu với thông tin doanh nghiệp các tuần có biến động trên ±20%.
- Tuần có ngày nghỉ (Tết, 30/4, 2/9) có khoảng cách 8–14 ngày giữa hai mốc.

## 3. Các tệp kết quả (tạo bởi `hrp_vn30.py`)

| Tệp | Nội dung |
| --- | --- |
| `HRP_VN30_ket_qua.xlsx` | 6 sheet: giá tuần, lợi suất log tuần, trọng số, kết quả ngoài mẫu tĩnh, kết quả ngoài mẫu cuốn, thời gian |
| `ket_qua_trong_so.csv` | Trọng số của 5 phương pháp |
| `ket_qua_ngoai_mau_tinh.csv` | Ước lượng 04/2021–12/2024, giữ nguyên trọng số 01/2025–10/2026 |
| `ket_qua_ngoai_mau_cuon.csv` | Cửa sổ 104 tuần, tái cân bằng mỗi 13 tuần |
| `ket_qua_thoi_gian.csv` | Thời gian bước sắp thứ tự, n = 100 đến 4.000 mã |
| `bieu_do_1..4_*.png` | 4 biểu đồ dùng cho slide |
| `chay_thu_log.txt` | Toàn bộ kết quả in ra khi chạy |

## 4. Cách chạy

```
pip install numpy pandas scipy matplotlib openpyxl
python hrp_vn30.py
```

## 5. Thiết kế thực nghiệm

- Lợi suất log tuần: r_t = ln(P_t / P_{t−1}).
- 5 phương pháp: HRP, FHRP, nghịch đảo phương sai (IVP), chia đều (EW), Markowitz phương sai nhỏ nhất chỉ mua (MinVar).
- FHRP: xếp hạng theo tương quan lớn nhất ngoài đường chéo, sắp tăng dần. Chiều sắp xếp là lựa chọn của người soạn vì bài báo không ghi rõ.
