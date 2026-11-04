# BÀI KIỂM TRA GIỮA KỲ
## Chủ đề lựa chọn: Nén dữ liệu tệp văn bản bằng thuật toán Huffman Codes (Text File Compression).
### Ngôn ngữ lập trình: Python 3.
# 1. YÊU CẦU ĐỀ BÀI VÀ TRÌNH BÀY THUẬT TOÁN
Dự án này vận dụng kiến thức Huffman Codes và Data Compression (Nén dữ liệu) để thực hiện nén dữ liệu không mất mát (lossless) cho tệp văn bản (.txt). Thuật toán Huffman là một giải thuật tham lam (greedy algorithm) tối ưu để xây dựng mã tiền tố (prefix code).

## 1.1. Quy trình Nén dữ liệu (Compression)
    Thống kê tần số: Đọc toàn bộ văn bản đầu vào và đếm số lần xuất hiện của mỗi ký tự.
    Khởi tạo Hàng đợi ưu tiên (Min-Heap): Tạo một nút lá cho mỗi ký tự chứa tần số của nó và đẩy vào Min-Heap.
### Xây dựng Cây Huffman:
    Lặp lại việc lấy ra hai nút có tần số nhỏ nhất từ Heap.
    Tạo một nút cha mới có tần số bằng tổng tần số của hai nút con.
    Gán hai nút vừa lấy ra làm con trái và con phải, sau đó đẩy nút cha trở lại Heap.
    Quá trình kết thúc khi Heap chỉ còn 1 nút (đó là nút Gốc của cây Huffman).
    Sinh Bảng mã Tiền tố (Prefix Codes): Duyệt từ gốc xuống lá. Mỗi lần rẽ trái gán bit 0, rẽ phải gán bit 1. Chuỗi bit tại lá chính là mã của ký tự đó.

    Mã hóa và Đóng gói: Thay thế các ký tự trong văn bản bằng chuỗi bit tương ứng. Lưu bảng mã (hoặc cấu trúc cây) vào phần Header của file nén, sau đó ghi luồng bit dữ liệu dưới dạng nhị phân ra file .bin.

## 1.2. Quy trình Giải nén (Decompression)
    Đọc Header: Tách phần siêu dữ liệu (metadata) từ đầu file nén để khôi phục lại bảng mã (từ điển ngược) hoặc cây Huffman.
    Dịch luồng bit: Đọc phần dữ liệu nhị phân, chuyển đổi thành chuỗi bit.
    Khôi phục văn bản: Khớp từng chuỗi bit theo bảng mã đã khôi phục để viết lại các ký tự nguyên gốc ra file .txt mới.

# 2. PHÂN TÍCH VÀ ĐÁNH GIÁ ĐỘ PHỨC TẠP
    Gọi N là tổng số ký tự của tệp văn bản gốc, và k là số lượng ký tự phân biệt (đối với bảng mã ASCII, k256).

## 2.1. Đánh giá thời gian thực thi (Time Complexity)
    Đếm tần số ký tự: Cần duyệt qua toàn bộ N ký tự (N).
    Xây dựng Hàng đợi ưu tiên (Min-Heap): Chèn k phần tử O(kk) hoặc (k) nếu dùng heapify.
    Xây dựng Cây Huffman: Gồm k-1 bước lặp, mỗi bước lấy ra 2 phần tử và chèn 1 phần tử vào Min-Heap có kích thước lớn nhất là k. Chi phí mỗi bước là O(k) (kk).
    Sinh bảng mã: Duyệt qua cây có 2k-1 nút (k).
    Mã hóa (Nén) và Giải nén: Tra cứu bảng mã và ghi ra tệp cho N ký tự (N).
### Tổng quát các giá trị tiệm cận toàn thuật toán:
    Trường hợp xấu nhất (Cận trên - O): O(N+kk).
    Trường hợp tốt nhất (Cận dưới - ): (N+kk).
    Cận chặt (): (N+kk).
(Lưu ý: Do trong thực tế kích thước tệp N thường rất lớn so với k, và k bị giới hạn bởi hằng số của bảng mã chữ cái (k256), nên độ phức tạp thời gian tổng quát tiệm cận tuyến tính là (N)).

## 2.2. Đánh giá không gian bộ nhớ (Space Complexity)
    Cây Huffman và Bảng mã: Cây có 2k-1 nút, bảng mã lưu trữ k chuỗi bit  Chi phí bộ nhớ phụ trợ là O(k).
    Tổng quát Không gian: O(k). Thuật toán cực kỳ tiết kiệm bộ nhớ.

# 3. CẤU TRÚC THƯ MỤC
├── huffman.py          # Mã nguồn thuật toán chính
├── README.md           # Báo cáo và Hướng dẫn
├── sample.txt          # Tệp văn bản mẫu dùng để nén
├── compressed.bin      # Tệp nhị phân đầu ra (sinh ra sau khi nén)
└── output.txt          # Tệp văn bản đầu ra (sinh ra sau khi giải nén)

# 4. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH TRÊN WINDOWS
    Yêu cầu: Máy tính đã cài đặt Python 3.x.
        Bước 1: Mở Command Prompt (CMD)
        Truy cập vào thư mục chứa mã nguồn (huffman.py) thông qua File Explorer.
        Nhấp chuột vào thanh địa chỉ đường dẫn ở trên cùng, gõ cmd và nhấn Enter. Cửa sổ lệnh sẽ mở ra tại đúng thư mục hiện tại.
        Bước 2: Thực thi mã nguồn Gõ lệnh sau vào cửa sổ CMD và nhấn Enter:
        python huffman.py
        Bước 3: Kiểm tra kết quả
    Ngay trên màn hình Terminal/CMD sẽ hiển thị thông số so sánh kích thước:
    Kích thước gốc: 58 bytes
    Kích thước sau nén: 43 bytes

    Kiểm tra trong thư mục, chương trình đã tự động sinh ra 2 tệp mới:
    compressed.bin: Tệp nén định dạng nhị phân, bạn có thể xem kích thước file này sẽ nhỏ hơn file sample.txt ban đầu.
    output.txt: Tệp được giải nén từ tệp .bin. Hãy mở lên, bạn sẽ thấy nội dung hoàn toàn khớp 100% với tệp sample.txt.
