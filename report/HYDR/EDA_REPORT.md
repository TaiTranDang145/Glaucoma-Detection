# Báo cáo EDA – Hillel Yaffe Glaucoma Dataset (HYGD)

## Tóm tắt

EDA trên `data/Labels.csv` ghi nhận **747 ảnh thuộc 288 bệnh nhân**: 548 ảnh GON+ (73,4%) và 199 ảnh GON− (26,6%). Nhãn và điểm chất lượng không có giá trị thiếu; tuy nhiên CSV có thêm một cột rỗng `Unnamed: 4`.

Kiểm tra bổ sung trên thư mục đã loại trùng tìm thấy **10 ảnh trùng nội dung chính xác**. Sáu ảnh trùng thuộc các mã bệnh nhân khác nhau; trong lần chia theo bệnh nhân với seed 42 của notebook, cả 6 cặp đều rơi vào các tập khác nhau, gây rò rỉ dữ liệu thực tế. Tập ảnh sau khử trùng khớp đủ 737 dòng nhãn và tất cả ảnh đều đọc được.

> Các số liệu chính bên dưới lấy từ CSV gốc 747 dòng như notebook EDA. Phần kiểm tra ảnh và trùng lặp dùng bản đã xử lý tại `outputs/deduplicated/`.

## 1. Phạm vi và nguồn dữ liệu

- Notebook phân tích: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
- Nhãn gốc: [data/Labels.csv](data/Labels.csv)
- Ảnh và báo cáo khử trùng có sẵn trong workspace: [outputs/deduplicated/](outputs/deduplicated/)
- Cách chia dữ liệu và lọc điểm chất lượng của pipeline: [src/dataset.py](src/dataset.py), cấu hình tại [configs/efficientnet_b3.yaml](configs/efficientnet_b3.yaml)

Trong workspace hiện tại không có thư mục `data/Images`; do đó việc kiểm tra toàn bộ file ảnh được thực hiện trên `outputs/deduplicated/Images`. Notebook đã lưu kết quả và biểu đồ EDA, nhưng cần đặt ảnh vào đúng đường dẫn mà notebook/pipeline cấu hình trước khi chạy lại toàn bộ từ đầu.

## 2. Cấu trúc và cân bằng nhãn

| Nhãn | Số ảnh | Tỷ lệ ảnh | Số bệnh nhân | Tỷ lệ bệnh nhân | Ảnh trung bình/bệnh nhân |
|---|---:|---:|---:|---:|---:|
| GON+ | 548 | 73,4% | 186 | 64,6% | 2,95 |
| GON− | 199 | 26,6% | 102 | 35,4% | 1,95 |
| **Tổng** | **747** | **100%** | **288** | **100%** | **2,59** |

Tỷ lệ mất cân bằng theo ảnh là khoảng **2,75:1**, nhưng theo bệnh nhân là **1,82:1**. GON+ có nhiều ảnh hơn trên mỗi bệnh nhân, nên thống kê theo ảnh làm chênh lệch hai lớp trông lớn hơn thống kê theo bệnh nhân. Mỗi bệnh nhân trong CSV chỉ thuộc một nhãn; không phát hiện bệnh nhân có nhãn mâu thuẫn.

Số ảnh trên mỗi bệnh nhân dao động từ 1 đến 14; trung vị là 2. Điều này củng cố yêu cầu chia dữ liệu theo bệnh nhân và cần thận trọng khi xem 747 ảnh là 747 quan sát độc lập.

## 3. Điểm chất lượng ảnh

Điểm chất lượng thực tế nằm trong khoảng **2,04–7,69**, trung bình 5,904, trung vị 6,18 và độ lệch chuẩn 1,007. Khoảng này khác với mô tả 1–10 trong phần tóm tắt notebook/README; khi báo cáo nên dùng khoảng quan sát được trong dữ liệu.

| Lớp | Trung bình | Trung vị | Độ lệch chuẩn | Nhỏ nhất | Lớn nhất |
|---|---:|---:|---:|---:|---:|
| GON+ | 5,844 | 6,15 | 1,046 | 2,04 | 7,69 |
| GON− | 6,070 | 6,22 | 0,873 | 3,20 | 7,68 |

GON− có điểm chất lượng trung bình cao hơn GON+ khoảng 0,23 điểm; hai lớp vẫn có độ phân tán lớn và khoảng điểm chồng lấp. Đây là khác biệt mô tả, không đủ để kết luận chất lượng ảnh gây ra nhãn bệnh.

| Ngưỡng giữ ảnh | Ảnh giữ lại | Tỷ lệ giữ | GON+ giữ lại | GON− giữ lại |
|---|---:|---:|---:|---:|
| ≥ 3 | 741/747 | 99,2% | 542/548 (98,9%) | 199/199 (100%) |
| ≥ 4 | 701/747 | 93,8% | 509/548 (92,9%) | 192/199 (96,5%) |
| ≥ 5 | 618/747 | 82,7% | 439/548 (80,1%) | 179/199 (89,9%) |

Ngưỡng ≥3 trong cấu hình loại 6 ảnh, tất cả đều là GON+. Ngưỡng cao hơn làm mất tỷ lệ ảnh GON+ lớn hơn GON−; vì vậy nên ghi rõ ngưỡng và phân bố nhãn sau lọc khi báo cáo kết quả huấn luyện.

## 4. Kiểm tra ảnh và trùng lặp

- CSV gốc có 747 tên ảnh duy nhất. Bản đã khử trùng có 737 dòng nhãn và 737 file ảnh; không thiếu ảnh, không có file ảnh thừa.
- Đọc thử toàn bộ 737 file: không phát hiện file hỏng. Tất cả ảnh đều có kích thước vuông; có 69 kích thước phân giải khác nhau. Kích thước phổ biến nhất là 1894 × 1894 px (52 ảnh).
- Theo nội dung file, có 726 JPEG và 11 PNG; cả 11 file PNG mang phần mở rộng `.jpg`. PIL đọc được chúng, nhưng phần mở rộng không phản ánh đúng định dạng bên trong.
- [Báo cáo khử trùng](outputs/deduplicated/duplicate_report.csv) ghi nhận 10 ảnh trùng chính xác theo SHA-256: 8 GON+ và 2 GON−. Trong đó 4 cặp cùng mã bệnh nhân, 6 cặp giữa các mã bệnh nhân khác nhau. Bản khử trùng còn 540 GON+ và 197 GON− thuộc 286 bệnh nhân.

**Insight quan trọng:** `GroupShuffleSplit` chỉ nhóm theo cột `Patient`; nó không nhận ra hai mã bệnh nhân khác nhau đang có ảnh giống hệt nhau. Ở lần chia seed 42 được lưu trong notebook, cả 6 cặp trùng khác bệnh nhân bị phân vào các tập khác nhau. Sau khi lọc `Quality Score >= 3` nhưng chưa khử trùng, 4 cặp vẫn đi xuyên qua các tập. Cần loại ảnh trùng trước khi chia. Hiện cấu hình mặc định đọc `data/Labels.csv` và `data/Images/`, nên không tự động dùng bản đã khử trùng trong `outputs/deduplicated/`.

## 5. Chia train/validation/test

Notebook chia 747 dòng gốc theo bệnh nhân, seed 42. Các tập không có bệnh nhân giao nhau, nhưng tỷ lệ nhãn giữa các tập không đồng đều:

| Tập | Ảnh | Bệnh nhân | GON+ | GON− | Tỷ lệ GON+ |
|---|---:|---:|---:|---:|---:|
| Train | 523 | 200 | 368 | 155 | 70,4% |
| Validation | 119 | 44 | 100 | 19 | 84,0% |
| Test | 105 | 44 | 80 | 25 | 76,2% |

`GroupShuffleSplit` bảo đảm không trùng bệnh nhân giữa các tập nhưng không phân tầng theo nhãn. Validation có tỷ lệ GON+ cao hơn rõ rệt mức 73,4% của toàn bộ dữ liệu; vì vậy metric có thể nhạy với lần chia cụ thể. Bảng split trong README (523/112/112) cũng không khớp với kết quả notebook (523/119/105).

Pipeline hiện lọc `Quality Score >= 3` trước khi chia. Áp dụng đúng thứ tự đó trên CSV gốc cho kết quả 514/110/117 ảnh ở train/validation/test, với lần lượt 379/84/79 ảnh GON+ và 135/26/38 ảnh GON−. Nếu khử trùng trước rồi lọc ngưỡng ≥3, kết quả theo seed hiện tại là 512/103/116 ảnh, với 376/69/89 GON+ và 136/34/27 GON−. Cả hai cách đều chia theo bệnh nhân nhưng không phân tầng nhãn.

## 6. Hình minh họa từ notebook

### Phân bố nhãn

![Phân bố nhãn](outputs/figures/eda_class_distribution.png)

### Điểm chất lượng

![Phân bố điểm chất lượng](outputs/figures/eda_quality_scores.png)

### Số ảnh trên mỗi bệnh nhân

![Số ảnh mỗi bệnh nhân](outputs/figures/eda_images_per_patient.png)

### Chia tập theo bệnh nhân trong notebook

![Phân bố nhãn trong các tập](outputs/figures/eda_splits.png)

## 7. Kết luận và khuyến nghị

1. Dùng bản đã loại trùng trước khi chia dữ liệu; đặc biệt cần loại các ảnh trùng SHA-256 nhưng mang mã bệnh nhân khác nhau.
2. Giữ patient-level split, đồng thời dùng cách chia có phân tầng theo nhãn ở cấp bệnh nhân hoặc đánh giá qua nhiều seed để giảm biến động phân bố lớp.
3. Báo cáo rõ số bệnh nhân bên cạnh số ảnh, ngưỡng chất lượng và phân bố nhãn sau lọc. Với dữ liệu hiện tại, ngưỡng ≥3 chỉ loại 0,8% ảnh và loại toàn ảnh ở hai bệnh nhân.
4. Cập nhật các số liệu tóm tắt/split trong README và notebook để khớp dữ liệu thực tế; bỏ cột `Unnamed: 4` rỗng khi đọc CSV.

EDA này mô tả cấu trúc dữ liệu, chất lượng ảnh, trùng lặp và rủi ro chia tập; không đánh giá độ chính xác lâm sàng hay khả năng tổng quát hóa của mô hình.
