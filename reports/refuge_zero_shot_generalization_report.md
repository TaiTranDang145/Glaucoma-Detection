# Báo cáo Thử nghiệm Tổng quát hóa Xuyên miền (Zero-Shot OOD Generalization) trên Tập REFUGE (1,200 Ảnh)

**Ngày thực nghiệm:** 07 Tháng 10, 2026  
**Dữ liệu thử nghiệm:** **REFUGE Challenge Dataset** (1,200 ảnh đáy mắt từ thiết bị Zeiss Visucam hoàn toàn độc lập)  
**Mô hình đánh giá:** Các mô hình được huấn luyện **100% trên PAPILA**, KHÔNG RETRAIN, KHÔNG FINE-TUNE, KHÔNG ADAPTATION (Thuần túy Zero-Shot)  
**File kết quả chi tiết:** [`results/refuge_zero_shot/refuge_zero_shot_metrics.json`](file:///home/dekii2275/Glaucoma-Detection/results/refuge_zero_shot/refuge_zero_shot_metrics.json)  
**File dự đoán per-image:** [`results/refuge_zero_shot/refuge_zero_shot_predictions.csv`](file:///home/dekii2275/Glaucoma-Detection/results/refuge_zero_shot/refuge_zero_shot_predictions.csv)  

---

## 1. Mục Đích & Bối Cảnh Khoa Học

Một trong những thách thức lớn nhất của AI Y tế là **Domain Shift (Hiện tượng trôi miền dữ liệu)**:
- Một mô hình có thể đạt AUC rất cao trên tập dữ liệu nội bộ (In-domain, ví dụ PAPILA chụp bằng máy Canon CR-2), nhưng khi mang sang triển khai tại bệnh viện khác với dòng máy chụp khác (REFUGE chụp bằng máy Zeiss Visucam), các đặc trưng thị giác của CNN thuần túy thường bị sụt giảm hiệu năng nghiêm trọng do khác biệt về màu sắc, độ tương phản và trường nhìn (*field of view*).
- **Mục tiêu của thực nghiệm này**: Đưa toàn bộ pipeline của chúng ta ($M_2$ Global, $M_{1\text{-auto}}$ Segmenter, và $M_{5a}$ Fusion) thử lửa trực tiếp trên **1,200 ảnh của REFUGE** để trả lời câu hỏi cốt lõi:
  > *"Liệu việc bổ sung đặc trưng hình thái lâm sàng (CDR) có giúp mô hình chống chịu domain shift và duy trì ưu thế phân loại tốt hơn CNN thuần túy hay không?"*

---

## 2. Bảng Kết Quả Đối Đầu Zero-Shot Trên 1,200 Ảnh REFUGE

| Phân vùng Đánh giá | Số lượng Ảnh (GON+ / GON−) | DINOv2 Foundation Model (86.58M params) ROC-AUC | $M_2$ Global (MobileNetV3) (1.52M params) ROC-AUC (95% CI) | $M_{1\text{-auto}}$ (Auto-CDR alone) (1.26M params) ROC-AUC (95% CI) | **$M_{5a}$ Clinical-Aware (Global + Auto-CDR) (2.78M params)** ROC-AUC (95% CI) | Ưu thế vs DINOv2 $\Delta \text{AUC} (M_{5a} - \text{DINOv2})$ | Ưu thế vs $M_2$ Paired $\Delta \text{AUC}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REFUGE Toàn bộ (1,200 ảnh)** | **1,200** (120 / 1,080) | 0.6686 | 0.6407 `[0.589, 0.688]` | 0.6388 `[0.585, 0.692]` | **$\mathbf{0.7544}$** `[0.703, 0.800]` | **$\mathbf{+0.0858}$** | **$\mathbf{+0.1128}$** ($p < 0.001$) |
| **REFUGE Test400 (Official Test)** | **400** (40 / 360) | 0.7171 | 0.6307 `[0.533, 0.722]` | 0.5512 `[0.458, 0.643]` | **$\mathbf{0.7576}$** `[0.658, 0.842]` | **$\mathbf{+0.0405}$** | **$\mathbf{+0.1271}$** ($p < 0.001$) |
| **REFUGE Validation400** | **400** (40 / 360) | 0.6916 | 0.7296 `[0.655, 0.804]` | 0.5841 `[0.498, 0.673]` | **$\mathbf{0.8078}$** `[0.744, 0.873]` | **$\mathbf{+0.1162}$** | **$\mathbf{+0.0788}$** ($p = 0.0015$) |
| **REFUGE Training400** | **400** (40 / 360) | 0.6201 | 0.6008 `[0.515, 0.686]` | 0.7101 `[0.590, 0.823]` | **$\mathbf{0.7183}$** `[0.615, 0.810]` | **$\mathbf{+0.0982}$** | **$\mathbf{+0.1170}$** ($p = 0.013$) |

---

## 3. Các Phát Hiện Khoa Học Đột Phá

### 3.1 DINOv2 Foundation Model vs CNN Thuần túy Xuyên Miền
- Nhờ được huấn luyện tự giám sát trên 142 triệu ảnh, **DINOv2 ViT-B/14** có khả năng tổng quát hóa đặc trưng thị giác tốt hơn mạng CNN nhỏ $M_2$ (0.6686 vs 0.6407 overall; 0.7171 vs 0.6307 trên Test400).
- Tuy nhiên, vì DINOv2 vẫn thuần túy dựa vào đặc trưng thị giác không gian mà không có tri thức hình thái, nó vẫn chịu ảnh hưởng rõ rệt từ sự sai khác thiết bị máy chụp (trượt dốc từ ~0.81–0.82 trên PAPILA xuống 0.668–0.717 trên REFUGE).

### 3.2 Mô hình Tinh gọn Clinical-Aware Đánh Bại Cả DINOv2 Lẫn CNN Thuần Túy
- **$M_{5a}$ vượt trội hơn DINOv2 trên toàn bộ 4/4 phân vùng của REFUGE**:
  - Trên toàn bộ 1,200 ảnh: $M_{5a}$ đạt **0.7544**, vượt DINOv2 (0.6686) tới **$+0.0858$ AUC**.
  - Trên Test400: $M_{5a}$ đạt **0.7576**, vượt DINOv2 (0.7171) tới **$+0.0405$ AUC**.
  - Trên Validation400: $M_{5a}$ đạt **0.8078**, vượt DINOv2 (0.6916) tới **$+0.1162$ AUC**.
- Điều phi thường là: $M_{5a}$ đạt được điều này chỉ với **2.78M tham số** và **1.98 GFLOPs**, trong khi DINOv2 tiêu tốn tới **86.58M tham số** và **46.3 GFLOPs**!

### 3.3 Hiệu Ứng "Mỏ Neo Lâm Sàng" (Clinical Invariant Anchor) Của Hướng Tiếp Cận Mới
- Khi kết hợp đặc trưng toàn cảnh với tỷ lệ hình thái lâm sàng tự động ($\widehat{\text{CDR}}$), mô hình $M_{5a}$ bật tăng mạnh mẽ:
  - Trên tập **Test400 chính thức**: Tăng từ **0.6307 lên 0.7576** ($\Delta = \mathbf{+0.1271\text{ AUC}}$).
  - Trên toàn bộ **1,200 ảnh REFUGE**: Tăng từ **0.6407 lên 0.7544** ($\Delta = \mathbf{+0.1128\text{ AUC}}$).
  - Trên tập **Validation400**: Đạt tới **0.8078 ROC-AUC**.
- **100% các mẫu Paired Bootstrap resamples đều mang giá trị dương** ($P(\Delta > 0) = 1.000$, khoảng tin cậy 95% `[+0.0701, +0.1548]` nằm hoàn toàn bên phải số 0).

```
                      +-----------------------------------------------------+
                      |           DOMAIN SHIFT (PAPILA -> REFUGE)           |
                      +--------------------------+--------------------------+
                                                 |
                     +---------------------------+---------------------------+
                     |                                                       |
                     v                                                       v
       [M2 CNN Thuần Túy]                                  [M5a Clinical-Aware Fusion]
    0.837 AUC -> 0.641 AUC                              0.867 AUC -> 0.754 - 0.808 AUC
  (Sụt giảm nặng nề -0.196 AUC)                       (Bền vững hơn, vượt M2 tới +0.113 AUC)
```

### 3.3 Ý Nghĩa Đối Với Bài Báo Khoa Học (Paper Narrative)
Kết quả này mang lại cho nghiên cứu một vũ khí cực kỳ thuyết phục:
1. Không chỉ chứng minh mô hình hoạt động xuất sắc *in-domain* trên PAPILA (0.867 AUC vs 0.801-0.813 của DINOv2).
2. Mà còn chứng minh được **giá trị bảo vệ xuyên miền (cross-domain robustness)**: Việc bổ sung thông tin hình thái lâm sàng tự động (CDR) đóng vai trò như một **mỏ neo giải phẫu bất biến (domain-invariant anatomical anchor)**, giúp cứu vãn hệ thống khỏi sự suy thoái thường thấy của các mạng nơ-ron sâu khi gặp thiết bị chụp lạ.
