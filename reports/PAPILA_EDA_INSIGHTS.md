# PAPILA v1 — EDA Insights và hàm ý cho modeling

## Tóm tắt điều hành

PAPILA v1 có **244 bệnh nhân**, mỗi bệnh nhân có đủ ảnh hai mắt, tương ứng **488 ảnh fundus**. Dữ liệu có chất lượng kỹ thuật tốt: tất cả ảnh đều đọc được, cùng độ phân giải `2576 × 1934`, đủ **1.952 contour** optic cup/disc từ hai chuyên gia và không phát hiện liên kết ảnh–clinical bị thiếu.

Các insight quan trọng nhất:

1. **Đơn vị độc lập thực tế là 244 bệnh nhân, không phải 488 ảnh.** Hai mắt của cùng bệnh nhân có chung nhiều đặc điểm và phải nằm trong cùng một data split.
2. **Nhãn mất cân bằng:** Healthy chiếm 68,2%, Glaucoma 17,8% và Suspect 13,9%. Accuracy đơn thuần sẽ dễ gây hiểu lầm.
3. **Missingness tạo tín hiệu leakage rất mạnh:** `VF_MD` thiếu ở 97,3% mắt Healthy nhưng không thiếu ở bất kỳ mắt Glaucoma/Suspect nào. Một rule chỉ dựa vào việc `VF_MD` có bị thiếu hay không đã tách Healthy và non-Healthy đúng **479/488 mắt (98,2%)**.
4. **Laterality cũng có tín hiệu hệ thống:** cả 7 bệnh nhân có chẩn đoán hai mắt khác nhau đều là `OD = Healthy`, `OS = Glaucoma`. Mô hình ảnh có thể học dấu hiệu mắt trái/phải thay vì pathology nếu không kiểm soát.
5. **Area cup-to-disc ratio có tín hiệu cấu trúc:** median tăng từ Healthy (~0,10) sang Suspect (~0,15) và Glaucoma (~0,25), nhưng các phân bố vẫn chồng lấn nên không thể dùng một ngưỡng đơn giản để thay thế nhãn lâm sàng.
6. **Hai expert đồng thuận cao nhưng không tuyệt đối:** tương quan area CDR là 0,941; median chênh lệch tuyệt đối là 0,019. Bài toán segmentation cần định nghĩa rõ ground truth.

> Đây là các quan sát mô tả trên PAPILA, không phải kết luận nhân quả hoặc khuyến nghị chẩn đoán y khoa.

## 1. Phạm vi và phương pháp

- Dataset: PAPILA phiên bản 1.
- Đơn vị ảnh: một mắt (`OD` = phải, `OS` = trái).
- Đơn vị bệnh nhân: hai mắt có cùng `patient_id`.
- Mapping diagnosis: `0 = Healthy`, `1 = Glaucoma`, `2 = Suspect`.
- Area CDR trong báo cáo là `diện tích cup / diện tích disc` tính từ polygon, **không phải vertical CDR lâm sàng**.
- Notebook tái lập phân tích: [`notebooks/03_papila_eda.ipynb`](../notebooks/03_papila_eda.ipynb).

Nguồn mô tả dataset: [Kovalyk et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01388-1).

## 2. Quy mô và tính toàn vẹn

| Thành phần | Kết quả |
|---|---:|
| Bệnh nhân | 244 |
| Mắt / ảnh fundus | 488 |
| Ảnh trên mỗi bệnh nhân | 2 |
| Độ phân giải | 2576 × 1934 RGB |
| Ảnh lỗi hoặc không đọc được | 0 |
| Ảnh thiếu so với bảng clinical | 0 |
| Contour cup/disc | 1.952 |
| Contour kỳ vọng bị thiếu | 0 |
| Expert trên mỗi cấu trúc | 2 |

Tất cả ảnh có cùng kích thước và mode màu. Điều này giảm nhu cầu xử lý nhiều resolution/camera format trong pipeline, nhưng không loại trừ khác biệt về ánh sáng, màu sắc hay chất lượng chụp.

## 3. Phân bố nhãn

| Diagnosis | Số mắt | Tỷ lệ |
|---|---:|---:|
| Healthy | 333 | 68,2% |
| Glaucoma | 87 | 17,8% |
| Suspect | 68 | 13,9% |
| **Tổng** | **488** | **100%** |

Healthy nhiều gấp khoảng **3,8 lần Glaucoma** và **4,9 lần Suspect**. Một mô hình luôn dự đoán Healthy đã đạt 68,2% accuracy, vì vậy nên ưu tiên macro-F1, balanced accuracy, sensitivity/specificity theo lớp và confusion matrix thay vì chỉ báo cáo accuracy.

## 4. Hai mắt và patient-level dependency

### 4.1. Chẩn đoán OD–OS

| OD \ OS | Healthy | Glaucoma | Suspect |
|---|---:|---:|---:|
| Healthy | 163 | 7 | 0 |
| Glaucoma | 0 | 40 | 0 |
| Suspect | 0 | 0 | 34 |

- 237/244 bệnh nhân (97,1%) có cùng nhãn ở hai mắt.
- 7/244 bệnh nhân (2,9%) có nhãn khác nhau.
- Cả 7 ca không đồng nhất đều có mắt phải Healthy và mắt trái Glaucoma.

### Insight

Hai ảnh của cùng bệnh nhân không độc lập. Random split theo ảnh sẽ đưa một mắt vào train và mắt còn lại vào validation/test, gây data leakage thông qua anatomy, camera characteristics và đặc điểm bệnh nhân.

Sự bất đối xứng của 7 ca discordant cũng khiến laterality trở thành một shortcut tiềm năng. OD có 40 mắt Glaucoma trong khi OS có 47. Nên:

- split bằng `patient_id`;
- báo cáo metric riêng cho OD và OS;
- kiểm tra mô hình sau khi chuẩn hóa hướng ảnh trái/phải;
- không để tên file hoặc biến `eye` đi vào mô hình nếu laterality không thuộc intended use case.

## 5. Nhân khẩu học

### 5.1. Toàn bộ bệnh nhân

| Thống kê tuổi | Giá trị |
|---|---:|
| Trung bình ± SD | 60,59 ± 13,08 |
| Median | 62 |
| IQR | 52–69,25 |
| Khoảng | 15–90 |

| Giới tính | Số bệnh nhân | Tỷ lệ |
|---|---:|---:|
| Female | 151 | 61,9% |
| Male | 93 | 38,1% |

### 5.2. Tuổi theo nhãn mắt

| Diagnosis | Mean | Median | Khoảng |
|---|---:|---:|---:|
| Healthy | 58,75 | 61 | 26–82 |
| Glaucoma | 70,10 | 71 | 50–90 |
| Suspect | 57,44 | 57 | 15–81 |

Các mắt Glaucoma thuộc nhóm bệnh nhân lớn tuổi hơn rõ rệt trong sample này. Vì vậy tuổi có thể là feature dự đoán mạnh, đồng thời là confounder khi so sánh các biến khác như lens status hoặc CDR. Kết quả image model cũng nên được stratify theo nhóm tuổi để kiểm tra performance không chỉ đến từ age-related appearance.

## 6. Missingness: rủi ro leakage lớn nhất

### 6.1. Missingness toàn bộ dataset

| Feature | Thiếu | Tỷ lệ thiếu |
|---|---:|---:|
| `iop_perkins` | 360 | 73,8% |
| `vf_md` | 324 | 66,4% |
| `iop_pneumatic` | 92 | 18,9% |
| `dioptre_1` | 27 | 5,5% |
| `pachymetry` | 14 | 2,9% |
| `lens_status` | 11 | 2,3% |
| `astigmatism` | 9 | 1,8% |
| `axial_length` | 9 | 1,8% |
| `dioptre_2` | 8 | 1,6% |

### 6.2. Missingness phụ thuộc mạnh vào diagnosis

| Diagnosis | `vf_md` thiếu | `iop_perkins` thiếu | `iop_pneumatic` thiếu |
|---|---:|---:|---:|
| Healthy | 97,3% | 95,5% | 4,5% |
| Glaucoma | 0,0% | 20,7% | 56,3% |
| Suspect | 0,0% | 35,3% | 41,2% |

Đây không phải missing completely at random. Pattern có thể phản ánh quy trình khám: nhóm được nghi ngờ hoặc xác nhận bệnh được đo thêm visual field/Perkins. Nếu mô hình clinical dùng mask missing, sentinel value, hoặc imputation kèm missing indicator, nó có thể học **quy trình thu thập dữ liệu** thay vì quan hệ sinh học.

Ví dụ leakage diagnostic:

```text
if VF_MD is missing: predict Healthy
else:                predict non-Healthy
```

Rule này đúng 479/488 mắt, tương đương 98,2% cho bài toán Healthy vs non-Healthy. Đây không phải một mô hình y khoa hợp lệ và không nên được dùng như baseline hiệu năng.

### Khuyến nghị

- Xác định rõ feature nào thực sự có tại thời điểm inference.
- So sánh baseline chỉ dùng giá trị quan sát với baseline có missing indicators.
- Báo cáo missingness theo split và theo lớp.
- Fit imputer chỉ trên train set, nhưng lưu ý patient-level split **không tự loại bỏ** shortcut missingness vì pattern tồn tại trên toàn dataset.
- Với bài toán đánh giá độc lập giá trị của ảnh fundus, không đưa clinical missingness vào image model.

## 7. Các biến lâm sàng

### 7.1. Median theo diagnosis

| Feature | Healthy | Glaucoma | Suspect |
|---|---:|---:|---:|
| IOP Pneumatic | 16,00 | 17,00 | 20,50 |
| IOP Perkins | 17,00 | 17,00 | 18,00 |
| Pachymetry | 533,00 | 536,50 | 558,50 |
| Axial length | 23,36 | 23,48 | 23,80 |
| VF_MD | 0,11 | -4,62 | -1,06 |
| Dioptre 1 | 0,75 | 1,00 | 0,50 |
| Dioptre 2 | -0,75 | -1,25 | -0,75 |

### Quan sát

- Glaucoma có `VF_MD` âm hơn nhóm Suspect, phù hợp với mức suy giảm visual field lớn hơn trong sample.
- Suspect có median IOP Pneumatic cao nhất.
- Median axial length khá gần nhau giữa ba nhóm.
- Tỷ lệ pseudophakic là 60,7% ở Healthy, 77,6% ở Glaucoma và 81,8% ở Suspect.

### Cảnh báo diễn giải

- `VF_MD` của Healthy chỉ có 9/333 giá trị quan sát, nên median 0,11 không đại diện đáng tin cho toàn bộ lớp.
- Hai loại IOP có pattern missing khác nhau rất mạnh theo nhãn; không nên so sánh hoặc gộp chúng mà bỏ qua quy trình đo.
- Lens status và nhiều biến clinical có thể bị confound bởi tuổi.
- Diagnosis được thiết lập từ đánh giá clinical, nên mô hình dùng chính các biến clinical để tái tạo diagnosis cần được mô tả đúng là bài toán tái lập quyết định lâm sàng, không phải bằng chứng độc lập của một biomarker mới.

## 8. Optic cup/disc segmentation

### 8.1. Area CDR theo lớp và expert

| Diagnosis | Expert | Q1 | Median | Q3 |
|---|---:|---:|---:|---:|
| Healthy | 1 | 0,050 | 0,099 | 0,168 |
| Healthy | 2 | 0,055 | 0,106 | 0,167 |
| Glaucoma | 1 | 0,131 | 0,246 | 0,378 |
| Glaucoma | 2 | 0,147 | 0,250 | 0,368 |
| Suspect | 1 | 0,093 | 0,159 | 0,233 |
| Suspect | 2 | 0,087 | 0,147 | 0,254 |

Median area CDR của Glaucoma cao hơn khoảng 2,4–2,5 lần Healthy. Suspect nằm giữa hai lớp theo metric này. Tuy nhiên các IQR chồng lấn đáng kể, nên area CDR đơn lẻ không đủ để phân loại chắc chắn từng mắt.

### 8.2. Agreement giữa expert

| Chỉ số | Giá trị |
|---|---:|
| Pearson correlation của area CDR | 0,941 |
| Mean absolute difference | 0,028 |
| Median absolute difference | 0,019 |
| Mean bias `(Expert 2 - Expert 1)` | 0,002 |

Hai expert có độ nhất quán cao và gần như không có bias trung bình, nhưng sai khác ở từng ảnh vẫn tồn tại. Với segmentation model, nên chọn và công bố một trong các chiến lược:

- train/evaluate riêng trên từng expert;
- consensus mask hoặc average contour;
- soft label thể hiện bất định;
- báo cáo performance của model cùng với inter-expert agreement để có upper-reference hợp lý.

## 9. Hàm ý cho thiết kế thí nghiệm

### Ưu tiên bắt buộc

1. **Group split theo `patient_id`.** Không split theo ảnh.
2. **Stratified group cross-validation.** Dataset chỉ có 244 đơn vị độc lập; k-fold patient-level thường ổn định hơn một lần chia train/validation/test.
3. **Khóa test set trước mọi imputation/normalization/feature selection.** Tất cả preprocessing phải fit trong từng training fold.
4. **Audit shortcut.** Kiểm tra performance theo eye side, age group, gender, missingness pattern và lens status.
5. **Chọn rõ cách xử lý Suspect.** Với binary classification, nên loại Suspect hoặc báo cáo riêng; không âm thầm gộp vào Healthy/Glaucoma.

### Metric nên báo cáo

- Multiclass: macro-F1, balanced accuracy, per-class recall, confusion matrix và one-vs-rest AUC.
- Binary: sensitivity, specificity, ROC-AUC, PR-AUC và confidence interval ở patient level.
- Segmentation: Dice/IoU cho cup và disc, CDR error, và metric riêng theo expert.
- Nếu dự đoán cả hai mắt: báo cáo metric theo mắt và theo bệnh nhân.

### Baseline nên xây dựng

1. Image-only, một mắt.
2. Image-only, ghép hai mắt cùng bệnh nhân.
3. Clinical-only nhưng loại missingness shortcut hoặc dùng feature availability giống điều kiện triển khai.
4. Image + clinical với ablation để biết gain đến từ giá trị thật hay từ pattern missing.
5. Baseline chỉ dùng metadata (`age`, `gender`, `eye`) để đo mức shortcut trước khi đánh giá deep model.

## 10. Giới hạn khi diễn giải

- Sample nhỏ: 244 bệnh nhân.
- Phân bố lớp không cân bằng.
- Tuổi và các yếu tố clinical khác nhau giữa lớp, tạo confounding.
- Missingness phụ thuộc mạnh vào quy trình khám và diagnosis.
- Dữ liệu đến từ một bối cảnh thu thập cụ thể; chưa có external validation trong phân tích này.
- Correlation trong PAPILA không chứng minh quan hệ nhân quả hoặc khả năng khái quát sang bệnh viện/camera/quần thể khác.

## 11. Kết luận

PAPILA phù hợp cho nghiên cứu glaucoma đa nguồn vì kết hợp ảnh hai mắt, clinical data và segmentation của hai expert. Tín hiệu hình thái ở optic cup/disc khá rõ ở mức quần thể, và cấu trúc paired-eye là điểm mạnh hiếm có.

Rủi ro lớn nhất không nằm ở chất lượng file mà ở **thiết kế đánh giá**: patient leakage, class imbalance, laterality shortcut và đặc biệt là clinical missingness gần như tiết lộ nhóm Healthy/non-Healthy. Một pipeline đáng tin cậy phải kiểm soát các shortcut này trước khi diễn giải metric như năng lực phát hiện glaucoma.
