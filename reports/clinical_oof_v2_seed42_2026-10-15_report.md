# Kết quả huấn luyện thực tế clinical OOF v2 — seed 42

Ngày: **15/10/2026**. Trạng thái: **huấn luyện hoàn tất, đã tạo và kiểm tra bundle**.

Mã nguồn của lượt chạy: commit `4bdbe1085ea93b66cba7d0b23893a9bd18fff341`, nhánh `ma/dev`. Các fingerprint mã nguồn trong protocol đã được đối chiếu với đúng commit này. Một lượt chuẩn bị trước đó được dừng trước checkpoint đầu tiên để bổ sung cache đọc ảnh; kết quả dưới đây thuộc lượt chạy mới từ đầu, không tiếp tục trọng số của lượt bị dừng.

## Cấu hình và dữ liệu

- GPU: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**, CUDA 13.0, PyTorch 2.14.0.
- Trọng số khởi tạo: ImageNet qua `timm/mobilenetv3_small_100.lamb_in1k`, tải từ cache có sẵn; không dùng checkpoint glaucoma lịch sử.
- Batch size 16, hai DataLoader workers, hai CPU threads, cache ảnh bất biến và worker được giữ giữa các epoch.
- Segmenter: tối đa 35 epoch cho mỗi lần chọn checkpoint; outer refit chạy số epoch được chọn từ inner-validation.
- Global backbone: 35 epoch; fusion head: 40 epoch. Checkpoint cuối của từng thành phần được chọn từ dữ liệu được protocol cho phép.
- Thời gian từ khi ghi protocol đến khi ghi summary: **884,53 giây**, khoảng **14 phút 45 giây**. Không gồm kiểm thử, lượt chuẩn bị bị dừng hoặc đóng gói.

| Vai trò | Ảnh | Bệnh nhân | Mắt dương tính |
|---|---:|---:|---:|
| Train | 236 | 118 | 49 |
| Calibration | 58 | 29 | 12 |
| Validation | 62 | 31 | 13 |
| Test | 64 | 32 | 13 |

Không trùng bệnh nhân giữa các vai trò. Calibration không tham gia train hoặc chọn checkpoint của backbone, segmenter hay head.

## OOF đã chạy thực tế

| Outer fold | Epoch chọn từ inner-validation | Epoch refit | Ảnh outer-held-out |
|---|---:|---:|---:|
| 0 | 28 | 28 | 46 |
| 1 | 29 | 29 | 46 |
| 2 | 27 | 27 | 48 |
| 3 | 19 | 19 | 48 |
| 4 | 35 | 35 | 48 |

Đã đối chiếu checkpoint của cả năm fold: inner-selection/refit có cùng khởi tạo, refit không dùng validation, các bệnh nhân outer-held-out không xuất hiện trong fit/selection. Tổng dự đoán OOF phủ đủ 236 ảnh train, không có ảnh calibration/validation/test trong các fold này. Scaler CDR khớp chính xác mean/std của CDR OOF train.

## Kết quả phân loại

Ngưỡng **0.5735082626** áp dụng trên `prob_raw`, chọn bằng Youden từ **validation** và giữ nguyên khi đánh giá test. Platt scaling được fit trên 58 ảnh calibration; không dùng nhãn test để hiệu chỉnh hoặc thay ngưỡng.

| Vai trò | ROC-AUC | Sensitivity | Specificity | Calibrated Brier |
|---|---:|---:|---:|---:|
| Validation, dùng chọn mô hình/ngưỡng | 0.8242 | 61.54% | 91.84% | 0.1293 |
| Test | **0.8658** | **46.15%** | **90.20%** | **0.1112** |

Confusion matrix test: **TP=6, FN=7, TN=46, FP=5**. AUC không đồng nghĩa độ nhạy cao tại ngưỡng triển khai; lượt chạy này bỏ sót 7/13 mắt dương tính ở ngưỡng đã chọn. Không thay ngưỡng theo kết quả test. Đây là kết quả một seed trên test có 13 mắt dương tính từ 7 bệnh nhân, chưa là bằng chứng hiệu năng đa seed, an toàn lâm sàng hoặc khả năng tổng quát hóa ngoài PAPILA. PAPILA test đã được xem trong nghiên cứu trước; không gọi nó là một cohort kiểm chứng mới chưa từng được sử dụng.

Metrics calibration được lưu trong JSON để chẩn đoán, nhưng là điểm số trên tập fit calibrator nên không được dùng làm kết quả kiểm chứng độc lập. Cỡ tập fit khác protocol cũ, vì vậy không lấy chênh lệch với mean AUC lịch sử làm bằng chứng hơn/kém của phương pháp.

## CDR

| Vai trò | MAE CDR | Pearson r |
|---|---:|---:|
| Train, chỉ dùng dự đoán OOF | 0.1110 | 0.6510 |
| Calibration | 0.0761 | 0.8183 |
| Validation | 0.0788 | 0.7094 |
| Test | **0.0649** | **0.8853** |

Không có disc dự đoán rỗng trong các vai trò. Train OOF có hai trường hợp chiều cao cup bằng 0; các vai trò khác không có. Chưa chạy REFUGE bằng bundle này.

## Bundle và kiểm tra bàn giao

- Global model: 1.518.881 parameters; segmenter: 1.256.050; head: 3.076; tổng **2.778.007**.
- Bundle: **11.523.437 bytes**, khoảng **10,99 MiB**.
- SHA-256: `abff4d7ab2dbc8309a1220745004586a6e693fb7e7cb97fe83355e0f125f2f51`.
- Pipeline đã đạt kiểm tra export/load ngay sau train.
- Đã nạp bundle lại và suy luận từ ảnh gốc trên **đủ 64 ảnh test**: AUC, sensitivity, specificity và toàn bộ quyết định nhị phân khớp; không có quyết định nào thay đổi.
- Chênh lệch xác suất tối đa khi replay: raw **4.37e-6**, calibrated **1.57e-5**. Sai khác số học nhỏ do cách batching; không khẳng định replay bit-for-bit. CDR khớp trong sai số biểu diễn CSV.
- **38/38 kiểm thử đạt** trước khi chạy nghiên cứu, gồm kiểm thử nested OOF, calibration tách biệt, cache bất biến và bundle toàn luồng.

Model và checkpoint lớn lưu cục bộ theo quy ước `.gitignore` của repo:

```text
outputs/clinical_oof_v2_seed42_20261015_run02/seed_42/bundle.pt
outputs/clinical_oof_v2_seed42_20261015_run02/seed_42/bundle.json
artifacts/clinical_oof_v2_seed42_20261015_bundle.zip
```

Các chỉ số tổng hợp và bằng chứng kiểm tra không gắn ID được lưu trong Git:

- [Summary](../results/clinical_oof_v2_seed42_20261015/summary.json), [artifact manifest/checksums](../results/clinical_oof_v2_seed42_20261015/artifact_manifest.json).
- [Replay đủ 64 ảnh test](../results/clinical_oof_v2_seed42_20261015/bundle_replay.json).
- [Integrity của checkpoint năm fold](../results/clinical_oof_v2_seed42_20261015/trained_fold_integrity.json).
- [Kiểm tra protocol theo số lượng và hash mã nguồn](../results/clinical_oof_v2_seed42_20261015/protocol_integrity.json).
- [CDR metrics](../results/clinical_oof_v2_seed42_20261015/morphometry_metrics.json), [training log](../results/clinical_oof_v2_seed42_20261015/training.log), [test log](../results/clinical_oof_v2_seed42_20261015/unittest.log).

Các file chi tiết gắn ID ảnh/bệnh nhân (predictions, roles, fold mapping, protocol đầy đủ, bundle metadata và replay từng ảnh) được giữ cục bộ trong `results/clinical_oof_v2_seed42_20261015/` và không được xuất trong commit kết quả này.

## Lệnh đã chạy

```bash
NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=42 \
venv/bin/python -u src/run_step6_oof_pipeline.py \
  --output-dir outputs/clinical_oof_v2_seed42_20261015_run02 \
  --seeds 42 --device cuda --workers 2 --threads 2 --batch-size 16
```

Để đánh giá REFUGE sau này, dùng đúng bundle này với `src/evaluate_zero_shot_refuge.py --bundle ...` và một thư mục kết quả mới. Lượt thực hiện hiện tại hoàn tất phần train một seed và bàn giao bundle; external benchmark và ablation ở cùng protocol chưa được chạy.
