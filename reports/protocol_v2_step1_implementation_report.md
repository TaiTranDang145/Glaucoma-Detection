# Kết quả triển khai bước 1: protocol và bundle mô hình v2

Ngày: **15/10/2026**.

**Đã hoàn tất phần sửa code/protocol và kiểm chứng kỹ thuật.** Đã chạy huấn luyện ngắn bằng dữ liệu giả lập để kiểm tra toàn luồng; chưa huấn luyện lại mô hình nghiên cứu trên PAPILA và chưa tạo kết quả AUC REFUGE mới.

## Những thay đổi đã thực hiện

| Hạng mục | Hành vi mới |
|---|---|
| OOF | Chọn epoch trên inner-validation thuộc outer-train; khởi tạo lại và refit toàn outer-train trước khi dự đoán outer-held-out. |
| Checkpoint | Lưu checkpoint inner-selection và refit cho từng fold; lưu riêng global backbone, final segmenter và head cho từng seed. |
| Calibration | Giữ riêng bệnh nhân calibration trước khi train bất kỳ thành phần nào; không dùng họ để chọn checkpoint. Fit Platt chỉ sau khi chọn xong predictor. |
| Preprocessing | Cùng hàm decode PIL RGB → square padding → resize/normalize cho train và inference. Nhánh global 224×224, segmenter 256×256. |
| Ghép feature | Kiểm tra unique/đủ image ID, patient/label/role, checkpoint hash và preprocessing; sắp theo ID trước khi nối CDR. Cache lịch sử thiếu provenance bị từ chối. |
| Bundle | Gói đủ ba thành phần mô hình, CDR scaler, Platt coefficients, ngưỡng raw probability, cấu hình, seed, vai trò bệnh nhân, fold mapping, checkpoint/input/code hashes và phiên bản môi trường. |
| External inference | Bắt buộc nạp bundle v2; không fallback sang oracle head và không fit lại scaler/calibrator/threshold từ REFUGE. Sửa tìm mask Training400 trong thư mục chẩn đoán. |
| Bảo toàn kết quả | Các lệnh mới yêu cầu thư mục output rỗng; không ghi đè predictions/metrics lịch sử. |

## Phân chia đã xác minh trên PAPILA thật

Giữ nguyên validation/test gốc. Trong 294 ảnh train gốc, giữ một group fold riêng để calibration:

| Vai trò | Ảnh | Bệnh nhân | Mắt dương tính |
|---|---:|---:|---:|
| Huấn luyện | 236 | 118 | 49 |
| Calibration | 58 | 29 | 12 |
| Validation | 62 | 31 | 13 |
| Test | 64 | 32 | 13 |

Đã xác minh không trùng bệnh nhân giữa các vai trò; outer fold không xuất hiện trong inner-selection/refit; image IDs phủ đúng tập huấn luyện. Plan-only đã kiểm tra hash ảnh và mask, đồng thời lưu mapping/config để xem lại:

- [Protocol JSON](../results/protocol_v2_validation/source_plan/protocol.json).
- [Vai trò từng ảnh](../results/protocol_v2_validation/source_plan/roles.csv).
- [Nested fold mapping](../results/protocol_v2_validation/source_plan/fold_plan.json).
- [Kiểm tra đầu vào ảnh thật](../results/protocol_v2_validation/source_input_check.json).

Do calibration tách từ train, phải train lại backbone và segmenter theo các vai trò này. Checkpoint M2 cũ đã học trên cả 294 ảnh không phù hợp với holdout calibration mới. Cỡ tập calibration nhỏ vẫn là một hạn chế của nghiên cứu.

## Kiểm thử đã chạy

Lệnh:

```bash
NO_ALBUMENTATIONS_UPDATE=1 PYTHONHASHSEED=42 venv/bin/python -m unittest discover -s tests -v
```

**37/37 kiểm thử đạt**, trong đó 10 kiểm thử mới cho protocol v2. [Log đầy đủ](../results/protocol_v2_validation/unittest.log).

Các kiểm tra chính:

- Từ chối patient overlap, fold chứa vai trò không được phép và fold selection trùng bệnh nhân held-out.
- Từ chối feature cache sai checkpoint, trùng ID hoặc lệch nhãn; reordering hợp lệ được xử lý đúng.
- Training và inference tạo tensor giống nhau từ cùng ảnh, đúng kích thước hai nhánh.
- Calibration từ chối bệnh nhân đã tham gia fit/selection.
- Thay nhãn **test** không làm thay đổi head, scaler, calibrator hay threshold.
- Chạy gradient updates thật trên segmenter nhỏ để xác minh nested selection/refit và export checkpoint.
- Suy luận thủ công với MobileNetV3/UNet/head thật khớp với bundle đã export và load lại.
- Kiểm thử toàn luồng dùng MobileNetV3/UNet/head thật, khởi tạo không pretrained, train một epoch trên ảnh/mask **giả lập**, xuất đủ checkpoint/bundle, load lại và gọi evaluator external. Calibrated probabilities và binary decisions khớp đúng coefficients/threshold lưu trong bundle.
- Từ chối bundle kiểu cũ hoặc provenance calibration không nhất quán; từ chối ghi đè kết quả external đã tồn tại.
- Tìm được mask Training400 trong thư mục `Glaucoma`/`Non-Glaucoma`.

Các checkpoint giả lập nằm trong thư mục kiểm thử tạm và đã được xóa sau test; chúng không được xuất như mô hình nghiên cứu. Kiểm thử chạy trên CPU vì môi trường hiện tại không có CUDA. Cú pháp Python và `git diff --check` cũng đã kiểm tra.

## Mã nguồn và cách chạy tiếp

- [Data contracts, preprocessing và bundle](../src/clinical_protocol.py).
- [Nested OOF, train và export](../src/clinical_training.py).
- [Entry point](../src/run_step6_oof_pipeline.py).
- [Evaluator REFUGE](../src/evaluate_zero_shot_refuge.py).
- [Hướng dẫn đầy đủ](../docs/clinical_protocol_v2.md).

Bước tiếp theo là chạy một seed trên PAPILA bằng pipeline đã sửa:

```bash
NO_ALBUMENTATIONS_UPDATE=1 venv/bin/python src/run_step6_oof_pipeline.py \
  --output-dir outputs/clinical_oof_v2_run01 --seeds 42 --device cuda
```

Lệnh này cần môi trường có CUDA; có thể đổi thành `--device cpu`. Nó sẽ train lại các thành phần đúng vai trò và tạo `seed_42/bundle.pt`. Chưa có bundle PAPILA v2 đã huấn luyện để công bố hiệu năng ở thời điểm báo cáo này; không dùng số AUC cũ thay cho kết quả lần chạy mới.
