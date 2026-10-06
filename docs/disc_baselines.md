# Baseline CDR/RDR từ OD/OC segmentation

Pipeline này tạo baseline hình thái cho bài GONet từ mask optic disc (OD) và optic cup (OC). Nó không thay thế hoặc sửa classifier GONet/DINOv2.

## Điều kiện cần

Repository chưa có segmenter LUNet OD/OC hoặc checkpoint tương thích. Bạn cần cung cấp cả hai trước khi chạy inference. Không dùng checkpoint LUNet artery/vein công khai hoặc segmentation ground truth của bộ dữ liệu làm dự đoán thay thế.

Adapter segmenter phải cung cấp factory có dạng:

```python
def make_segmenter(checkpoint: str, device: str):
    ...
```

Factory trả về object có method:

```python
from typing import Tuple

def predict_masks(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Trả về optic_disc_mask, optic_cup_mask theo đúng thứ tự.
    # Cả hai là mask nhị phân 2D, cùng kích thước ảnh RGB đầu vào.
    ...
```

Adapter chịu trách nhiệm preprocessing, threshold output, mapping channel OD/OC và đưa mask về tọa độ/kích thước ảnh gốc. Ảnh truyền vào là RGB `uint8`. Trong Kaggle, đặt module adapter ở một thư mục có thể import được, ví dụ thêm thư mục dataset adapter vào `sys.path` trong notebook, rồi điền cấu hình:

```yaml
segmenter:
  factory: my_adapter:make_segmenter
  checkpoint: /kaggle/input/my-segmenter/model.pth
  device: cuda
```

Đường dẫn tương đối trong config được tính từ thư mục repository; đường dẫn tuyệt đối được giữ nguyên. Nếu thiếu factory hoặc checkpoint, inference dừng với thông báo cấu hình cần bổ sung.

## CDR

Với foreground mask OD và OC hợp lệ:

```text
vertical_CDR = vertical_height(OC) / vertical_height(OD)
vertical_height(mask) = max(y) - min(y) + 1
```

Hai mép trên/dưới được tính vào chiều cao. Mask phải có giá trị hữu hạn và nhị phân (encoding `0/1` hoặc `0/255`). Trước khi đo, hàm giữ lại connected component 8-lân cận lớn nhất của mỗi mask để bỏ chấm nhiễu rời. Mask rỗng, mask chỉ có một pixel, giá trị không nhị phân, shape không hợp lệ hoặc cup nằm ngoài disc trả về `cdr=NaN`, `segmentation_valid=false` và `segmentation_status` tương ứng. Cup không bị cắt/clamp vào disc.

## RDR

RDR hiện được xuất là `NaN`, với `rdr_status=unverified_mask_level_definition`. Paper GONet dẫn reference [10] về narrowest neuroretinal rim và hình học tâm đường tròn; repository chưa xác nhận cách chuyển định nghĩa đó thành phép đo trên mask pixel. Vì vậy pipeline không vẽ ROC hoặc báo AUROC cho RDR. Nếu sau này xác nhận được định nghĩa mask-level và RDR thấp biểu thị nguy cơ glaucoma cao, score ROC sẽ là `-RDR`.

Reference: [Rim-to-Disc Ratio Outperforms Cup-to-Disc Ratio for Glaucoma Prescreening](https://doi.org/10.1038/s41598-019-43385-2) và [supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-019-43385-2/MediaObjects/41598_2019_43385_MOESM1_ESM.pdf).

## Input manifest và output inference

Manifest cần có các cột `image_path`, `patient_id`, `domain`, `label`. `image_path` là đường dẫn tương đối dưới `data_root`; label dùng `1 = GON+`, `0 = GON-`. Nếu `patient_id` trống thì output để trống. Manifest hiện tại không có patient ID cho REFUGE; pipeline không suy ra ID từ tên file.

Mặc định đọc đường dẫn trong `configs/disc_baselines.yaml`. Chạy từ thư mục gốc repository:

```bash
python -m src.infer_disc_baselines --config configs/disc_baselines.yaml
```

Ví dụ override đường dẫn khi dữ liệu nằm trong Kaggle Input:

```bash
python -m src.infer_disc_baselines \
  --config configs/disc_baselines.yaml \
  --data-root /kaggle/input/my-fundus-data \
  --manifest /kaggle/input/my-fundus-data/manifests/gonet_msd.csv \
  --output-dir /kaggle/working/disc-baselines
```

Inference ghi một dòng mỗi ảnh vào `disc_baselines.csv`, gồm:

| Cột | Ý nghĩa |
| --- | --- |
| `image_path`, `patient_id`, `domain` | Metadata từ manifest |
| `ground_truth` | `GON+` hoặc `GON-` |
| `cdr` | Vertical CDR; `NaN` nếu segmentation không hợp lệ |
| `rdr`, `rdr_status` | RDR để `NaN` và trạng thái chưa xác minh |
| `segmentation_valid`, `segmentation_status` | Cờ và lý do validity |
| `od_mask_path`, `oc_mask_path` | Mask grayscale nhị phân, đường dẫn tương đối với output dir |

Mask được lưu dưới `masks/<domain>/...`. Mask có hình dạng sai so với ảnh không được lưu; các mask cùng đúng kích thước ảnh vẫn được lưu để có thể kiểm tra kể cả khi cup không nằm trong disc.

## Evaluation CDR và so sánh classifier

CDR được dùng trực tiếp làm score cho lớp GON+, không đảo dấu. Evaluation loại CDR không hữu hạn, segmentation invalid và domain chỉ có một lớp. AUC table bao gồm `model`, `domain`, `auc`, `n`, cùng bootstrap 95% CI. Nếu truyền probability CSV, các dòng được ghép bằng cả `image_path` và `domain`; file cần các cột `image_path`, `domain`, `gonet_prob`. Unmatched hoặc non-finite probability không tham gia đánh giá.

Chạy CDR:

```bash
python -m src.evaluate_disc_baselines \
  --results /kaggle/working/disc-baselines/disc_baselines.csv \
  --output-dir /kaggle/working/disc-baselines
```

Thêm xác suất GONet/DINOv2:

```bash
python -m src.evaluate_disc_baselines \
  --results /kaggle/working/disc-baselines/disc_baselines.csv \
  --probabilities /kaggle/input/gonet-predictions/predictions.csv \
  --output-dir /kaggle/working/disc-baselines
```

Kết quả gồm `auc_comparison.csv` và `roc_<domain>.png` cho mỗi domain đủ hai lớp. Khi chưa có probability CSV, chỉ có dòng CDR.

## Visualization

Lấy mẫu ngẫu nhiên có seed cố định từ các dòng có cả hai mask đã lưu:

```bash
python -m src.visualize_disc_baselines \
  --results /kaggle/working/disc-baselines/disc_baselines.csv \
  --data-root /kaggle/input/my-fundus-data \
  --results-root /kaggle/working/disc-baselines \
  --output-dir /kaggle/working/disc-baselines/visualizations \
  --count 16 \
  --seed 42
```

Mỗi ảnh overlay hiển thị fundus gốc, contour OD/OC, CDR và trạng thái `RDR unavailable`. Script báo rõ nếu ảnh/mask thiếu, mask không phải grayscale 2D, hoặc mask không cùng kích thước với ảnh.

## Kiểm tra unit tests

```bash
python -m unittest discover -s tests -v
```

## Giới hạn tái lập

Kết quả inference end-to-end phụ thuộc adapter và checkpoint OD/OC bạn cung cấp; preprocessing/channel mapping của mô hình không thể xác nhận từ code trong repository. CDR là phép đo pixel theo công thức vertical trong paper, nhưng cách xử lý component lớn nhất là quy ước triển khai để loại chấm nhiễu. RDR chưa được báo cáo cho tới khi phép đo mask-level được đối chiếu với reference [10].
