# Baseline CDR/RDR từ OD/OC segmentation

Pipeline này tạo baseline hình thái cho bài GONet từ mask optic disc (OD) và optic cup (OC). Nó không thay thế hoặc sửa classifier GONet/DINOv2.

## Segmenter và phạm vi baseline

Config mặc định dùng `FunduSegmenter_OriginalImage.pth` qua `src.models.fundu_segmenter_adapter`. Đây là segmenter OD/OC thay thế do bạn cung cấp, **không phải LUNet đã được GONet fine-tune**; vì vậy kết quả là CDR baseline từ FunduSegmenter, không phải tái lập chính xác segmentation branch của paper GONet. Adapter bám theo mã test chính thức: resize 256×256, chuẩn hóa ImageNet, logits nội suy bicubic về kích thước gốc; class 1 (rim) và 2 (cup) hợp thành OD, class 2 là OC. Xem [mã inference chính thức](https://github.com/JusticeZzy/FunduSegmenter/blob/main/test_nopadding.py), [transform chính thức](https://github.com/JusticeZzy/FunduSegmenter/blob/main/utils/transform.py), và [mã nguồn model](https://github.com/JusticeZzy/FunduSegmenter).

Các model weights có license CC BY-NC 4.0. Tác giả ghi checkpoint `FunduSegmenter_OriginalImage.pth` được train trên Drishti-GS, RIM-ONE-r3, REFUGE train và REFUGE validation; hiệu năng checkpoint này chưa được kiểm chứng rộng. Do đó cần đánh dấu Drishti-GS và REFUGE là domain có nguy cơ leakage khi báo cáo AUROC, không coi chúng là đánh giá độc lập. Xem [license và phạm vi train trong README chính thức](https://github.com/JusticeZzy/FunduSegmenter#6-evaluation-only).

Checkpoint mặc định được tìm ở `outputs/FunduSegmenter_OriginalImage.pth`. Factory cần thêm source repo chính thức vào môi trường Python qua biến `FUNDUS_SEGMENTER_REPO`; code kiến trúc không bị sao chép vào project. Ảnh đầu vào adapter là RGB `uint8`, output gồm hai mask nhị phân cùng kích thước ảnh: optic disc rồi optic cup.

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

`src.prepare_data` bỏ qua dataset folder không có mặt. REFUGE2 vẫn chỉ được audit và loại khỏi manifest vì hiện không có glaucoma label; nó không thay cho REFUGE có nhãn. HYDR, DRISHTI-GS và PAPILA cần được đặt dưới cùng một `data_root` với tên folder `HYDR/`, `DRISHTI-GS/`, `PAPILA/`. HYDR cần `Labels.csv` và `Images/`; PAPILA cần `FundusImages/` và `ClinicalData/`.

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

## Chạy trên Kaggle

1. Tạo private Kaggle Dataset chứa `FunduSegmenter_OriginalImage.pth`, rồi gắn nó cùng với các bộ ảnh bằng **Add Input**. Ổ `D:` của máy cá nhân không được Kaggle mount tự động.
2. Đảm bảo notebook dùng phiên bản repository có adapter này. Nếu notebook clone `main`, các commit chỉ có trong local checkout sẽ chưa xuất hiện cho tới khi bạn push chúng lên GitHub.
3. Trong Kaggle, clone source FunduSegmenter và cài dependency bổ sung (không cài đè PyTorch CUDA của Kaggle):

```python
from pathlib import Path
import os
import subprocess

FUNDUS_REPO = Path("/kaggle/working/FunduSegmenter")
subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/JusticeZzy/FunduSegmenter.git", str(FUNDUS_REPO)
], check=True)
os.environ["FUNDUS_SEGMENTER_REPO"] = str(FUNDUS_REPO)
%pip install -q einops ml-collections timm
```

4. Chạy cell sau để xem input folder, tự tìm slug theo tên dataset, rồi tạo alias folder. Nó tìm HYDR theo `Labels.csv` có `Images/` bên cạnh; kiểm tra kết quả log trước khi tiếp tục:

```python
from pathlib import Path

input_roots = list(Path("/kaggle/input").iterdir())
for item in input_roots:
    print(item.name, [child.name for child in item.iterdir()][:12])

data_root = Path("/kaggle/working/gonet-data")
data_root.mkdir(parents=True, exist_ok=True)
tokens = {"DRISHTI-GS": "drishti", "PAPILA": "papila", "REFUGE2": "refuge2"}
for alias, token in tokens.items():
    matches = [item for item in input_roots if token in item.name.casefold()]
    if len(matches) == 1:
        (data_root / alias).symlink_to(matches[0], target_is_directory=True)
    else:
        print(f"{alias}: expected one input matching {token!r}; found {matches}")

hydr_inputs = [item for item in input_roots if "hydr" in item.name.casefold()]
hydr_dirs = [
    label_file.parent
    for item in hydr_inputs
    for label_file in item.rglob("Labels.csv")
    if (label_file.parent / "Images").is_dir()
]
if len(hydr_dirs) == 1:
    (data_root / "HYDR").symlink_to(hydr_dirs[0], target_is_directory=True)
else:
    print(f"HYDR: expected one Labels.csv + Images/ pair; found {hydr_dirs}")
```

PAPILA phải chứa `FundusImages/` và `ClinicalData/`; nếu chúng không nằm trong cùng input folder được tìm thấy, dừng và kiểm tra tree thay vì chạy manifest sai.

5. Dùng đoạn sau để tìm checkpoint và ghi config riêng ở thư mục output. `DATA_ROOT` là alias folder đã tạo:

```python
from pathlib import Path
from omegaconf import OmegaConf

REPO = Path("/kaggle/working/Glaucoma-Detection")
DATA_ROOT = Path("/kaggle/working/gonet-data")
checkpoints = list(Path("/kaggle/input").rglob("FunduSegmenter_OriginalImage.pth"))
assert len(checkpoints) == 1, f"Expected one checkpoint, found: {checkpoints}"

cfg = OmegaConf.load(REPO / "configs/disc_baselines.yaml")
cfg.segmenter.checkpoint = str(checkpoints[0])
cfg.segmenter.device = "cuda"
cfg.paths.data_root = str(DATA_ROOT)
cfg.paths.manifest = "/kaggle/working/gonet_msd.csv"
cfg.paths.output_dir = "/kaggle/working/disc-baselines"
OmegaConf.save(cfg, "/kaggle/working/disc_baselines.yaml")
```

Sau đó chạy:

```bash
python -m src.prepare_data --data-root /kaggle/working/gonet-data --output /kaggle/working/gonet_msd.csv
python -m src.infer_disc_baselines --config /kaggle/working/disc_baselines.yaml
python -m src.evaluate_disc_baselines --results /kaggle/working/disc-baselines/disc_baselines.csv --output-dir /kaggle/working/disc-baselines
```

Kaggle phải bật Internet để clone source và bật GPU để inference thuận tiện. Khi xong, kiểm tra số mẫu/label trong manifest trước khi chạy hết dataset; inference chỉ tạo mask và CDR, không train segmenter hay classifier.

## Kiểm tra unit tests

```bash
python -m unittest discover -s tests -v
```

## Giới hạn tái lập

Kết quả inference end-to-end phụ thuộc adapter và checkpoint OD/OC bạn cung cấp; preprocessing/channel mapping của mô hình không thể xác nhận từ code trong repository. CDR là phép đo pixel theo công thức vertical trong paper, nhưng cách xử lý component lớn nhất là quy ước triển khai để loại chấm nhiễu. RDR chưa được báo cáo cho tới khi phép đo mask-level được đối chiếu với reference [10].
