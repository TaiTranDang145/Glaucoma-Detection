"""Inspect saved checkpoints and repeat a fixed inference sample on CPU.

This diagnostic uses no pretrained downloads, training, or model selection.
It reproduces the existing REFUGE route, including its oracle-trained head and
256px classifier input, to identify the provenance of the stored results.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pandas as pd
import torch

# Support both `python -m src...` and `python src/...`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_research_claims import fingerprint
from src.evaluate_zero_shot_refuge import get_eval_transforms
from src.models.mobilenetv3 import MobileNetV3Classifier
from src.models.segmenter import MobileNetV3UNet
from src.train_fusion import FusionHead
from src.train_lightweight import FundusDataset, get_transforms
from src.train_segmenter import compute_predicted_cdr, get_seg_transforms


def main():
    torch.set_num_threads(2)
    paths = [Path(p) for p in ["outputs/models/M2_global/best_model.pt",
             "outputs/models/M5a_global_cdr/best_model.pt",
             "results/segmentation/best_mobilenetv3_unet.pt",
             "data/features/papila_m2_global_features.pt"]]
    m2_checkpoint, fusion_checkpoint, seg_weights, features = [
        torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    m2 = MobileNetV3Classifier(pretrained=False).eval()
    m2.load_state_dict(m2_checkpoint["model_state_dict"])
    segmenter = MobileNetV3UNet(pretrained=False).eval()
    segmenter.load_state_dict(seg_weights)
    fusion = FusionHead(in_features=1025, hidden_dim=0, dropout=.2).eval()
    fusion.load_state_dict(fusion_checkpoint["model_state_dict"])
    morphology = pd.read_csv("data/manifests/papila_morphology.csv")
    oracle_train = morphology.loc[morphology.split == "train", "cdr_mean"].to_numpy()
    mean, std = oracle_train.mean(), oracle_train.std()
    records = []
    source_sample = morphology[morphology.split == "test"].iloc[:4].reset_index(drop=True)
    source_dataset = FundusDataset(source_sample, "path", get_transforms("val", 224))
    source_predictions = pd.read_csv("results/M2_global_test_predictions.csv").set_index("image_id")
    cdr_predictions = pd.read_csv("data/manifests/papila_predicted_cdr.csv").set_index("image_id")
    with torch.inference_mode():
        for index, row in enumerate(source_sample.itertuples(index=False)):
            x = source_dataset[index]["image"].unsqueeze(0)
            f = m2.extract_features(x).squeeze(0)
            cache = cv2.cvtColor(cv2.imread(f"data/cache/fundus_256/{row.image_id}.png"), cv2.COLOR_BGR2RGB)
            cache_x = get_transforms("val", 224)(image=cache)["image"].unsqueeze(0)
            cache_f = m2.extract_features(cache_x).squeeze(0)
            seg_x = get_seg_transforms("val")(image=cache)["image"].unsqueeze(0)
            masks = torch.sigmoid(segmenter(seg_x))[0].numpy()
            cdr, _, _ = compute_predicted_cdr(masks[0] >= .5, masks[1] >= .5)
            saved_f = features["features"][features["image_ids"].index(row.image_id)]
            records.append({"dataset": "PAPILA", "image_id": row.image_id,
                "feature_max_abs_error_original_route": float((f - saved_f).abs().max()),
                "feature_max_abs_error_cache_route": float((cache_f - saved_f).abs().max()),
                "m2_prob_reinferred": float(torch.sigmoid(m2(x)).item()),
                "m2_prob_stored": float(source_predictions.loc[row.image_id, "prob_raw"]),
                "cdr_reinferred": cdr, "cdr_stored": float(cdr_predictions.loc[row.image_id, "cdr_pred"])})
        refuge = pd.read_csv("results/refuge_zero_shot/refuge_zero_shot_predictions.csv")
        for _, group in refuge.groupby("subfolder", sort=False):
            for row in group.iloc[:4].itertuples(index=False):
                raw = cv2.cvtColor(cv2.imread(str(Path("data") / row.image_path)), cv2.COLOR_BGR2RGB)
                x = get_eval_transforms(256)(image=raw)["image"].unsqueeze(0)
                masks = torch.sigmoid(segmenter(x))[0].numpy()
                cdr, _, _ = compute_predicted_cdr(masks[0] >= .5, masks[1] >= .5)
                z = torch.cat([m2.extract_features(x), torch.tensor([[(cdr - mean) / std]], dtype=torch.float32)], dim=1)
                records.append({"dataset": "REFUGE", "image_id": row.image_id,
                    "cdr_reinferred": cdr, "cdr_stored": row.pred_cdr,
                    "m2_prob_reinferred": float(torch.sigmoid(m2(x)).item()), "m2_prob_stored": row.m2_prob,
                    "m5a_prob_reinferred": float(torch.sigmoid(fusion(z)).item()), "m5a_prob_stored": row.m5a_prob})
    summary = {"device": "CPU", "torch_version": torch.__version__,
               "scope": "First four PAPILA test images and first four images in each REFUGE partition; existing route, not a corrected external benchmark.",
               "papila_feature_id_order_matches": features["image_ids"] == morphology.image_id.tolist(),
               "M2_checkpoint": {k: v for k, v in m2_checkpoint.items() if k != "model_state_dict"},
               "fusion_checkpoint": {k: v for k, v in fusion_checkpoint.items() if k != "model_state_dict"},
               "oof_checkpoint_files_found": [str(p) for root in [Path("outputs"), Path("results")]
                                              for p in root.rglob("*.pt") if "oof" in str(p).lower()],
               "oracle_scaler_mean": float(mean), "oracle_scaler_std_ddof0": float(std),
               "input_fingerprints": [fingerprint(p) for p in paths], "records": records}
    out = Path("results/research_claims_audit/checkpoint_inference_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(out), "sample_size": len(records),
                      "papila_feature_id_order_matches": summary["papila_feature_id_order_matches"],
                      "records": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
