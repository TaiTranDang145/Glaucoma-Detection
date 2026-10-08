"""Generate high-resolution 2x1 Input vs Output visualization grid with rich clinical annotations.
Publication-grade Pure White Theme (Clean, 100% white background, medical journal aesthetic).

Pipeline: Proposed Autonomous Clinical-Aware System (M5a-auto-OOF).
Sample: RET025OS (True GON+, Severe cupping, predicted P(GON+) = 99.8%, CDR = 0.83).
"""

import os
from pathlib import Path
import shutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.segmenter import MobileNetV3UNet
from src.train_segmenter import compute_predicted_cdr, keep_largest_component

def generate_visualization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load trained Segmenter
    seg = MobileNetV3UNet(pretrained=False, num_classes=2).to(device)
    ckpt_path = "results/segmentation/best_mobilenetv3_unet.pt"
    seg.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    seg.eval()

    # 2. Load Sample: RET025OS (Glaucoma Positive)
    manifest = pd.read_csv("data/manifests/papila_morphology.csv")
    sample_id = "RET025OS"
    row = manifest[manifest["image_id"] == sample_id].iloc[0]

    img_raw_bgr = cv2.imread(row["path"])
    img_raw_rgb = cv2.cvtColor(img_raw_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for segmenter (256x256)
    img_256 = cv2.resize(img_raw_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (img_256.astype(np.float32) / 255.0 - mean) / std
    t = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = seg(t)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Masks at 256x256
    disc_m256 = (probs[0] >= 0.5).astype(np.uint8)
    cup_m256 = (probs[1] >= 0.5).astype(np.uint8)

    disc_clean256 = keep_largest_component(disc_m256)
    cup_inside256 = (cup_m256 > 0) & (disc_clean256 > 0)
    cup_clean256 = keep_largest_component(cup_inside256.astype(np.uint8))

    cdr_val, hd, hc = compute_predicted_cdr(disc_clean256, cup_clean256)

    # High display resolution (1024x1024)
    disp_size = 1024
    img_disp = cv2.resize(img_raw_rgb, (disp_size, disp_size), interpolation=cv2.INTER_AREA)

    disc_disp = cv2.resize(disc_clean256, (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)
    cup_disp = cv2.resize(cup_clean256, (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)

    # Contours
    contours_disc, _ = cv2.findContours(disc_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_cup, _ = cv2.findContours(cup_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Bounding box & calipers in display coords
    rows_d = np.flatnonzero(disc_disp.any(axis=1))
    cols_d = np.flatnonzero(disc_disp.any(axis=0))
    d_ymin, d_ymax = rows_d[0], rows_d[-1]
    d_xmin, d_xmax = cols_d[0], cols_d[-1]
    d_xcenter = int((d_xmin + d_xmax) / 2)
    d_ycenter = int((d_ymin + d_ymax) / 2)

    rows_c = np.flatnonzero(cup_disp.any(axis=1))
    c_ymin, c_ymax = rows_c[0], rows_c[-1]

    # Create annotated image with overlay
    annotated = img_disp.copy()
    overlay = annotated.copy()
    overlay[disc_disp > 0] = overlay[disc_disp > 0] * 0.72 + np.array([0, 255, 120]) * 0.28
    overlay[cup_disp > 0] = overlay[cup_disp > 0] * 0.58 + np.array([0, 220, 255]) * 0.42
    annotated = cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0)

    # Draw crisp contour boundaries
    cv2.drawContours(annotated, contours_disc, -1, (0, 200, 60), 3, cv2.LINE_AA)
    cv2.drawContours(annotated, contours_cup, -1, (0, 180, 240), 3, cv2.LINE_AA)

    # Extract OD Crop for Inset (zoom in)
    crop_radius = int(max(d_xmax - d_xmin, d_ymax - d_ymin) * 0.75)
    cy_min = max(0, d_ycenter - crop_radius)
    cy_max = min(disp_size, d_ycenter + crop_radius)
    cx_min = max(0, d_xcenter - crop_radius)
    cx_max = min(disp_size, d_xcenter + crop_radius)

    crop_raw = img_disp[cy_min:cy_max, cx_min:cx_max]
    crop_annotated = annotated[cy_min:cy_max, cx_min:cx_max]

    # Elliptical mask to convert pixels outside retina into pure white
    cx, cy = 512.0, 512.0
    rx, ry = 470.0, 492.0
    Y, X = np.ogrid[:disp_size, :disp_size]
    norm_dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    retina_mask = np.clip((1.0 - norm_dist) * 150.0 + 0.5, 0.0, 1.0)[:, :, None]

    img_disp_white = (img_disp.astype(np.float32) * retina_mask + 255.0 * (1.0 - retina_mask)).astype(np.uint8)
    cv2.ellipse(img_disp_white, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, (203, 213, 225), 2, cv2.LINE_AA)

    annotated_white = (annotated.astype(np.float32) * retina_mask + 255.0 * (1.0 - retina_mask)).astype(np.uint8)
    cv2.ellipse(annotated_white, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, (203, 213, 225), 2, cv2.LINE_AA)

    # Embed insets into the display images directly in OpenCV
    inset_size = 320
    inset_x, inset_y = disp_size - inset_size - 40, 80

    # Panel 1 display image
    p1_img = img_disp_white.copy()
    crop_raw_resized = cv2.resize(crop_raw, (inset_size, inset_size))
    p1_img[inset_y:inset_y+inset_size, inset_x:inset_x+inset_size] = crop_raw_resized
    cv2.rectangle(p1_img, (inset_x, inset_y), (inset_x+inset_size, inset_y+inset_size), (245, 158, 11), 3)

    # Panel 2 display image
    p2_img = annotated_white.copy()
    crop_ann_resized = cv2.resize(crop_annotated, (inset_size, inset_size))
    p2_img[inset_y:inset_y+inset_size, inset_x:inset_x+inset_size] = crop_ann_resized
    cv2.rectangle(p2_img, (inset_x, inset_y), (inset_x+inset_size, inset_y+inset_size), (16, 185, 129), 3)

    # Model prediction metrics for RET025OS
    prob_gon = 0.998
    thresh = 0.027
    diagnosis = "GON+ (Glaucoma Detected)"
    risk_level = "HIGH RISK / PATHOLOGICAL CUPPING"

    meta_text = (
        f"Subject ID: {sample_id} (Left Eye / OS)\n"
        f"Acquisition: Canon CR-2 Digital Retinal Camera (PAPILA)\n"
        f"Clinical Reference GT: Glaucomatous Optic Neuropathy (GON+)"
    )

    hud_text = (
        "┌────────────────────────────────────────────────────────┐\n"
        "│           CLINICAL DECISION & MORPHOMETRY REPORT       │\n"
        "├────────────────────────────────────────────────────────┤\n"
        f"│  • AI Classification : {diagnosis:<31}│\n"
        f"│  • Predicted Prob    : P(GON+) = {prob_gon*100:.1f}% [HIGH CONFIDENCE]   │\n"
        f"│  • Vertical CDR (vCDR): {cdr_val:.2f} (Threshold: > 0.41)       │\n"
        f"│  • Clinical Status   : {risk_level:<31}│\n"
        f"│  • Operating Point   : τ = {thresh:.3f} (Val Youden's J)         │\n"
        f"│  • Model Efficiency  : 2.78M Params | 1.98 GFLOPs | 5.5 ms     │\n"
        "└────────────────────────────────────────────────────────┘"
    )

    legend_elements = [
        Line2D([0], [0], color="#16A34A", lw=3.5, label="Optic Disc (OD) Contour"),
        Line2D([0], [0], color="#0284C7", lw=3.5, label="Optic Cup (OC) Contour"),
        Line2D([0], [0], color="#F59E0B", lw=3.5, label="Vertical Cup Caliper (h_cup)"),
        Line2D([0], [0], color="#16A34A", lw=3.5, label="Vertical Disc Caliper (h_disc)"),
    ]

    # ─────────────────────────────────────────────────────────────────────────────
    # BUILD 2x1 VERTICAL GRID FIGURE (100% PURE WHITE THEME)
    # ─────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 20), facecolor="#FFFFFF")
    plt.subplots_adjust(hspace=0.07, top=0.97, bottom=0.02, left=0.03, right=0.97)

    # ── PANEL 1: INPUT IMAGE ─────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    ax0.imshow(p1_img)
    ax0.set_xlim(0, disp_size)
    ax0.set_ylim(disp_size, 0)
    ax0.axis("off")

    # OD ROI Bounding Box on Input
    rect_roi = patches.Rectangle(
        (cx_min, cy_min), cx_max - cx_min, cy_max - cy_min,
        linewidth=2.5, edgecolor="#D97706", facecolor="none", linestyle="--", alpha=0.95
    )
    ax0.add_patch(rect_roi)
    ax0.text(
        cx_min, cy_min - 12,
        "OPTIC DISC REGION (ROI)",
        color="#B45309", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#D97706", linewidth=1.5, alpha=0.98)
    )

    # Inset label
    ax0.text(
        inset_x + 12, inset_y + 28,
        "ZOOM-IN: OPTIC DISC (RAW)",
        color="#B45309", fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#D97706", linewidth=1.5, alpha=0.98)
    )

    # Title Banner for Panel 1 (Clean white theme)
    ax0.text(
        30, 48,
        "(1) INPUT: COLOR FUNDUS PHOTOGRAPHY",
        color="#0F172A", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.45", facecolor="#F8FAFC", edgecolor="#0284C7", linewidth=2.0)
    )

    # Metadata Badge
    ax0.text(
        30, disp_size - 40,
        meta_text,
        color="#1E293B", fontsize=10.5, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.5, alpha=0.98)
    )

    # ── PANEL 2: ANNOTATED OUTPUT IMAGE ──────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    ax1.imshow(p2_img)
    ax1.set_xlim(0, disp_size)
    ax1.set_ylim(disp_size, 0)
    ax1.axis("off")

    # Calipers on main image
    caliper_x = d_xcenter + 85
    ax1.plot([caliper_x, caliper_x], [c_ymin, c_ymax], color="#F59E0B", linewidth=3.5, solid_capstyle="round")
    ax1.plot([caliper_x - 10, caliper_x + 10], [c_ymin, c_ymin], color="#F59E0B", linewidth=2.5)
    ax1.plot([caliper_x - 10, caliper_x + 10], [c_ymax, c_ymax], color="#F59E0B", linewidth=2.5)

    caliper_x_d = d_xcenter + 145
    ax1.plot([caliper_x_d, caliper_x_d], [d_ymin, d_ymax], color="#16A34A", linewidth=3.5, solid_capstyle="round")
    ax1.plot([caliper_x_d - 10, caliper_x_d + 10], [d_ymin, d_ymin], color="#16A34A", linewidth=2.5)
    ax1.plot([caliper_x_d - 10, caliper_x_d + 10], [d_ymax, d_ymax], color="#16A34A", linewidth=2.5)

    # Caliper Labels (Non-overlapping, white pills)
    ax1.text(
        caliper_x - 14, (c_ymin + c_ymax) / 2,
        f"h_cup\n{hc}px",
        color="#B45309", fontsize=9.5, fontweight="bold", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#F59E0B", linewidth=1.5, alpha=0.98)
    )
    ax1.text(
        caliper_x_d + 14, (d_ymin + d_ymax) / 2,
        f"h_disc\n{hd}px",
        color="#15803D", fontsize=9.5, fontweight="bold", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.5, alpha=0.98)
    )

    # Inset labels on Panel 2
    ax1.text(
        inset_x + 12, inset_y + 28,
        "ZOOM-IN: ANNOTATED OD / OC",
        color="#047857", fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.5, alpha=0.98)
    )
    ax1.text(
        inset_x + 12, inset_y + inset_size - 22,
        f"Measured Vertical CDR = {cdr_val:.2f}",
        color="#FFFFFF", fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#DC2626", edgecolor="#B91C1C", linewidth=1.2, alpha=0.98)
    )

    # Title Banner for Panel 2
    ax1.text(
        30, 48,
        "(2) OUTPUT: CLINICAL-AWARE PIPELINE ANNOTATION (M5a-auto-OOF)",
        color="#0F172A", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.45", facecolor="#F8FAFC", edgecolor="#10B981", linewidth=2.0)
    )

    # Diagnostic HUD Box (Bottom-Left, White Card with Red Alert Border)
    ax1.text(
        30, disp_size - 40,
        hud_text,
        family="monospace",
        color="#0F172A", fontsize=9.2, fontweight="medium", va="bottom",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFFFFF", edgecolor="#DC2626", linewidth=2.2, alpha=0.98)
    )

    # Legend at bottom right (below inset, White Card)
    ax1.legend(
        handles=legend_elements, loc="center right",
        bbox_to_anchor=(0.96, 0.45),
        facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#0F172A",
        fontsize=9.5, framealpha=0.98
    )

    # Save 2x1 grid
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "input_vs_output_2x1_grid.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Saved white-theme 2x1 grid figure to: {out_file}")

    # Optionally copy to artifacts directory if available
    artifact_env = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")
    artifact_dir = Path(artifact_env) if artifact_env else Path("/home/dekii2275/.gemini/antigravity-ide/brain/71125d97-fcba-4816-8c2c-08749fdea017")
    if artifact_dir.is_dir():
        artifact_copy = artifact_dir / "input_vs_output_2x1_grid.png"
        shutil.copyfile(out_file, artifact_copy)
        print(f"Copied figure to IDE artifacts: {artifact_copy}")

    # ─────────────────────────────────────────────────────────────────────────────
    # BUILD 1x2 HORIZONTAL SIDE-BY-SIDE FIGURE (WHITE THEME)
    # ─────────────────────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 10.5), facecolor="#FFFFFF")
    plt.subplots_adjust(wspace=0.03, top=0.94, bottom=0.03, left=0.02, right=0.98)

    # Left: Input
    axes2[0].set_facecolor("#FFFFFF")
    axes2[0].imshow(p1_img)
    axes2[0].set_xlim(0, disp_size)
    axes2[0].set_ylim(disp_size, 0)
    axes2[0].axis("off")

    rect_roi2 = patches.Rectangle(
        (cx_min, cy_min), cx_max - cx_min, cy_max - cy_min,
        linewidth=2.5, edgecolor="#D97706", facecolor="none", linestyle="--", alpha=0.95
    )
    axes2[0].add_patch(rect_roi2)
    axes2[0].text(
        cx_min, cy_min - 12, "OPTIC DISC ROI",
        color="#B45309", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#D97706", linewidth=1.5, alpha=0.98)
    )
    axes2[0].text(
        inset_x + 12, inset_y + 28, "ZOOM-IN: OPTIC DISC (RAW)",
        color="#B45309", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#D97706", linewidth=1.5, alpha=0.98)
    )
    axes2[0].text(
        25, 45, "(A) INPUT: COLOR FUNDUS PHOTOGRAPHY",
        color="#0F172A", fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#F8FAFC", edgecolor="#0284C7", linewidth=2.0)
    )
    axes2[0].text(
        25, disp_size - 40, meta_text,
        color="#1E293B", fontsize=9.5, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.5, alpha=0.98)
    )

    # Right: Output
    axes2[1].set_facecolor("#FFFFFF")
    axes2[1].imshow(p2_img)
    axes2[1].set_xlim(0, disp_size)
    axes2[1].set_ylim(disp_size, 0)
    axes2[1].axis("off")

    axes2[1].plot([caliper_x, caliper_x], [c_ymin, c_ymax], color="#F59E0B", linewidth=3.5, solid_capstyle="round")
    axes2[1].plot([caliper_x - 10, caliper_x + 10], [c_ymin, c_ymin], color="#F59E0B", linewidth=2.5)
    axes2[1].plot([caliper_x - 10, caliper_x + 10], [c_ymax, c_ymax], color="#F59E0B", linewidth=2.5)

    axes2[1].plot([caliper_x_d, caliper_x_d], [d_ymin, d_ymax], color="#16A34A", linewidth=3.5, solid_capstyle="round")
    axes2[1].plot([caliper_x_d - 10, caliper_x_d + 10], [d_ymin, d_ymin], color="#16A34A", linewidth=2.5)
    axes2[1].plot([caliper_x_d - 10, caliper_x_d + 10], [d_ymax, d_ymax], color="#16A34A", linewidth=2.5)

    axes2[1].text(
        caliper_x - 14, (c_ymin + c_ymax) / 2, f"h_cup\n{hc}px",
        color="#B45309", fontsize=9.5, fontweight="bold", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#F59E0B", linewidth=1.5, alpha=0.98)
    )
    axes2[1].text(
        caliper_x_d + 14, (d_ymin + d_ymax) / 2, f"h_disc\n{hd}px",
        color="#15803D", fontsize=9.5, fontweight="bold", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=1.5, alpha=0.98)
    )
    axes2[1].text(
        inset_x + 12, inset_y + 28, "ZOOM-IN: ANNOTATED OD / OC",
        color="#047857", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.5, alpha=0.98)
    )
    axes2[1].text(
        inset_x + 12, inset_y + inset_size - 22, f"Measured Vertical CDR = {cdr_val:.2f}",
        color="#FFFFFF", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#DC2626", edgecolor="#B91C1C", linewidth=1.2, alpha=0.98)
    )
    axes2[1].text(
        25, 45, "(B) OUTPUT: CLINICAL-AWARE PIPELINE ANNOTATION",
        color="#0F172A", fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#F8FAFC", edgecolor="#10B981", linewidth=2.0)
    )
    axes2[1].text(
        25, disp_size - 30, hud_text, family="monospace",
        color="#0F172A", fontsize=8.2, fontweight="medium", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor="#DC2626", linewidth=2.0, alpha=0.98)
    )
    axes2[1].legend(
        handles=legend_elements, loc="center right",
        bbox_to_anchor=(0.96, 0.45),
        facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#0F172A",
        fontsize=9.5, framealpha=0.98
    )

    out_file_h = out_dir / "input_vs_output_side_by_side.png"
    plt.savefig(out_file_h, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    if artifact_dir.is_dir():
        artifact_copy_h = artifact_dir / "input_vs_output_side_by_side.png"
        shutil.copyfile(out_file_h, artifact_copy_h)
    print(f"Saved white-theme side-by-side figure to: {out_file_h}")

if __name__ == "__main__":
    generate_visualization()
