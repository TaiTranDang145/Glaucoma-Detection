"""Comprehensive computational efficiency benchmark for lightweight GON classification.

Measures:
1. Total & trainable parameter count
2. Model weight size on disk (MB)
3. FLOPs and MACs (torch.utils.flop_counter.FlopCounterMode)
4. Inference latency (ms/image) and throughput (FPS) on RTX 3050 GPU and CPU (batch size = 1)
5. Comparison against standard architectures (ResNet-50, MobileNetV3-Large) and foundation models (DINOv2)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.mobilenetv3 import MobileNetV3Classifier


class M5aFusedModel(nn.Module):
    """End-to-end wrapper for M5a (Global Fundus + CDR scalar)."""

    def __init__(self, backbone: MobileNetV3Classifier):
        super().__init__()
        self.backbone = backbone
        in_dim = 1024 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.2),
            nn.Linear(in_dim, 1),
        )

    def forward(self, img: torch.Tensor, cdr: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.extract_features(img)
        fused = torch.cat([feat, cdr], dim=1)
        return self.head(fused)


def measure_flops_params(model: nn.Module, dummy_inputs: tuple) -> dict:
    """Measures parameters and FLOPs using PyTorch 2.x native FlopCounterMode."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            model(*dummy_inputs)
    total_flops = fcm.get_total_flops()
    macs = total_flops / 2.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_m": round(total_params / 1e6, 3),
        "total_flops": total_flops,
        "mflops": round(total_flops / 1e6, 2),
        "gflops": round(total_flops / 1e9, 4),
        "mmacs": round(macs / 1e6, 2),
    }


def measure_latency_gpu(model: nn.Module, dummy_inputs: tuple, warmup: int = 50, runs: int = 200) -> dict:
    """Measures GPU latency with CUDA events."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    dev_inputs = tuple(x.to(device) for x in dummy_inputs)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*dev_inputs)
    torch.cuda.synchronize()

    # Timing
    timings = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(runs):
            start_event.record()
            _ = model(*dev_inputs)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))

    timings = np.array(timings)
    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    median_ms = float(np.median(timings))
    p95_ms = float(np.percentile(timings, 95))
    fps = 1000.0 / mean_ms

    return {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "fps": round(fps, 1),
    }


def measure_latency_cpu(model: nn.Module, dummy_inputs: tuple, warmup: int = 20, runs: int = 50) -> dict:
    """Measures CPU latency with perf_counter."""
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    dev_inputs = tuple(x.to(device) for x in dummy_inputs)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*dev_inputs)

    # Timing
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(*dev_inputs)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    timings = np.array(timings)
    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    median_ms = float(np.median(timings))
    p95_ms = float(np.percentile(timings, 95))
    fps = 1000.0 / mean_ms

    return {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "fps": round(fps, 1),
    }


def main():
    print("=== Benchmarking Computational Efficiency ===")
    out_dir = Path("results/efficiency")
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_cdr = torch.randn(1, 1)

    results = {}

    # 1. M2 Global MobileNetV3-Small
    print("\n--- 1. Evaluating M2 (MobileNetV3-Small Global) ---")
    m2 = MobileNetV3Classifier(pretrained=False, dropout_rate=0.2, num_classes=1)
    flops_m2 = measure_flops_params(m2, (dummy_img,))
    lat_gpu_m2 = measure_latency_gpu(m2, (dummy_img,))
    lat_cpu_m2 = measure_latency_cpu(m2, (dummy_img,))

    # Save dummy state dict to measure checkpoint size
    ckpt_path = out_dir / "temp_m2.pt"
    torch.save(m2.state_dict(), ckpt_path)
    ckpt_size_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    results["M2_Global_MobileNetV3_Small"] = {
        "params": flops_m2,
        "checkpoint_size_mb": ckpt_size_mb,
        "latency_gpu_rtx3050": lat_gpu_m2,
        "latency_cpu": lat_cpu_m2,
    }

    # 2. M5a Global + CDR Fusion
    print("\n--- 2. Evaluating M5a (MobileNetV3-Small + CDR Fusion) ---")
    m5a = M5aFusedModel(m2)
    flops_m5a = measure_flops_params(m5a, (dummy_img, dummy_cdr))
    lat_gpu_m5a = measure_latency_gpu(m5a, (dummy_img, dummy_cdr))
    lat_cpu_m5a = measure_latency_cpu(m5a, (dummy_img, dummy_cdr))

    ckpt_path = out_dir / "temp_m5a.pt"
    torch.save(m5a.state_dict(), ckpt_path)
    ckpt_size_m5a_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    results["M5a_Global_CDR_Fusion"] = {
        "params": flops_m5a,
        "checkpoint_size_mb": ckpt_size_m5a_mb,
        "latency_gpu_rtx3050": lat_gpu_m5a,
        "latency_cpu": lat_cpu_m5a,
    }

    # 3. Standard ResNet-50 (Reference)
    print("\n--- 3. Evaluating Standard ResNet-50 (Reference Baseline) ---")
    import torchvision.models as models
    resnet50 = models.resnet50()
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 1)
    flops_rn50 = measure_flops_params(resnet50, (dummy_img,))
    lat_gpu_rn50 = measure_latency_gpu(resnet50, (dummy_img,))
    lat_cpu_rn50 = measure_latency_cpu(resnet50, (dummy_img,))

    ckpt_path = out_dir / "temp_rn50.pt"
    torch.save(resnet50.state_dict(), ckpt_path)
    ckpt_size_rn50_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    results["Reference_ResNet50"] = {
        "params": flops_rn50,
        "checkpoint_size_mb": ckpt_size_rn50_mb,
        "latency_gpu_rtx3050": lat_gpu_rn50,
        "latency_cpu": lat_cpu_rn50,
    }

    # 4. MobileNetV3-Large (Reference)
    print("\n--- 4. Evaluating MobileNetV3-Large (Reference) ---")
    import timm
    mnv3_large = timm.create_model("mobilenetv3_large_100", num_classes=1)
    flops_large = measure_flops_params(mnv3_large, (dummy_img,))
    lat_gpu_large = measure_latency_gpu(mnv3_large, (dummy_img,))
    lat_cpu_large = measure_latency_cpu(mnv3_large, (dummy_img,))

    results["Reference_MobileNetV3_Large"] = {
        "params": flops_large,
        "checkpoint_size_mb": round(flops_large["total_params"] * 4 / (1024 * 1024), 2),
        "latency_gpu_rtx3050": lat_gpu_large,
        "latency_cpu": lat_cpu_large,
    }

    # Save to JSON
    out_file = out_dir / "efficiency_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")

    # Print Markdown Table
    print("\n### Computational Efficiency Comparison Table")
    print("| Model Architecture | Params (M) | Size (MB) | MFLOPs | MMACs | GPU Latency (ms) | GPU FPS | CPU Latency (ms) | CPU FPS |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, m in results.items():
        p = m["params"]
        gpu = m["latency_gpu_rtx3050"]
        cpu = m["latency_cpu"]
        print(f"| **{name}** | {p['params_m']}M | {m['checkpoint_size_mb']} MB | {p['mflops']} | {p['mmacs']} | {gpu['mean_ms']} ± {gpu['std_ms']} | {gpu['fps']} | {cpu['mean_ms']} ± {cpu['std_ms']} | {cpu['fps']} |")


if __name__ == "__main__":
    main()
