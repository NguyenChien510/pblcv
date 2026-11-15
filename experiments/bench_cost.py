"""
Measures the cost claim of the paper (section 5.1, "Về tài nguyên") instead of quoting it:
peak GPU memory of one training step and inference latency per video for

  ours     SpaceTimeViT video encoder (6 divided space-time blocks, 16 frames, 224x224), as trained
           (gradient checkpointing on) and without checkpointing
  vit_s3d  the video side of MFGF [24]: ViT-B/16 on 4 frames + S3D on 16 frames (torchvision implementations)

Text encoders are identical (BERT-base) in both systems and are excluded. Mixed precision (FP16) as in training.

    python experiments/bench_cost.py --batch 8 --out results/bench.json
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, save_json  # noqa: E402

sys.path.insert(0, str(REPO))
import models.spacetime_vit as stv  # noqa: E402


class ViTS3D(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vit_b_16
        from torchvision.models.video import s3d
        self.vit = vit_b_16(weights=None)
        self.s3d = s3d(weights=None)

    def forward(self, frames16):
        B, T = frames16.shape[:2]
        keyframes = frames16[:, torch.linspace(0, T - 1, 4).long()]              # 4 discrete frames
        f_vis = self.vit(keyframes.flatten(0, 1)).view(B, 4, -1).mean(1)
        f_mot = self.s3d(frames16.permute(0, 2, 1, 3, 4))                         # [B, C, T, H, W]
        return torch.cat([f_vis, f_mot], dim=-1)


class Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = stv.SpaceTimeViT(num_frames=16, depth=6, num_heads=12, pretrained=False)

    def forward(self, frames16):
        a, m, v = self.enc(frames16)
        return v


def peak_train_gb(model, batch, device, steps=3):
    model.train().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(batch, 16, 3, 224, 224, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    for _ in range(steps):
        with torch.amp.autocast("cuda"):
            loss = model(x).float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    del opt, x
    torch.cuda.empty_cache()
    return peak


@torch.no_grad()
def infer_ms(model, device, reps=30):
    model.eval().to(device)
    x = torch.randn(1, 16, 3, 224, 224, device=device)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "results" / "bench.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"device": device.type, "batch": args.batch, "frames": 16, "img_size": 224, "precision": "fp16 autocast"}
    if device.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name(device)
    for name, ctor in (("ours", Ours), ("vit_s3d", ViTS3D)):
        model = ctor()
        r = {"params_m": sum(p.numel() for p in model.parameters()) / 1e6}
        if device.type == "cuda":
            try:
                r["train_peak_gb"] = peak_train_gb(model, args.batch, device)
            except torch.cuda.OutOfMemoryError:
                r["train_peak_gb"] = float("nan")
                r["oom"] = True
                torch.cuda.empty_cache()
            if name == "ours":   # same encoder without gradient checkpointing (architecture cost only)
                orig = stv.checkpoint.checkpoint
                stv.checkpoint.checkpoint = lambda fn, *a, **k: fn(*a)
                try:
                    r["train_peak_gb_no_checkpointing"] = peak_train_gb(model, args.batch, device)
                except torch.cuda.OutOfMemoryError:
                    r["train_peak_gb_no_checkpointing"] = float("nan")
                    torch.cuda.empty_cache()
                finally:
                    stv.checkpoint.checkpoint = orig
        r["infer_ms_per_video"] = infer_ms(model, device)
        res[name] = r
        print(name, r)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type != "cuda":
        print("[!] no GPU: only CPU latency was measured; the paper needs the GPU numbers")
    save_json(res, args.out)
    print(f"[+] saved {args.out}")


if __name__ == "__main__":
    main()
