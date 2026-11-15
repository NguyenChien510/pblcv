"""
Text-image person pre-training on the CUHK-PEDES-style set of the Kaggle dataset (recipe m3c, stage 1).

    captions/train.json : {"train_000000.jpg": {"id": 0, "text": "..."}, ...}   (238,768 images, one caption each)
    images/<name>.jpg

Only the parts that an image can teach are trained: BERT, the spatial layers of the Space-Time ViT, app_head and
fusion_head, with L = InfoNCE(T_full, V_video) + lambda_app * InfoNCE(T_app, V_app). The temporal attention, S3D
and mot_head are frozen (models/mfgf_video.py image mode: one frame, V_mot = 0). No TVPReid data is read here.

The schedule is time-boxed: progress = max(iter / total_iters, elapsed / max_hours), cosine LR on progress, so the
run ends inside the Kaggle budget whatever the GPU speed. Output (train.py checkpoint format, usable with
train.py --init_from):  <out_dir>/best.pth, <out_dir>/train_history.json ({"epochs": [1]} marks the run finished).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import swap_left_right           # noqa: E402
from data.text_decomposer import TextDecomposer    # noqa: E402
from loss.criterion import DAMRCriterion           # noqa: E402
from models.damr_main import DAMRModel             # noqa: E402

FROZEN_KEYS = ("timeattn", "norm3", "temporal_embed", "video_encoder.s3d.", "mot_head")


def find_cuhk_root(data_root: str):
    """<root>/captions/train.json + <root>/images, searched next to / above the TVPReid data root."""
    start = Path(data_root).resolve()
    for base in [start, *start.parents][:5]:
        for cand in (base, base / "CUHK-PEDES", base / "CUHK-PEDES" / "CUHK-PEDES"):
            if (cand / "captions" / "train.json").exists() and (cand / "images").is_dir():
                return cand
    return None


class CuhkDataset(Dataset):
    def __init__(self, root: Path, dsca_mode: str = "clause", img_size: int = 224, limit: int = 0):
        ann = json.loads((root / "captions" / "train.json").read_text(encoding="utf-8"))
        self.items = [(name, str(v.get("text", "")).strip()) for name, v in ann.items() if str(v.get("text", "")).strip()]
        if limit:
            self.items = self.items[:limit]
        self.img_dir = root / "images"
        self.decomposer = TextDecomposer(mode=dsca_mode)
        self.tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, caption = self.items[i]
        try:
            img = self.tf(Image.open(self.img_dir / name).convert("RGB"))
        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
        if random.random() < 0.5:                              # same clip-level flip rule as recipe m3
            img = torch.flip(img, dims=[-1])
            caption = swap_left_right(caption)
        q_app, _ = self.decomposer.decompose(caption)
        return {"image": img.unsqueeze(0), "caption": caption, "q_app": q_app, "key": name}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="TVPReid data root; the CUHK folder is searched next to it")
    ap.add_argument("--cuhk_root", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=4.0, help="passes over the set if the time budget allows")
    ap.add_argument("--max_hours", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lambda_app", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N images")
    ap.add_argument("--no_pretrained", action="store_true", help="smoke test: random init (no downloads)")
    ap.add_argument("--save_every_min", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_history.json").exists() and (out / "best.pth").exists():
        print(f"[=] {out}: CUHK pre-training already finished")
        return
    root = Path(args.cuhk_root) if args.cuhk_root else find_cuhk_root(args.data_root)
    if root is None:
        sys.exit(f"[x] no CUHK folder (captions/train.json + images/) found near {args.data_root}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CuhkDataset(root, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    total_iters = max(1, int(args.epochs * len(loader)))
    print(f"[*] CUHK pre-training: {len(ds)} images from {root} | batch {args.batch_size} | {len(loader)} it/epoch | "
          f"up to {total_iters} it or {args.max_hours} h | device {device}")

    cfg = {"model": {"backbone": "mfgf", "embed_dim": 768, "dsca_mode": "clause", "mim_mode": "centered",
                     "temperature": 0.07, "use_dsca": True, "use_temporal_attn": True,
                     "pretrained_vit": not args.no_pretrained},
           "train": {"seed": args.seed, "stage": "cuhk_pretrain", "batch_size": args.batch_size, "lr": args.lr}}
    model = DAMRModel(backbone="mfgf", mim_mode="centered", pretrained_vit=not args.no_pretrained).to(device)
    cfg["model"]["visual_weights"] = getattr(model.video_encoder, "weights_source", None)
    criterion = DAMRCriterion(temperature=0.07, lambda_app=args.lambda_app, lambda_mot=0.0, lambda_mim=0.0,
                              learnable_tau=True).to(device)

    backbone, heads = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in FROZEN_KEYS):
            p.requires_grad_(False)
            continue
        (backbone if ("text_encoder" in name or "video_encoder.vit." in name) else heads).append(p)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr * 0.2, "weight_decay": args.weight_decay},
        {"params": heads, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": list(criterion.parameters()), "lr": args.lr, "weight_decay": 0.0}])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    it, elapsed0 = 0, 0.0
    latest = out / "latest.pth"
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); criterion.load_state_dict(ck["criterion_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"]); scaler.load_state_dict(ck["scaler_state_dict"])
        it, elapsed0 = ck["iter"], ck["elapsed_h"]
        print(f"[+] resumed at iteration {it} ({elapsed0:.2f} h used)")

    def save(path, full):
        data = {"model_state_dict": model.state_dict(), "criterion_state_dict": criterion.state_dict(),
                "config": cfg, "epoch": 1, "iter": it, "elapsed_h": elapsed0 + (time.time() - t0) / 3600}
        if full:
            data.update(optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict())
        tmp = f"{path}.tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)

    model.train()
    for m in model.video_encoder.s3d.modules():
        m.eval()
    t0, last_save, warm = time.time(), time.time(), 500
    log, done = [], False
    while not done:
        for batch in loader:
            hours = elapsed0 + (time.time() - t0) / 3600
            progress = max(it / total_iters, hours / args.max_hours)
            if progress >= 1.0:
                done = True
                break
            scale = min(1.0, (it + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
            for g, b in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b * scale
            frames = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                o = model(batch["caption"], frames, q_app=batch["q_app"], q_mot=batch["q_app"])
                loss, parts = criterion(T=o["T_full"], V=o["V_video"], V_app=o["V_app"], V_mot=None,
                                        l_mim=torch.zeros((), device=device), T_app=o["T_app"],
                                        video_keys=batch["key"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update()
            it += 1
            if it % 50 == 0 or it == 1:
                speed = (it - (log[0][0] if log else 0)) / max(1e-6, time.time() - (log[0][1] if log else t0))
                if not log:
                    log.append((it, time.time()))
                print(f"\r[cuhk] it {it}/{total_iters} | {hours:.2f}/{args.max_hours} h | progress {100 * progress:4.1f}% | "
                      f"{speed:.2f} it/s | Loss: {parts['loss_total']:.4f} (App: {parts['loss_app']:.4f}) | "
                      f"tau {parts.get('tau', 0):.3f}", end="", flush=True)
            if it % 1000 == 0:
                print()
            if time.time() - last_save > args.save_every_min * 60:
                save(latest, full=True); last_save = time.time()
    print(f"\n[+] CUHK pre-training done: {it} iterations ({it * args.batch_size / max(1, len(ds)):.2f} epochs), "
          f"{elapsed0 + (time.time() - t0) / 3600:.2f} h")
    for p in model.parameters():
        p.requires_grad_(True)
    save(out / "best.pth", full=False)
    (out / "train_history.json").write_text(json.dumps({"epochs": [1], "iters": it, "images": len(ds),
                                                        "samples_seen": it * args.batch_size}), encoding="utf-8")
    latest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
