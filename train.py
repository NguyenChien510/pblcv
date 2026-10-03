"""
Training Script for DAMR-LLM (srccv)
Disentangled Appearance-Motion Representation Learning for Text-to-Video Person Retrieval

Features:
  - Supports selecting specific sub-dataset (TVPReid-Duke, TVPReid-iLIDs, TVPReid-PRID, or all)
  - Auto-creates dedicated subfolders in checkpoints/<sub_dataset>/ and reports/<sub_dataset>/
  - AdamW Optimizer with Warmup + Cosine Annealing Learning Rate Scheduler (Faster Learning)
  - PyTorch Automatic Mixed Precision (AMP FP16)
  - High GPU Utilization on Colab: cuDNN benchmark, non_blocking CUDA transfers, prefetch
  - Single-line real-time progress update (2s interval, carriage return \r, no line jumping)
  - Configurable eval_interval to skip unnecessary evaluation overhead
  - Auto-saves latest.pth and best.pth into sub-dataset folder
  - Automatic export of 3 individual benchmark charts & 1 consolidated 4-in-1 dashboard into reports/<sub_dataset>/
"""

import os
import sys
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass
import argparse
import time
import json
import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from utils.paths import paths
from data.dataset import create_dataloaders
from models.damr_main import DAMRModel
from loss.criterion import DAMRCriterion
from utils.metrics import compute_similarity_matrix, evaluate_rank_metrics
from utils.plot_benchmarks import save_training_history, plot_training_dashboard

DEFAULT_CONFIG = {
    "model": {
        "embed_dim": 512, "keyframe_num": 4, "motion_frame_num": 16, "img_size": 224,
        "temperature": 0.05, "lambda_mot": 1.0, "lambda_mim": 0.2, "lambda_cross_neg": 0.1
    },
    "data": {
        "dataset_name": "TVPReid_dataset",
        "sub_dataset": "TVPReid-Duke",
        "batch_size": 32,
        "num_workers": 2
    },
    "train": {
        "seed": 42, "device": "cuda", "epochs": 30, "warmup_epochs": 1,
        "lr": 0.0003, "min_lr": 0.00001, "weight_decay": 0.0001,
        "grad_clip_norm": 5.0, "use_amp": True, "eval_interval": 2
    },
    "eval": {
        "batch_size": 64
    }
}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def get_lr_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, base_lr: float, min_lr: float):
    """Creates a Warmup + Cosine Annealing Learning Rate Scheduler for accelerated learning."""
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (float(epoch + 1) / float(max(1, warmup_epochs)))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = min_lr / max(base_lr, 1e-8)
            return min_ratio + (1.0 - min_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
    total_epochs: int = 30,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 2
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_app = 0.0
    total_mot = 0.0
    total_mim = 0.0
    start_time = time.time()
    last_update_time = 0.0
    total_steps = len(dataloader)

    current_lr = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        captions = batch.get("caption", batch.get("q_app"))
        q_app = batch.get("q_app", None)
        q_mot = batch.get("q_mot", None)
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(captions, video_frames, q_app=q_app, q_mot=q_mot)
                loss, breakdown = criterion(
                    T=out["T"], V=out["V_video"],
                    V_app=out["V_app"], V_mot=out["V_mot"],
                    l_mim=out["l_mim"],
                    T_app=out.get("T_app"), T_mot=out.get("T_mot")
                )
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            out = model(captions, video_frames, q_app=q_app, q_mot=q_mot)
            loss, breakdown = criterion(
                T=out["T"], V=out["V_video"],
                V_app=out["V_app"], V_mot=out["V_mot"],
                l_mim=out["l_mim"],
                T_app=out.get("T_app"), T_mot=out.get("T_mot")
            )
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        total_app += breakdown.get("loss_app", 0.0)
        total_mot += breakdown.get("loss_mot", 0.0)
        total_mim += breakdown.get("loss_mim", 0.0)

        # Single-line in-place update every 2 seconds or at final step
        now = time.time()
        if (now - last_update_time >= 2.0) or (step + 1) == total_steps:
            last_update_time = now
            elapsed = now - start_time
            speed = (step + 1) / elapsed if elapsed > 0 else 1.0
            rem_sec = int((total_steps - (step + 1)) / speed)
            m, s = divmod(rem_sec, 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            pct = int(((step + 1) / total_steps) * 100)

            msg = (
                f"\r[{epoch:02d}/{total_epochs:02d}] {step+1:03d}/{total_steps:03d} [{pct:2d}% | ETA: {eta_str} | {speed:.2f} it/s | lr: {current_lr:.1e}] - "
                f"Loss: {loss.item():.4f} (App: {breakdown['loss_app']:.4f}, Mot: {breakdown['loss_mot']:.4f}, MIM: {breakdown['loss_mim']:.4f})"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    avg_app = total_app / total_steps if total_steps > 0 else 0.0
    avg_mot = total_mot / total_steps if total_steps > 0 else 0.0
    avg_mim = total_mim / total_steps if total_steps > 0 else 0.0
    print(f"--> Epoch {epoch:02d}/{total_epochs:02d} Complete | Average Loss: {avg_loss:.4f} | Elapsed: {time.time() - start_time:.2f}s")
    return avg_loss, avg_app, avg_mot, avg_mim


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 0.5
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluates Text-to-Video Person Retrieval following standard TVPR benchmark protocol:
      - All N_queries text queries are evaluated.
      - Gallery contains UNIQUE video tracks (deduplicated by video_id).
      - Multi-level Disentangled Concept Alignment scoring:
          Score = 0.50 * CosSim(T_full, V_video) + 0.35 * CosSim(T_app, V_app) + 0.15 * CosSim(T_mot, V_mot)
    """
    model.eval()
    all_t = []
    all_t_app = []
    all_t_mot = []
    query_pids = []

    gallery_v = []
    gallery_v_app = []
    gallery_v_mot = []
    gallery_pids = []
    visited_videos = set()

    for batch in dataloader:
        captions = batch.get("caption", batch.get("q_app"))
        q_app = batch.get("q_app", None)
        q_mot = batch.get("q_mot", None)
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)
        video_ids = batch.get("video_id")
        pids = batch["person_id"].numpy()

        out = model(captions, video_frames, q_app=q_app, q_mot=q_mot)

        # 1. Text queries (all N_queries evaluated)
        all_t.append(out["T"].cpu())
        all_t_app.append(out["T_app"].cpu())
        all_t_mot.append(out["T_mot"].cpu())
        query_pids.extend(pids)

        # 2. Gallery videos (Deduplicated: exactly 1 entry per unique video track)
        v_vid_cpu = out["V_video"].cpu()
        v_app_cpu = out["V_app"].cpu()
        v_mot_cpu = out["V_mot"].cpu()

        for i in range(len(pids)):
            vid = video_ids[i] if video_ids is not None else pids[i]
            if vid not in visited_videos:
                visited_videos.add(vid)
                gallery_v.append(v_vid_cpu[i:i+1])
                gallery_v_app.append(v_app_cpu[i:i+1])
                gallery_v_mot.append(v_mot_cpu[i:i+1])
                gallery_pids.append(pids[i])

    T_full = torch.cat(all_t, dim=0)
    T_app = torch.cat(all_t_app, dim=0)
    T_mot = torch.cat(all_t_mot, dim=0)
    pids_text = np.array(query_pids)

    V = torch.cat(gallery_v, dim=0)
    V_app = torch.cat(gallery_v_app, dim=0)
    V_mot = torch.cat(gallery_v_mot, dim=0)
    pids_video = np.array(gallery_pids)

    T_full_n = F.normalize(T_full, p=2, dim=-1)
    T_app_n = F.normalize(T_app, p=2, dim=-1)
    T_mot_n = F.normalize(T_mot, p=2, dim=-1)

    V_n = F.normalize(V, p=2, dim=-1)
    V_app_n = F.normalize(V_app, p=2, dim=-1)
    V_mot_n = F.normalize(V_mot, p=2, dim=-1)

    sim_joint = torch.matmul(T_full_n, V_n.t())
    sim_app = torch.matmul(T_app_n, V_app_n.t())
    sim_mot = torch.matmul(T_mot_n, V_mot_n.t())

    # Multi-level Disentangled Concept Scoring
    sim_matrix = (0.50 * sim_joint + 0.35 * sim_app + 0.15 * sim_mot).cpu().numpy()
    metrics = evaluate_rank_metrics(sim_matrix, pids_text, pids_video)
    return metrics, sim_matrix, pids_video, pids_text


def main():
    parser = argparse.ArgumentParser(description="Train DAMR-LLM Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root folder")
    parser.add_argument("--sub_dataset", type=str, default=None, help="Choose specific subdata: TVPReid-Duke, TVPReid-iLIDs, TVPReid-PRID, or all")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size override")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader num_workers override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=None, help="Validation interval in epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path ('best', 'latest', or path/to/file.pth)")
    parser.add_argument("--pretrained_vit", action="store_true", default=None, help="Load ImageNet pretrained weights into ViT spatial layers")
    parser.add_argument("--no_pretrained_vit", dest="pretrained_vit", action="store_false", help="Disable ViT spatial pretraining")
    
    # Use parse_known_args to gracefully handle any extra/unknown flags
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[*] Note: Ignoring unparsed extra CLI arguments: {unknown}")

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_cfg = yaml.safe_load(f)
                if loaded_cfg and isinstance(loaded_cfg, dict):
                    for sec, vals in loaded_cfg.items():
                        if sec in cfg and isinstance(vals, dict):
                            cfg[sec].update(vals)
                        else:
                            cfg[sec] = vals
        except Exception as e:
            print(f"[!] Warning reading yaml config: {e}")

    # Determine sub-dataset and configure dedicated folders defensively
    dataset_name = cfg.get("data", {}).get("dataset_name", "TVPReid_dataset")
    sub_dataset = args.sub_dataset or cfg.get("data", {}).get("sub_dataset", "TVPReid-Duke")
    
    if hasattr(paths, "set_sub_dataset"):
        paths.set_sub_dataset(sub_dataset)
    else:
        norm_sub = str(sub_dataset).strip()
        paths.sub_dataset = norm_sub
        sub_folder = "" if norm_sub.lower() in ["all", "combined", "*"] else norm_sub
        paths.checkpoints_dir = (paths.srccv_dir / "checkpoints" / (sub_folder if sub_folder else "all"))
        paths.reports_dir = (paths.srccv_dir / "reports" / (sub_folder if sub_folder else "all"))
        paths.outputs_dir = (paths.srccv_dir / "outputs" / (sub_folder if sub_folder else "all"))
        paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        paths.reports_dir.mkdir(parents=True, exist_ok=True)
        paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        paths.get_checkpoint_path = lambda filename="best.pth": paths.checkpoints_dir / filename
        paths.get_report_path = lambda filename="eval_results.json": paths.reports_dir / filename

    # Determine data_root
    custom_data_root = args.data_root or cfg.get("data", {}).get("data_root")
    if custom_data_root and custom_data_root != "auto":
        if hasattr(paths, "set_data_root"):
            paths.set_data_root(custom_data_root)
        else:
            paths.data_root = Path(custom_data_root)

    batch_size = args.batch_size or int(cfg.get("data", {}).get("batch_size", 8))
    grad_accum_steps = args.grad_accum_steps or int(cfg.get("train", {}).get("grad_accum_steps", 2))
    eval_batch_size = int(cfg.get("eval", {}).get("batch_size", 16))
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("data", {}).get("num_workers", 2)
    lr = args.lr or float(cfg.get("train", {}).get("lr", 0.0001))
    min_lr = float(cfg.get("train", {}).get("min_lr", 0.000001))
    warmup_epochs = int(cfg.get("train", {}).get("warmup_epochs", 2))
    epochs = args.epochs or int(cfg.get("train", {}).get("epochs", 35))
    eval_interval = args.eval_interval or int(cfg.get("train", {}).get("eval_interval", 1))
    grad_clip_norm = float(cfg.get("train", {}).get("grad_clip_norm", 1.0))

    set_seed(cfg.get("train", {}).get("seed", 42))
    device = torch.device(cfg.get("train", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("🚀 SPACETIME-DAMR TRAINING PIPELINE INITIALIZATION")
    print(f"  • Computing Device  : {device} (cuDNN benchmark={torch.backends.cudnn.benchmark})")
    print(f"  • Primary Dataset   : {dataset_name}")
    print(f"  • Selected Sub-data : {getattr(paths, 'sub_dataset', sub_dataset)}")
    print(f"  • Checkpoints Dir   : {paths.checkpoints_dir}")
    print(f"  • Reports Dir       : {paths.reports_dir}")
    print(f"  • Data Root Path    : {paths.data_root}")
    print(f"  • Batch Size        : {batch_size} (Train, Accum={grad_accum_steps} -> Eff={batch_size*grad_accum_steps}) | {eval_batch_size} (Eval)")
    print(f"  • Learning Rate     : {lr} (Warmup {warmup_epochs} eps -> Cosine decay to {min_lr})")
    print(f"  • Eval Interval     : Every {eval_interval} epoch(s)")
    print("=" * 65 + "\n")

    # Create Dataloaders for specific sub_dataset
    active_sub = getattr(paths, "sub_dataset", sub_dataset)
    try:
        train_loader, val_loader = create_dataloaders(
            data_root=str(paths.data_root),
            dataset_name=dataset_name,
            sub_dataset=active_sub,
            batch_size=batch_size,
            num_workers=num_workers,
            keyframe_num=cfg.get("model", {}).get("keyframe_num", 4),
            motion_frame_num=cfg.get("model", {}).get("motion_frame_num", 16),
            img_size=cfg.get("model", {}).get("img_size", 224)
        )
    except TypeError:
        train_loader, val_loader = create_dataloaders(
            data_root=str(paths.data_root),
            batch_size=batch_size,
            num_workers=num_workers
        )

    # Initialize Model & Criterion (SpaceTimeViT + BERT)
    model_cfg = cfg.get("model", {})
    pretrained_vit_flag = getattr(args, "pretrained_vit", None)
    if pretrained_vit_flag is None:
        pretrained_vit_flag = model_cfg.get("pretrained_vit", True)

    model = DAMRModel(
        embed_dim=model_cfg.get("embed_dim", 768),
        text_model_name=model_cfg.get("text_model_name", "bert-base-uncased"),
        pretrained=model_cfg.get("pretrained", True),
        motion_frame_num=model_cfg.get("num_frames", model_cfg.get("motion_frame_num", 16)),
        img_size=model_cfg.get("img_size", 224),
        vit_depth=model_cfg.get("vit_depth", 6),
        vit_heads=model_cfg.get("vit_heads", 12),
        drop_rate=model_cfg.get("drop_rate", 0.1),
        pretrained_vit=pretrained_vit_flag
    ).to(device)

    criterion = DAMRCriterion(
        temperature=cfg.get("model", {}).get("temperature", 0.05),
        lambda_app=cfg.get("model", {}).get("lambda_app", 0.5),
        lambda_mot=cfg.get("model", {}).get("lambda_mot", 0.25),
        lambda_mim=cfg.get("model", {}).get("lambda_mim", 0.1)
    ).to(device)

    # 3-Tier Differential Learning Rates:
    # 1. Pretrained backbones (BERT + ViT spatial blocks + PatchEmbed): fine-tune gently at lr * 0.2 (2e-5)
    # 2. Freshly initialized heads (app_head, mot_head, fusion_head, motion_gru, app_attn, attn_temp): train at lr * 1.0 (1e-4)
    pretrained_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_pretrained = (
            "text_encoder" in name or
            "patch_embed" in name or
            "spatial_pos" in name or
            "cls_token" in name or
            ("video_encoder.blocks" in name and ("attn_spat" in name or "norm_spat" in name or "mlp" in name or "norm_mlp" in name))
        )
        if is_pretrained:
            pretrained_params.append(param)
        else:
            head_params.append(param)

    wd = float(cfg.get("train", {}).get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": lr * 0.2, "weight_decay": wd},
        {"params": head_params, "lr": lr * 1.0, "weight_decay": wd},
    ])
    scheduler = get_lr_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs, base_lr=lr, min_lr=min_lr)

    use_amp_flag = cfg.get("train", {}).get("use_amp", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_flag)

    history = {
        "epochs": [],
        "lr": [],
        "loss": [],
        "loss_app": [],
        "loss_mot": [],
        "loss_mim": [],
        "eval_epochs": [],
        "rank1": [],
        "rank5": [],
        "rank10": [],
        "rank50": [],
        "mdr": [],
        "sim_matrix": None
    }

    start_epoch = 1
    best_r1 = 0.0

    # Resume handling
    if args.resume:
        resume_path = args.resume
        if resume_path.lower() in ["best", "best_damr"]:
            resume_path = str(paths.get_checkpoint_path("best.pth"))
            if not os.path.exists(resume_path):
                resume_path = str(paths.get_checkpoint_path("best_damr.pth"))
        elif resume_path.lower() in ["latest", "last"]:
            resume_path = str(paths.get_checkpoint_path("latest.pth"))

        if os.path.exists(resume_path):
            print(f"[*] Resuming training state from checkpoint: {resume_path}")
            try:
                checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] and scaler:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "metrics" in checkpoint:
                best_r1 = checkpoint["metrics"].get("R@1", 0.0)
            if "history" in checkpoint and checkpoint["history"]:
                history = checkpoint["history"]
                for k in ["epochs", "lr", "loss", "loss_app", "loss_mot", "loss_mim", "eval_epochs", "rank1", "rank5", "rank10", "rank50", "mdr"]:
                    if k not in history:
                        history[k] = []
            resumed_lr = optimizer.param_groups[0]["lr"]
            print(f"[+] Successfully resumed from Epoch {start_epoch-1} (Previous Best Rank@1: {best_r1:.2f}% | Current LR: {resumed_lr:.2e})")
        else:
            print(f"[!] Warning: Specified resume checkpoint '{resume_path}' not found. Starting from Epoch 1.")

    for epoch in range(start_epoch, epochs + 1):
        avg_loss, avg_app, avg_mot, avg_mim = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler,
            device, epoch, epochs, use_amp=use_amp_flag, grad_clip_norm=grad_clip_norm,
            grad_accum_steps=grad_accum_steps
        )
        scheduler.step()

        history.setdefault("epochs", []).append(epoch)
        history.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
        history.setdefault("loss", []).append(avg_loss)
        history.setdefault("loss_app", []).append(avg_app)
        history.setdefault("loss_mot", []).append(avg_mot)
        history.setdefault("loss_mim", []).append(avg_mim)

        do_eval = (epoch % eval_interval == 0) or (epoch == epochs)
        if do_eval:
            eval_alpha = float(cfg.get("model", {}).get("alpha", 0.85))
            metrics, sim_matrix, pids, pids_text = evaluate(model, val_loader, device, alpha=eval_alpha)
            r1 = metrics.get("R@1", 0.0)
            mdr = metrics.get("MdR", 0.0)
            print(f"📊 Validation Results [Epoch {epoch:02d}]: Rank@1 = {r1:.2f}% | Rank@5 = {metrics.get('R@5', 0):.2f}% | MdR = {mdr:.1f} (alpha={eval_alpha})")

            history.setdefault("eval_epochs", []).append(epoch)
            history.setdefault("rank1", []).append(r1)
            history.setdefault("rank5", []).append(metrics.get("R@5", 0.0))
            history.setdefault("rank10", []).append(metrics.get("R@10", 0.0))
            history.setdefault("rank50", []).append(metrics.get("R@50", 0.0))
            history.setdefault("mdr", []).append(mdr)
            history["sim_matrix"] = sim_matrix
            history["pids"] = pids
            history["pids_text"] = pids_text

            try:
                save_training_history(history)
                plot_training_dashboard(history)
            except Exception as e:
                print(f"[!] Warning: Failed to save training history / dashboard plots: {e}", flush=True)

            if r1 > best_r1:
                best_r1 = r1
                best_path = paths.get_checkpoint_path("best.pth")
                best_data = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "metrics": metrics,
                    "config": cfg,
                    "history": history
                }
                torch.save(best_data, best_path)
                print(f"🏆 New Best Model Saved to: {best_path} (Rank@1: {best_r1:.2f}%)")

        checkpoint_data = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics if do_eval else (checkpoint_data.get("metrics") if "checkpoint_data" in locals() else {}),
            "config": cfg,
            "history": history
        }
        latest_path = paths.get_checkpoint_path("latest.pth")
        torch.save(checkpoint_data, latest_path)


if __name__ == "__main__":
    main()
