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
import random
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
    random.seed(seed)
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


class ModelEMA:
    """Exponential moving average of the float tensors of model.state_dict() (recipe m3, --ema).
    swap() exchanges the live weights with the averaged ones in place (no extra copy of the model)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, e in self.shadow.items():
            e.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, e in self.shadow.items():
            tmp = msd[k].detach().clone()
            msd[k].copy_(e)
            e.copy_(tmp)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.updates = int(sd.get("updates", 0))
        for k, v in sd.get("shadow", {}).items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


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
    grad_accum_steps: int = 2,
    use_dsca: bool = True,
    ema: Optional["ModelEMA"] = None
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
        if not use_dsca:
            # Ablation "- DSCA": no decomposition, all three text streams use the full caption
            q_app, q_mot = None, None
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(captions, video_frames, q_app=q_app, q_mot=q_mot)
                loss, breakdown = criterion(
                    T=out["T"], V=out["V_video"],
                    V_app=out["V_app"], V_mot=out["V_mot"],
                    l_mim=out["l_mim"],
                    T_app=out.get("T_app"), T_mot=out.get("T_mot"),
                    video_keys=batch.get("video_key"),
                    app_texts=q_app, mot_texts=q_mot
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
                if ema is not None:
                    ema.update(model)
        else:
            out = model(captions, video_frames, q_app=q_app, q_mot=q_mot)
            loss, breakdown = criterion(
                T=out["T"], V=out["V_video"],
                V_app=out["V_app"], V_mot=out["V_mot"],
                l_mim=out["l_mim"],
                T_app=out.get("T_app"), T_mot=out.get("T_mot"),
                video_keys=batch.get("video_key"),
                app_texts=q_app, mot_texts=q_mot
            )
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

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


def atomic_save(obj, path) -> None:
    """Write to a temp file then rename, so an interrupted save never leaves a truncated checkpoint."""
    tmp = f"{path}.tmp"
    if os.environ.get("PBL_LOW_DISK") == "1" and str(path).endswith("latest.pth") and os.path.exists(path):
        os.remove(path)          # no room for two copies: an interrupted save then restarts the run (best.pth is kept)
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 0.5,
    weights: Tuple[float, float, float] = (0.50, 0.35, 0.15),
    use_dsca: bool = True
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluates Text-to-Video Person Retrieval following standard TVPR benchmark protocol:
      - All N_queries text queries are evaluated.
      - Gallery contains UNIQUE video tracks (deduplicated by video_id).
      - Multi-level Disentangled Concept Alignment scoring (default weights w = 0.50 / 0.35 / 0.15):
          Score = w1 * CosSim(T_full, V_video) + w2 * CosSim(T_app, V_app) + w3 * CosSim(T_mot, V_mot)
    Used on the validation split only (checkpoint selection); the test split is evaluated by experiments/.
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
    visited_order = []

    for batch in dataloader:
        captions = batch.get("caption", batch.get("q_app"))
        q_app = batch.get("q_app", None)
        q_mot = batch.get("q_mot", None)
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)
        video_ids = batch.get("video_key", batch.get("video_id"))
        pids = batch["person_id"].numpy()
        if not use_dsca:
            q_app, q_mot = None, None

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
                visited_order.append(vid)
                gallery_v.append(v_vid_cpu[i:i+1])
                gallery_v_app.append(v_app_cpu[i:i+1])
                gallery_v_mot.append(v_mot_cpu[i:i+1])
                gallery_pids.append(pids[i])

    query_keys = list(getattr(dataloader.dataset, "samples", None) or [])
    query_keys = [f"{q.get('dataset', '')}/{q.get('video_id')}" for q in query_keys]
    if len({k.split("/", 1)[0] for k in visited_videos if isinstance(k, str)}) > 1 and len(query_keys) == len(query_pids):
        # Several sub-datasets: person ids repeat across them, so match on the video key instead
        key_id = {k: i for i, k in enumerate(dict.fromkeys(query_keys))}
        query_pids = [key_id[k] for k in query_keys]
        gallery_pids = [key_id[k] for k in visited_order]

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
    w1, w2, w3 = weights
    sim_matrix = (w1 * sim_joint + w2 * sim_app + w3 * sim_mot).cpu().numpy()
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
    # Revision experiments (seeds / ablations). Without these flags training is unchanged.
    parser.add_argument("--seed", type=int, default=None, help="Random seed override (train.seed in the config)")
    parser.add_argument("--tag", type=str, default=None, help="Run name: checkpoints/reports go to <dir>/<sub_dataset>/<tag>/")
    parser.add_argument("--lambda_app", type=float, default=None, help="Override lambda_app (appearance InfoNCE weight)")
    parser.add_argument("--lambda_mot", type=float, default=None, help="Override lambda_mot (motion InfoNCE weight)")
    parser.add_argument("--lambda_mim", type=float, default=None, help="Override lambda_mim (orthogonality weight)")
    parser.add_argument("--no_temporal_attn", action="store_true", help="Ablation: skip temporal attention in every SpaceTimeViT block")
    parser.add_argument("--no_dsca", action="store_true", help="Ablation: no text decomposition, T_app = T_mot = T_full")
    # Recipe v2 (all off by default; see experiments/plan.py RECIPES)
    parser.add_argument("--backbone", choices=["spacetime", "mfgf"], default=None,
                        help="Video encoder: spacetime (default) or mfgf = Frozen-in-Time ViT (4 frames) + S3D (16 frames)")
    parser.add_argument("--s3d_motion", action="store_true", help="Add a Kinetics-400 S3D clip feature to the motion stream")
    # Recipe m3 (all off by default; without them training is unchanged)
    parser.add_argument("--clip_flip", action="store_true", help="Flip whole clips (and swap left/right in the caption) instead of every frame on its own")
    parser.add_argument("--text_pos", action="store_true", help="app/mot InfoNCE: batch items with the same Q_app/Q_mot string are positives")
    parser.add_argument("--dsca_mode", choices=["keyword", "clause"], default=None, help="DSCA sub-prompts: keyword bags (original) or whole clauses")
    parser.add_argument("--temporal_head_lr", action="store_true", help="Train the new temporal attention / temporal embedding at the head lr, not the backbone lr")
    parser.add_argument("--mim_mode", choices=["raw", "centered"], default=None, help="L_MIM on raw or batch-centred V_app/V_mot")
    parser.add_argument("--ema", type=float, default=None, help="EMA decay of the weights used for validation and best.pth (0 = off)")
    parser.add_argument("--unique_video_batches", action="store_true", help="No two captions of the same video in a batch")
    parser.add_argument("--queue_size", type=int, default=None, help="Cross-batch memory of detached embeddings (extra negatives); 0 = off")
    parser.add_argument("--queue_start_epoch", type=int, default=None, help="First epoch that uses the queue (default: after warmup)")
    parser.add_argument("--learnable_tau", action="store_true", help="Learn the InfoNCE temperature (init = model.temperature)")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weight decay override")
    parser.add_argument("--select_metric", choices=["r1", "mean_r1_5_10"], default=None,
                        help="Validation score used to keep best.pth (default r1)")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Start from these model weights (e.g. a joint all-subset run) with a fresh optimizer/schedule")
    parser.add_argument("--eval_weights", type=float, nargs=3, default=None, metavar=("J", "A", "M"),
                        help="Fusion weights of the per-epoch validation score used for checkpoint selection (default 0.50 0.35 0.15)")

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

    # CLI overrides for revision experiments; written into cfg so they are stored in every checkpoint
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    for key in ("lambda_app", "lambda_mot", "lambda_mim"):
        if getattr(args, key) is not None:
            cfg.setdefault("model", {})[key] = getattr(args, key)
    if args.no_temporal_attn:
        cfg.setdefault("model", {})["use_temporal_attn"] = False
    if args.no_dsca:
        cfg.setdefault("model", {})["use_dsca"] = False
    if args.s3d_motion:
        cfg.setdefault("model", {})["s3d_motion"] = True
    if args.backbone:
        cfg.setdefault("model", {})["backbone"] = args.backbone
    if args.eval_weights is not None:
        cfg.setdefault("eval", {})["weights"] = list(args.eval_weights)
    if args.pretrained_vit is not None:
        cfg.setdefault("model", {})["pretrained_vit"] = args.pretrained_vit
    tcfg = cfg.setdefault("train", {})
    if args.unique_video_batches:
        tcfg["unique_video_batches"] = True
    if args.clip_flip:
        tcfg["clip_flip"] = True
    if args.text_pos:
        tcfg["text_pos"] = True
    if args.temporal_head_lr:
        tcfg["temporal_head_lr"] = True
    if args.ema is not None:
        tcfg["ema"] = args.ema
    if args.dsca_mode is not None:
        cfg.setdefault("model", {})["dsca_mode"] = args.dsca_mode
    if args.mim_mode is not None:
        cfg.setdefault("model", {})["mim_mode"] = args.mim_mode
    if args.queue_size is not None:
        tcfg["queue_size"] = args.queue_size
    if args.queue_start_epoch is not None:
        tcfg["queue_start_epoch"] = args.queue_start_epoch
    if args.learnable_tau:
        tcfg["learnable_tau"] = True
    if args.weight_decay is not None:
        tcfg["weight_decay"] = args.weight_decay
    if args.select_metric is not None:
        tcfg["select_metric"] = args.select_metric
    if args.init_from:
        tcfg["init_from"] = args.init_from
    select_metric = tcfg.get("select_metric", "r1")
    use_dsca = bool(cfg.get("model", {}).get("use_dsca", True))
    eval_weights = tuple(cfg.get("eval", {}).get("weights", (0.50, 0.35, 0.15)))

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

    if args.tag:
        # Each tagged run (seed / ablation) gets its own folders and never falls back to another run's checkpoint
        paths.checkpoints_dir = paths.checkpoints_dir / args.tag
        paths.reports_dir = paths.reports_dir / args.tag
        paths.outputs_dir = paths.outputs_dir / args.tag
        for d in (paths.checkpoints_dir, paths.reports_dir, paths.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)
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
    print(f"  • Seed / Tag        : {cfg.get('train', {}).get('seed', 42)} / {args.tag or '-'}")
    print(f"  • Ablation flags    : temporal_attn={cfg.get('model', {}).get('use_temporal_attn', True)} | dsca={use_dsca} | "
          f"pretrained_vit={cfg.get('model', {}).get('pretrained_vit', True)} | s3d={cfg.get('model', {}).get('s3d_motion', False)} | backbone={cfg.get('model', {}).get('backbone', 'spacetime')} | val weights={eval_weights}")
    print(f"  • Recipe            : unique_video={tcfg.get('unique_video_batches', False)} | queue={tcfg.get('queue_size', 0)} | "
          f"learnable_tau={tcfg.get('learnable_tau', False)} | wd={tcfg.get('weight_decay', 0.0001)} | "
          f"select={select_metric} | init_from={tcfg.get('init_from') or '-'}")
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
            img_size=cfg.get("model", {}).get("img_size", 224),
            unique_video=bool(tcfg.get("unique_video_batches", False)),
            seed=int(tcfg.get("seed", 42)),
            frame_layout="mfgf" if cfg.get("model", {}).get("backbone") == "mfgf" else "uniform",
            clip_flip=bool(tcfg.get("clip_flip", False)),
            dsca_mode=cfg.get("model", {}).get("dsca_mode", "keyword")
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
        pretrained_vit=pretrained_vit_flag,
        use_temporal_attn=model_cfg.get("use_temporal_attn", True),
        s3d_motion=model_cfg.get("s3d_motion", False),
        backbone=model_cfg.get("backbone", "spacetime"),
        mim_mode=model_cfg.get("mim_mode", "raw")
    ).to(device)

    if getattr(model.video_encoder, "weights_source", None):
        cfg.setdefault("model", {})["visual_weights"] = model.video_encoder.weights_source   # stored in every checkpoint
        print(f"  • Visual weights    : {cfg['model']['visual_weights']}")

    criterion = DAMRCriterion(
        temperature=cfg.get("model", {}).get("temperature", 0.05),
        lambda_app=cfg.get("model", {}).get("lambda_app", 0.5),
        lambda_mot=cfg.get("model", {}).get("lambda_mot", 0.25),
        lambda_mim=cfg.get("model", {}).get("lambda_mim", 0.1),
        learnable_tau=bool(tcfg.get("learnable_tau", False)),
        queue_size=int(tcfg.get("queue_size", 0) or 0),
        text_pos=bool(tcfg.get("text_pos", False))
    ).to(device)
    queue_start_epoch = int(tcfg.get("queue_start_epoch", warmup_epochs + 1))

    if tcfg.get("init_from"):
        init_ck = torch.load(tcfg["init_from"], map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(init_ck["model_state_dict"], strict=False)
        missing = [k for k in missing if not k.startswith(("video_encoder.s3d.", "video_encoder.s3d_proj."))]
        if missing or unexpected:     # only modules added on top of the init checkpoint may be missing
            raise RuntimeError(f"init_from mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        if init_ck.get("criterion_state_dict"):
            criterion.load_state_dict(init_ck["criterion_state_dict"], strict=False)
        print(f"[+] Initialised from {tcfg['init_from']} (epoch {init_ck.get('epoch')}); optimizer and schedule start fresh")
        del init_ck

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
            "video_encoder.s3d." in name or
            "video_encoder.vit." in name or
            ("video_encoder.blocks" in name and ("attn_spat" in name or "norm_spat" in name or "mlp" in name or "norm_mlp" in name))
        )
        if tcfg.get("temporal_head_lr") and any(k in name for k in ("timeattn", "norm3", "temporal_embed")):
            is_pretrained = False          # recipe m3: new (zero-initialised) temporal parts learn at the head lr
        if is_pretrained:
            pretrained_params.append(param)
        else:
            head_params.append(param)

    wd = float(cfg.get("train", {}).get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": lr * 0.2, "weight_decay": wd},
        {"params": head_params, "lr": lr * 1.0, "weight_decay": wd},
    ] + ([{"params": list(criterion.parameters()), "lr": lr * 1.0, "weight_decay": 0.0}]
         if any(True for _ in criterion.parameters()) else []))
    scheduler = get_lr_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs, base_lr=lr, min_lr=min_lr)

    use_amp_flag = cfg.get("train", {}).get("use_amp", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_flag)
    ema_decay = float(tcfg.get("ema", 0.0) or 0.0)
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"  • EMA               : decay {ema_decay} (validation + best.pth use the averaged weights)")

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
            if checkpoint.get("criterion_state_dict"):
                criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] and scaler:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if ema is not None and checkpoint.get("ema_state_dict"):
                ema.load_state_dict(checkpoint["ema_state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "best_score" in checkpoint:
                best_r1 = checkpoint["best_score"]
            elif "metrics" in checkpoint:
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
        if hasattr(getattr(train_loader, "batch_sampler", None), "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        criterion.queue_active = criterion.queues is not None and epoch >= queue_start_epoch
        avg_loss, avg_app, avg_mot, avg_mim = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler,
            device, epoch, epochs, use_amp=use_amp_flag, grad_clip_norm=grad_clip_norm,
            grad_accum_steps=grad_accum_steps, use_dsca=use_dsca, ema=ema
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
            if ema is not None:
                ema.swap(model)            # validate and save best.pth with the averaged weights
            eval_alpha = float(cfg.get("model", {}).get("alpha", 0.85))
            metrics, sim_matrix, pids, pids_text = evaluate(model, val_loader, device, alpha=eval_alpha,
                                                            weights=eval_weights, use_dsca=use_dsca)
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

            score = r1 if select_metric == "r1" else \
                (metrics.get("R@1", 0.0) + metrics.get("R@5", 0.0) + metrics.get("R@10", 0.0)) / 3.0
            if select_metric != "r1":
                print(f"    selection score ({select_metric}) = {score:.2f} (best so far {best_r1:.2f})")
            if score > best_r1:
                best_r1 = score
                best_path = paths.get_checkpoint_path("best.pth")
                best_data = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "metrics": metrics,
                    "best_score": best_r1,
                    "criterion_state_dict": criterion.state_dict(),
                    "config": cfg,
                    "history": history
                }
                if ema is not None:
                    # best.pth holds the EMA weights; the optimizer state belongs to the live weights and only
                    # latest.pth needs it (resume). Dropping it saves ~1.7 GB per run (Kaggle /kaggle/working is 20 GB).
                    for k in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"):
                        best_data.pop(k, None)
                atomic_save(best_data, best_path)
                print(f"🏆 New Best Model Saved to: {best_path} (Rank@1: {r1:.2f}%)")

        if do_eval and ema is not None:
            ema.swap(model)                # back to the live weights for training / latest.pth
        checkpoint_data = {
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics if do_eval else (checkpoint_data.get("metrics") if "checkpoint_data" in locals() else {}),
            "best_score": best_r1,
            "criterion_state_dict": criterion.state_dict(),
            "config": cfg,
            "history": history
        }
        latest_path = paths.get_checkpoint_path("latest.pth")
        atomic_save(checkpoint_data, latest_path)


if __name__ == "__main__":
    main()
