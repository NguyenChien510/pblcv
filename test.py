"""
Evaluation & Inference Script for Spatio-Temporal DAMR (srccv)
Pure ViT + BERT Architecture for Text-to-Video Person Retrieval (TVPR)

Usage:
  1. Evaluate Checkpoint on Sub-dataset:
     python test.py --sub_dataset TVPReid-Duke
  2. Evaluate specific checkpoint file:
     python test.py --checkpoint checkpoints/TVPReid-Duke/best.pth
  3. Single Query Retrieval:
     python test.py --query "A woman in a red jacket walking towards the gate"
"""

import os
import sys
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from utils.paths import paths, normalize_sub_dataset_name
from data.dataset import create_dataloaders
from models.damr_main import DAMRModel
from utils.metrics import compute_similarity_matrix, evaluate_rank_metrics
from utils.plot_benchmarks import plot_comprehensive_confusion_matrix, plot_training_dashboard


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 0.5,
    confusion_matrix: bool = False,
    cm_size: int = 10,
    output_dir: str = None,
    history: dict = None,
    export_dashboard: bool = False
):
    """
    Evaluates Text-to-Video Person Retrieval following standard TVPR benchmark protocol:
      - All N_queries text queries evaluated against unique video gallery.
      - Gallery deduplicated by video_id (37 videos for iLIDs, 302 for Duke).
      - Multi-level Disentangled Concept Scoring.
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

    print("🔍 Running Evaluation on Dataset...")
    for batch in val_loader:
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

    T_full_n = torch.nn.functional.normalize(T_full, p=2, dim=-1)
    T_app_n = torch.nn.functional.normalize(T_app, p=2, dim=-1)
    T_mot_n = torch.nn.functional.normalize(T_mot, p=2, dim=-1)

    V_n = torch.nn.functional.normalize(V, p=2, dim=-1)
    V_app_n = torch.nn.functional.normalize(V_app, p=2, dim=-1)
    V_mot_n = torch.nn.functional.normalize(V_mot, p=2, dim=-1)

    sim_joint = torch.matmul(T_full_n, V_n.t())
    sim_app = torch.matmul(T_app_n, V_app_n.t())
    sim_mot = torch.matmul(T_mot_n, V_mot_n.t())

    # Multi-level Disentangled Concept Scoring
    sim_matrix = (0.50 * sim_joint + 0.35 * sim_app + 0.15 * sim_mot).cpu().numpy()
    metrics = evaluate_rank_metrics(sim_matrix, pids_text, pids_video)

    print("\n==========================================")
    print("📊 SPACE-TIME DAMR RETRIEVAL BENCHMARK")
    print("==========================================")
    print(f"Rank@1  : {metrics.get('R@1', 0.0):.2f}%")
    print(f"Rank@5  : {metrics.get('R@5', 0.0):.2f}%")
    print(f"Rank@10 : {metrics.get('R@10', 0.0):.2f}%")
    print(f"Rank@50 : {metrics.get('R@50', 0.0):.2f}%")
    print(f"MdR     : {metrics.get('MdR', 0.0):.1f}")
    print(f"mAP     : {metrics.get('mAP', 0.0):.2f}%")
    print("==========================================\n")

    out_dir = output_dir or str(paths.reports_dir)
    if confusion_matrix:
        print(f"🎨 Generating Comprehensive Cross-Modal Confusion Matrices in: {out_dir}")
        cm_outputs = plot_comprehensive_confusion_matrix(
            sim_matrix=sim_matrix,
            pids_text=pids_text,
            pids_video=pids_video,
            output_dir=out_dir,
            max_identities=cm_size
        )

    if (confusion_matrix or export_dashboard) and history:
        try:
            print(f"📈 Regenerating Training Dashboard with Evaluated Similarity Matrix...")
            hist_copy = dict(history)
            hist_copy["sim_matrix"] = sim_matrix
            hist_copy["pids"] = pids_video
            hist_copy["pids_text"] = pids_text
            out_dash = os.path.join(out_dir, "training_dashboard.png")
            plot_training_dashboard(hist_copy, output_path=out_dash)
        except Exception as e:
            print(f"[!] Warning: Could not regenerate training dashboard: {e}")

    return metrics


@torch.no_grad()
def run_query_inference(model: torch.nn.Module, text_query: str, val_loader: torch.utils.data.DataLoader, device: torch.device, alpha: float = 0.85):
    model.eval()
    print(f"📝 Original Query : {text_query}")

    # Encode full natural language query with BERT
    T = model.encode_text([text_query], device=device)

    all_v = []
    all_v_app = []
    video_ids = []

    for batch in val_loader:
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)
        V_app, V_mot, V_video = model.encode_video(video_frames)
        all_v.append(V_video.cpu())
        all_v_app.append(V_app.cpu())
        video_ids.extend(batch["video_id"])

    V = torch.cat(all_v, dim=0)
    V_app = torch.cat(all_v_app, dim=0)

    sim_matrix = compute_similarity_matrix(T.cpu(), V, V_app, beta=alpha)
    scores = sim_matrix[0]
    top_indices = np.argsort(-scores)[:5]

    print("\n🏆 TOP-5 RETRIEVED PERSON VIDEOS:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  Rank #{rank}: Video ID = {video_ids[idx]} | Similarity Score = {scores[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpaceTime-DAMR Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument("--query", type=str, default=None, help="Text query string for single-query retrieval demo")
    parser.add_argument("--sub_dataset", type=str, default=None, help="Sub-dataset name (e.g. TVPReid-Duke, TVPReid-iLIDs, TVPReid-PRID, all)")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Weight blend between fused video and appearance (default: 0.85)")
    parser.add_argument("--confusion_matrix", "--cm", action="store_true", help="Generate and export comprehensive cross-modal confusion matrices")
    parser.add_argument("--cm_size", type=int, default=10, help="Number of identities to display in confusion matrix heatmap (default: 10)")
    parser.add_argument("--dashboard", action="store_true", help="Regenerate training dashboard from checkpoint history")
    parser.add_argument("--reports_dir", type=str, default=None, help="Custom output directory to save reports and confusion matrix plots")
    args, unknown = parser.parse_known_args()

    # Configure sub-dataset paths if provided
    sub_data = args.sub_dataset or getattr(paths, "sub_dataset", "all")
    if hasattr(paths, "set_sub_dataset"):
        paths.set_sub_dataset(sub_data)
    else:
        paths.sub_dataset = normalize_sub_dataset_name(sub_data)

    if args.data_root and hasattr(paths, "set_data_root"):
        paths.set_data_root(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAMRModel(
        embed_dim=768,
        text_model_name="bert-base-uncased",
        motion_frame_num=16,
        vit_depth=6,
        vit_heads=12
    ).to(device)

    checkpoint_path = args.checkpoint or str(paths.get_checkpoint_path("best.pth"))
    history = None
    if os.path.exists(checkpoint_path):
        print(f"📦 Loading Checkpoint from {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        history = ckpt.get("history", None)
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}. Running evaluation with initialized weights.")

    active_sub = getattr(paths, "sub_dataset", "all")
    train_loader, val_loader = create_dataloaders(
        data_root=str(paths.data_root),
        sub_dataset=active_sub,
        batch_size=args.batch_size,
        num_frames=16
    )

    if args.query:
        run_query_inference(model, args.query, val_loader, device, alpha=args.alpha)
    else:
        run_evaluation(
            model=model,
            val_loader=val_loader,
            device=device,
            alpha=args.alpha,
            confusion_matrix=args.confusion_matrix,
            cm_size=args.cm_size,
            output_dir=args.reports_dir,
            history=history,
            export_dashboard=args.dashboard
        )


if __name__ == "__main__":
    main()
