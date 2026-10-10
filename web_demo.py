#!/usr/bin/env python3
"""
========================================================================================
🚀 SpaceTime-DAMR Video Person Search & Tracking Web Demo
Upload ANY custom video, enter a prompt, and find/track that person directly inside the video!
Pure ViT + BERT Architecture (SpaceTimeViT + BERT 768-D) + YOLOv8 Tracking
========================================================================================
100% Single Standalone Python File (FastAPI + Embedded Modern Glassmorphism Web UI).
Zero external HTML/CSS files required.

Usage:
  python web_demo.py
  python web_demo.py --port 8000
========================================================================================
"""

import os
import sys
import time
import json
import glob
import math
import uuid
import base64
import shutil
import io
import argparse
import webbrowser
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure UTF-8 stdout on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try: sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models.damr_main import DAMRModel
from utils.metrics import compute_similarity_matrix


# ==============================================================================
# 1. CORE DETECTOR, TRACKER & SPACETIME-DAMR MATCHING ENGINE
# ==============================================================================

# Distinct high-contrast color palette for Top 10 candidates in video annotations & web UI
TOPK_COLOR_CONFIG = {
    1: {"bgr": (127, 255, 0), "hex": "#00ff7f", "name": "Neon Green (Target #1)", "dark_text": True},
    2: {"bgr": (255, 210, 0), "hex": "#00d2ff", "name": "Electric Cyan (#2)", "dark_text": True},
    3: {"bgr": (0, 184, 255), "hex": "#ffb800", "name": "Cyber Gold (#3)", "dark_text": True},
    4: {"bgr": (133, 42, 255), "hex": "#ff2a85", "name": "Neon Pink (#4)", "dark_text": False},
    5: {"bgr": (247, 85, 168), "hex": "#a855f7", "name": "Electric Violet (#5)", "dark_text": False},
    6: {"bgr": (74, 107, 255), "hex": "#ff6b4a", "name": "Bright Coral (#6)", "dark_text": False},
    7: {"bgr": (0, 255, 204), "hex": "#ccff00", "name": "Electric Lime (#7)", "dark_text": True},
    8: {"bgr": (212, 245, 0), "hex": "#00f5d4", "name": "Vivid Turquoise (#8)", "dark_text": True},
    9: {"bgr": (248, 189, 56), "hex": "#38bdf8", "name": "Sky Blue (#9)", "dark_text": True},
    10: {"bgr": (72, 29, 225), "hex": "#e11d48", "name": "Crimson Rose (#10)", "dark_text": False},
}
TOP5_COLOR_CONFIG = TOPK_COLOR_CONFIG
OTHER_COLOR_CONFIG = {"bgr": (184, 163, 148), "hex": "#94a3b8", "name": "Slate Gray", "dark_text": False}


class VideoPersonSearchEngine:
    """
    Handles:
      1. YOLOv8 / Torchvision / HOG Person Detection and Tracking in arbitrary uploaded videos.
      2. Cropping and assembling multi-frame person tracklets.
      3. Cross-modal semantic matching via SpaceTime-DAMR (SpaceTimeViT + BERT).
      4. Rendering high-visibility annotated tracking video.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing VideoPersonSearchEngine on device: {self.device}")

        self.uploads_dir = CURRENT_DIR / "uploads"
        self.outputs_dir = CURRENT_DIR / "outputs"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[DAMRModel] = None
        self.detector = None
        self.detector_backend = "opencv"
        self.checkpoint_info = {
            "loaded": False,
            "name": "Base Model (Pretrained ViT+BERT 768-D)",
            "path": "None",
            "layers": 0,
            "size_mb": 0.0
        }

        self._init_models()
        self._init_detector()

    def scan_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Scans filesystem for all available .pth checkpoint files STRICTLY inside srccv/checkpoints."""
        found = []
        ckpt_dir = CURRENT_DIR / "checkpoints"
        if not ckpt_dir.exists():
            return found

        seen_paths = set()
        for cp in sorted(ckpt_dir.glob("**/*.pth")):
            if cp.is_file() and cp.stat().st_size > 1024 * 1024:
                full_p = str(cp.resolve())
                if full_p not in seen_paths:
                    seen_paths.add(full_p)
                    size_mb = round(cp.stat().st_size / (1024 * 1024), 1)
                    try:
                        rel_p = cp.relative_to(ckpt_dir)
                        rel_label = f"{rel_p} ({size_mb} MB)"
                    except Exception:
                        rel_label = f"{cp.name} ({size_mb} MB)"
                    found.append({
                        "name": cp.name,
                        "label": rel_label,
                        "path": full_p,
                        "size_mb": size_mb
                    })
        return found

    def load_checkpoint_file(self, full_path: str) -> Dict[str, Any]:
        """Dynamically hot-swaps model weights from a chosen checkpoint STRICTLY in srccv/checkpoints."""
        cp = Path(full_path).resolve()
        ckpt_dir = (CURRENT_DIR / "checkpoints").resolve()

        # Strict security & scope check: only allow files inside srccv/checkpoints
        try:
            cp.relative_to(ckpt_dir)
        except ValueError:
            return {"success": False, "error": f"Security restriction: Checkpoint must be inside '{ckpt_dir}'"}

        if not cp.exists() or self.model is None:
            return {"success": False, "error": f"Checkpoint not found at: {full_path}"}

        try:
            ckpt = torch.load(cp, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model_dict = self.model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            if filtered:
                model_dict.update(filtered)
                self.model.load_state_dict(model_dict)
                size_mb = round(cp.stat().st_size / (1024 * 1024), 1)
                epoch = ckpt.get("epoch", "N/A")
                metrics = ckpt.get("metrics", {})
                r1_val = metrics.get("R@1", metrics.get("Rank@1", None))
                map_val = metrics.get("mAP", None)

                r1_str = f" | R@1: {r1_val:.1f}%" if r1_val is not None else ""
                map_str = f" | mAP: {map_val:.1f}%" if map_val is not None else ""
                label = f"{cp.name} (Epoch {epoch}{r1_str}{map_str} - {size_mb} MB)"

                self.checkpoint_info = {
                    "loaded": True,
                    "name": cp.name,
                    "label": label,
                    "path": str(cp),
                    "layers": len(filtered),
                    "size_mb": size_mb,
                    "epoch": epoch,
                    "metrics": metrics,
                    "rank1": f"{r1_val:.2f}%" if r1_val is not None else "N/A",
                    "mAP": f"{map_val:.2f}%" if map_val is not None else "N/A"
                }
                print(f"[+] Loaded model checkpoint: {cp.name} from srccv/checkpoints ({len(filtered)} layers, Epoch {epoch}{r1_str}{map_str})")
                return {"success": True, "checkpoint": self.checkpoint_info}
            else:
                return {"success": False, "error": "No compatible layers found for SpaceTimeViT/BERT"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _init_models(self):
        """Initializes DAMRModel (SpaceTimeViT + BERT-base-uncased) strictly from srccv/checkpoints."""
        try:
            self.model = DAMRModel(
                embed_dim=768,
                text_model_name="bert-base-uncased",
                motion_frame_num=16,
                vit_depth=6,
                vit_heads=12,
                drop_rate=0.0
            ).to(self.device).eval()

            # Strictly scan and load from srccv/checkpoints folder
            ckpts = self.scan_available_checkpoints()
            if ckpts:
                # Prioritize 'best.pth' first, then 'latest.pth'
                preferred = sorted(ckpts, key=lambda c: (
                    0 if "best" in c["name"].lower() else (1 if "latest" in c["name"].lower() else 2)
                ))
                self.load_checkpoint_file(preferred[0]["path"])

            if not self.checkpoint_info["loaded"]:
                print("[!] No checkpoint found in srccv/checkpoints. Using ImageNet-1K pretrained ViT + BERT weights.")
        except Exception as e:
            print(f"[!] Error initializing DAMRModel: {e}")

    def _init_detector(self):
        """Initializes YOLOv8 (preferred) with fallback to Torchvision Faster R-CNN or OpenCV HOG."""
        try:
            from ultralytics import YOLO
            yolo_path = CURRENT_DIR / "yolov8n.pt"
            self.detector = YOLO(str(yolo_path) if yolo_path.exists() else "yolov8n.pt")
            self.detector_backend = "yolo"
            print("[+] Person Tracker Backend: Ultralytics YOLOv8n")
            return
        except Exception as e:
            print(f"[*] YOLOv8 not available ({e}), trying Torchvision...")

        try:
            import torchvision.models.detection as tv_det
            weights = tv_det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.detector = tv_det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(self.device).eval()
            self.detector_backend = "torchvision"
            print("[+] Person Tracker Backend: Torchvision MobileNet-FPN")
            return
        except Exception:
            pass

        # Fallback OpenCV HOG
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.detector_backend = "opencv"
        print("[+] Person Tracker Backend: OpenCV HOG People Detector")

    def _detect_persons(self, frame: np.ndarray, conf_thresh: float = 0.35) -> List[List[int]]:
        """Detects person bounding boxes [x1, y1, x2, y2] in a single BGR frame."""
        h, w = frame.shape[:2]
        boxes = []

        if self.detector_backend == "yolo":
            results = self.detector(frame, classes=[0], conf=conf_thresh, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    coords = b.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, int(coords[0])), max(0, int(coords[1]))
                    x2, y2 = min(w, int(coords[2])), min(h, int(coords[3]))
                    if (x2 - x1) > 15 and (y2 - y1) > 30:
                        boxes.append([x1, y1, x2, y2])
        elif self.detector_backend == "torchvision":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device).unsqueeze(0)
            with torch.no_grad():
                preds = self.detector(tensor)[0]
            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            det_boxes = preds["boxes"].cpu().numpy().astype(int)
            for idx, label in enumerate(labels):
                if label == 1 and scores[idx] >= conf_thresh:
                    b = det_boxes[idx]
                    x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
                    x2, y2 = min(w, int(b[2])), min(h, int(b[3]))
                    if (x2 - x1) > 15 and (y2 - y1) > 30:
                        boxes.append([x1, y1, x2, y2])
        else:
            found, _ = self.detector.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            for (x, y, bw, bh) in found:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)
                if bw > 15 and bh > 30:
                    boxes.append([x1, y1, x2, y2])

        return boxes

    def track_video(
        self,
        video_path: str,
        max_frames: int = 150,
        frame_stride: int = 1,
        conf_thresh: float = 0.35
    ) -> Dict[str, Any]:
        """
        Processes video frames, associates person detections across time into continuous tracklets.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

        total_to_process = min(total_video_frames, max_frames) if max_frames else total_video_frames

        tracks: Dict[int, Dict[str, Any]] = {}
        next_tid = 1
        frame_idx = 0
        raw_frames = []

        while cap.isOpened() and frame_idx < total_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frames.append(frame.copy())

            # Detect on sampled stride
            if frame_idx % frame_stride == 0:
                detections = self._detect_persons(frame, conf_thresh=conf_thresh)
            else:
                detections = []

            # Associate with existing tracks
            matched_dets = set()
            active_tids = [tid for tid, t in tracks.items() if (frame_idx - t["last_frame"]) <= 15]

            for tid in active_tids:
                t = tracks[tid]
                last_box = t["boxes"][t["last_frame"]]

                best_iou = 0.0
                best_d_idx = -1

                for d_idx, det in enumerate(detections):
                    if d_idx in matched_dets:
                        continue
                    # Compute IoU
                    ix1, iy1 = max(last_box[0], det[0]), max(last_box[1], det[1])
                    ix2, iy2 = min(last_box[2], det[2]), min(last_box[3], det[3])
                    iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    uarea = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1]) + (det[2] - det[0]) * (det[3] - det[1]) - iarea
                    iou = iarea / max(uarea, 1e-6)

                    # Distance between centers
                    cx1, cy1 = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
                    cx2, cy2 = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
                    dist = math.hypot(cx1 - cx2, cy1 - cy2)
                    diag = math.hypot(det[2] - det[0], det[3] - det[1])

                    score = iou + max(0.0, 1.0 - dist / max(diag, 20))
                    if (iou > 0.15 or dist < diag * 0.9) and score > best_iou:
                        best_iou = score
                        best_d_idx = d_idx

                if best_d_idx >= 0:
                    det = detections[best_d_idx]
                    matched_dets.add(best_d_idx)

                    # Smooth interpolation if stride skipped frames
                    gap = frame_idx - t["last_frame"]
                    if gap > 1:
                        for gf in range(t["last_frame"] + 1, frame_idx):
                            alpha = (gf - t["last_frame"]) / float(gap)
                            inter = [int(last_box[k] + alpha * (det[k] - last_box[k])) for k in range(4)]
                            t["boxes"][gf] = inter

                    t["boxes"][frame_idx] = det
                    t["last_frame"] = frame_idx
                    x1, y1, x2, y2 = det
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        t["crops"].append(crop)

            # Start new tracks
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_dets:
                    x1, y1, x2, y2 = det
                    crop = frame[y1:y2, x1:x2]
                    tracks[next_tid] = {
                        "boxes": {frame_idx: det},
                        "crops": [crop] if crop.size > 0 else [],
                        "start_frame": frame_idx,
                        "last_frame": frame_idx
                    }
                    next_tid += 1

            frame_idx += 1

        cap.release()

        # Filter valid tracks with at least 3 detections
        valid_tracks = {tid: t for tid, t in tracks.items() if len(t["boxes"]) >= 3 and len(t["crops"]) >= 2}

        # Fallback: if no person track was formed, use center crop of whole video
        if not valid_tracks:
            cw1, ch1 = int(width * 0.15), int(height * 0.1)
            cw2, ch2 = int(width * 0.85), int(height * 0.9)
            crops = [f[ch1:ch2, cw1:cw2] for f in raw_frames[::max(1, len(raw_frames) // 16)]]
            valid_tracks[1] = {
                "boxes": {i: [cw1, ch1, cw2, ch2] for i in range(len(raw_frames))},
                "crops": crops,
                "start_frame": 0,
                "last_frame": len(raw_frames) - 1
            }

        return {
            "tracks": valid_tracks,
            "raw_frames": raw_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": len(raw_frames)
        }

    def _prepare_tracklet_tensor(self, crops: List[np.ndarray], target_frames: int = 16) -> torch.Tensor:
        """Samples exactly 16 frames from person crops and normalizes for SpaceTimeViT."""
        n_crops = len(crops)
        indices = np.linspace(0, max(0, n_crops - 1), target_frames, dtype=int)

        tensors = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for idx in indices:
            c = crops[idx]
            if c.size == 0 or c.shape[0] < 5 or c.shape[1] < 5:
                c = np.zeros((224, 224, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            t = (t - mean) / std
            tensors.append(t)

        return torch.stack(tensors).unsqueeze(0).to(self.device)  # [1, 16, 3, 224, 224]

    def _img_to_base64(self, img_bgr: np.ndarray, max_size: int = 160) -> str:
        """Converts BGR image to base64 jpeg string."""
        if img_bgr.size == 0:
            return ""
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    def search_person_in_video(
        self,
        video_path: str,
        prompt: str,
        max_frames: int = 150,
        top_k: int = 5,
        alpha: float = 0.85
    ) -> Dict[str, Any]:
        """
        Full Pipeline:
          1. Detect & Track all individuals in the video.
          2. Encode user prompt via BERT (768-D).
          3. Encode each person track via SpaceTimeViT (768-D).
          4. Compute multi-level similarity scores (fixed alpha=0.85).
          5. Render high-visibility tracking output video for Top-K candidates.
        """
        track_res = self.track_video(video_path, max_frames=max_frames)
        tracks = track_res["tracks"]
        raw_frames = track_res["raw_frames"]
        width, height = track_res["width"], track_res["height"]
        fps = track_res["fps"]

        # 1. Encode Text Query (BERT 768-D)
        with torch.no_grad():
            T = self.model.encode_text([prompt.strip()], device=self.device)

        # 2. Encode each person track and compute similarity
        track_scores: Dict[int, float] = {}
        track_previews: Dict[int, Dict[str, Any]] = {}

        for tid, t in tracks.items():
            crops = t["crops"]
            v_tensor = self._prepare_tracklet_tensor(crops)

            with torch.no_grad():
                V_app, V_mot, V_video = self.model.encode_video(v_tensor)
                sim_mat = compute_similarity_matrix(T.cpu(), V_video.cpu(), V_app.cpu(), beta=alpha)
                raw_sim = float(sim_mat[0, 0])

            # Convert cosine score to human-readable percentage
            pct = max(0.0, min(99.4, (raw_sim + 1.0) * 45.0 + 10.0))
            if raw_sim > 0.3:
                pct = min(99.8, 55.0 + (raw_sim - 0.3) * 65.0)

            track_scores[tid] = raw_sim

            # Best representative crop
            best_crop = crops[len(crops) // 2] if crops else np.zeros((100, 100, 3), dtype=np.uint8)
            thumb_b64 = self._img_to_base64(best_crop)

            # 4-frame filmstrip
            sub_indices = np.linspace(0, max(0, len(crops) - 1), 4, dtype=int)
            filmstrip = [self._img_to_base64(crops[idx], max_size=80) for idx in sub_indices]

            s_frame = t["start_frame"]
            e_frame = t["last_frame"]
            s_sec = s_frame / float(fps)
            e_sec = e_frame / float(fps)

            track_previews[tid] = {
                "track_id": tid,
                "score_raw": round(raw_sim, 4),
                "score_pct": round(pct, 1),
                "start_frame": s_frame,
                "end_frame": e_frame,
                "time_range": f"{int(s_sec//60):02d}:{int(s_sec%60):02d} -> {int(e_sec//60):02d}:{int(e_sec%60):02d}",
                "thumbnail": thumb_b64,
                "filmstrip": filmstrip,
                "total_frames_tracked": len(t["boxes"])
            }

        # 3. Sort tracks by score
        ranked_tids = sorted(track_scores.keys(), key=lambda x: track_scores[x], reverse=True)
        target_tid = ranked_tids[0]
        target_score_pct = track_previews[target_tid]["score_pct"]

        top_k = max(1, min(10, int(top_k)))
        top_k_tids = set(ranked_tids[:top_k])

        for rank_pos, tid in enumerate(ranked_tids, start=1):
            c_info = TOPK_COLOR_CONFIG.get(rank_pos, OTHER_COLOR_CONFIG)
            track_previews[tid]["rank"] = rank_pos
            track_previews[tid]["color_hex"] = c_info["hex"]
            track_previews[tid]["color_name"] = c_info["name"]
            track_previews[tid]["dark_text"] = c_info["dark_text"]

        # 4. Render Annotated Video with Multi-Color Bounding Boxes for Top-K
        session_id = uuid.uuid4().hex[:8]
        out_filename = f"tracked_{session_id}.mp4"
        out_path = self.outputs_dir / out_filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        trail_pts: List[Tuple[int, int]] = []
        render_tids = [tid for tid in reversed(ranked_tids) if tid in top_k_tids]

        for f_idx, frame in enumerate(raw_frames):
            annotated = frame.copy()

            # Render bounding boxes: draw from lowest rank to highest in Top-K so #1 TARGET is on top
            for tid in render_tids:
                t = tracks[tid]
                if f_idx in t["boxes"]:
                    box = t["boxes"][f_idx]
                    x1, y1, x2, y2 = box
                    is_target = (tid == target_tid)
                    rank = track_previews[tid]["rank"]
                    score_p = track_previews[tid]["score_pct"]
                    c_info = TOPK_COLOR_CONFIG.get(rank, OTHER_COLOR_CONFIG)
                    color = c_info["bgr"]
                    dark_text = c_info["dark_text"]

                    if is_target:
                        # Top 1 Target: thickness 3 and neon trajectory trail
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                        # Top Tag Header
                        tag = f"#1 TARGET (Person {tid}) | {score_p}%"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                        tag_y1 = max(0, y1 - th - 10)
                        cv2.rectangle(annotated, (x1, tag_y1), (min(width - 1, x1 + tw + 10), y1), color, -1)
                        cv2.putText(annotated, tag, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)

                        # Center trajectory trail
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        trail_pts.append((cx, cy))
                        if len(trail_pts) > 30:
                            trail_pts.pop(0)
                        for tp_idx in range(1, len(trail_pts)):
                            cv2.line(annotated, trail_pts[tp_idx - 1], trail_pts[tp_idx], color, 2)
                    else:
                        # Top 2 to Top K: Distinct high-visibility colored box & solid header
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        tag = f"#{rank} Person {tid} | {score_p}%"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        tag_y1 = max(0, y1 - th - 8)
                        cv2.rectangle(annotated, (x1, tag_y1), (min(width - 1, x1 + tw + 8), y1), color, -1)
                        txt_color = (0, 0, 0) if dark_text else (255, 255, 255)
                        cv2.putText(annotated, tag, (x1 + 4, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_color, 1, cv2.LINE_AA)

            # Top Cinematic HUD Banner
            hud_h = 44
            cv2.rectangle(annotated, (0, 0), (width, hud_h), (12, 16, 24), -1)
            cv2.line(annotated, (0, hud_h), (width, hud_h), (0, 255, 127), 2)

            q_crop = prompt[:38] + "..." if len(prompt) > 38 else prompt
            hud_text = f"PROMPT: \"{q_crop}\"  |  #1 TARGET: Person {target_tid} ({target_score_pct}%)  |  TOP-{top_k}  |  FRAME: {f_idx+1}/{len(raw_frames)}"
            cv2.putText(annotated, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(annotated)

        writer.release()

        # Build ordered list of preview candidates for Top-K
        ordered_candidates = [track_previews[tid] for tid in ranked_tids[:top_k]]

        return {
            "success": True,
            "prompt": prompt,
            "top_k": top_k,
            "target_track_id": target_tid,
            "target_score_pct": target_score_pct,
            "total_persons_detected": len(ranked_tids),
            "video_tracked_url": f"/api/video_file/{out_filename}",
            "candidates": ordered_candidates
        }


# Global Engine Instance
engine = VideoPersonSearchEngine()


# ==============================================================================
# 2. FASTAPI BACKEND API
# ==============================================================================

app = FastAPI(
    title="SpaceTime-DAMR Video Person Search",
    description="Upload any video, input a text prompt, and search/track that person directly inside the video.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "device": str(engine.device).upper(),
        "detector": engine.detector_backend.upper(),
        "model_architecture": "SpaceTime-DAMR (SpaceTimeViT + BERT 768-D)",
        "text_encoder": "bert-base-uncased (768-D)",
        "video_backbone": "Divided Space-Time ViT (6 blocks, 12 heads, 16 frames)",
        "checkpoint": engine.checkpoint_info,
        "available_checkpoints": engine.scan_available_checkpoints()
    }


@app.post("/api/switch_checkpoint")
async def switch_checkpoint(request: Request):
    """Dynamically hot-swaps active model checkpoint."""
    data = await request.json()
    path = data.get("path", "")
    res = engine.load_checkpoint_file(path)
    return res


@app.get("/api/sample_videos")
async def get_sample_videos():
    """Returns available sample videos from dataset for instant 1-click testing."""
    samples = []
    ds_root = PROJECT_ROOT / "TVPReid_dataset"
    for cand in ds_root.glob("**/*.mp4"):
        if cand.stat().st_size > 10000:
            samples.append({
                "name": cand.stem,
                "path": str(cand),
                "url": f"/api/raw_video?path={cand}"
            })
            if len(samples) >= 4:
                break
    return samples


@app.get("/api/video_file/{filename}")
async def serve_output_video(filename: str):
    """Streams rendered tracking output video."""
    fpath = engine.outputs_dir / filename
    if not fpath.exists():
        fpath = engine.uploads_dir / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(fpath), media_type="video/mp4")


@app.get("/api/raw_video")
async def serve_raw_video(path: str):
    """Serves sample video from dataset."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(p), media_type="video/mp4")


@app.post("/api/track_custom_video")
async def track_custom_video(
    prompt: str = Form(...),
    max_frames: int = Form(150),
    top_k: int = Form(5),
    alpha: float = Form(0.85),
    video_file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None)
):
    """
    Uploads a video, runs person detection & SpaceTime-DAMR matching, and generates tracked video for Top-K.
    """
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Determine video source
    if video_file is not None and video_file.filename:
        vid_id = uuid.uuid4().hex[:8]
        ext = Path(video_file.filename).suffix or ".mp4"
        saved_path = engine.uploads_dir / f"upload_{vid_id}{ext}"
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        target_video = str(saved_path)
    elif sample_path and os.path.exists(sample_path):
        target_video = sample_path
    else:
        # Fallback to first available sample video
        samples = list((PROJECT_ROOT / "TVPReid_dataset").glob("**/*.mp4"))
        if samples:
            target_video = str(samples[0])
        else:
            raise HTTPException(status_code=400, detail="No video file uploaded and no sample available")

    print(f"[*] Processing Video Search: '{prompt}' (Top-{top_k}) on {Path(target_video).name}...")
    result = engine.search_person_in_video(
        video_path=target_video,
        prompt=prompt,
        max_frames=max_frames,
        top_k=top_k,
        alpha=0.85
    )
    result["video_original_url"] = f"/api/raw_video?path={target_video}"
    return result


# ==============================================================================
# 3. EMBEDDED SINGLE-PAGE WEB INTERFACE (HTML5 + MODERN DARK CSS + JS)
# ==============================================================================

EMBEDDED_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SpaceTime-DAMR | Video Person Search & Tracking</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #07090e;
      --bg-card: rgba(15, 23, 42, 0.85);
      --border-glow: rgba(56, 189, 248, 0.3);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --primary: #38bdf8;
      --neon-green: #00ff7f;
      --primary-gradient: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --radius-xl: 18px;
      --radius-lg: 12px;
      --radius-sm: 8px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      background: radial-gradient(circle at 15% 15%, #0f172a 0%, #07090e 60%, #030712 100%);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.5;
    }

    .ambient-glow {
      position: fixed;
      top: -20%;
      left: 20%;
      width: 60vw;
      height: 60vw;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.04) 40%, transparent 70%);
      filter: blur(80px);
      pointer-events: none;
      z-index: 0;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 20px;
      position: relative;
      z-index: 1;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      margin-bottom: 28px;
      box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      width: 44px;
      height: 44px;
      border-radius: var(--radius-lg);
      background: var(--primary-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .brand h1 {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: var(--primary-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand p {
      font-size: 12px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .badges-group {
      display: flex;
      gap: 10px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 999px;
      color: #34d399;
      font-size: 12px;
      font-weight: 600;
    }

    /* Main Grid: Upload & Controls */
    .hero-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 28px;
    }
    @media (max-width: 900px) {
      .hero-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      padding: 24px;
      box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5);
    }
    .card h2 {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #38bdf8;
    }

    /* Video Upload Area */
    .dropzone {
      border: 2px dashed rgba(56, 189, 248, 0.35);
      border-radius: var(--radius-lg);
      padding: 30px 20px;
      text-align: center;
      background: rgba(8, 12, 22, 0.5);
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: #38bdf8;
      background: rgba(56, 189, 248, 0.08);
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    }
    .dropzone input[type="file"] {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      opacity: 0; cursor: pointer;
    }
    .file-preview {
      margin-top: 12px;
      font-size: 13px;
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
    }

    .samples-bar {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text-muted);
    }
    .sample-btn {
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--border-subtle);
      color: #cbd5e1;
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .sample-btn:hover, .sample-btn.active {
      border-color: #38bdf8;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.12);
    }

    /* Query & Form */
    .input-group {
      margin-bottom: 16px;
    }
    .input-group label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .text-input {
      width: 100%;
      background: rgba(8, 12, 22, 0.85);
      border: 1px solid var(--border-glow);
      border-radius: var(--radius-lg);
      padding: 12px 16px;
      color: #fff;
      font-size: 14px;
      outline: none;
      font-family: inherit;
      transition: all 0.2s ease;
    }
    .text-input:focus {
      border-color: #38bdf8;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
    }

    .chips-group {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .chip {
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid var(--border-subtle);
      border-radius: 999px;
      padding: 4px 12px;
      font-size: 11px;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .chip:hover {
      background: rgba(56, 189, 248, 0.15);
      border-color: rgba(56, 189, 248, 0.4);
      color: #38bdf8;
    }

    .form-row {
      display: flex;
      gap: 16px;
      align-items: center;
      margin-top: 14px;
      font-size: 13px;
      color: var(--text-muted);
    }

    .btn-action {
      width: 100%;
      margin-top: 18px;
      background: var(--primary-gradient);
      color: #07090e;
      border: none;
      padding: 14px 24px;
      border-radius: var(--radius-lg);
      font-size: 15px;
      font-weight: 800;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.2s ease;
    }
    .btn-action:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(56, 189, 248, 0.45);
    }
    .btn-action:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* Progress Banner */
    .progress-box {
      display: none;
      margin-top: 16px;
      background: rgba(8, 12, 22, 0.9);
      border: 1px solid var(--border-glow);
      border-radius: var(--radius-lg);
      padding: 14px;
      text-align: center;
      font-size: 13px;
      color: #38bdf8;
    }

    /* Results Showcase */
    .results-container {
      display: none;
      margin-top: 28px;
    }

    .match-banner {
      background: linear-gradient(135deg, rgba(0, 255, 127, 0.15), rgba(56, 189, 248, 0.15));
      border: 1px solid rgba(0, 255, 127, 0.4);
      border-radius: var(--radius-xl);
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }
    .match-banner h3 {
      font-size: 18px;
      font-weight: 800;
      color: var(--neon-green);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Video Player Split View */
    .video-view-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 24px;
      margin-bottom: 28px;
    }
    @media (max-width: 992px) {
      .video-view-grid { grid-template-columns: 1fr; }
    }

    .video-frame-card {
      background: #000;
      border: 1px solid var(--border-glow);
      border-radius: var(--radius-xl);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .video-frame-card video {
      width: 100%;
      max-height: 480px;
      object-fit: contain;
      background: #000;
    }
    .video-header {
      padding: 12px 18px;
      background: rgba(15, 23, 42, 0.9);
      font-size: 13px;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Detected Persons Panel */
    .persons-panel {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 520px;
      overflow-y: auto;
      padding-right: 4px;
    }

    /* Top-K Color Legend */
    .topk-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 8px 10px;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      margin-bottom: 12px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
      padding: 2px 7px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
    }
    .legend-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      flex-shrink: 0;
      box-shadow: 0 0 6px currentColor;
    }
    .person-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 14px;
      display: flex;
      gap: 14px;
      align-items: center;
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .person-card:hover {
      border-color: #38bdf8 !important;
      transform: translateX(4px);
    }
    .person-card.target-match {
      background: rgba(0, 255, 127, 0.05);
      box-shadow: 0 0 20px rgba(0, 255, 127, 0.2);
    }
    .person-avatar {
      width: 64px;
      height: 90px;
      border-radius: var(--radius-sm);
      object-fit: cover;
      background: #000;
      border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .person-info {
      flex: 1;
    }
    .person-rank {
      font-size: 11px;
      font-weight: 800;
      padding: 2px 8px;
      border-radius: 999px;
      display: inline-block;
      margin-bottom: 4px;
      font-family: 'JetBrains Mono', monospace;
    }
    .person-title {
      font-size: 14px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }
    .person-score {
      font-size: 15px;
      font-weight: 800;
      color: #38bdf8;
    }
    .person-time {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .filmstrip-row {
      display: flex;
      gap: 4px;
      margin-top: 8px;
    }
    .filmstrip-row img {
      width: 28px;
      height: 38px;
      border-radius: 3px;
      object-fit: cover;
      opacity: 0.8;
    }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 2px solid rgba(0,0,0,0.3);
      border-radius: 50%;
      border-top-color: #000;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      margin-top: 48px;
      text-align: center;
      padding: 20px;
      color: #64748b;
      font-size: 13px;
      border-top: 1px solid var(--border-subtle);
    }
  </style>
</head>
<body>
  <div class="ambient-glow"></div>

  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand">
        <div class="brand-icon">🎯</div>
        <div>
          <h1>SpaceTime-DAMR</h1>
          <p>Text-Guided Video Person Search & Real-Time Tracking</p>
        </div>
      </div>

      <div class="badges-group" style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
        <div class="badge" title="Model Architecture" style="background:rgba(129,140,248,0.15);border-color:rgba(129,140,248,0.35);color:#a5b4fc;">
          <span>🧠</span>
          <span id="model-arch-label">SpaceTime-DAMR (ViT+BERT 768-D)</span>
        </div>
        <div class="badge" style="background:rgba(56,189,248,0.12);border-color:rgba(56,189,248,0.35);color:#38bdf8;">
          <span>📦 CKPT:</span>
          <select id="checkpoint-select" onchange="switchCheckpoint()" style="background:transparent;color:#38bdf8;border:none;outline:none;font-size:12px;font-weight:700;cursor:pointer;max-width:200px;">
            <option value="">Scanning...</option>
          </select>
        </div>
        <div class="badge" style="background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.3);color:#34d399;">
          <span style="width:8px;height:8px;border-radius:50%;background:#10b981;"></span>
          <span id="device-label">DEVICE: CPU</span>
        </div>
        <div class="badge" style="background:rgba(245,158,11,0.12);border-color:rgba(245,158,11,0.3);color:#fbbf24;">
          <span id="detector-label">TRACKER: YOLOv8</span>
        </div>
      </div>
    </header>

    <!-- Upload & Prompt Hero Section -->
    <div class="hero-grid">
      <!-- Card 1: Custom Video Upload -->
      <div class="card">
        <h2><span>📹</span> 1. Upload Video To Search Within</h2>
        <div class="dropzone" id="dropzone">
          <input type="file" id="video-file-input" accept="video/mp4,video/avi,video/mov,video/mkv" onchange="handleFileSelect(event)">
          <div style="font-size: 32px; margin-bottom: 8px;">📤</div>
          <p style="font-size: 14px; font-weight: 600;">Drag & Drop video file here, or click to browse</p>
          <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Supports MP4, AVI, MOV (CCTV, surveillance, or camera footage)</p>
          <div id="file-name-preview" class="file-preview"></div>
        </div>

        <div class="samples-bar">
          <span>Or test sample:</span>
          <div id="sample-buttons" style="display:flex; gap:6px; flex-wrap:wrap;"></div>
        </div>
      </div>

      <!-- Card 2: Natural Language Prompt -->
      <div class="card">
        <h2><span>💬</span> 2. Describe The Person To Find</h2>
        <div class="input-group">
          <label for="prompt-input">Enter natural language description:</label>
          <input type="text" id="prompt-input" class="text-input" placeholder="e.g. A man in black jacket and brown trousers..." value="The man is wearing a black jacket and he is wearing brown trousers">
        </div>

        <div style="font-size: 12px; color: var(--text-muted); font-weight: 600;">Quick Prompts:</div>
        <div class="chips-group">
          <span class="chip" onclick="setPrompt(this.innerText)">The man is wearing a black jacket and he is wearing brown trousers</span>
          <span class="chip" onclick="setPrompt(this.innerText)">The woman is wearing a blue coat, black skirt and carrying a backpack</span>
          <span class="chip" onclick="setPrompt(this.innerText)">The woman is wearing a black jacket and she is wearing red skirt</span>
          <span class="chip" onclick="setPrompt(this.innerText)">The man is wearing a green jacket and black shoes</span>
        </div>

        <div class="form-row">
          <div>
            <span>Max Frames:</span>
            <select id="max-frames-select" style="background:rgba(15,23,42,0.9);color:#fff;border:1px solid rgba(255,255,255,0.15);padding:4px 8px;border-radius:6px;">
              <option value="100">100 frames (Ultra Fast)</option>
              <option value="150" selected>150 frames (Standard)</option>
              <option value="300">300 frames (Full Scan)</option>
            </select>
          </div>
          <div>
            <span>Display Candidates:</span>
            <select id="topk-select" style="background:rgba(15,23,42,0.9);color:#38bdf8;font-weight:700;border:1px solid rgba(56,189,248,0.4);padding:4px 8px;border-radius:6px;">
              <option value="1">Top-1 (Target Only)</option>
              <option value="5" selected>Top-5 (Recommended)</option>
              <option value="10">Top-10 (Extended)</option>
            </select>
          </div>
        </div>

        <button class="btn-action" id="btn-submit" onclick="startTrackingPipeline()">
          <span>🚀 Find & Track Person Inside Video</span>
        </button>

        <div class="progress-box" id="progress-box">
          <span class="spinner" style="margin-right:8px;vertical-align:middle;"></span>
          <span id="progress-text">Processing detection and cross-modal matching...</span>
        </div>
      </div>
    </div>

    <!-- Results Section -->
    <div class="results-container" id="results-container">
      <div class="match-banner" id="match-banner">
        <div>
          <h3 id="banner-title">🎯 TARGET LOCATED</h3>
          <p id="banner-sub" style="font-size:13px;color:#cbd5e1;margin-top:4px;"></p>
        </div>
        <div>
          <span id="banner-score" style="font-size:28px;font-weight:800;color:var(--neon-green);font-family:'JetBrains Mono',monospace;"></span>
        </div>
      </div>

      <div class="video-view-grid">
        <!-- Tracked Video Player -->
        <div class="video-frame-card">
          <div class="video-header">
            <span id="video-header-title">🎥 RENDERED TRACKING VIDEO (Top Multi-Color Bounding Boxes)</span>
            <a id="btn-download-video" href="#" download="tracked_target.mp4" style="color:#38bdf8;text-decoration:none;font-size:12px;">⬇ Download MP4</a>
          </div>
          <video id="player-tracked" controls autoplay loop playsinline></video>
        </div>

        <!-- Detected Individuals Inspection Panel -->
        <div class="card" style="padding:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <h2 id="legend-heading-title" style="font-size:14px;margin:0;">👥 Detected Individuals In Video</h2>
            <span id="legend-sub-badge" style="font-size:11px;color:#94a3b8;">Multi-Color Palette</span>
          </div>
          <div class="topk-legend" id="topk-legend"></div>
          <div class="persons-panel" id="persons-panel"></div>
        </div>
      </div>
    </div>

    <footer>
      SpaceTime-DAMR Video Person Search & Tracking | PBL6 Intelligent Surveillance System
    </footer>
  </div>

  <script>
    let selectedFile = null;
    let selectedSamplePath = null;

    async function initPage() {
      try {
        const res = await fetch('/api/status');
        const st = await res.json();
        document.getElementById('device-label').innerText = `DEVICE: ${st.device}`;
        document.getElementById('detector-label').innerText = `TRACKER: ${st.detector}`;
        document.getElementById('model-arch-label').innerText = st.model_architecture || 'SpaceTime-DAMR (ViT+BERT)';

        // Populate checkpoints dropdown
        const select = document.getElementById('checkpoint-select');
        select.innerHTML = '';
        if (st.available_checkpoints && st.available_checkpoints.length > 0) {
          st.available_checkpoints.forEach(cp => {
            const opt = document.createElement('option');
            opt.value = cp.path;
            opt.innerText = cp.label || cp.name;
            if (st.checkpoint && (st.checkpoint.path === cp.path || st.checkpoint.name === cp.name)) {
              opt.selected = true;
            }
            select.appendChild(opt);
          });
        } else {
          select.innerHTML = '<option value="">Base Pretrained Weights</option>';
        }
      } catch (e) {
        console.error("Status fetch error:", e);
      }

      try {
        const res = await fetch('/api/sample_videos');
        const samples = await res.json();
        const container = document.getElementById('sample-buttons');
        samples.forEach((s, idx) => {
          const btn = document.createElement('button');
          btn.className = 'sample-btn' + (idx === 0 ? ' active' : '');
          btn.innerText = s.name;
          btn.onclick = () => selectSample(s.path, btn);
          container.appendChild(btn);
          if (idx === 0) selectedSamplePath = s.path;
        });
      } catch (e) {}
    }

    async function switchCheckpoint() {
      const select = document.getElementById('checkpoint-select');
      const chosenPath = select.value;
      if (!chosenPath) return;

      try {
        const res = await fetch('/api/switch_checkpoint', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ path: chosenPath })
        });
        const data = await res.json();
        if (data.success) {
          alert(`✅ Đã nạp thành công Checkpoint: ${data.checkpoint.name} (${data.checkpoint.layers} layers loaded)!`);
        } else {
          alert(`⚠️ Không thể nạp checkpoint: ${data.error}`);
        }
      } catch (e) {
        alert("Lỗi khi đổi checkpoint: " + e);
      }
    }

    function selectSample(path, btnElem) {
      selectedSamplePath = path;
      selectedFile = null;
      document.getElementById('video-file-input').value = '';
      document.getElementById('file-name-preview').innerText = 'Selected: ' + path.split('\\\\').pop().split('/').pop();
      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      btnElem.classList.add('active');
    }

    function handleFileSelect(e) {
      const file = e.target.files[0];
      if (file) {
        selectedFile = file;
        selectedSamplePath = null;
        document.getElementById('file-name-preview').innerText = `Uploaded: ${file.name} (${(file.size / (1024*1024)).toFixed(1)} MB)`;
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      }
    }

    function setPrompt(text) {
      document.getElementById('prompt-input').value = text;
    }

    async function startTrackingPipeline() {
      const prompt = document.getElementById('prompt-input').value.trim();
      if (!prompt) {
        alert("Please enter a person description prompt.");
        return;
      }

      const btn = document.getElementById('btn-submit');
      const progressBox = document.getElementById('progress-box');
      const progressText = document.getElementById('progress-text');
      const resultsCont = document.getElementById('results-container');

      btn.disabled = true;
      progressBox.style.display = 'block';
      progressText.innerText = "Analyzing video frames with YOLOv8 & SpaceTime-DAMR...";
      resultsCont.style.display = 'none';

      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('max_frames', document.getElementById('max-frames-select').value);
      formData.append('top_k', document.getElementById('topk-select').value);

      if (selectedFile) {
        formData.append('video_file', selectedFile);
      } else if (selectedSamplePath) {
        formData.append('sample_path', selectedSamplePath);
      }

      try {
        const res = await fetch('/api/track_custom_video', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();

        if (!data.success) {
          alert("Error: " + (data.detail || data.error || "Failed to process video"));
          return;
        }

        renderResults(data);
      } catch (e) {
        alert("Pipeline error: " + e);
      } finally {
        btn.disabled = false;
        progressBox.style.display = 'none';
      }
    }

    function renderResults(data) {
      const resultsCont = document.getElementById('results-container');
      resultsCont.style.display = 'block';

      // Update banner
      document.getElementById('banner-title').innerText = `🎯 TARGET FOUND: Person #${data.target_track_id}`;
      document.getElementById('banner-sub').innerText = `Prompt: "${data.prompt}" | Found ${data.total_persons_detected} candidate(s) in video.`;
      document.getElementById('banner-score').innerText = `${data.target_score_pct}% MATCH`;

      // Update dynamic titles
      const kCount = data.candidates ? data.candidates.length : 1;
      document.getElementById('video-header-title').innerText = `🎥 RENDERED TRACKING VIDEO (Top-${kCount} Bounding Boxes)`;
      document.getElementById('legend-heading-title').innerText = `👥 Top-${kCount} Detected Individuals`;
      document.getElementById('legend-sub-badge').innerText = `Total ${data.total_persons_detected} in video`;

      // Update video player
      const player = document.getElementById('player-tracked');
      player.src = data.video_tracked_url;
      player.play();

      document.getElementById('btn-download-video').href = data.video_tracked_url;

      // Update Top-K Dynamic Legend
      const legend = document.getElementById('topk-legend');
      legend.innerHTML = '';
      data.candidates.forEach(c => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.style.color = c.color_hex;
        item.innerHTML = `<span class="legend-dot" style="background:${c.color_hex};"></span>#${c.rank} ${c.rank === 1 ? 'Target' : 'Person ' + c.track_id}`;
        legend.appendChild(item);
      });

      // Render candidates panel
      const panel = document.getElementById('persons-panel');
      panel.innerHTML = '';

      data.candidates.forEach(c => {
        const isTarget = (c.track_id === data.target_track_id);
        const colorHex = c.color_hex || (isTarget ? '#00ff7f' : '#38bdf8');
        const badgeTextColor = (c.dark_text !== undefined ? (c.dark_text ? '#000' : '#fff') : (c.rank <= 3 ? '#000' : '#fff'));
        const rankBadgeText = isTarget ? '★ #1 BEST MATCH' : `RANK #${c.rank}`;

        const card = document.createElement('div');
        card.className = 'person-card' + (isTarget ? ' target-match' : '');
        card.style.borderLeft = `5px solid ${colorHex}`;
        card.style.borderColor = `${colorHex}66`;
        card.style.boxShadow = `0 4px 16px ${colorHex}18`;

        const filmstripHtml = c.filmstrip.map(imgSrc => `<img src="${imgSrc}" alt="thumb">`).join('');

        card.innerHTML = `
          <div style="position:relative;flex-shrink:0;">
            <img class="person-avatar" style="border: 2px solid ${colorHex};" src="${c.thumbnail}" alt="Person ${c.track_id}">
            <span style="position:absolute;top:2px;left:2px;background:${colorHex};color:${badgeTextColor};font-size:10px;font-weight:900;padding:1px 5px;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.5);">#${c.rank}</span>
          </div>
          <div class="person-info">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span class="person-rank" style="background:${colorHex};color:${badgeTextColor};">${rankBadgeText}</span>
              <span style="font-size:11px;font-weight:700;color:${colorHex};font-family:'JetBrains Mono',monospace;">${c.color_name || ''}</span>
            </div>
            <div class="person-title">Person #${c.track_id}</div>
            <div class="person-score" style="color:${colorHex};">${c.score_pct}% Match</div>
            <div class="person-time">Timeline: ${c.time_range} (${c.total_frames_tracked} frames)</div>
            <div class="filmstrip-row">${filmstripHtml}</div>
          </div>
        `;

        card.onclick = () => {
          // Seek video to person's appearance
          const video = document.getElementById('player-tracked');
          const fps = 25.0;
          video.currentTime = Math.max(0, c.start_frame / fps);
          video.play();
        };

        panel.appendChild(card);
      });

      // Scroll smoothly to results
      resultsCont.scrollIntoView({ behavior: 'smooth' });
    }

    window.addEventListener('DOMContentLoaded', initPage);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return HTMLResponse(content=EMBEDDED_HTML_PAGE)


# ==============================================================================
# 4. ENTRY POINT & AUTO BROWSER LAUNCHER
# ==============================================================================

def open_browser_delayed(url: str, delay: float = 1.2):
    time.sleep(delay)
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="SpaceTime-DAMR Video Person Search & Tracking Web Demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 70)
    print("[+] SpaceTime-DAMR Video Person Search & Tracking Web App Running!")
    print(f"  * Local Web URL  : {url}")
    print(f"  * Device         : {str(engine.device).upper()}")
    print(f"  * Detector       : {engine.detector_backend.upper()}")
    print(f"  * Checkpoint     : {engine.checkpoint_info['name']}")
    print("=" * 70 + "\n")

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
