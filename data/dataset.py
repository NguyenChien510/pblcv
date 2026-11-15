"""
Dataset Loader for DAMR-LLM (Text-to-Video Person Retrieval)
Handles:
  - Multimodal Dataset scanning across subfolders (TVPReid-Duke, TVPReid-PRID, TVPReid-iLIDs, CUHK-PEDES)
  - Selective sub-dataset filtering (e.g. Duke only, PRID only, iLIDs only, or all)
  - Seamless linking between split CSV (train.csv/val.csv/test.csv) and JSON captions (TVPReid-Duke.json, etc.)
  - Sampling K=4 keyframes (F_key) for Appearance Stream
  - Sampling L=16 consecutive frames (F) for Motion-Difference Stream
  - Optimized OpenCV cap.grab() fast-forward frame decoding for maximum GPU throughput
  - Decomposing text query into Q_app and Q_mot via TextDecomposer
"""

import os
import json
import re
import random
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from .text_decomposer import TextDecomposer


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()


def get_default_video_transforms(img_size: int = 224, is_train: bool = True):
    """Spatio-temporal data transformations."""
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


_LR = {"left": "right", "right": "left"}


def swap_left_right(text: str) -> str:
    """'turns left' -> 'turns right' (keeps the case of the first letter); used with a horizontally flipped clip."""
    def rep(m):
        w = _LR[m.group(0).lower()]
        return w.capitalize() if m.group(0)[0].isupper() else w
    return re.sub(r"\b(left|right)\b", rep, text, flags=re.IGNORECASE)


class TVPRDataset(Dataset):
    """
    Multimodal Dataset for Text-to-Video Person Retrieval (TVPR / DAMR-LLM).
    Supports selective sub-dataset filtering (TVPReid-Duke, TVPReid-PRID, TVPReid-iLIDs, all).
    """
    def __init__(
        self,
        data_root: str = "auto",
        dataset_name: str = "TVPReid_dataset",
        sub_dataset: str = "all",
        split: str = "train",
        keyframe_num: int = 4,      # K = 4 keyframes for Appearance
        motion_frame_num: int = 16,  # L = 16 frames for Motion Difference
        num_frames: int = 16,        # T = 16 frames for SpaceTimeViT
        img_size: int = 224,
        language: str = "en",
        llm_cache: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs
    ):
        super().__init__()
        self.data_root = os.path.abspath(data_root) if (data_root and data_root != "auto") else os.path.abspath(".")
        self.dataset_name = dataset_name
        self.sub_dataset = str(sub_dataset).strip() if sub_dataset else "all"
        self.split = split.lower()
        self.num_frames = num_frames if num_frames else motion_frame_num
        # "mfgf": 4 discrete frames (Visual Encoder) followed by 16 consecutive frames (Motion Encoder), as in MFGF
        self.frame_layout = kwargs.get("frame_layout", "uniform") or "uniform"
        if self.frame_layout == "mfgf":
            self.num_frames = 4 + 16
        self.keyframe_num = keyframe_num
        self.motion_frame_num = self.num_frames
        self.img_size = img_size
        self.is_train = (self.split == "train")
        # clip_flip (recipe m3): flip all frames of a clip together (and swap left/right in the caption) instead of
        # flipping every frame independently inside the per-frame transform
        self.clip_flip = bool(kwargs.get("clip_flip", False)) and self.is_train
        self.transform = get_default_video_transforms(img_size, is_train=self.is_train and not self.clip_flip)
        self.decomposer = TextDecomposer(language=language, llm_cache=llm_cache,
                                         mode=kwargs.get("dsca_mode", "keyword") or "keyword")

        # Discover subfolder dataset directories (CUHK-PEDES, TVPReid-Duke, TVPReid-PRID, TVPReid-iLIDs)
        self.target_dirs: List[Tuple[str, str]] = []

        def is_valid_ds_dir(d_path: str) -> bool:
            return os.path.exists(os.path.join(d_path, "captions"))

        def find_ds_dirs(root: str, depth: int = 0) -> List[Tuple[str, str]]:
            dirs = []
            if not os.path.exists(root) or depth > 3:
                return dirs
            if is_valid_ds_dir(root):
                dirs.append((os.path.basename(os.path.normpath(root)), root))
                return dirs
            try:
                for entry in sorted(os.scandir(root), key=lambda e: e.name):
                    if entry.is_dir():
                        if is_valid_ds_dir(entry.path):
                            dirs.append((entry.name, entry.path))
                        else:
                            dirs.extend(find_ds_dirs(entry.path, depth + 1))
            except PermissionError:
                pass
            return dirs

        all_dirs = find_ds_dirs(self.data_root)
        if not all_dirs:
            parent_root = os.path.dirname(self.data_root)
            all_dirs = find_ds_dirs(parent_root)

        # Filter by sub_dataset if specified
        norm_sub = self.sub_dataset.lower()
        if norm_sub not in ["all", "combined", "*", "both", "none", "", "tvpreid", "tvpreid_dataset", "tvp-reid"]:
            filtered_dirs = []
            for name, path in all_dirs:
                low_name = name.lower()
                if ("duke" in norm_sub and "duke" in low_name) or \
                   ("ilids" in norm_sub and ("ilids" in low_name or "i-lids" in low_name)) or \
                   ("prid" in norm_sub and "prid" in low_name) or \
                   ("cuhk" in norm_sub and "cuhk" in low_name) or \
                   (norm_sub in low_name):
                    filtered_dirs.append((name, path))
            self.target_dirs = filtered_dirs if filtered_dirs else all_dirs
        else:
            self.target_dirs = all_dirs

        print(f"[*] TVPRDataset [{self.split.upper()}]: Scanning sub-dataset='{self.sub_dataset}'. "
              f"Matched {len(self.target_dirs)} dataset folders: {[d[0] for d in self.target_dirs]}")

        # Index dataset samples
        self.samples: List[Dict[str, Any]] = self._build_index()
        print(f"[+] Loaded {len(self.samples)} valid multimodal samples for split='{self.split}'.")

    def _build_index(self) -> List[Dict[str, Any]]:
        samples = []

        def extract_pid(vid_name: str, fallback_idx: int) -> int:
            m = re.search(r'(\d+)', str(vid_name))
            return int(m.group(1)) if m else fallback_idx

        for ds_name, ds_path in self.target_dirs:
            captions_dir = os.path.join(ds_path, "captions")
            if not os.path.exists(captions_dir):
                continue

            # 1. Load JSON captions in captions_dir (e.g. TVPReid-PRID.json, TVPReid-Duke.json)
            caption_dict: Dict[str, Any] = {}
            for jf in glob.glob(os.path.join(captions_dir, "*.json")):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        jdata = json.load(f)
                        if isinstance(jdata, dict):
                            caption_dict.update(jdata)
                except Exception as e:
                    print(f"[!] Warning reading JSON {jf}: {e}")

            # 2. Locate split CSV file (train.csv, val.csv, test.csv)
            csv_candidates = [
                os.path.join(captions_dir, f"{self.split}.csv"),
                os.path.join(captions_dir, f"{self.split}_captions.csv"),
                os.path.join(captions_dir, "captions.csv")
            ]
            csv_path = None
            for cp in csv_candidates:
                if os.path.exists(cp):
                    csv_path = cp
                    break

            def find_video_file(vid_key: str) -> Optional[str]:
                clean_k = os.path.basename(vid_key)
                clean_no_ext = os.path.splitext(clean_k)[0]
                candidates = [
                    os.path.join(ds_path, "videos", f"{clean_no_ext}.mp4"),
                    os.path.join(ds_path, "videos", clean_k),
                    os.path.join(ds_path, "raw_videos", f"{clean_no_ext}.mp4"),
                    os.path.join(ds_path, "raw_videos", clean_k),
                    os.path.join(ds_path, "images", clean_k),
                    os.path.join(ds_path, "images", f"{clean_no_ext}.jpg"),
                    os.path.join(ds_path, clean_k),
                    os.path.join(ds_path, f"{clean_no_ext}.mp4")
                ]
                for c in candidates:
                    if os.path.exists(c):
                        return c
                return os.path.join(ds_path, "videos", f"{clean_no_ext}.mp4")

            # Case A: Split CSV exists
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    has_direct_caption = any(c in df.columns for c in ["caption", "text", "description"])

                    if has_direct_caption:
                        cap_col = next(c for c in ["caption", "text", "description"] if c in df.columns)
                        for _, row in df.iterrows():
                            cap = clean_text(str(row.get(cap_col, "")))
                            if not cap:
                                continue
                            vid_rel = str(row.get("video_path", row.get("video", row.get("image", row.get("video_id", "")))))
                            v_full = find_video_file(vid_rel)
                            pid = int(row.get("person_id", row.get("id", extract_pid(vid_rel, len(samples)))))
                            samples.append({
                                "dataset": ds_name,
                                "video_path": v_full,
                                "video_id": str(row.get("video_id", os.path.splitext(os.path.basename(vid_rel))[0])),
                                "person_id": pid,
                                "caption": cap
                            })
                    elif "video_id" in df.columns and caption_dict:
                        # Standard TVPReid format: split.csv lists video_id, JSON contains list of captions
                        for _, row in df.iterrows():
                            vid_id = str(row["video_id"]).strip()
                            if not vid_id:
                                continue
                            clean_id = os.path.splitext(os.path.basename(vid_id))[0]
                            # Match in JSON dictionary
                            caps = caption_dict.get(vid_id) or caption_dict.get(f"{clean_id}.mp4") or caption_dict.get(clean_id) or []
                            if isinstance(caps, str):
                                caps = [caps]
                            
                            v_full = find_video_file(clean_id)
                            pid = extract_pid(clean_id, len(samples))

                            for cap_text in caps:
                                cap_clean = clean_text(str(cap_text))
                                if cap_clean:
                                    samples.append({
                                        "dataset": ds_name,
                                        "video_path": v_full,
                                        "video_id": clean_id,
                                        "person_id": pid,
                                        "caption": cap_clean
                                    })
                except Exception as e:
                    print(f"[!] Warning reading CSV {csv_path}: {e}")

            # Case B: JSON-only annotations (CUHK-PEDES or dataset without CSV)
            if not samples and caption_dict:
                for k, val in caption_dict.items():
                    clean_k = os.path.splitext(os.path.basename(k))[0]
                    v_full = find_video_file(k)
                    if isinstance(val, dict):
                        cap_text = clean_text(val.get("text", val.get("caption", "")))
                        if cap_text:
                            pid = int(val.get("id", extract_pid(clean_k, len(samples))))
                            samples.append({
                                "dataset": ds_name,
                                "video_path": v_full,
                                "video_id": clean_k,
                                "person_id": pid,
                                "caption": cap_text
                            })
                    elif isinstance(val, list):
                        pid = extract_pid(clean_k, len(samples))
                        for cap_text in val:
                            cap_clean = clean_text(str(cap_text))
                            if cap_clean:
                                samples.append({
                                    "dataset": ds_name,
                                    "video_path": v_full,
                                    "video_id": clean_k,
                                    "person_id": pid,
                                    "caption": cap_clean
                                })

        # Fallback if no samples found
        if len(samples) == 0:
            print(f"[!] Notice: No caption annotations found under {self.data_root}. Generating dummy index for test/dry-run.")
            for i in range(32 if self.split == "train" else 16):
                samples.append({
                    "dataset": "Synthetic",
                    "video_path": None,
                    "video_id": f"dummy_vid_{i}",
                    "person_id": i % 8,
                    "caption": f"A pedestrian with backpack and blue shirt walking quickly #{i}"
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_indices(self, total_frames: int, num_frames: int) -> List[int]:
        """
        Segment-based temporal sampling:
        Divides the video sequence into num_frames equal segments.
        - Training: randomly samples 1 frame per segment (temporal jitter augmentation).
        - Evaluation: samples the deterministic midpoint of each segment.
        """
        if total_frames <= 0:
            return [0] * num_frames
        if total_frames <= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            return indices

        seg_size = float(total_frames) / float(num_frames)
        indices = []
        for i in range(num_frames):
            seg_start = int(i * seg_size)
            seg_end = int((i + 1) * seg_size)
            if self.is_train:
                idx = random.randint(seg_start, max(seg_start, seg_end - 1))
            else:
                idx = (seg_start + seg_end) // 2
            indices.append(min(idx, total_frames - 1))
        return indices

    def _layout_indices(self, total_frames: int) -> List[int]:
        if self.frame_layout != "mfgf":
            return self._sample_frame_indices(total_frames, self.num_frames)
        discrete = self._sample_frame_indices(total_frames, 4)
        if total_frames <= 16:
            consecutive = np.linspace(0, max(total_frames - 1, 0), 16, dtype=int).tolist()
        else:
            start = random.randint(0, total_frames - 16) if self.is_train else (total_frames - 16) // 2
            consecutive = list(range(start, start + 16))
        return discrete + consecutive

    def _read_video_frames(self, video_path: Optional[str]) -> torch.Tensor:
        """
        Extracts T = self.num_frames (default 16) frames as a tensor of shape [T, 3, H, W].
        Uses segment-based temporal sampling and OpenCV cap.grab() fast seeking.
        """
        t = self.num_frames
        default_frame = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
        default_tensor = self.transform(default_frame)

        if not video_path or not os.path.exists(video_path):
            return torch.stack([default_tensor] * t)

        # 1. Image sequence folder (.jpg / .png)
        if os.path.isdir(video_path):
            img_files = sorted(
                glob.glob(os.path.join(video_path, "*.jpg")) +
                glob.glob(os.path.join(video_path, "*.png"))
            )
            if not img_files:
                return torch.stack([default_tensor] * t)

            num_f = len(img_files)
            target_indices = self._layout_indices(num_f)
            frame_tensors = []
            for idx in target_indices:
                try:
                    img = Image.open(img_files[idx]).convert("RGB")
                    frame_tensors.append(self.transform(img))
                except Exception:
                    frame_tensors.append(default_tensor)
            return torch.stack(frame_tensors)

        # 2. Video file (.mp4 / .avi)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.stack([default_tensor] * t)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = t

        target_indices = self._layout_indices(total_frames)
        needed_set = set(target_indices)
        max_needed = max(target_indices) if target_indices else 0

        frames_dict: Dict[int, torch.Tensor] = {}
        cur_idx = 0
        while cap.isOpened() and cur_idx <= max_needed:
            if cur_idx in needed_set:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames_dict[cur_idx] = self.transform(pil_img)
            else:
                if not cap.grab():
                    break
            cur_idx += 1

        cap.release()

        frame_tensors = [frames_dict.get(idx, default_tensor) for idx in target_indices]
        while len(frame_tensors) < t:
            frame_tensors.append(frame_tensors[-1] if frame_tensors else default_tensor)

        return torch.stack(frame_tensors[:t])

    def _read_frames(self, video_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible helper returning (keyframes, motion_frames)."""
        video_frames = self._read_video_frames(video_path)
        k = min(self.keyframe_num, video_frames.size(0))
        return video_frames[:k], video_frames

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        caption = item["caption"]
        video_frames = self._read_video_frames(item.get("video_path"))
        if self.clip_flip and random.random() < 0.5:
            video_frames = torch.flip(video_frames, dims=[-1])
            caption = swap_left_right(caption)

        # Disentangled Semantic Concept Alignment (DSCA) decomposition
        q_app, q_mot = self.decomposer.decompose(caption)

        k = min(self.keyframe_num, video_frames.size(0))
        return {
            "video_id": item.get("video_id", f"vid_{index}"),
            # video_id repeats across sub-datasets (person0001 exists in PRID and iLIDs); the key does not
            "video_key": f"{item.get('dataset', '')}/{item.get('video_id', f'vid_{index}')}",
            "person_id": item.get("person_id", index),
            "caption": caption,
            "q_app": q_app,
            "q_mot": q_mot,
            "video_frames": video_frames,        # [T, C, H, W]
            # Backward-compatibility keys
            "keyframes": video_frames[:k],       # [K, C, H, W]
            "motion_frames": video_frames,       # [T, C, H, W]
        }


class UniqueVideoBatchSampler(torch.utils.data.Sampler):
    """
    Shuffled batches in which no two captions belong to the same video. With plain shuffling the two captions of a
    video can share a batch, and InfoNCE then treats one of them as a negative of the other's video.
    Reshuffled every epoch from a generator seeded once (set_epoch keeps the order reproducible after a resume).
    """

    def __init__(self, video_keys: List[str], batch_size: int, seed: int = 0):
        self.keys = list(video_keys)
        self.batch_size = batch_size
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.keys) // self.batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000003 + self.epoch)
        self.epoch += 1
        pending = torch.randperm(len(self.keys), generator=g).tolist()
        n_out = 0
        while pending and n_out < len(self):
            batch, used, rest = [], set(), []
            for i in pending:
                if len(batch) < self.batch_size and self.keys[i] not in used:
                    batch.append(i)
                    used.add(self.keys[i])
                else:
                    rest.append(i)
            if len(batch) < self.batch_size:
                break                                  # drop_last: not enough distinct videos left
            yield batch
            n_out += 1
            pending = rest


def create_dataloaders(
    data_root: str,
    dataset_name: str = "TVPReid_dataset",
    sub_dataset: str = "all",
    batch_size: int = 16,
    num_workers: int = 2,
    keyframe_num: int = 4,
    motion_frame_num: int = 16,
    num_frames: int = 16,
    img_size: int = 224,
    language: str = "en",
    pin_memory: bool = True,
    unique_video: bool = False,
    seed: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch Train and Val DataLoaders for Spatio-Temporal DAMR."""
    target_frames = num_frames if num_frames else motion_frame_num

    train_dataset = TVPRDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        sub_dataset=sub_dataset,
        split="train",
        keyframe_num=keyframe_num,
        motion_frame_num=target_frames,
        num_frames=target_frames,
        img_size=img_size,
        language=language,
        frame_layout=kwargs.get("frame_layout", "uniform"),
        clip_flip=kwargs.get("clip_flip", False),
        dsca_mode=kwargs.get("dsca_mode", "keyword")
    )
    val_dataset = TVPRDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        sub_dataset=sub_dataset,
        split="val",
        keyframe_num=keyframe_num,
        motion_frame_num=target_frames,
        num_frames=target_frames,
        img_size=img_size,
        language=language,
        frame_layout=kwargs.get("frame_layout", "uniform"),
        dsca_mode=kwargs.get("dsca_mode", "keyword")
    )

    is_cuda = torch.cuda.is_available() and pin_memory
    persistent = (num_workers > 0)
    prefetch = 2 if num_workers > 0 else None

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": is_cuda
    }
    if persistent:
        loader_kwargs["persistent_workers"] = True
    if prefetch:
        loader_kwargs["prefetch_factor"] = prefetch

    if unique_video:
        keys = [f"{s.get('dataset', '')}/{s.get('video_id', i)}" for i, s in enumerate(train_dataset.samples)]
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=UniqueVideoBatchSampler(keys, batch_size, seed=seed),
            **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    return train_loader, val_loader
