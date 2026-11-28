"""
Person search engine for the web demo.

Splits the work into a one-off ANALYZE step per video (detect -> track -> crop -> encode
V_app/V_mot/V_video per track, all cached) and a cheap SEARCH step per prompt (decompose into
Q_app/Q_mot with TextDecomposer, encode T_full/T_app/T_mot, three cosine scores per track).
Fusion weights are applied client-side so they can be tuned live; the defaults match
train.py::evaluate (0.50 joint / 0.35 app / 0.15 mot).
"""

import base64
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.text_decomposer import TextDecomposer
from models.damr_main import DAMRModel

from . import vi_query

ProgressCb = Callable[[str, float, float], None]

DEFAULT_WEIGHTS = {"joint": 0.45, "app": 0.45, "mot": 0.09}   # the three DSCA branches; the UI tunes only these
CLIP_WEIGHT = 0.2           # hidden CLIP branch: fused = (DSCA + CLIP_WEIGHT * clip) / (1 + CLIP_WEIGHT), not user-tunable
CLIP_MODEL = "openai/clip-vit-base-patch16"
# SPEED MODE (enabled 2026-11-21 on request). UNDO: CLIP_CROPS = 3 and ENCODER_INT8 = False.
# Measured on the two hand-labelled videos (each change on its own): CLIP on the thumb only instead of thumb +
# first/last film crop cut CLIP time on the station clip 105 s -> 36 s; dynamic int8 Linear layers in the video
# encoder cut its encoding 266 s -> 195 s. Cost: top-1 8/9 -> 7/9 (TESTVIDEO, from the CLIP change) and 15/15 -> 14/15
# (station clip, from either change).
CLIP_CROPS = 1              # preview crops per track for CLIP (1 = thumb; 3 = thumb + first/last film crop); ~230 ms each on CPU
ENCODER_INT8 = True         # CPU only: torch dynamic int8 quantisation of the video encoder's nn.Linear layers
CLIP_BATCH = 16
CLIP_PREFIX = "a photo of "
# Loaded at startup when present: recipe m3c joint model (MFGF backbones, trained on the train splits of all three
# TVPReid subsets after CUHK-PEDES text-image pre-training). On the three test splits it retrieves far better than
# the per-subset SpaceTimeViT checkpoints, which barely transfer between subsets.
PREFERRED_CHECKPOINT = "m3c_joint_seed42.pth"
NUM_TRACK_FRAMES = 16
MFGF_VISUAL_FRAMES = 4      # backbone "mfgf": 4 discrete frames for the Space-Time ViT ...
MFGF_MOTION_FRAMES = 16     # ... followed by 16 consecutive frames for S3D (data/dataset.py, frame_layout="mfgf")
MFGF_MOTION_FPS = 15.0      # frame rate of the TVPReid tracklets the motion encoder was trained on
PREVIEW_SLOTS = (0, 5, 8, 10, 15)   # of the 16 uniform picks: film crops, with slot 8 as the thumbnail
CROP_SIZE = 224
ENCODE_BATCH = 4
SOFTMAX_TAU = 0.07          # same temperature as the InfoNCE training loss

# --- tracking (YOLO + BoT-SORT with native ReID) ---
TRACKER_VERSION = "botsort-reid-v9-direction"   # part of the analysis cache key; bump when analysis output changes
DETECT_IMGSZ = 960          # upscaled inference finds small, distant people in crowds (Accurate mode)
DETECT_IMGSZ_FAST = 640     # Balanced/Fast modes: skips tiny far-away people -> fewer tracks to encode
DETECT_CONF_LOW = 0.1       # low-score boxes feed BoT-SORT's second association stage
TRACK_BUFFER_S = 2.0        # seconds a lost track is kept for re-association
MAX_INTERP_GAP_S = 2.0      # fill boxes across gaps up to this long
MIN_TRACK_DETECTIONS = 12   # drop tracks seen fewer times than this (~0.5s at 25 fps)
MIN_TRACK_HEIGHT = 40       # px; median box height below this gives useless crops

# --- tracklet stitching with the model's own appearance embedding (V_app) ---
STITCH_MAX_GAP_S = 8.0      # max time between the end of A and the start of B (occlusion length)
STITCH_SPEED = 1.5          # max movement in box-heights per second (plus half a height of slack)
STITCH_MIN_SIM = 0.92       # fallback without the re-ID model: cosine(V_app_A, V_app_B) (different people peak ~0.92)
REID_MODEL = "person-reidentification-retail-0277"   # OpenVINO person re-ID, 256-D embedding of a 256x128 crop
REID_MIN_SIM = 0.60         # cosine of mean re-ID embeddings to join tracklets; on a crowded station clip different
                            # people p99 ~0.40, the same person median ~0.79; 0.52-0.56 produced visible wrong joins
REID_SAMPLE_S = 0.25        # during tracking, one re-ID sample per track every this many seconds...
REID_SAMPLE_MAX_OCCL = 0.3  # ...only from detections with at most this much overlap with other boxes
SPLIT_MAX_SIM = 0.40        # split a track where the mean re-ID embeddings before/after differ this much
SPLIT_MIN_SAMPLES = 2       # samples needed on each side of a split
STITCH_GENDER_GAP = 0.6     # never join tracklets whose P(male) differ by more than this

# --- pedestrian attributes (gender etc.) with OpenVINO person-attributes-recognition-crossroad-0230 ---
ATTR_MODEL = "person-attributes-recognition-crossroad-0230"
OMZ_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{0}/FP32/{0}"
ATTR_NAMES = ["male", "bag", "backpack", "hat", "longsleeves", "longpants", "longhair", "coat"]
MALE_WORDS = {"man", "men", "male", "males", "guy", "guys", "boy", "boys", "gentleman", "gentlemen", "he", "him", "his"}
FEMALE_WORDS = {"woman", "women", "female", "females", "lady", "ladies", "girl", "girls", "she", "her", "hers"}
# PA-100K pedestrian attributes: PaddleClas PULC person_attribute (PP-LCNet x1.0, Apache-2.0), read by OpenVINO's
# Paddle frontend from checkpoints/attributes/person_attribute_infer/ (downloaded once). 192x256 RGB, ImageNet
# normalisation, 26 sigmoid scores in PA-100K order (PAR_INDEX lists the ones used). Ranking AUC on all 6,559 TVPReid
# tracklets, labels taken from the captions (crossroad-0230 -> PA-100K): female 0.80 -> 0.90, hat 0.68 -> 0.77,
# backpack 0.76 -> 0.79 (0.81 when averaged with crossroad-0230, which is what backpack uses), long trousers
# 0.83 -> 0.95, shorts 0.91 -> 0.95 (only 7 positives), skirt/dress - -> 0.95, long coat 0.58 -> 0.68 (captions call
# jackets coats), bag 0.61 -> 0.74. crossroad-0230 still gives long hair, long sleeves and the colour points.
PAR_DIR = "person_attribute_infer"
PAR_URL = "https://paddleclas.bj.bcebos.com/models/PULC/inference/person_attribute_infer.tar"
PAR_INDEX = {"hat": 0, "longcoat": 10, "longpants": 11, "shorts": 12, "skirt": 13, "handbag": 15, "shoulderbag": 16,
             "backpack": 17, "female": 22}
PAR_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
PAR_STD = np.array([0.229, 0.224, 0.225], np.float32)
POSE_MODEL = "yolov8n-pose.pt"  # shoulders/hips/knees locate the top & bottom garment inside each crop

# Query constraints matched against the attribute model; fused -= weight * (1 - p_match). The weights are small on
# purpose: on TVPReid test tracklets these attributes separate the described person only weakly (ranking AUC
# 0.6-0.7), and with the m3c joint checkpoint weights 3x larger cost 13 points of R@1 (60 -> 47, mean of the three
# subsets), while this scale is neutral there and still reorders close calls. "coat" (p ~0.1 for everyone) and
# "wears long pants" (AUC 0.52) carried no signal and are not used as constraints.
# Ranking weights re-chosen 2026-11-28 with the PA-100K attributes, on TVPReid val (mean R@1 over the subsets, demo
# scoring) and half of a 46-query set written for the station clip, checked on TVPReid test and the other half.
# Gender and top colour carry no ranking weight: alone each lowered TVPReid val R@1 by 1.3; dropping both raised it
# from 55.6 to 60.2 (test 55.7 -> 57.2), above using no constraints (59.9 / 57.0). On the station clip the top-1 was
# never confidently of the wrong gender without constraints (66 gender queries), so DSCA + CLIP already get gender
# right there, and dropping top colour helped there too (EXT 27 -> 29 of 46, GT15 14 -> 15). With these weights the
# real-footage sets give TESTVIDEO 7/9, GT15 15/15, EXT 27/46 against 7/9, 14/15, 24/46 with no constraints. Zero-
# weight constraints still show as chips and still count for the "matching attributes only" filter.
CONSTRAINT_WEIGHTS = {"gender": 0.0, "top_color": 0.0, "bottom_color": 0.06, "backpack": 0.05, "bag": 0.03,
                      "hat": 0.05, "longsleeves": 0.03, "longpants": 0.03, "longhair": 0.05, "suitcase": 0.15,
                      "shorts": 0.05, "skirt": 0.05, "longcoat": 0.03}
LOOSE_COLOR_WEIGHT = 0.0    # "a man in red" (color without a garment) -> top_color; no ranking weight (see above)
ATTR_PATTERNS = {           # key -> (regex meaning "yes" or None, regex meaning "no" or None)
    "backpack": (r"\b(backpacks?|rucksacks?)\b", None),
    "bag": (r"\b(bags?|handbags?|purses?|totes?)\b", None),
    "hat": (r"\b(hats?|caps?|beanies?|helmets?)\b", None),
    "longsleeves": (r"\blong[- ]?sleeve[ds]?\b", r"\b(short[- ]?sleeve[ds]?|sleeveless|t-?shirts?|tees?|tank tops?)\b"),
    "longpants": (r"\b(trousers|pants|jeans|slacks|leggings|sweatpants|chinos|joggers)\b", None),
    "shorts": (r"\bshorts\b", None),
    "skirt": (r"\b(skirts?|dress|dresses|gowns?)\b", None),           # \b: "dressed in" is not a dress
    "longcoat": (r"\b(coats?|overcoats?|trench ?coats?|raincoats?|parkas?)\b", None),
    "longhair": (r"\b(long[- ]hair(ed)?|ponytail)\b", r"\b(short[- ]hair(ed)?|bald)\b"),
}
# Actions are measured from the track itself, not by the text model: TVPReid captions mention walking in 57% of
# cases but running in 0.8%, standing in 0.3% and sitting in 0.02% (3 captions, all wheelchairs), so the DSCA
# scores barely react to these words. Speed = displacement of the box centre over ACTION_WINDOW_S, in box heights
# per second (plus the change of log box height, i.e. moving towards/away from the camera). Sitting comes from pose
# keypoints: knees close to hip height relative to the torso.
# Calibrated by eye on two real videos (2026-11-21): 34 people standing in a station crowd measured a median of
# 0.06 (max 0.125); 18 people walking far from the camera, mostly along its axis, measured 0.13-0.9. A 1 s window
# separated them best (2-3 s windows were worse). Running is set from physics only: no runner was available.
ACTION_WINDOW_S = 1.0       # long enough to average out box jitter of a standing person
ACTION_STEP_S = 0.25
ACTION_MIN_TRACK_S = 1.0    # less observed time than this gives no action (short tracks at the frame edge looked fast)
ACTION_EDGE_PX = 3          # windows whose box touches the frame border are skipped (a body entering the frame "grows")
STAND_MAX_SPEED = 0.13      # body heights per second; below -> standing still
RUN_MIN_SPEED = 1.5         # above -> running (walking across the image is about 0.7-0.9)
ACTION_MIN_FRACTION = 0.7   # standing/running for this fraction of the track counts fully (p = 1)
SIT_THIGH_RATIO = 0.45      # (knee_y - hip_y) / (hip_y - shoulder_y) below this -> sitting (standing ~0.7-1.0)
ACTION_PATTERNS = {         # constraint key -> query regex
    "act_run": r"\b(run|runs|running|ran|jog|jogs|jogging|sprint|sprints|sprinting|rushing|hurrying)\b",
    "act_walk": r"\b(walk|walks|walking|walked|strolling|strolls)\b",
    "act_stand": r"\b(stand|stands|standing|stood|waiting|waits)\b",
    "act_sit": r"\b(sit|sits|sitting|sat|seated)\b",
}
ACTION_WEIGHTS = {"act_run": 0.10, "act_walk": 0.10, "act_stand": 0.10, "act_sit": 0.08}

# Direction of motion in the image (viewer's left/right; towards/away from the camera), from the track's boxes:
# least-squares slopes of the box centre (lateral, body heights/s), of log box height and of the feet position, per
# contiguous run that keeps off the frame border, weighted by run length. TVPReid captions describe motion mostly as
# direction ("walking to the left", "towards the camera"). Checked against drawn trajectories: all 16 measurable
# TESTVIDEO tracks and the most lateral station-clip tracks went the way the slopes say. Approach uses both log-height
# growth and the feet moving down the image, because either alone disagreed for people cut by the bottom border.
DIR_LATERAL_MID, DIR_LATERAL_SCALE = 0.10, 0.035      # |lateral slope| in body heights/s at which p = 0.5
DIR_APPROACH_UNITS = (0.05, 0.15)                     # log-height /s and feet body-heights/s that count as one unit
DIR_APPROACH_MID, DIR_APPROACH_SCALE = 1.0, 0.35      # summed units at which p(towards) = 0.5
DIRECTION_PATTERNS = {
    "act_toward": r"\b(towards?|to) (the )?camera\b|\btowards? (us|the viewer)\b|\b(approach|approaches|approaching) (the )?camera\b"
                  r"|\bcoming (closer|towards?|this way)\b",
    "act_away": r"\baway from (the )?camera\b|\b(walk|walks|walking|walked|move|moves|moving|head|heads|heading|go|goes|"
                r"going|run|runs|running) away\b",
    "act_left": r"\b(to|towards?) the left\b|\bleftwards?\b|\bfrom (the )?right to (the )?left\b"
                r"|\b(heading|going|walking|moving|turning) left\b",
    "act_right": r"\b(to|towards?) the right\b|\brightwards?\b|\bfrom (the )?left to (the )?right\b"
                 r"|\b(heading|going|walking|moving|turning) right\b",
}

# Objects that travel with a person, found by the person detector in the same forward pass (YOLO COCO classes,
# _split_objects keeps them away from the tracker). Each object box goes to the best-fitting visible person of its
# frame (_object_fit); a track's score is the fraction of its frames with that object. Checked by eye on the station
# clip (ok.mp4): all 9 tracks with a suitcase fraction >= 0.35 had a suitcase, 0.1-0.3 was mixed (crowds of luggage);
# handbags are often hidden by the body, so their fractions run lower. Tried and not used: bicycles (COCO 1; a walked
# bicycle is mostly hidden by its owner and got 5 frames for the owner and for a passer-by alike), an AVA action model
# (SlowFast R50 fired only on walk/stand/carry on this footage) and CLIP zero-shot (3/8 suitcase carriers in its top 8
# against 7/8 here). COCO has no stroller class. The suitcase weight (0.15, larger than the attribute weights because
# the detector is far more precise than the attribute model) was picked on the same clip: suitcase carriers in the top
# 5 of six suitcase queries went 5/30 without the constraint, 14/30 at 0.08, 20/30 at 0.16 (0.24: 23/30, but then
# the suitcase starts to outweigh the clothing words).
OBJECT_CLASSES = {28: "suitcase", 26: "handbag"}
OBJECT_MIN_CONF = 0.25
OBJECT_FRAC_MID = {"suitcase": 0.25, "handbag": 0.20}   # fraction of the track's frames at which p = 0.5
OBJECT_FRAC_SCALE = 0.06
SUITCASE_PATTERN = (r"\b(trolley bags?|roller bags?|rolling bags?|wheeled bags?|suitcases?|luggage|trolleys?|"
                    r"carry-ons?)\b")   # longer phrases first, so "trolley bag" is removed whole before the bag check
TOP_GARMENTS = {"shirt", "shirts", "t-shirt", "tshirt", "tee", "top", "jacket", "coat", "sweater", "hoodie", "blouse",
                "vest", "dress", "suit", "cardigan", "blazer", "jumper", "sweatshirt", "polo", "parka", "overcoat"}
BOTTOM_GARMENTS = {"pants", "trousers", "jeans", "shorts", "skirt", "leggings", "sweatpants"}
OTHER_ITEMS = {"bag", "bags", "backpack", "handbag", "purse", "hat", "cap", "shoes", "sneakers", "boots", "hair",
               "umbrella", "suitcase", "suitcases", "luggage", "trolley", "glasses", "sunglasses", "helmet", "beanie", "scarf", "mask", "socks"}
COLOR_ALIASES = {"black": "black", "white": "white", "grey": "grey", "gray": "grey", "silver": "grey",
                 "red": "red", "maroon": "red", "burgundy": "red", "orange": "orange", "yellow": "yellow",
                 "gold": "yellow", "green": "green", "olive": "green", "blue": "blue", "navy": "blue",
                 "denim": "blue", "turquoise": "blue", "teal": "blue",
                 "cyan": "blue", "purple": "purple", "violet": "purple", "pink": "pink", "brown": "brown",
                 "beige": "beige", "khaki": "beige", "tan": "beige", "cream": "beige"}
COLOR_MODIFIERS = {"dark", "light", "bright", "pale", "deep", "and", "or", "/"}
# A few sRGB prototypes per colour name; pixels are assigned to the nearest one in (weighted) Lab space.
COLOR_PROTOS = {
    "black": [(20, 20, 20), (45, 45, 50)],
    "white": [(240, 240, 240), (210, 210, 215)],
    "grey": [(128, 128, 128), (95, 95, 100), (170, 170, 172)],
    "red": [(200, 30, 35), (135, 20, 30)],
    "orange": [(240, 130, 30)],
    "yellow": [(235, 210, 40)],
    "green": [(40, 140, 60), (75, 85, 45)],
    "blue": [(40, 80, 190), (30, 40, 90), (110, 150, 205),
             (55, 70, 100), (140, 160, 190), (60, 165, 175)],   # + dark/light denim, turquoise
    "purple": [(120, 50, 140), (90, 60, 120)],
    "pink": [(240, 140, 180), (215, 90, 140)],
    "brown": [(110, 70, 40), (80, 55, 40)],
    "beige": [(215, 195, 160), (185, 165, 125)],
}
COLOR_NAMES = list(COLOR_PROTOS)
COLOR_SIM = {("grey", "black"): 0.5, ("grey", "white"): 0.5, ("brown", "beige"): 0.5, ("brown", "orange"): 0.4,
             ("pink", "red"): 0.5, ("purple", "pink"): 0.4, ("purple", "blue"): 0.4, ("beige", "white"): 0.4,
             ("yellow", "orange"): 0.5, ("beige", "yellow"): 0.3, ("brown", "black"): 0.3, ("red", "orange"): 0.3}

# Legacy greedy tracker (only for the non-YOLO fallback detectors)
MAX_TRACK_GAP = 15

# Distinct rank colours shared by the rendered MP4 (BGR) and the web UI (hex)
RANK_COLORS = [
    ((127, 255, 0), "#00ff7f"),
    ((255, 210, 0), "#00d2ff"),
    ((0, 184, 255), "#ffb800"),
    ((133, 42, 255), "#ff2a85"),
    ((247, 85, 168), "#a855f7"),
    ((74, 107, 255), "#ff6b4a"),
    ((0, 255, 204), "#ccff00"),
    ((212, 245, 0), "#00f5d4"),
    ((248, 189, 56), "#38bdf8"),
    ((72, 29, 225), "#e11d48"),
]
OTHER_COLOR = ((184, 163, 148), "#94a3b8")
DARK_TEXT_RANKS = {1, 2, 3, 7, 8, 9}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rank_color(rank: int) -> Tuple[Tuple[int, int, int], str]:
    return RANK_COLORS[rank - 1] if 1 <= rank <= len(RANK_COLORS) else OTHER_COLOR


def _jpeg(img_bgr: np.ndarray, max_size: int, quality: int = 85) -> bytes:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        s = max_size / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _iou(a: List[int], b: List[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def _max_overlap(xyxy: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by any other box in the same frame."""
    n = len(xyxy)
    if n < 2:
        return np.zeros(n)
    x1, y1, x2, y2 = (xyxy[:, i] for i in range(4))
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    cover = iw * ih / np.maximum((x2 - x1) * (y2 - y1), 1e-6)[:, None]
    np.fill_diagonal(cover, 0)
    return np.clip(cover.max(axis=1), 0, 1)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """[N, 3] uint8 RGB -> [N, 3] float Lab (L 0..100), with L down-weighted so shading matters less than hue."""
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    return lab * np.array([0.6, 1.0, 1.0], dtype=np.float32)


_PROTO_LAB = _lab(np.array([c for n in COLOR_NAMES for c in COLOR_PROTOS[n]], dtype=np.uint8))
_PROTO_IDX = np.array([i for i, n in enumerate(COLOR_NAMES) for _ in COLOR_PROTOS[n]])


def _color_hist(region_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Fraction of the region's pixels nearest to each colour name (len(COLOR_NAMES),), or None if too small."""
    if region_bgr is None or region_bgr.shape[0] < 3 or region_bgr.shape[1] < 3:
        return None
    if region_bgr.shape[0] * region_bgr.shape[1] > 900:
        region_bgr = cv2.resize(region_bgr, (30, 30), interpolation=cv2.INTER_AREA)
    lab = _lab(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3))
    nearest = _PROTO_IDX[((lab[:, None] - _PROTO_LAB[None]) ** 2).sum(-1).argmin(1)]
    return np.bincount(nearest, minlength=len(COLOR_NAMES)) / len(nearest)


def _garment_regions(kp: Optional[np.ndarray], w: int, h: int, top_pt, bottom_pt):
    """
    Top/bottom garment boxes inside a crop of size (w, h). kp: COCO keypoints [17, 3] in crop coordinates.
    Uses shoulders -> hips for the top and hips -> knees for the bottom; the bottom is None when the hips are
    not visible (crop shows only the upper body). Falls back to the attribute model's colour points.
    """
    def box(x1, y1, x2, y2):
        x1, x2 = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
        y1, y2 = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
        return (x1, y1, x2, y2) if x2 - x1 >= 3 and y2 - y1 >= 3 else None

    vis = (lambda i: kp is not None and kp[i, 2] > 0.5)
    top = bottom = None
    if vis(5) and vis(6):
        sx1, sx2 = sorted((kp[5, 0], kp[6, 0]))
        sw = max(sx2 - sx1, 0.2 * w)
        sy = (kp[5, 1] + kp[6, 1]) / 2
        hips = [i for i in (11, 12) if vis(i)]
        hy = float(np.mean([kp[i, 1] for i in hips])) if hips else min(h, sy + 1.5 * sw)
        L = max(hy - sy, 1.0)
        top = box(sx1 + 0.15 * sw, sy + 0.15 * L, sx2 - 0.15 * sw, hy - 0.1 * L)
        if hips:
            hx = [kp[i, 0] for i in hips]
            hx1, hx2 = min(hx), max(hx)
            hw = max(hx2 - hx1, 0.5 * sw)
            cx = (hx1 + hx2) / 2
            knees = [i for i in (13, 14) if vis(i)]
            ky = float(np.mean([kp[i, 1] for i in knees])) if knees else min(h, hy + 0.9 * L)
            if ky - hy >= 0.4 * L:
                Lb = ky - hy
                bottom = box(cx - 0.6 * hw, hy + 0.1 * Lb, cx + 0.6 * hw, ky - 0.05 * Lb)
    elif top_pt is not None:
        tx, ty = top_pt[0] * w, top_pt[1] * h
        top = box(tx - 0.15 * w, ty - 0.06 * h, tx + 0.15 * w, ty + 0.06 * h)
    return top, bottom


def _parse_colors(words: List[str], i: int) -> List[str]:
    """Colour names directly before position i ("dark blue", "black and white")."""
    out, j = [], i - 1
    while j >= 0 and (words[j] in COLOR_ALIASES or words[j] in COLOR_MODIFIERS):
        if words[j] in COLOR_ALIASES:
            out.append(COLOR_ALIASES[words[j]])
        j -= 1
    return out


def query_attributes(prompt: str, with_attrs: bool = True) -> List[Dict[str, Any]]:
    """
    Structured constraints found in the prompt: gender, garment colours, attribute-model attributes (only when
    with_attrs, i.e. the attribute model is loaded) and actions measured from the tracks (always).
    Each is {"key", "value", "weight"}; value is "male"/"female", a list of colour names, or True/False.
    """
    text = prompt.lower()
    out: List[Dict[str, Any]] = [{"key": key, "value": True, "weight": ACTION_WEIGHTS[key]}
                                 for key, rx in ACTION_PATTERNS.items() if re.search(rx, text)]
    out += [{"key": key, "value": True, "weight": ACTION_WEIGHTS["act_walk"]}
            for key, rx in DIRECTION_PATTERNS.items() if re.search(rx, text)]
    if re.search(SUITCASE_PATTERN, text):     # measured from the detector's object boxes, not the attribute model
        out.append({"key": "suitcase", "value": True, "weight": CONSTRAINT_WEIGHTS["suitcase"]})
    if not with_attrs:
        return out
    words = re.findall(r"[a-z]+(?:-[a-z]+)?|/", text)
    wset = set(words)

    male, female = bool(wset & MALE_WORDS), bool(wset & FEMALE_WORDS)
    if male != female:
        out.append({"key": "gender", "value": "male" if male else "female", "weight": CONSTRAINT_WEIGHTS["gender"]})

    top_c, bottom_c, loose, used = [], [], [], set()
    for i, wd in enumerate(words):
        if wd in TOP_GARMENTS or wd in BOTTOM_GARMENTS or wd in OTHER_ITEMS:
            cols = _parse_colors(words, i)
            for j in range(max(0, i - 4), i):
                used.add(j)
            (top_c if wd in TOP_GARMENTS else bottom_c if wd in BOTTOM_GARMENTS else []).extend(cols)
    for i, wd in enumerate(words):
        if wd in COLOR_ALIASES and i not in used:
            loose.append(COLOR_ALIASES[wd])
    if top_c:
        out.append({"key": "top_color", "value": sorted(set(top_c)), "weight": CONSTRAINT_WEIGHTS["top_color"]})
    elif loose:
        out.append({"key": "top_color", "value": sorted(set(loose)), "weight": LOOSE_COLOR_WEIGHT})
    if bottom_c:
        out.append({"key": "bottom_color", "value": sorted(set(bottom_c)), "weight": CONSTRAINT_WEIGHTS["bottom_color"]})

    no_luggage = re.sub(SUITCASE_PATTERN, " ", text)    # "trolley bag" is luggage, not a handbag
    for key, (yes, no) in ATTR_PATTERNS.items():
        if yes and re.search(yes, no_luggage if key == "bag" else text):
            out.append({"key": key, "value": True, "weight": CONSTRAINT_WEIGHTS[key]})
        elif no and re.search(no, text):
            out.append({"key": key, "value": False, "weight": CONSTRAINT_WEIGHTS[key]})
    return out


def _sit_score(kp: Optional[np.ndarray]) -> Optional[float]:
    """
    P(sitting) from one crop's COCO keypoints [17, 3]. Seen from the side a sitting person's thighs are horizontal,
    seen from the front they are foreshortened; either way the knees sit close to hip height relative to the torso
    (standing or walking: about 0.7-1.0 torso lengths lower). None when shoulders, hips or knees are not visible.
    """
    if kp is None:
        return None
    groups = [[i for i in g if kp[i, 2] > 0.5] for g in ((5, 6), (11, 12), (13, 14))]
    if not all(groups):
        return None
    sy, hy, ky = (float(np.mean([kp[i, 1] for i in g])) for g in groups)
    torso = hy - sy
    if torso < 4:
        return None
    r = (ky - hy) / torso
    return 1.0 / (1.0 + math.exp((r - SIT_THIGH_RATIO) / 0.08))


def track_speeds(boxes: Dict[int, List[int]], fps: float, width: int = 0, height: int = 0) -> np.ndarray:
    """
    Speed of a track in box heights per second, one value every ACTION_STEP_S over windows of ACTION_WINDOW_S.
    Each contiguous run of frames is measured on its own (stitched tracks have gaps); windows whose box touches the
    frame border (when width/height are given) are skipped. Empty when there is too little to measure.
    """
    frames = sorted(boxes)
    if len(frames) < ACTION_MIN_TRACK_S * fps:
        return np.zeros(0)
    win = max(2, int(round(ACTION_WINDOW_S * fps)))
    step = max(1, int(round(ACTION_STEP_S * fps)))
    e = ACTION_EDGE_PX
    inside = (lambda b: not width or (b[0] > e and b[1] > e and b[2] < width - e and b[3] < height - e))
    runs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    runs.append(cur)
    out = []
    for run in runs:
        w = min(win, len(run) - 1)
        if w < max(2, win // 2):        # shorter than half a window: jitter would dominate
            continue
        for a in range(run[0], run[-1] - w + 1, step):
            b0, b1 = boxes[a], boxes[a + w]
            h0, h1 = b0[3] - b0[1], b1[3] - b1[1]
            if min(h0, h1) <= 0 or not (inside(b0) and inside(b1)):
                continue
            dx = ((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0
            dy = ((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0
            lateral = math.hypot(dx, dy) / ((h0 + h1) / 2.0)
            axial = abs(math.log(h1 / h0))
            out.append(math.hypot(lateral, axial) * fps / w)
    return np.array(out)


def track_direction(boxes: Dict[int, List[int]], fps: float, width: int, height: int) -> Optional[Tuple[float, float, float]]:
    """
    (lateral, log-height, feet) slopes per second of a track, lateral and feet in body heights: least squares per
    contiguous run of frames whose box keeps off the frame border, weighted by run duration. None when no run of at
    least ACTION_MIN_TRACK_S * 0.75 s is left.
    """
    frames = sorted(boxes)
    if len(frames) < ACTION_MIN_TRACK_S * fps:
        return None
    e = ACTION_EDGE_PX
    runs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    runs.append(cur)
    acc, total = np.zeros(3), 0.0
    for run in runs:
        ok = [f for f in run if boxes[f][0] > e and boxes[f][1] > e and boxes[f][2] < width - e and boxes[f][3] < height - e
              and boxes[f][3] > boxes[f][1]]
        if len(ok) < 0.75 * ACTION_MIN_TRACK_S * fps:
            continue
        t = (np.asarray(ok, float) - ok[0]) / fps
        dur = float(t[-1])
        if dur < 0.5:
            continue
        b = np.asarray([boxes[f] for f in ok], float)
        hs = b[:, 3] - b[:, 1]
        hm = float(np.median(hs))
        series = ((b[:, 0] + b[:, 2]) / 2 / hm, np.log(hs), b[:, 3] / hm)
        acc += dur * np.array([np.polyfit(t, v, 1)[0] for v in series])
        total += dur
    return None if total == 0 else tuple(float(v) for v in acc / total)


def direction_probs(slopes: Tuple[float, float, float]) -> Dict[str, float]:
    """act_left/act_right/act_toward/act_away from track_direction slopes (image coordinates, viewer's left/right)."""
    lateral, dlogh, feet = slopes
    sig = lambda z: float(1.0 / (1.0 + math.exp(-z)))
    approach = dlogh / DIR_APPROACH_UNITS[0] + feet / DIR_APPROACH_UNITS[1]
    return {"act_right": sig((lateral - DIR_LATERAL_MID) / DIR_LATERAL_SCALE),
            "act_left": sig((-lateral - DIR_LATERAL_MID) / DIR_LATERAL_SCALE),
            "act_toward": sig((approach - DIR_APPROACH_MID) / DIR_APPROACH_SCALE),
            "act_away": sig((-approach - DIR_APPROACH_MID) / DIR_APPROACH_SCALE)}


def action_probs(speeds: np.ndarray, sit_pose: Optional[float]) -> Dict[str, float]:
    """
    act_* probabilities of a track from the soft fraction of its windows in each speed band. Standing and running
    count fully once done for ACTION_MIN_FRACTION of the track; walking is the plain fraction, because scaling it up
    gave standing people p(walk) of 0.3-0.4 on the calibration videos. Standing is still and not sitting.
    """
    if not len(speeds):
        return {}
    p_still = 1.0 / (1.0 + np.exp((speeds - STAND_MAX_SPEED) / 0.015))
    p_run = 1.0 / (1.0 + np.exp((RUN_MIN_SPEED - speeds) / 0.2))
    p_walk = np.clip(1.0 - p_still - p_run, 0.0, 1.0)
    frac = lambda p: float(min(1.0, p.mean() / ACTION_MIN_FRACTION))
    out = {"act_run": frac(p_run), "act_walk": float(p_walk.mean()), "act_stand": frac(p_still)}
    if sit_pose is not None:
        out["act_sit"] = float(sit_pose)
        out["act_stand"] *= 1.0 - float(sit_pose)
    return out


def _color_match(dist: Optional[Dict[str, float]], wanted: List[str]) -> Optional[float]:
    """How well a colour distribution matches any wanted colour, 0..1 (1 = the dominant colour)."""
    if not dist:
        return None
    top = max(dist.values()) or 1.0
    best = 0.0
    for c in wanted:
        for k, v in dist.items():
            sim = 1.0 if k == c else COLOR_SIM.get((c, k), COLOR_SIM.get((k, c), 0.0))
            best = max(best, sim * v / top)
    return min(1.0, best)


def _object_fit(kind: str, o: List[float], p: List[int]) -> Optional[float]:
    """
    How well object box o belongs to person box p (higher is better), or None when it cannot be theirs. Bags must lie
    mostly inside the person's box below the head; a suitcase stands beside or in front of the legs with its bottom
    near the feet. The score is 1 - the horizontal gap in body heights, so the nearest person wins.
    """
    ox1, oy1, ox2, oy2 = o
    px1, py1, px2, py2 = p
    h = py2 - py1
    if h <= 0:
        return None
    gap = max(px1 - ox2, ox1 - px2, 0.0) / h
    if kind == "handbag":
        inter = max(0.0, min(ox2, px2) - max(ox1, px1)) * max(0.0, min(oy2, py2) - max(oy1, py1))
        inside = inter / max(1e-6, (ox2 - ox1) * (oy2 - oy1))
        return 1.0 - gap if inside >= 0.5 and py1 + 0.1 * h <= (oy1 + oy2) / 2 <= py2 else None
    if kind == "suitcase":
        ok = (gap <= 0.4 and (oy1 + oy2) / 2 >= py1 + 0.45 * h and py1 + 0.7 * h <= oy2 <= py2 + 0.2 * h)
        return 1.0 - gap if ok else None
    return None


def object_fractions(tracks: Dict[int, "Track"],
                     objects: Dict[int, List[Tuple[str, float, List[float]]]]) -> Dict[int, Dict[str, float]]:
    """Per track and object kind, the fraction of the track's frames in which such an object was given to it."""
    frames: Dict[Tuple[int, str], set] = {}
    for f, olist in objects.items():
        visible = [(tid, t.boxes[f]) for tid, t in tracks.items() if f in t.boxes]
        if not visible:
            continue
        for kind, _, box in olist:
            fits = [(score, tid) for tid, b in visible if (score := _object_fit(kind, box, b)) is not None]
            if fits:
                frames.setdefault((max(fits)[1], kind), set()).add(f)
    out: Dict[int, Dict[str, float]] = {}
    for (tid, kind), fs in frames.items():
        out.setdefault(tid, {})[kind] = len(fs) / max(1, len(tracks[tid].boxes))
    return out


def object_p(kind: str, frac: float) -> float:
    """Probability that a track has the object, from the fraction of its frames with it."""
    return float(1.0 / (1.0 + math.exp(-(frac - OBJECT_FRAC_MID[kind]) / OBJECT_FRAC_SCALE)))


def constraint_p(track: "Track", c: Dict[str, Any]) -> Optional[float]:
    """Probability that the track satisfies one query constraint (None when the track has no such information)."""
    key, val = c["key"], c["value"]
    if key == "gender":
        p = track.attrs.get("male")
        return None if p is None else (p if val == "male" else 1.0 - p)
    if key in ("top_color", "bottom_color"):
        return _color_match(track.attrs.get(key), val)
    p = track.attrs.get(key)
    if p is None and key == "shorts" and "longpants" in track.attrs:   # without the PA-100K model
        p = 1.0 - track.attrs["longpants"]
    return None if p is None else (p if val else 1.0 - p)


def par_attrs(p: np.ndarray, old: Dict[str, Any]) -> Dict[str, float]:
    """Track attributes from the mean PA-100K scores p [26]; old: the crossroad-0230 attributes (may be empty)."""
    g = lambda k: float(p[PAR_INDEX[k]])
    out = {"male": 1.0 - g("female"), "hat": g("hat"), "longpants": g("longpants"), "shorts": g("shorts"),
           "skirt": g("skirt"), "longcoat": g("longcoat"), "bag": max(g("handbag"), g("shoulderbag")),
           "backpack": g("backpack")}
    if "backpack" in old:          # the two models disagree on different people; their mean ranks best
        out["backpack"] = 0.5 * (out["backpack"] + float(old["backpack"]))
    return out


@dataclass
class Track:
    id: int
    boxes: Dict[int, List[int]]          # frame -> [x1, y1, x2, y2], incl. interpolated gaps
    det_frames: List[int]                # frames with a real detection
    quality: Dict[int, float] = field(default_factory=dict)  # det frame -> crop quality (conf, occlusion, size)
    attrs: Dict[str, Any] = field(default_factory=dict)      # male/backpack/... -> prob; top_color/bottom_color -> {colour: frac}
    reid: Optional[np.ndarray] = None    # L2-normalised mean re-ID embedding (tracking samples or preview crops)
    reid_n: int = 0                      # number of samples behind `reid` (weights merges)
    clip_emb: Optional[torch.Tensor] = None  # L2-normalised mean CLIP image embedding of preview crops
    clip: Optional[np.ndarray] = None    # [16, 224, 224, 3] RGB uint8, model input
    thumb: bytes = b""                   # JPEG, representative crop
    film: List[bytes] = field(default_factory=list)  # 4 JPEG crops across the track

    @property
    def start(self) -> int:
        return min(self.boxes)

    @property
    def end(self) -> int:
        return max(self.boxes)


@dataclass
class Analysis:
    id: str
    video_id: str
    params: Dict[str, Any]
    fps: float
    width: int
    height: int
    frames_processed: int
    frames_total: int
    tracks: Dict[int, Track]
    ckpt_key: str = ""
    emb: Dict[str, torch.Tensor] = field(default_factory=dict)  # V_app/V_mot/V_video [N, D], L2-normalised
    track_order: List[int] = field(default_factory=list)          # row order of emb
    track_emb: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)  # per-track V_* rows
    video_path: Optional[Path] = None    # proxy video, re-read when a checkpoint needs another clip layout
    layout: str = "uniform"              # frame layout of Track.clip ("uniform" or "mfgf")
    stitch_log: List[Dict[str, Any]] = field(default_factory=list)  # re-ID merges (for inspection)


class SearchEngine:
    def __init__(self, root: Path, use_clip: bool = True):
        self.root = root
        self.checkpoints_dir = root / "checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_lock = threading.RLock()
        self.decomposer = TextDecomposer(language="en")
        self.detector = None
        self.detector_backend = "opencv"
        self.checkpoint_info: Dict[str, Any] = {"loaded": False, "name": None}
        print(f"[*] SearchEngine on device: {self.device}")

        # The architecture follows the checkpoint (config stored inside it); without any checkpoint the
        # original SpaceTimeViT is built with ImageNet weights.
        self.model: Optional[DAMRModel] = None
        self.arch_key: Tuple = ()
        self.encoder_int8 = False
        self.frame_layout = "uniform"
        self._load_default_checkpoint()
        if self.model is None:
            self.model, self.arch_key = self._build_model({}, pretrained_vit=True), self._arch_key({})
        if getattr(self.model.text_encoder, "fallback", False):
            print("[!] WARNING: bert-base-uncased could not be loaded; the text encoder fell back to an "
                  "untrained model and checkpoints will not match. Check HF_HOME / network access.")
        self._init_detector()
        self._init_attr_model()
        self._init_clip(use_clip)

    # ------------------------------------------------------------------ checkpoints
    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        if not self.checkpoints_dir.exists():
            return []
        found = []
        for cp in sorted(self.checkpoints_dir.glob("**/*.pth")):
            if cp.is_file() and cp.stat().st_size > 1024 * 1024:
                found.append({
                    "name": str(cp.relative_to(self.checkpoints_dir)),
                    "size_mb": round(cp.stat().st_size / (1024 * 1024), 1),
                })
        return found

    def _load_default_checkpoint(self) -> None:
        def priority(c: Dict[str, Any]) -> int:
            name = Path(c["name"]).name.lower()
            if c["name"] == PREFERRED_CHECKPOINT:
                return -1
            if c["name"] == "best.pth":       # GitHub release v2.0 asset (= m3c_joint_seed42.pth)
                return 0
            if name == "best_weights.pth":    # release v1.0 (older, Duke-trained)
                return 1
            if "best" in name:
                return 2
            if "latest" in name:
                return 3
            return 4

        for c in sorted(self.scan_checkpoints(), key=priority):
            try:
                self.load_checkpoint(c["name"])
                return
            except Exception as e:
                print(f"[!] Skipping checkpoint {c['name']}: {e}")
        print("[!] No usable checkpoint in checkpoints/. Using ImageNet ViT + BERT initial weights.")

    @staticmethod
    def _arch_key(mcfg: Dict[str, Any]) -> Tuple:
        return (mcfg.get("backbone", "spacetime") or "spacetime", int(mcfg.get("embed_dim", 768)),
                int(mcfg.get("vit_depth", 6)), int(mcfg.get("vit_heads", 12)),
                bool(mcfg.get("use_temporal_attn", True)), bool(mcfg.get("s3d_motion", False)))

    def _build_model(self, mcfg: Dict[str, Any], pretrained_vit: bool = False) -> DAMRModel:
        """DAMRModel with the architecture of a checkpoint config (same fields as experiments/common.load_model)."""
        backbone, embed_dim, depth, heads, temporal, s3d = self._arch_key(mcfg)
        return DAMRModel(
            embed_dim=embed_dim,
            text_model_name=mcfg.get("text_model_name", "bert-base-uncased"),
            motion_frame_num=NUM_TRACK_FRAMES,
            vit_depth=depth,
            vit_heads=heads,
            drop_rate=0.0,
            pretrained_vit=pretrained_vit,     # backbone weights are fully overwritten by a checkpoint
            use_temporal_attn=temporal,
            s3d_motion=s3d,
            backbone=backbone,
        ).to(self.device).eval()

    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """
        Loads checkpoints/<name>, rebuilding the model when the checkpoint was trained with another architecture
        (backbone "spacetime" or "mfgf"). Rejects paths outside checkpoints/ and partial matches; on failure the
        current model stays active.
        """
        base = self.checkpoints_dir.resolve()
        cp = (base / name).resolve()
        if base not in cp.parents or not cp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {name}")

        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        cfg = (ckpt.get("config") or {}) if isinstance(ckpt, dict) else {}
        mcfg = dict(cfg.get("model") or {})
        if any(k.startswith("video_encoder.vit.") for k in state):   # weights-only files carry no config
            mcfg["backbone"] = "mfgf"
        arch = self._arch_key(mcfg)
        # a quantised encoder has no fp32 weights to load into, so speed mode always rebuilds
        reuse = self.model is not None and arch == self.arch_key and not self.encoder_int8
        model = self.model if reuse else self._build_model(mcfg)
        model_state = model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        if len(matched) < len(model_state):
            raise ValueError(f"Architecture mismatch: {len(matched)}/{len(model_state)} layers match")
        dsca_mode = mcfg.get("dsca_mode", "keyword") or "keyword"
        with self.model_lock:
            model.load_state_dict(matched)
            int8 = self._quantize_encoder(model)
            self.model, self.arch_key, self.encoder_int8 = model, arch, int8
            self.frame_layout = "mfgf" if arch[0] == "mfgf" else "uniform"
            if dsca_mode != self.decomposer.mode:      # sub-prompts must be built the way the checkpoint was trained
                self.decomposer = TextDecomposer(language="en", mode=dsca_mode)

        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        self.checkpoint_info = {
            "loaded": True,
            "name": str(cp.relative_to(base)),
            "backbone": arch[0],
            "dsca_mode": dsca_mode,
            "layers": len(matched),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            # joint runs train on all three subsets, but their config keeps the name of one of them
            "sub_dataset": "all" if "joint" in cp.name else (cfg.get("data") or {}).get("sub_dataset"),
            "metrics": {k: round(float(v), 2) for k, v in metrics.items() if isinstance(v, (int, float))},
            "encoder_int8": int8,
        }
        print(f"[+] Loaded checkpoint {self.checkpoint_info['name']} ({len(matched)} layers, "
              f"epoch {self.checkpoint_info['epoch']}{', int8 video encoder' if int8 else ''})")
        return self.checkpoint_info

    def _quantize_encoder(self, model: DAMRModel) -> bool:
        """Speed mode (ENCODER_INT8): dynamic int8 nn.Linear layers in the video encoder, in place. CPU only; the
        text encoder stays fp32. Returns whether the encoder is quantised."""
        if not ENCODER_INT8 or self.device.type != "cpu":
            return False
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # torch.ao eager quantisation prints a torchao migration notice
                torch.ao.quantization.quantize_dynamic(model.video_encoder, {torch.nn.Linear}, dtype=torch.qint8,
                                                       inplace=True)
            return True
        except Exception as e:
            print(f"[!] int8 video encoder unavailable ({e}): using fp32.")
            return False

    @property
    def ckpt_key(self) -> str:
        return str(self.checkpoint_info.get("name"))

    # ------------------------------------------------------------------ detection
    def _init_detector(self) -> None:
        try:
            from ultralytics import YOLO
            yolo_path = self.root / "yolov8n.pt"
            self.detector = YOLO(str(yolo_path) if yolo_path.exists() else "yolov8n.pt")
            self.detector_backend = "yolo"
            # Registered before the first track() call, so it runs ahead of ultralytics' tracker callback
            self._objects_sink: Optional[List[List[Tuple[str, float, List[float]]]]] = None
            self.detector.add_callback("on_predict_postprocess_end", self._split_objects)
            print("[+] Person detector: Ultralytics YOLOv8n")
            return
        except Exception as e:
            print(f"[*] YOLOv8 unavailable ({e}), trying torchvision...")
        try:
            import torchvision.models.detection as tv_det
            weights = tv_det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.detector = tv_det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(self.device).eval()
            self.detector_backend = "torchvision"
            print("[+] Person detector: torchvision Faster R-CNN MobileNetV3")
            return
        except Exception:
            pass
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.detector_backend = "opencv"
        print("[+] Person detector: OpenCV HOG")

    def detect(self, frame: np.ndarray, conf: float) -> List[List[int]]:
        """Person boxes [x1, y1, x2, y2] in a BGR frame."""
        h, w = frame.shape[:2]
        raw: List[Tuple[float, float, float, float]] = []
        if self.detector_backend == "yolo":
            res = self.detector(frame, classes=[0], conf=conf, verbose=False)
            if res and res[0].boxes is not None:
                raw = [tuple(b) for b in res[0].boxes.xyxy.cpu().numpy().tolist()]
        elif self.detector_backend == "torchvision":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            with torch.no_grad():
                p = self.detector([t])[0]
            for box, label, score in zip(p["boxes"].cpu().numpy(), p["labels"].cpu().numpy(), p["scores"].cpu().numpy()):
                if label == 1 and score >= conf:
                    raw.append(tuple(box))
        else:
            found, _ = self.detector.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            raw = [(x, y, x + bw, y + bh) for (x, y, bw, bh) in found]

        boxes = []
        for x1, y1, x2, y2 in raw:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if (x2 - x1) > 15 and (y2 - y1) > 30:
                boxes.append([x1, y1, x2, y2])
        return boxes

    # ------------------------------------------------------------------ attributes
    def _ov_model(self, name: str, folder: str, size_mb: int, dynamic_batch: bool = False):
        """Compiles an Open Model Zoo IR from checkpoints/<folder>/, downloading it once. Raises on failure."""
        import openvino as ov
        xml = self.checkpoints_dir / folder / f"{name}.xml"
        if not xml.exists() or not xml.with_suffix(".bin").exists():
            import urllib.request
            xml.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Downloading {name} (~{size_mb} MB)...")
            for ext in (".xml", ".bin"):
                tmp = xml.with_suffix(ext + ".part")
                urllib.request.urlretrieve(OMZ_URL.format(name) + ext, tmp)
                tmp.replace(xml.with_suffix(ext))
        core = ov.Core()
        model = core.read_model(str(xml))
        if dynamic_batch:
            shape = list(model.input(0).shape)
            model.reshape([-1, *shape[1:]])
        device = "GPU" if "GPU" in core.available_devices else "CPU"
        return core.compile_model(model, device), device

    def _init_attr_model(self) -> None:
        """OpenVINO pedestrian-attribute and re-ID models, pose model for garment regions. All optional."""
        self.attr_model = self.reid_model = self.pose_model = self.par_model = None
        try:
            import openvino  # noqa: F401
        except ImportError:
            print("[!] openvino not installed: attribute recognition and re-ID disabled.")
            return
        try:
            self.reid_model, device = self._ov_model(REID_MODEL, "reid", 8, dynamic_batch=True)
            print(f"[+] Re-ID model: {REID_MODEL} on {device}")
        except Exception as e:
            print(f"[!] Re-ID model unavailable ({e}): tracklet stitching falls back to V_app similarity.")
        try:
            self.attr_model, device = self._ov_model(ATTR_MODEL, "attributes", 3)
            self.attr_output = self.attr_model.output("453")
            self.attr_points = (self.attr_model.output("456"), self.attr_model.output("459"))
            print(f"[+] Attribute model: {ATTR_MODEL} on {device}")
        except Exception as e:
            print(f"[!] Attribute model unavailable ({e}): gender/attribute recognition disabled.")
            self.attr_model = None
            return
        self._init_par_model()
        try:
            from ultralytics import YOLO
            self.pose_model = YOLO(str(self.checkpoints_dir / POSE_MODEL))
            print(f"[+] Pose model: {POSE_MODEL} (garment colour regions)")
        except Exception as e:
            print(f"[!] Pose model unavailable ({e}): colours fall back to the attribute model's colour points.")

    def _init_par_model(self) -> None:
        """PA-100K attribute model (PAR_URL), downloaded once to checkpoints/attributes/PAR_DIR. Optional."""
        try:
            import openvino as ov
            model_file = self.checkpoints_dir / "attributes" / PAR_DIR / "inference.pdmodel"
            if not model_file.exists():
                import tarfile
                import urllib.request
                print(f"[*] Downloading the PA-100K attribute model (~7 MB)...")
                model_file.parent.parent.mkdir(parents=True, exist_ok=True)
                tmp = model_file.parent.parent / "person_attribute_infer.tar.part"
                urllib.request.urlretrieve(PAR_URL, tmp)
                with tarfile.open(tmp) as tar:
                    tar.extractall(model_file.parent.parent, filter="data")
                tmp.unlink()
            core = ov.Core()
            device = "GPU" if "GPU" in core.available_devices else "CPU"
            self.par_model = core.compile_model(core.read_model(str(model_file)), device)
            print(f"[+] PA-100K attribute model: PULC person_attribute on {device}")
        except Exception as e:
            print(f"[!] PA-100K attribute model unavailable ({e}): attributes come from {ATTR_MODEL} only.")
            self.par_model = None

    def _par_probs(self, crops: List[np.ndarray]) -> np.ndarray:
        """[N, 26] PA-100K scores of BGR crops."""
        x = np.stack([((cv2.resize(c, (192, 256))[:, :, ::-1].astype(np.float32) / 255.0 - PAR_MEAN) / PAR_STD)
                      .transpose(2, 0, 1) for c in crops])
        with self.model_lock:
            return self.par_model([x])[self.par_model.output(0)]

    def _embed_reid(self, crops: List[np.ndarray]) -> np.ndarray:
        """[N, 256] L2-normalised re-ID embeddings of BGR crops."""
        x = np.stack([cv2.resize(c, (128, 256)).transpose(2, 0, 1) for c in crops]).astype(np.float32)
        with self.model_lock:
            v = self.reid_model([x])[self.reid_model.output(0)]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)

    def _reid(self, analysis: Analysis, tids: List[int]) -> None:
        """Fallback re-ID embedding from the preview crops, for tracks without samples taken during tracking."""
        if self.reid_model is None:
            return
        for tid in tids:
            t = analysis.tracks[tid]
            if t.reid is not None:
                continue
            crops = [img for j in t.film + ([t.thumb] if t.thumb else [])
                     if (img := cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)) is not None]
            if not crops:
                continue
            m = self._embed_reid(crops).mean(axis=0)
            t.reid, t.reid_n = m / max(float(np.linalg.norm(m)), 1e-6), len(crops)

    def _init_clip(self, enabled: bool) -> None:
        """Optional open-vocabulary scorer (fourth fused score). Downloaded once to the HF cache (~600 MB)."""
        self.clip_model = self.clip_processor = None
        if not enabled:
            print("[*] CLIP disabled (--no_clip).")
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            print(f"[+] CLIP: {CLIP_MODEL}")
        except Exception as e:
            print(f"[!] CLIP unavailable ({e}): ranking uses the DSCA scores only.")
            self.clip_model = None

    @property
    def default_weights(self) -> Dict[str, float]:
        return dict(DEFAULT_WEIGHTS)

    @staticmethod
    def _letterbox(img_bgr: np.ndarray) -> np.ndarray:
        """Pads a (tall) person crop to a square RGB image so CLIP's centre crop keeps head and feet."""
        h, w = img_bgr.shape[:2]
        side = max(h, w)
        out = np.full((side, side, 3), 127, np.uint8)
        y, x = (side - h) // 2, (side - w) // 2
        out[y:y + h, x:x + w] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out

    def _clip_embed(self, analysis: Analysis, tids: List[int], progress: Optional[ProgressCb] = None) -> None:
        """Mean CLIP image embedding over up to CLIP_CROPS preview crops of each track."""
        if self.clip_model is None or not tids:
            return
        items = []  # (tid, letterboxed RGB crop)
        for tid in tids:
            t = analysis.tracks[tid]
            jpegs = ([t.thumb] if t.thumb else []) + ([t.film[0], t.film[-1]] if len(t.film) >= 2 else t.film)
            for j in jpegs[:CLIP_CROPS]:
                img = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    items.append((tid, self._letterbox(img)))
        feats: Dict[int, List[torch.Tensor]] = {}
        for b in range(0, len(items), CLIP_BATCH):
            chunk = items[b:b + CLIP_BATCH]
            x = self.clip_processor(images=[im for _, im in chunk], return_tensors="pt")["pixel_values"].to(self.device)
            with self.model_lock, torch.inference_mode():
                f = F.normalize(self.clip_model.get_image_features(pixel_values=x).float().cpu(), dim=-1)
            for (tid, _), v in zip(chunk, f):
                feats.setdefault(tid, []).append(v)
            if progress:
                progress("clip", b + len(chunk), len(items))
        for tid, vs in feats.items():
            analysis.tracks[tid].clip_emb = F.normalize(torch.stack(vs).mean(0), dim=-1)

    def _pose_keypoints(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """COCO keypoints [17, 3] (crop coordinates) of the person filling each crop, or None."""
        if getattr(self, "pose_model", None) is None or not crops:
            return [None] * len(crops)
        padded, pads = [], []
        for c in crops:
            pad = int(0.3 * c.shape[0])   # context so a tightly cropped person is still detected
            padded.append(cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114)))
            pads.append(pad)
        with self.model_lock:
            results = self.pose_model(padded, imgsz=320, verbose=False)
        out = []
        for c, pad, r in zip(crops, pads, results):
            if r.keypoints is None or not len(r.boxes):
                out.append(None)
                continue
            h, w = c.shape[:2]
            target = [pad, pad, pad + w, pad + h]
            ious = [_iou([int(v) for v in b], target) for b in r.boxes.xyxy.cpu().numpy()]
            k = r.keypoints.data[int(np.argmax(ious))].cpu().numpy().copy()
            k[:, :2] -= pad
            out.append(k)
        return out

    def _attributes(self, analysis: Analysis, tids: List[int]) -> None:
        """
        Per track, over its preview crops (least-occluded frames): averages the 8 attribute probabilities and
        builds top/bottom garment colour distributions from pose-located regions.
        """
        if self.attr_model is None and self.par_model is None:
            return
        for tid in tids:
            t = analysis.tracks[tid]
            crops = [img for j in t.film + ([t.thumb] if t.thumb else [])
                     if (img := cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)) is not None]
            if not crops:
                continue
            probs, hists, sits = [], {"top_color": [], "bottom_color": []}, []
            for img, kp in zip(crops, self._pose_keypoints(crops)):
                if (sp := _sit_score(kp)) is not None:
                    sits.append(sp)
                if self.attr_model is None:      # colours need its colour points when there is no pose
                    continue
                x = cv2.resize(img, (80, 160)).transpose(2, 0, 1)[None].astype(np.float32)  # BGR, NCHW, 0..255
                res = self.attr_model([x])
                probs.append(res[self.attr_output].reshape(-1))
                h, w = img.shape[:2]
                regions = _garment_regions(kp, w, h, res[self.attr_points[0]].reshape(-1), res[self.attr_points[1]].reshape(-1))
                for key, r in zip(("top_color", "bottom_color"), regions):
                    hist = _color_hist(img[r[1]:r[3], r[0]:r[2]]) if r else None
                    if hist is not None:
                        hists[key].append(hist)
            t.attrs = {k: float(v) for k, v in zip(ATTR_NAMES, np.mean(probs, axis=0))} if probs else {}
            if self.par_model is not None:
                t.attrs.update(par_attrs(self._par_probs(crops).mean(axis=0), t.attrs))
            for key, hs in hists.items():
                if hs:
                    t.attrs[key] = {c: round(float(v), 4) for c, v in zip(COLOR_NAMES, np.mean(hs, axis=0))}
            if len(sits) >= 2:   # one stray pose is not enough
                t.attrs["sit_pose"] = float(np.mean(sits))

    @staticmethod
    def _actions(analysis: Analysis) -> None:
        """act_run/act_walk/act_stand(/act_sit) and the median speed per track, from its final boxes."""
        for t in analysis.tracks.values():
            speeds = track_speeds(t.boxes, analysis.fps, analysis.width, analysis.height)
            if len(speeds):
                t.attrs["speed"] = float(np.median(speeds))
                t.attrs.update(action_probs(speeds, t.attrs.get("sit_pose")))
            if (slopes := track_direction(t.boxes, analysis.fps, analysis.width, analysis.height)) is not None:
                t.attrs["dir_slopes"] = [round(v, 4) for v in slopes]
                t.attrs.update(direction_probs(slopes))

    @staticmethod
    def _objects(analysis: Analysis, objects: Dict[int, List[Tuple[str, float, List[float]]]]) -> None:
        """
        obj_<kind> fractions and the suitcase probability per track; the bag probability becomes the mean of the
        attribute model and the detector's handbags (each missed bags the other saw).
        """
        fractions = object_fractions(analysis.tracks, objects)
        for tid, t in analysis.tracks.items():
            fr = fractions.get(tid, {})
            for kind in OBJECT_CLASSES.values():
                t.attrs[f"obj_{kind}"] = round(fr.get(kind, 0.0), 4)
            t.attrs["suitcase"] = object_p("suitcase", fr.get("suitcase", 0.0))
            handbag = object_p("handbag", fr.get("handbag", 0.0))
            if "bag" in t.attrs:
                t.attrs["bag_attr"] = t.attrs["bag"]
                t.attrs["bag"] = 0.5 * (t.attrs["bag"] + handbag)

    # ------------------------------------------------------------------ analysis
    def analyze(self, video_path: Path, video_id: str, analysis_id: str, params: Dict[str, Any],
                progress: ProgressCb) -> Analysis:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = int(params["max_frames"]) or total
        n_frames = min(total, max_frames) if total else max_frames
        stride = max(1, int(params.get("stride", 1)))
        conf = float(params.get("conf", 0.35))

        # Pass 1: detect + track. Only boxes and per-detection quality are kept, never frames.
        objects = None   # OBJECT_CLASSES boxes per frame (YOLO only)
        if self.detector_backend == "yolo":
            raw, frames_processed, objects = self._track_botsort(cap, n_frames, stride, conf, fps, progress)
        else:
            raw, frames_processed = self._track_greedy(cap, n_frames, stride, conf, progress)
        cap.release()
        raw = self._split_by_appearance(raw)

        max_gap = int(MAX_INTERP_GAP_S * fps)
        kept = []
        for t in raw.values():
            dets = sorted(t["boxes"])
            heights = [t["boxes"][f][3] - t["boxes"][f][1] for f in dets]
            if len(dets) < MIN_TRACK_DETECTIONS or float(np.median(heights)) < MIN_TRACK_HEIGHT:
                continue
            boxes = dict(t["boxes"])
            for f0, f1 in zip(dets, dets[1:]):
                if 1 < f1 - f0 <= max_gap:
                    b0, b1 = boxes[f0], boxes[f1]
                    for gf in range(f0 + 1, f1):
                        a = (gf - f0) / float(f1 - f0)
                        boxes[gf] = [int(b0[k] + a * (b1[k] - b0[k])) for k in range(4)]
            track = Track(id=0, boxes=boxes, det_frames=dets, quality=t["q"])
            if t.get("emb"):
                m = np.mean([v for _, v in t["emb"]], axis=0)
                track.reid, track.reid_n = m / max(float(np.linalg.norm(m)), 1e-6), len(t["emb"])
            kept.append(track)
        kept.sort(key=lambda t: t.start)
        tracks = {}
        for i, t in enumerate(kept, start=1):
            t.id = i
            tracks[i] = t

        analysis = Analysis(
            id=analysis_id, video_id=video_id, params=params, fps=float(fps), width=width, height=height,
            frames_processed=frames_processed, frames_total=total or frames_processed, tracks=tracks,
            video_path=Path(video_path), layout=self.frame_layout,
        )
        if tracks:
            self._extract_crops(video_path, analysis, list(tracks), progress)
            self._attributes(analysis, list(tracks))
            self._reid(analysis, list(tracks))
            if self.reid_model is not None:
                # Stitch first (it needs only re-ID and gender), then run CLIP and the video model once per final
                # track. Encoding before stitching repeated about half of that work (station clip: 280 encodes
                # for 149 final tracks).
                self._stitch(video_path, analysis, progress, encode=False)
                if params.get("clip", True):
                    self._clip_embed(analysis, list(analysis.tracks), progress)
                self.encode_tracks(analysis, progress=progress)
            else:   # without the re-ID model, stitching compares V_app, so the tracks must be encoded first
                if params.get("clip", True):
                    self._clip_embed(analysis, list(tracks), progress)
                self.encode_tracks(analysis, progress=progress)
                self._stitch(video_path, analysis, progress)
            self._actions(analysis)   # after stitching: merged tracks have their final boxes
            if objects is not None:
                self._objects(analysis, objects)
        return analysis

    @staticmethod
    def _split_by_appearance(raw: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Fixes identity switches inside a track: when the tracker's box jumps to another person (typically
        while they cross and occlude each other), the re-ID samples before and after the switch disagree.
        Recursively cuts each track at the sample boundary with the lowest cosine between the mean embeddings
        of the two sides, while that cosine is below SPLIT_MAX_SIM. The pieces are re-joined by _stitch.
        """
        def norm_mean(vs):
            m = np.mean(vs, axis=0)
            return m / max(float(np.linalg.norm(m)), 1e-6)

        def cuts(samples):
            n = len(samples)
            if n < 2 * SPLIT_MIN_SAMPLES:
                return []
            vs = np.stack([v for _, v in samples])
            best_k, best_sim = None, SPLIT_MAX_SIM
            for k in range(SPLIT_MIN_SAMPLES, n - SPLIT_MIN_SAMPLES + 1):
                sim = float(norm_mean(vs[:k]) @ norm_mean(vs[k:]))
                if sim < best_sim:
                    best_k, best_sim = k, sim
            if best_k is None:
                return []
            return cuts(samples[:best_k]) + [samples[best_k][0]] + cuts(samples[best_k:])

        out: Dict[int, Dict[str, Any]] = {}
        next_id = max(raw, default=0) + 1
        for tid, t in raw.items():
            starts = cuts(t.get("emb", []))
            if not starts:
                out[tid] = t
                continue
            bounds = [-1] + starts + [10 ** 9]
            for lo, hi in zip(bounds, bounds[1:]):
                part = {"boxes": {f: b for f, b in t["boxes"].items() if lo <= f < hi},
                        "q": {f: q for f, q in t["q"].items() if lo <= f < hi},
                        "emb": [(f, v) for f, v in t["emb"] if lo <= f < hi]}
                if part["boxes"]:
                    out[next_id] = part
                    next_id += 1
        return out

    def _split_objects(self, predictor) -> None:
        """
        Detector callback that runs before ultralytics' tracker callback: moves the frame's OBJECT_CLASSES boxes into
        self._objects_sink, so BoT-SORT only sees people. The person boxes are the same as in a person-only run,
        because NMS gives each box its best class before the class filter and runs per class.
        """
        sink = self._objects_sink
        if sink is None:
            return
        for r in predictor.results:
            b = r.boxes
            if b is None or not len(b):
                sink.append([])
                continue
            person = b.cls == 0
            other = ~person
            sink.append([(OBJECT_CLASSES[int(c)], float(sc), [float(v) for v in xy])
                         for xy, c, sc in zip(b.xyxy[other].tolist(), b.cls[other].tolist(), b.conf[other].tolist())
                         if sc >= OBJECT_MIN_CONF and int(c) in OBJECT_CLASSES])
            r.boxes = b[person]
            feats = getattr(r, "feats", None)      # per-detection ReID features, aligned with the boxes
            if isinstance(feats, torch.Tensor) and len(feats) == len(person):
                r.feats = feats[person]

    def _track_botsort(self, cap, n_frames: int, stride: int, conf: float, fps: float, progress: ProgressCb
                       ) -> Tuple[Dict[int, Dict[str, Any]], int, Dict[int, List[Tuple[str, float, List[float]]]]]:
        """
        YOLO + BoT-SORT: Kalman motion, two-stage (high/low score) Hungarian matching, native ReID. Also returns the
        OBJECT_CLASSES boxes per processed frame, from the same detector pass.
        """
        cfg = (
            "tracker_type: botsort\n"
            f"track_high_thresh: {conf}\n"
            f"track_low_thresh: {DETECT_CONF_LOW}\n"
            f"new_track_thresh: {conf}\n"
            # BYTETracker scales track_buffer by frame_rate / 30; calls happen every `stride` frames
            f"track_buffer: {max(10, int(TRACK_BUFFER_S * 30 / stride))}\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n"
            "gmc_method: none\n"           # surveillance cameras are static
            "proximity_thresh: 0.5\n"
            "appearance_thresh: 0.8\n"
            "with_reid: True\n"
            "model: auto\n"                # ReID features straight from the detector, no extra model
        )
        fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="botsort_")
        with os.fdopen(fd, "w") as f:
            f.write(cfg)

        tracks: Dict[int, Dict[str, Any]] = {}
        objects: Dict[int, List[Tuple[str, float, List[float]]]] = {}
        idx, first = 0, True
        sample_every = max(stride, int(round(REID_SAMPLE_S * fps)))
        self._objects_sink = sink = []
        try:
            while idx < n_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    res = self.detector.track(frame, persist=not first, tracker=cfg_path, conf=DETECT_CONF_LOW,
                                              imgsz=DETECT_IMGSZ if stride == 1 else DETECT_IMGSZ_FAST,
                                              classes=[0, *OBJECT_CLASSES], verbose=False)[0]
                    first = False
                    if sink:
                        objects[idx] = sink.pop()
                        sink.clear()
                    b = res.boxes
                    if b is not None and b.id is not None and len(b):
                        xyxy = b.xyxy.cpu().numpy()
                        ids = b.id.int().cpu().tolist()
                        scores = b.conf.cpu().numpy()
                        occl = _max_overlap(xyxy)
                        h_img, w_img = frame.shape[:2]
                        to_embed = []
                        for (x1, y1, x2, y2), tid, s, oc in zip(xyxy, ids, scores, occl):
                            box = [max(0, int(x1)), max(0, int(y1)), min(w_img, int(x2)), min(h_img, int(y2))]
                            if box[2] - box[0] < 8 or box[3] - box[1] < 16:
                                continue
                            t = tracks.setdefault(tid, {"boxes": {}, "q": {}, "emb": []})
                            t["boxes"][idx] = box
                            t["q"][idx] = float(s) * (1.0 - float(oc)) * min(1.0, (box[3] - box[1]) / 120.0)
                            if (self.reid_model is not None and oc <= REID_SAMPLE_MAX_OCCL
                                    and box[3] - box[1] >= MIN_TRACK_HEIGHT
                                    and (not t["emb"] or idx - t["emb"][-1][0] >= sample_every)):
                                to_embed.append((tid, frame[box[1]:box[3], box[0]:box[2]]))
                        if to_embed:
                            for (tid, _), v in zip(to_embed, self._embed_reid([c for _, c in to_embed])):
                                tracks[tid]["emb"].append((idx, v))
                idx += 1
                if idx % 5 == 0 or idx == n_frames:
                    progress("detect", idx, n_frames)
        finally:
            self._objects_sink = None
            os.unlink(cfg_path)
        return tracks, idx, objects

    def _track_greedy(self, cap, n_frames: int, stride: int, conf: float,
                      progress: ProgressCb) -> Tuple[Dict[int, Dict[str, Any]], int]:
        """Fallback IoU/centre tracker for the torchvision and HOG detectors (no track ids of their own)."""
        tracks: Dict[int, Dict[str, Any]] = {}
        next_tid, idx = 1, 0
        while idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.detect(frame, conf) if idx % stride == 0 else []
            matched = set()
            for t in tracks.values():
                if idx - t["last"] > MAX_TRACK_GAP:
                    continue
                last = t["boxes"][t["last"]]
                best, best_d = 0.0, -1
                for d_i, det in enumerate(detections):
                    if d_i in matched:
                        continue
                    iou = _iou(last, det)
                    if iou > 0.3 and iou > best:
                        best, best_d = iou, d_i
                if best_d >= 0:
                    matched.add(best_d)
                    t["boxes"][idx] = detections[best_d]
                    t["q"][idx] = 1.0
                    t["last"] = idx
            for d_i, det in enumerate(detections):
                if d_i not in matched:
                    tracks[next_tid] = {"boxes": {idx: det}, "q": {idx: 1.0}, "last": idx}
                    next_tid += 1
            idx += 1
            if idx % 5 == 0 or idx == n_frames:
                progress("detect", idx, n_frames)
        return tracks, idx

    @staticmethod
    def _pick_spread(t: Track, n: int) -> List[int]:
        """n frames spread over the track; in each time bin take the best-quality (least occluded) detection."""
        dets = t.det_frames
        if len(dets) <= n:
            return [dets[i] for i in np.linspace(0, len(dets) - 1, n).round().astype(int)]
        bins = np.array_split(np.array(dets), n)
        return [int(max(b, key=lambda f: t.quality.get(int(f), 0.0))) for b in bins]

    def _pick_clip_frames(self, t: Track, fps: float) -> List[int]:
        """
        Frames of the model input clip. "uniform": 16 frames spread over the track. "mfgf": 4 spread frames, then
        16 consecutive frames at about the tracklet frame rate of the training data, taken from the stretch of
        the track with the best mean crop quality (boxes interpolated across short gaps count as present).
        """
        if self.frame_layout != "mfgf":
            return self._pick_spread(t, NUM_TRACK_FRAMES)
        visual = self._pick_spread(t, MFGF_VISUAL_FRAMES)
        step = max(1, int(round(fps / MFGF_MOTION_FPS)))
        dets = np.array(t.det_frames)
        q = np.array([t.quality.get(int(f), 0.0) for f in dets])
        csum = np.concatenate([[0.0], np.cumsum(q)])
        span = step * (MFGF_MOTION_FRAMES - 1)
        best, best_q = None, -1.0
        for i, f0 in enumerate(dets):
            if f0 + span > dets[-1]:
                break
            j = int(np.searchsorted(dets, f0 + span, side="right"))
            mean_q = (csum[j] - csum[i]) / (j - i)
            if mean_q > best_q and all(int(f0 + step * k) in t.boxes for k in range(MFGF_MOTION_FRAMES)):
                best, best_q = int(f0), mean_q
        if best is None:        # track shorter than the window (or broken by long gaps): spread over what exists
            frames = sorted(t.boxes)
            motion = [frames[i] for i in np.linspace(0, len(frames) - 1, MFGF_MOTION_FRAMES).round().astype(int)]
        else:
            motion = [best + step * k for k in range(MFGF_MOTION_FRAMES)]
        return visual + motion

    def _extract_crops(self, video_path: Path, analysis: Analysis, tids: List[int], progress: ProgressCb,
                       previews: bool = True) -> None:
        """Pass 2: re-reads only the frames needed for the given tracks' model clips and preview crops."""
        wanted: Dict[int, List[Tuple[int, int, bool]]] = {}  # frame -> [(track_id, slot, is_preview)]
        for tid in tids:
            t = analysis.tracks[tid]
            picks = self._pick_clip_frames(t, analysis.fps)
            t.clip = np.zeros((len(picks), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            for slot, f in enumerate(picks):
                wanted.setdefault(f, []).append((tid, slot, False))
            if previews:
                spread = self._pick_spread(t, NUM_TRACK_FRAMES)
                for slot in PREVIEW_SLOTS:
                    wanted.setdefault(spread[slot], []).append((tid, slot, True))
        thumb_slot = NUM_TRACK_FRAMES // 2
        film: Dict[int, Dict[int, bytes]] = {tid: {} for tid in tids}

        cap = cv2.VideoCapture(str(video_path))
        first_needed, last_needed = min(wanted), max(wanted)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_needed)
        idx = first_needed
        while idx <= last_needed:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, slot, is_preview in wanted.get(idx, []):
                t = analysis.tracks[tid]
                x1, y1, x2, y2 = t.boxes[idx]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if not is_preview:
                    t.clip[slot] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (CROP_SIZE, CROP_SIZE),
                                              interpolation=cv2.INTER_LINEAR)
                elif slot == thumb_slot:
                    t.thumb = _jpeg(crop, 256)
                else:
                    film[tid][slot] = _jpeg(crop, 256)
            idx += 1
            if idx % 10 == 0:
                progress("crops", idx - first_needed, last_needed - first_needed + 1)
        cap.release()
        if previews:
            for tid in tids:
                analysis.tracks[tid].film = [film[tid][s] for s in sorted(film[tid])]
        progress("crops", 1, 1)

    def encode_tracks(self, analysis: Analysis, tids: Optional[List[int]] = None,
                      progress: Optional[ProgressCb] = None, stage: str = "encode") -> None:
        """Encodes track clips with SpaceTimeViT (all tracks by default) and rebuilds the search matrices."""
        if tids is None:
            tids = sorted(analysis.tracks)
            analysis.track_emb = {}
            if tids and analysis.layout != self.frame_layout:   # the active checkpoint reads another clip layout
                if analysis.video_path is None or not analysis.video_path.exists():
                    raise RuntimeError("Video file is gone; analyse the video again with this checkpoint")
                self._extract_crops(analysis.video_path, analysis, tids, lambda *a: None, previews=False)
                analysis.layout = self.frame_layout
        for b in range(0, len(tids), ENCODE_BATCH):
            ids = tids[b:b + ENCODE_BATCH]
            clips = torch.from_numpy(np.stack([analysis.tracks[i].clip for i in ids]))  # [B,T,H,W,3]
            x = clips.permute(0, 1, 4, 2, 3).float().div(255.0)                          # [B,T,3,H,W]
            x = ((x.flatten(0, 1) - _MEAN) / _STD).view_as(x).to(self.device)
            with self.model_lock, torch.no_grad():
                v_app, v_mot, v_video = self.model.encode_video(x)
            for j, tid in enumerate(ids):
                analysis.track_emb[tid] = {
                    "V_app": F.normalize(v_app[j].float().cpu(), dim=-1),
                    "V_mot": F.normalize(v_mot[j].float().cpu(), dim=-1),
                    "V_video": F.normalize(v_video[j].float().cpu(), dim=-1),
                }
            if progress:
                progress(stage, b + len(ids), len(tids))
        self._rebuild_matrices(analysis)
        analysis.ckpt_key = self.ckpt_key

    @staticmethod
    def _rebuild_matrices(analysis: Analysis) -> None:
        order = sorted(analysis.tracks)
        analysis.track_order = order
        analysis.emb = {k: torch.stack([analysis.track_emb[t][k] for t in order]) for k in ("V_app", "V_mot", "V_video")}
        if order and all(analysis.tracks[t].clip_emb is not None for t in order):
            analysis.emb["CLIP"] = torch.stack([analysis.tracks[t].clip_emb for t in order])

    def _stitch(self, video_path: Path, analysis: Analysis, progress: ProgressCb, encode: bool = True) -> None:
        """
        Re-ID across gaps (occlusions): joins tracklet B onto A when B starts at most STITCH_MAX_GAP_S after A
        ends, near where A was last seen, the attribute model does not see a different gender, and the OpenVINO
        re-ID embeddings agree (V_app similarity is the fallback when the re-ID model is unavailable). Merged tracks
        are re-cropped (and, with encode, CLIP/video-encoded), then ids are renumbered by start time. With
        encode=False the caller encodes every track afterwards.
        """
        fps = analysis.fps
        tr = analysis.tracks
        center = lambda b: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        cands = []
        for a in tr.values():
            a_box = a.boxes[a.det_frames[-1]]
            for b in tr.values():
                gap = b.det_frames[0] - a.det_frames[-1]
                if gap <= 0 or gap > STITCH_MAX_GAP_S * fps:
                    continue
                b_box = b.boxes[b.det_frames[0]]
                h = max(a_box[3] - a_box[1], b_box[3] - b_box[1])
                (ax, ay), (bx, by) = center(a_box), center(b_box)
                if math.hypot(ax - bx, ay - by) > h * (STITCH_SPEED * gap / fps + 0.5):
                    continue
                if "male" in a.attrs and "male" in b.attrs and abs(a.attrs["male"] - b.attrs["male"]) > STITCH_GENDER_GAP:
                    continue
                if a.reid is not None and b.reid is not None:
                    sim = float(a.reid @ b.reid)
                    ok = sim >= REID_MIN_SIM
                else:
                    ea, eb = analysis.track_emb.get(a.id), analysis.track_emb.get(b.id)
                    if ea is None or eb is None:   # not encoded yet (stitching before encoding)
                        continue
                    sim = float(ea["V_app"] @ eb["V_app"])
                    ok = sim >= STITCH_MIN_SIM
                if ok:
                    cands.append((sim, a.id, b.id, gap))

        succ: Dict[int, int] = {}
        pred: Dict[int, int] = {}
        analysis.stitch_log = []
        for sim, a, b, gap in sorted(cands, reverse=True):
            if a in succ or b in pred:
                continue
            succ[a], pred[b] = b, a
            analysis.stitch_log.append({"from": a, "to": b, "sim": round(sim, 4), "gap_s": round(gap / fps, 2)})
        if not succ:
            progress("stitch", 1, 1)
            return

        merged = []
        head_of: Dict[int, int] = {}
        for head in [t for t in tr if t not in pred and t in succ]:
            cur = head
            head_of[head] = head
            while cur in succ:
                head_of[succ[cur]] = head
                nxt = tr.pop(succ[cur])
                h = tr[head]
                h.boxes.update(nxt.boxes)
                h.quality.update(nxt.quality)
                h.det_frames = sorted(set(h.det_frames) | set(nxt.det_frames))
                if h.reid is not None and nxt.reid is not None:
                    m = h.reid * h.reid_n + nxt.reid * nxt.reid_n
                    h.reid, h.reid_n = m / max(float(np.linalg.norm(m)), 1e-6), h.reid_n + nxt.reid_n
                analysis.track_emb.pop(nxt.id, None)
                cur = nxt.id
            merged.append(head)
        progress("stitch", 0, 1)
        self._extract_crops(video_path, analysis, merged, lambda *a: None)
        self._attributes(analysis, merged)
        self._reid(analysis, merged)
        if encode:
            if analysis.params.get("clip", True):
                self._clip_embed(analysis, merged)
            self.encode_tracks(analysis, tids=merged, progress=progress, stage="stitch")

        # Renumber 1..N by start time so ids read naturally in the UI
        order = sorted(tr.values(), key=lambda t: t.start)
        remap = {t.id: i for i, t in enumerate(order, start=1)}
        analysis.tracks = {remap[t.id]: t for t in order}
        analysis.track_emb = {remap[k]: v for k, v in analysis.track_emb.items()}
        for t in order:
            t.id = remap[t.id]
        for s in analysis.stitch_log:
            s["into"] = remap[head_of[s["from"]]]
        if encode:
            self._rebuild_matrices(analysis)
        else:
            progress("stitch", 1, 1)

    # ------------------------------------------------------------------ search
    def search(self, analysis: Analysis, prompt: str) -> Dict[str, Any]:
        """Three cosine scores per track; ranking/fusion happens in the client."""
        prompt = prompt.strip()
        # Vietnamese -> the English every model and word list below expects (vi_query); English is left unchanged
        text, lang, unknown = vi_query.translate(prompt)
        concepts = self.decomposer.extract_concepts(text)       # keywords shown in the UI
        concepts["q_app"], concepts["q_mot"] = self.decomposer.decompose(text)   # follows the checkpoint's dsca_mode
        if analysis.tracks and analysis.ckpt_key != self.ckpt_key:
            self.encode_tracks(analysis)

        result = {
            "prompt": prompt,
            "prompt_en": text,
            "prompt_lang": lang,
            "unknown_words": unknown,
            "q_app": concepts["q_app"],
            "q_mot": concepts["q_mot"],
            "app_keywords": concepts["app_keywords"],
            "mot_keywords": concepts["mot_keywords"],
            "constraints": query_attributes(text, with_attrs=self.attr_model is not None or self.par_model is not None),
            "scores": [],
        }
        if not analysis.tracks:
            return result

        with self.model_lock, torch.no_grad():
            t = self.model.encode_text([text, concepts["q_app"], concepts["q_mot"]], device=self.device)
        t = F.normalize(t.float().cpu(), dim=-1)
        sim_joint = (analysis.emb["V_video"] @ t[0]).tolist()
        sim_app = (analysis.emb["V_app"] @ t[1]).tolist()
        sim_mot = (analysis.emb["V_mot"] @ t[2]).tolist()
        clip_raw = clip_norm = None
        if self.clip_model is not None and "CLIP" in analysis.emb:
            tok = self.clip_processor(text=[CLIP_PREFIX + text], return_tensors="pt", padding=True,
                                      truncation=True, max_length=77)
            with self.model_lock, torch.inference_mode():
                ct = F.normalize(self.clip_model.get_text_features(**{k: v.to(self.device) for k, v in tok.items()})
                                 .float().cpu(), dim=-1)[0]
            raw = analysis.emb["CLIP"] @ ct
            # CLIP cosines live in a narrow band (~0.2-0.35); map them per query onto the joint score's
            # mean/std so the weight slider is meaningful and the softmax temperature still applies.
            j = torch.tensor(sim_joint)
            z = (raw - raw.mean()) / raw.std().clamp_min(1e-6) if len(raw) > 1 else raw * 0
            clip_raw, clip_norm = raw.tolist(), (j.mean() + j.std().nan_to_num(0.0) * z).tolist()
        result["scores"] = []
        for i, tid in enumerate(analysis.track_order):
            attr_p = {c["key"]: constraint_p(analysis.tracks[tid], c) for c in result["constraints"]}
            known = [(c["weight"], attr_p[c["key"]]) for c in result["constraints"] if attr_p[c["key"]] is not None]
            result["scores"].append({
                "track_id": tid, "joint": round(sim_joint[i], 5), "app": round(sim_app[i], 5),
                "mot": round(sim_mot[i], 5),
                **({"clip": round(clip_norm[i], 5), "clip_raw": round(clip_raw[i], 5)} if clip_norm else {}),
                "attr_p": {k: (None if v is None else round(v, 4)) for k, v in attr_p.items()},
                "penalty": round(sum(w * (1.0 - p) for w, p in known), 5),      # subtracted from the fused score
                "attr_min": round(min(p for _, p in known), 4) if known else None,  # used by the attribute filter
            })
        return result

    @staticmethod
    def rank(scores: List[Dict[str, Any]], weights: Dict[str, float], attr_filter: bool = False) -> List[Dict[str, Any]]:
        """
        Fuses the three DSCA scores with normalised weights, blends in the hidden CLIP score (CLIP_WEIGHT) when the
        scores carry one, subtracts the attribute penalty and adds softmax match probability. attr_filter drops tracks
        failing any query constraint (p < 0.5).
        Mirrored by static/js/app.js::rerank, keep the two in sync.
        """
        keys = list(DEFAULT_WEIGHTS)
        cw = CLIP_WEIGHT if scores and scores[0].get("clip") is not None else 0.0   # absent without CLIP
        wsum = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys) or 1.0
        w = {k: max(0.0, float(weights.get(k, 0.0))) / wsum for k in keys}
        if attr_filter:
            scores = [s for s in scores if s.get("attr_min") is None or s["attr_min"] >= 0.5]
        ranked = [dict(s, fused=(sum(w[k] * s[k] for k in w) + cw * (s.get("clip") or 0.0)) / (1.0 + cw)
                       - s.get("penalty", 0.0)) for s in scores]
        ranked.sort(key=lambda s: s["fused"], reverse=True)
        if ranked:
            m = max(s["fused"] for s in ranked)
            exps = [math.exp((s["fused"] - m) / SOFTMAX_TAU) for s in ranked]
            z = sum(exps)
            for r, (s, e) in enumerate(zip(ranked, exps), start=1):
                s["rank"] = r
                s["prob"] = e / z
        return ranked

    # ------------------------------------------------------------------ render
    def render(self, video_path: Path, analysis: Analysis, ranked: List[Dict[str, Any]], prompt: str,
               top_k: int, out_path: Path, progress: ProgressCb) -> None:
        """Draws top-K boxes, labels and a HUD onto the proxy video (mp4v; caller converts to H.264)."""
        cap = cv2.VideoCapture(str(video_path))
        w, h = analysis.width, analysis.height
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), analysis.fps, (w, h))
        top = ranked[:top_k]
        draw_order = list(reversed(top))  # #1 drawn last, on top
        target = top[0] if top else None
        trail: List[Tuple[int, int]] = []
        scale = max(0.4, min(1.0, w / 1280.0))
        n = analysis.frames_processed

        for f_idx in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            for s in draw_order:
                box = analysis.tracks[s["track_id"]].boxes.get(f_idx)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                color = rank_color(s["rank"])[0]
                thick = 3 if s["rank"] == 1 else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                tag = f"#{s['rank']} P{s['track_id']}  {s['fused']:.3f} ({s['prob'] * 100:.0f}%)"
                fs = 0.5 * scale + 0.1
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                ty = max(th + 8, y1)
                cv2.rectangle(frame, (x1, ty - th - 8), (min(w - 1, x1 + tw + 8), ty), color, -1)
                txt = (0, 0, 0) if s["rank"] in DARK_TEXT_RANKS else (255, 255, 255)
                cv2.putText(frame, tag, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, txt, 1, cv2.LINE_AA)
                if s["rank"] == 1:
                    trail.append(((x1 + x2) // 2, y2))
                    trail = trail[-30:]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], RANK_COLORS[0][0], 2, cv2.LINE_AA)

            hud_h = int(36 * scale + 10)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 16, 12), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            max_chars = max(20, int(w / (11 * scale)) - 40)
            q = prompt if len(prompt) <= max_chars else prompt[:max_chars - 1] + "…"
            hud = f'"{q}"   #1: P{target["track_id"]}   {f_idx + 1}/{n}' if target else f"{f_idx + 1}/{n}"
            cv2.putText(frame, hud, (12, int(hud_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale + 0.08,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if f_idx % 10 == 0:
                progress("render", f_idx + 1, n)
        cap.release()
        writer.release()

    # ------------------------------------------------------------------ serialisation
    def tracks_payload(self, analysis: Analysis) -> Dict[str, Any]:
        """Compact per-frame boxes and previews for the browser overlay/timeline."""
        tracks = []
        for tid in sorted(analysis.tracks):
            t = analysis.tracks[tid]
            frames = sorted(t.boxes)
            tracks.append({
                "id": tid,
                "start": t.start,
                "end": t.end,
                "n_det": len(t.det_frames),
                # numbers rounded; colour dicts and lists (dir_slopes) passed through
                "attrs": {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in t.attrs.items()},
                "boxes": [[f, *t.boxes[f]] for f in frames],
                "thumb": _data_url(t.thumb) if t.thumb else "",
                "film": [_data_url(j) for j in t.film],
            })
        return {
            "analysis_id": analysis.id,
            "video_id": analysis.video_id,
            "params": analysis.params,
            "fps": analysis.fps,
            "width": analysis.width,
            "height": analysis.height,
            "frames_processed": analysis.frames_processed,
            "frames_total": analysis.frames_total,
            "tracks": tracks,
        }
