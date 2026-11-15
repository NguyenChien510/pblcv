"""
Shared helpers for the revision experiments (review DG25, submission 2).

  - load_model / build_split_dataset / encode_split: encode one split (val or test) with a checkpoint.
    Every caption becomes a query; every unique video_id is encoded exactly once (gallery).
  - ranks_for: rank of the correct gallery item for each query (video-level or person-ID-level match).
  - metrics_from_ranks, bootstrap_summary, paired_bootstrap: Rank@K / MdR and their bootstrap CIs.

Embeddings are stored L2-normalised, so any fusion weights can be applied offline without re-encoding.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KS = (1, 5, 10, 50)
DEFAULT_WEIGHTS = (0.50, 0.35, 0.15)
SUBSETS = ("TVPReid-PRID", "TVPReid-iLIDs", "TVPReid-Duke")

# Published numbers of [24] (MFGF, ACM MM'24, Table 1 and Table 2), test split, one caption per video
MFGF_PUBLISHED = {
    "TVPReid-PRID": {"R@1": 32.3, "R@5": 76.3, "R@10": 85.1, "R@50": 100.0, "MdR": 2.0},
    "TVPReid-iLIDs": {"R@1": 30.0, "R@5": 71.7, "R@10": 91.7, "R@50": 100.0, "MdR": 4.7},
    "TVPReid-Duke": {"R@1": 35.2, "R@5": 62.2, "R@10": 84.0, "R@50": 90.4, "MdR": 3.0},
}
MFGF_SPLIT_VIDEOS = {  # train / val / test videos, Table 2 of [24]
    "TVPReid-PRID": (921, 71, 142),
    "TVPReid-iLIDs": (488, 37, 75),
    "TVPReid-Duke": (3920, 302, 603),
}


def short_name(sub: str) -> str:
    return {"TVPReid-PRID": "PRID", "TVPReid-iLIDs": "ILIDS", "TVPReid-Duke": "DUKE"}.get(sub, sub)


# ----------------------------------------------------------------------------------------------
# Model / data
# ----------------------------------------------------------------------------------------------
def load_checkpoint(path: str) -> dict:
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(ckpt_path: str, device):
    """Rebuilds DAMRModel with the architecture stored in the checkpoint config and loads its weights."""
    from models.damr_main import DAMRModel

    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    model = DAMRModel(
        embed_dim=mcfg.get("embed_dim", 768),
        text_model_name=mcfg.get("text_model_name", "bert-base-uncased"),
        pretrained=True,                      # loads the BERT tokenizer; weights are overwritten below
        motion_frame_num=mcfg.get("num_frames", mcfg.get("motion_frame_num", 16)),
        img_size=mcfg.get("img_size", 224),
        vit_depth=mcfg.get("vit_depth", 6),
        vit_heads=mcfg.get("vit_heads", 12),
        drop_rate=mcfg.get("drop_rate", 0.1),
        pretrained_vit=False,                 # ImageNet weights would be overwritten anyway
        use_temporal_attn=mcfg.get("use_temporal_attn", True),
        s3d_motion=mcfg.get("s3d_motion", False),
        backbone=mcfg.get("backbone", "spacetime"),
        mim_mode=mcfg.get("mim_mode", "raw"),
    )
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()
    info = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "val_metrics": ckpt.get("metrics"),
        "seed": (cfg.get("train", {}) or {}).get("seed"),
        "use_dsca": bool(mcfg.get("use_dsca", True)),
        "use_temporal_attn": bool(mcfg.get("use_temporal_attn", True)),
        "pretrained_vit": bool(mcfg.get("pretrained_vit", True)),
        "lambda_app": mcfg.get("lambda_app"),
        "lambda_mot": mcfg.get("lambda_mot"),
        "lambda_mim": mcfg.get("lambda_mim"),
        "num_frames": mcfg.get("num_frames", mcfg.get("motion_frame_num", 16)),
        "img_size": mcfg.get("img_size", 224),
        "backbone": mcfg.get("backbone", "spacetime"),
        "frame_layout": "mfgf" if mcfg.get("backbone") == "mfgf" else "uniform",
        "dsca_mode": mcfg.get("dsca_mode", "keyword"),
    }
    return model, info


def build_split_dataset(sub_dataset: str, split: str, data_root: Optional[str], num_frames: int = 16, img_size: int = 224,
                        frame_layout: str = "uniform", dsca_mode: str = "keyword"):
    from data.dataset import TVPRDataset
    from utils.paths import paths

    if data_root:
        paths.set_data_root(data_root)
    ds = TVPRDataset(
        data_root=str(paths.data_root),
        sub_dataset=sub_dataset,
        split=split,
        num_frames=num_frames,
        motion_frame_num=num_frames,
        img_size=img_size,
        frame_layout=frame_layout,
        dsca_mode=dsca_mode or "keyword",
    )
    if ds.samples and ds.samples[0].get("dataset") == "Synthetic":
        print("[!] WARNING: no captions found under the data root -> TVPRDataset built a DUMMY index. "
              "Results are meaningless (smoke test only).")
    return ds


def _caption_table(ds) -> Tuple[List[dict], List[dict]]:
    """Splits dataset samples into a query table (one row per caption) and a gallery table (one row per video)."""
    queries, gallery, seen, cap_counter = [], [], {}, {}
    for s in ds.samples:
        vid = str(s["video_id"])
        cap_idx = cap_counter.get(vid, 0)
        cap_counter[vid] = cap_idx + 1
        q_app, q_mot = ds.decomposer.decompose(s["caption"])
        queries.append({"video_id": vid, "pid": int(s["person_id"]), "cap_idx": cap_idx,
                        "caption": s["caption"], "q_app": q_app, "q_mot": q_mot})
        if vid not in seen:
            seen[vid] = len(gallery)
            gallery.append({"video_id": vid, "pid": int(s["person_id"]), "video_path": s.get("video_path")})
    return queries, gallery


class _VideoOnlyDataset:
    def __init__(self, ds, gallery):
        self.ds, self.gallery = ds, gallery

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, i):
        return self.ds._read_video_frames(self.gallery[i]["video_path"])


def encode_texts(model, texts: Sequence[str], device, batch_size: int = 64):
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            t = model.text_encoder.encode_text(list(texts[i:i + batch_size]), device=device)
            out.append(F.normalize(t.float(), dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, model.embed_dim), dtype=np.float32)


def encode_split(model, info: dict, ds, device, batch_size: int = 16, num_workers: int = 2) -> Dict[str, np.ndarray]:
    """Returns L2-normalised query/gallery embeddings plus identifiers (see module docstring)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    queries, gallery = _caption_table(ds)
    use_dsca = info.get("use_dsca", True)
    captions = [q["caption"] for q in queries]
    t_full = encode_texts(model, captions, device)
    if use_dsca:
        t_app = encode_texts(model, [q["q_app"] for q in queries], device)
        t_mot = encode_texts(model, [q["q_mot"] for q in queries], device)
    else:  # "- DSCA" models were trained with T_app = T_mot = T_full
        t_app, t_mot = t_full.copy(), t_full.copy()

    loader = DataLoader(_VideoOnlyDataset(ds, gallery), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    v_app, v_mot, v_vid = [], [], []
    with torch.no_grad():
        for frames in loader:
            a, m, v = model.encode_video(frames.to(device, non_blocking=True))
            v_app.append(F.normalize(a.float(), dim=-1).cpu())
            v_mot.append(F.normalize(m.float(), dim=-1).cpu())
            v_vid.append(F.normalize(v.float(), dim=-1).cpu())
            print(f"\r    encoded {sum(len(x) for x in v_vid)}/{len(gallery)} videos", end="", flush=True)
    print()
    return {
        "T_full": t_full.astype(np.float32), "T_app": t_app.astype(np.float32), "T_mot": t_mot.astype(np.float32),
        "V_video": torch.cat(v_vid).numpy().astype(np.float32),
        "V_app": torch.cat(v_app).numpy().astype(np.float32),
        "V_mot": torch.cat(v_mot).numpy().astype(np.float32),
        "q_vid": np.array([q["video_id"] for q in queries]),
        "q_pid": np.array([q["pid"] for q in queries], dtype=np.int64),
        "q_cap": np.array([q["cap_idx"] for q in queries], dtype=np.int64),
        "g_vid": np.array([g["video_id"] for g in gallery]),
        "g_pid": np.array([g["pid"] for g in gallery], dtype=np.int64),
        "q_text": np.array(captions),
        "q_app_text": np.array([q["q_app"] for q in queries]),
        "q_mot_text": np.array([q["q_mot"] for q in queries]),
    }


def encode_bank(model, info: dict, ds, device) -> Dict[str, np.ndarray]:
    """T_full / T_app / T_mot of the TRAIN captions of a split dataset (reference bank for hubness reduction)."""
    captions = [s["caption"] for s in ds.samples]
    t_full = encode_texts(model, captions, device)
    if not info.get("use_dsca", True):
        return {"T_full": t_full, "T_app": t_full.copy(), "T_mot": t_full.copy()}
    q_app, q_mot = zip(*(ds.decomposer.decompose(c) for c in captions)) if captions else ((), ())
    return {"T_full": t_full, "T_app": encode_texts(model, list(q_app), device),
            "T_mot": encode_texts(model, list(q_mot), device)}


def hub_vectors(E: Dict[str, np.ndarray], bank: Dict[str, np.ndarray], k: int = 10) -> np.ndarray:
    """[3, G]: per branch, mean top-k cosine of every gallery video to the train-caption bank (its 'hubness')."""
    out = []
    for v_key, t_key in (("V_video", "T_full"), ("V_app", "T_app"), ("V_mot", "T_mot")):
        sims = E[v_key] @ bank[t_key].T
        kk = min(k, sims.shape[1])
        out.append(np.sort(sims, axis=1)[:, -kk:].mean(axis=1))
    return np.stack(out).astype(np.float32)


def with_hubness(E: Dict[str, np.ndarray], hub: np.ndarray, beta: float) -> Dict[str, np.ndarray]:
    """A view of E whose fused score is lowered by beta * centred hubness of each gallery video (utils/rerank.py)."""
    return {**E, "_hub": hub, "_hub_beta": np.float32(beta)}


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------------
# Scoring / ranking
# ----------------------------------------------------------------------------------------------
def fused_similarity(E: Dict[str, np.ndarray], weights: Sequence[float], q_idx: Optional[np.ndarray] = None) -> np.ndarray:
    w1, w2, w3 = weights
    sl = slice(None) if q_idx is None else q_idx
    S = (w1 * E["T_full"][sl] @ E["V_video"].T
         + w2 * E["T_app"][sl] @ E["V_app"].T
         + w3 * E["T_mot"][sl] @ E["V_mot"].T)
    if "_hub" in E and float(E["_hub_beta"]) != 0.0:
        r = w1 * E["_hub"][0] + w2 * E["_hub"][1] + w3 * E["_hub"][2]
        S = S - float(E["_hub_beta"]) * (r - r.mean())[None, :]
    return S


def ranks_for(sim: np.ndarray, q_keys: np.ndarray, g_keys: np.ndarray) -> np.ndarray:
    """1-based rank of the first gallery item whose key equals the query key (video_id or pid)."""
    order = np.argsort(-sim, axis=1, kind="stable")
    hits = g_keys[order] == q_keys[:, None]
    has = hits.any(axis=1)
    first = hits.argmax(axis=1) + 1
    return np.where(has, first, sim.shape[1] + 1).astype(np.int64)


PROTOCOLS = {
    # name: (description, caption filter)
    "mfgf": ("test protocol of [24]: one caption per video (first caption), gallery = all videos of the split", 0),
    "mfgf_cap2": ("same as mfgf but with the second caption of every video", 1),
    "all2": ("both captions of every video are queries, gallery deduplicated by video_id", None),
}


def protocol_query_index(E: Dict[str, np.ndarray], protocol: str) -> np.ndarray:
    cap = PROTOCOLS[protocol][1]
    if cap is None:
        return np.arange(len(E["q_vid"]))
    idx = np.where(E["q_cap"] == cap)[0]
    return idx


def protocol_ranks(E: Dict[str, np.ndarray], weights: Sequence[float], protocol: str, match: str = "video"):
    idx = protocol_query_index(E, protocol)
    sim = fused_similarity(E, weights, idx)
    if match == "video":
        r = ranks_for(sim, E["q_vid"][idx], E["g_vid"])
    else:
        r = ranks_for(sim, E["q_pid"][idx], E["g_pid"])
    return r, E["q_vid"][idx], E["q_cap"][idx]


# ----------------------------------------------------------------------------------------------
# Metrics and bootstrap
# ----------------------------------------------------------------------------------------------
def metrics_from_ranks(r: np.ndarray, ks: Sequence[int] = KS) -> Dict[str, float]:
    r = np.asarray(r)
    out = {f"R@{k}": float(np.mean(r <= k) * 100.0) for k in ks}
    out["MdR"] = float(np.median(r))
    out["MnR"] = float(np.mean(r))
    return out


def _resampler(n: int, clusters: Optional[np.ndarray], rng):
    """Returns (draw, n_units): draw() gives query indices of one bootstrap sample (queries or whole clusters)."""
    if clusters is None:
        return (lambda: rng.integers(0, n, size=n)), n
    _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    G = len(counts)

    def draw():
        pick = rng.integers(0, G, size=G)
        return np.concatenate([order[starts[g]:starts[g] + counts[g]] for g in pick])
    return draw, G


def bootstrap_summary(R: np.ndarray, clusters: Optional[np.ndarray] = None, B: int = 10000, seed: int = 0,
                      ks: Sequence[int] = KS, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """
    R: [n_seeds, n_queries] ranks of the same queries under several seeds (n_seeds may be 1).
    Point estimate = metric averaged over seeds. CI = percentile bootstrap of that seed-averaged metric,
    resampling queries (or whole clusters, e.g. both captions of a video) with replacement.
    """
    R = np.atleast_2d(np.asarray(R))
    rng = np.random.default_rng(seed)
    n = R.shape[1]
    draw, G = _resampler(n, clusters, rng)
    per_seed = [metrics_from_ranks(R[s], ks) for s in range(R.shape[0])]
    out = {}
    names = [f"R@{k}" for k in ks] + ["MdR"]
    point = {m: float(np.mean([p[m] for p in per_seed])) for m in names}
    std = {m: float(np.std([p[m] for p in per_seed], ddof=1)) if R.shape[0] > 1 else 0.0 for m in names}
    boot = {m: np.empty(B) for m in names}
    for b in range(B):
        Rb = R[:, draw()]
        for k in ks:
            boot[f"R@{k}"][b] = np.mean(Rb <= k) * 100.0
        boot["MdR"][b] = np.mean(np.median(Rb, axis=1))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for m in names:
        out[m] = {"mean": point[m], "std": std[m],
                  "ci_low": float(np.percentile(boot[m], lo)), "ci_high": float(np.percentile(boot[m], hi)),
                  "per_seed": [p[m] for p in per_seed]}
    out["_n_queries"] = n
    out["_n_seeds"] = int(R.shape[0])
    out["_n_clusters"] = G
    return out


def paired_bootstrap(Ra: np.ndarray, Rb: np.ndarray, metric: str = "R@1", clusters: Optional[np.ndarray] = None,
                     B: int = 10000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Difference (a - b) of a seed-averaged metric on the SAME queries, with a paired percentile bootstrap CI."""
    Ra, Rb = np.atleast_2d(Ra), np.atleast_2d(Rb)
    assert Ra.shape[1] == Rb.shape[1], "paired bootstrap needs the same query set"
    rng = np.random.default_rng(seed)
    n = Ra.shape[1]
    draw, _ = _resampler(n, clusters, rng)

    def stat(A, idx):
        if metric == "MdR":
            return float(np.mean(np.median(A[:, idx], axis=1)))
        k = int(metric.split("@")[1])
        return float(np.mean(A[:, idx] <= k) * 100.0)

    full = np.arange(n)
    diff = stat(Ra, full) - stat(Rb, full)
    boot = np.empty(B)
    for b in range(B):
        idx = draw()
        boot[b] = stat(Ra, idx) - stat(Rb, idx)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": diff, "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))}


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
