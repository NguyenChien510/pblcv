"""
Gallery store: analysed videos kept on Supabase (Postgres rows + private Storage bucket) with a local disk cache.

- An analysis is saved as a zip of JSON (tracks, boxes, attributes, parameters), one npz of float32 arrays (loaded with
  allow_pickle=False) and the preview JPEGs. No pickle: loading a stored analysis cannot run code. The model clips
  (Track.clip, 2.4 MB per track) are not stored; SearchEngine.encode_tracks crops them again from the proxy when a
  checkpoint needs them. Size: 0.7 MB for 18 tracks, 5.7 MB for 149 (vs 55 / 454 MB with the clips).
- The proxy video goes to Storage as one object, or as parts of PART_BYTES when it is larger (Supabase Free caps one
  file at 50 MB).
- Everything is cached under uploads/videos/<video_id>/ and fetched from Supabase only on a cache miss; the cache is
  trimmed to a size limit, least recently used first. Nothing on Supabase is ever deleted automatically.
"""
import io
import json
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .engine import Analysis, Track
from .supabase_client import SupabaseClient, SupabaseError

FORMAT_VERSION = 1
BUCKET = "videos"
PART_BYTES = int(float(os.environ.get("STORE_PART_MB", "45")) * 2**20)


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dump_analysis(a: Analysis) -> bytes:
    """Analysis -> zip bytes (no model clips)."""
    order = list(a.track_order) or sorted(a.tracks)
    meta = {
        "format": FORMAT_VERSION, "id": a.id, "video_id": a.video_id, "params": a.params, "fps": a.fps,
        "width": a.width, "height": a.height, "frames_processed": a.frames_processed, "frames_total": a.frames_total,
        "ckpt_key": a.ckpt_key, "track_order": order, "layout": a.layout, "stitch_log": a.stitch_log,
        "tracks": [{
            "id": t.id, "boxes": {str(f): list(map(int, b)) for f, b in t.boxes.items()},
            "det_frames": list(map(int, t.det_frames)), "quality": {str(f): float(q) for f, q in t.quality.items()},
            "attrs": t.attrs, "reid_n": int(t.reid_n), "film": len(t.film), "thumb": bool(t.thumb),
        } for t in (a.tracks[i] for i in sorted(a.tracks))],
    }
    arrays: Dict[str, np.ndarray] = {f"emb_{k}": v.float().numpy() for k, v in a.emb.items()}
    for t in a.tracks.values():
        if t.reid is not None:
            arrays[f"reid_{t.id}"] = np.asarray(t.reid, dtype=np.float32)
        if t.clip_emb is not None:
            arrays[f"clipemb_{t.id}"] = t.clip_emb.float().numpy()
    npz = io.BytesIO()
    np.savez(npz, **arrays)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(meta, default=_jsonable))
        z.writestr("arrays.npz", npz.getvalue())
        for t in a.tracks.values():
            if t.thumb:
                z.writestr(f"jpg/{t.id}_thumb.jpg", t.thumb)
            for i, j in enumerate(t.film):
                z.writestr(f"jpg/{t.id}_film{i}.jpg", j)
    return buf.getvalue()


def load_analysis(data: bytes, video_path: Optional[Path]) -> Analysis:
    """zip bytes -> Analysis whose tracks have no model clips (encode_tracks re-crops them when needed)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("meta.json"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported analysis format {meta.get('format')}")
        with np.load(io.BytesIO(z.read("arrays.npz")), allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        tracks: Dict[int, Track] = {}
        for m in meta["tracks"]:
            tid = int(m["id"])
            t = Track(id=tid, boxes={int(f): b for f, b in m["boxes"].items()}, det_frames=m["det_frames"],
                      quality={int(f): q for f, q in m["quality"].items()}, attrs=m["attrs"])
            t.reid_n = m["reid_n"]
            if f"reid_{tid}" in arrays:
                t.reid = arrays[f"reid_{tid}"]
            if f"clipemb_{tid}" in arrays:
                t.clip_emb = torch.from_numpy(arrays[f"clipemb_{tid}"])
            if m["thumb"]:
                t.thumb = z.read(f"jpg/{tid}_thumb.jpg")
            t.film = [z.read(f"jpg/{tid}_film{i}.jpg") for i in range(m["film"])]
            tracks[tid] = t
    order = [int(i) for i in meta["track_order"]]
    emb = {k[4:]: torch.from_numpy(v) for k, v in arrays.items() if k.startswith("emb_")}
    a = Analysis(id=meta["id"], video_id=meta["video_id"], params=meta["params"], fps=meta["fps"],
                 width=meta["width"], height=meta["height"], frames_processed=meta["frames_processed"],
                 frames_total=meta["frames_total"], tracks=tracks, ckpt_key=meta["ckpt_key"], emb=emb,
                 track_order=order, video_path=video_path, layout=meta["layout"], stitch_log=meta["stitch_log"])
    a.track_emb = {tid: {k: emb[k][row] for k in ("V_app", "V_mot", "V_video") if k in emb}
                   for row, tid in enumerate(order)}
    return a


def params_key(params: Dict[str, Any]) -> tuple:
    """What makes two analyses of the same video interchangeable."""
    return (int(params.get("max_frames", 0)), int(params.get("stride", 1)), round(float(params.get("conf", 0.35)), 2),
            bool(params.get("clip", True)), params.get("tracker", ""))


class LocalCache:
    """uploads/videos/<id>/ folders, trimmed to max_bytes by last use (folders in use are never removed)."""

    def __init__(self, root: Path, max_bytes: int):
        self.root = root
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        for p in self.root.glob("incoming_*"):
            p.unlink(missing_ok=True)

    def folder(self, vid: str) -> Path:
        f = self.root / vid
        f.mkdir(parents=True, exist_ok=True)
        os.utime(f)                     # last use, for trimming
        return f

    def size(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def trim(self, in_use: set) -> None:
        folders = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        total = self.size()
        for f in folders:
            if total <= self.max_bytes:
                break
            if f.name in in_use:
                continue
            n = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
            shutil.rmtree(f, ignore_errors=True)
            total -= n


class Gallery:
    """Rows in `videos` / `analyses`, objects in the private `videos` bucket (deploy/supabase/schema.sql)."""

    def __init__(self, client: SupabaseClient, limit_bytes: int):
        self.db = client
        self.limit_bytes = limit_bytes
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ reading
    def list(self) -> List[Dict[str, Any]]:
        rows = self.db.select("videos", {"select": "*,analyses(*)", "order": "last_used_at.desc"})
        for r in rows:
            r["analyses"] = sorted(r.get("analyses") or [], key=lambda x: x["created_at"], reverse=True)
        return rows

    def get(self, vid: str) -> Optional[Dict[str, Any]]:
        rows = self.db.select("videos", {"select": "*,analyses(*)", "id": f"eq.{vid}"})
        return rows[0] if rows else None

    def used_bytes(self) -> int:
        rows = self.db.select("videos", {"select": "proxy_bytes,thumb_bytes,analyses(blob_bytes)"})
        return sum((r["proxy_bytes"] or 0) + (r.get("thumb_bytes") or 0)
                   + sum(a["blob_bytes"] or 0 for a in r.get("analyses") or []) for r in rows)

    def has_room(self, extra: int) -> bool:
        return self.used_bytes() + extra <= self.limit_bytes

    def find_analysis(self, vid: str, key: tuple) -> Optional[Dict[str, Any]]:
        row = self.get(vid)
        if not row:
            return None
        cands = [a for a in row.get("analyses") or [] if params_key(a["params"]) == key]
        return max(cands, key=lambda a: a["created_at"]) if cands else None

    # ------------------------------------------------------------------ writing
    def save_video(self, vid: str, name: str, proxy: Path, info: Dict[str, Any], thumb: bytes, ip: str) -> None:
        size = proxy.stat().st_size
        if self.get(vid):
            return
        if not self.has_room(size + len(thumb)):
            raise SupabaseError("gallery storage is full")
        data = proxy.read_bytes()
        parts = 0 if size <= PART_BYTES else -(-size // PART_BYTES)
        if parts:
            for i in range(parts):
                self.db.upload(BUCKET, f"{vid}/proxy.part{i:03d}", data[i * PART_BYTES:(i + 1) * PART_BYTES],
                               "application/octet-stream")
        else:
            self.db.upload(BUCKET, f"{vid}/proxy.mp4", data, "video/mp4")
        if thumb:
            self.db.upload(BUCKET, f"{vid}/thumb.jpg", thumb, "image/jpeg")
        self.db.insert("videos", {
            "id": vid, "name": name, "proxy_bytes": size, "proxy_parts": parts, "thumb_bytes": len(thumb),
            "duration": info.get("duration"), "fps": info.get("fps"), "width": info.get("width"),
            "height": info.get("height"), "uploader_ip": ip,
        })

    def save_analysis(self, vid: str, a: Analysis) -> Dict[str, Any]:
        """Stores the analysis, then removes older analyses of this video with the same parameters (re-analysis)."""
        blob = dump_analysis(a)
        if not self.has_room(len(blob)):
            raise SupabaseError("gallery storage is full")
        self.db.upload(BUCKET, f"{vid}/{a.id}.zip", blob, "application/zip")
        row = {"id": a.id, "video_id": vid, "params": a.params, "tracker_version": a.params.get("tracker", ""),
               "ckpt_key": a.ckpt_key, "n_tracks": len(a.tracks), "blob_bytes": len(blob)}
        self.db.insert("analyses", row)
        key = params_key(a.params)
        old = [x for x in (self.get(vid) or {}).get("analyses") or [] if x["id"] != a.id and params_key(x["params"]) == key]
        for x in old:
            self.delete_analysis(vid, x["id"])
        return row

    def delete_analysis(self, vid: str, aid: str) -> None:
        self.db.delete("analyses", {"id": f"eq.{aid}"})
        self.db.remove(BUCKET, [f"{vid}/{aid}.zip"])

    def delete_video(self, vid: str) -> None:
        names = [f"{vid}/{o['name']}" for o in self.db.list_objects(BUCKET, f"{vid}/")]
        if names:
            self.db.remove(BUCKET, names)
        self.db.delete("videos", {"id": f"eq.{vid}"})       # analyses go with it (on delete cascade)

    def touch(self, vid: str, opens: int) -> None:
        self.db.update("videos", {"id": f"eq.{vid}"},
                       {"last_used_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "opens": opens})

    # ------------------------------------------------------------------ fetching into the cache
    def fetch_proxy(self, row: Dict[str, Any], folder: Path) -> Path:
        dst = folder / "proxy.mp4"
        if dst.exists() and dst.stat().st_size == row["proxy_bytes"]:
            return dst
        tmp = folder / "proxy.download"
        with open(tmp, "wb") as f:
            if row["proxy_parts"]:
                for i in range(row["proxy_parts"]):
                    f.write(self.db.download(BUCKET, f"{row['id']}/proxy.part{i:03d}"))
            else:
                f.write(self.db.download(BUCKET, f"{row['id']}/proxy.mp4"))
        if tmp.stat().st_size != row["proxy_bytes"]:
            tmp.unlink(missing_ok=True)
            raise SupabaseError("downloaded proxy has the wrong size")
        tmp.replace(dst)
        return dst

    def fetch_thumb(self, vid: str, folder: Path) -> bytes:
        dst = folder / "thumb.jpg"
        if not dst.exists():
            dst.write_bytes(self.db.download(BUCKET, f"{vid}/thumb.jpg"))
        return dst.read_bytes()

    def fetch_analysis(self, vid: str, arow: Dict[str, Any], folder: Path, video_path: Path) -> Analysis:
        dst = folder / f"{arow['id']}.zip"
        if not (dst.exists() and dst.stat().st_size == arow["blob_bytes"]):
            dst.write_bytes(self.db.download(BUCKET, f"{vid}/{arow['id']}.zip"))
        return load_analysis(dst.read_bytes(), video_path)
