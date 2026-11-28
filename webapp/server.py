"""
FastAPI app: video registry, background analysis jobs, search, static UI, the shared gallery
(webapp/store.py, Supabase) and the /admin page (webapp/admin.py, webapp/monitor.py).
"""

import hashlib
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import video_io
from .admin import COOKIE, SESSION_S, AdminAuth
from .engine import CLIP_WEIGHT, RANK_COLORS, TRACKER_VERSION, Analysis, SearchEngine
from .jobs import JobManager
from .monitor import Monitor
from .store import Gallery, LocalCache, params_key
from .supabase_client import SupabaseClient, SupabaseError

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_VIDEOS = 20
MAX_ANALYSES = 8
ANALYZE_STAGES = {"prepare": (0, 3), "detect": (3, 45), "crops": (45, 50), "stitch": (50, 55), "clip": (55, 62),
                  "encode": (62, 100)}


@dataclass
class ServerConfig:
    """From the environment / .env (web_demo.py). Without SUPABASE_URL there is no gallery (old in-memory mode)."""
    supabase_url: str = ""
    supabase_key: str = ""
    admin_password_hash: str = ""
    admin_session_secret: str = ""
    cache_max_gb: float = 5.0          # local copies of gallery videos (uploads/videos)
    storage_limit_mb: float = 950.0    # gallery size on Supabase (Free plan: 1 GB)
    max_queue: int = 5                 # queued + running jobs before new analyses are refused
    uploads_per_hour: int = 10         # per IP

    @classmethod
    def from_env(cls) -> "ServerConfig":
        e = os.environ.get
        return cls(e("SUPABASE_URL", ""), e("SUPABASE_SECRET_KEY", ""), e("ADMIN_PASSWORD_HASH", ""),
                   e("ADMIN_SESSION_SECRET", ""), float(e("CACHE_MAX_GB", "5")), float(e("STORAGE_LIMIT_MB", "950")),
                   int(e("MAX_QUEUE", "5")), int(e("UPLOADS_PER_HOUR", "10")))


class VideoEntry:
    def __init__(self, vid: str, name: str, source: Optional[Path], folder: Path, ip: str = "", save: bool = False):
        self.id = vid
        self.name = name
        self.source = source            # None for a video opened from the gallery (only the proxy exists)
        self.folder = folder
        self.proxy = folder / "proxy.mp4"
        self.info: Optional[Dict[str, Any]] = None  # probe of the proxy
        self.prepare_job: Optional[str] = None
        self.ip = ip
        self.save = save                # keep it in the shared gallery
        self.gallery = "off"            # off | pending | saved | full | error

    @property
    def ready(self) -> bool:
        return self.info is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.id,
            "name": self.name,
            "ready": self.ready,
            "info": self.info,
            "prepare_job": None if self.ready else self.prepare_job,
            "stream_url": f"/api/videos/{self.id}/stream" if self.ready else None,
            "gallery": self.gallery,
        }


class AnalyzeParams(BaseModel):
    max_frames: int = Field(300, ge=0, le=20000)   # 0 = whole video
    stride: int = Field(1, ge=1, le=4)
    conf: float = Field(0.35, ge=0.1, le=0.9)
    clip: bool = True      # encode CLIP image embeddings during analysis (skippable to save time)
    reanalyse: bool = False  # ignore results kept in memory or in the gallery


class SearchBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)


class CheckpointBody(BaseModel):
    name: str


class PasswordBody(BaseModel):
    password: str = Field(..., max_length=200)


class NewPasswordBody(BaseModel):
    current: str = Field(..., max_length=200)
    new: str = Field(..., min_length=8, max_length=200)


class BlockBody(BaseModel):
    ip: str = Field(..., min_length=3, max_length=64)
    reason: str = Field("", max_length=200)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "?"


def create_app(engine: SearchEngine, uploads_dir: Path, max_upload_mb: int = 1024,
               config: Optional[ServerConfig] = None) -> FastAPI:
    cfg = config or ServerConfig()
    db = SupabaseClient(cfg.supabase_url, cfg.supabase_key) if cfg.supabase_url and cfg.supabase_key else None
    gallery = Gallery(db, int(cfg.storage_limit_mb * 2**20)) if db else None
    monitor = Monitor(db, cfg.uploads_per_hour)

    app = FastAPI(title="SpaceTime-DSCA Person Search", docs_url="/api/docs", redoc_url=None)
    admin = AdminAuth(cfg.admin_session_secret, cfg.admin_password_hash, db)
    jobs = JobManager()            # CPU work, one at a time (shared model and detector)
    net_jobs = JobManager()        # Supabase transfers, so a download never waits behind a 10-minute analysis
    videos: "OrderedDict[str, VideoEntry]" = OrderedDict()
    analyses: "OrderedDict[str, Analysis]" = OrderedDict()
    analysis_keys: Dict[tuple, str] = {}
    analysis_meta: Dict[str, Dict[str, Any]] = {}     # aid -> {"from_gallery", "stored_at"}
    registry_lock = threading.Lock()
    gallery_cache: Dict[str, Any] = {"t": 0.0, "rows": [], "error": None}

    store = uploads_dir / "videos"
    if gallery is None:
        shutil.rmtree(store, ignore_errors=True)  # session-scoped: index lives in memory
    cache = LocalCache(store, int(cfg.cache_max_gb * 2**30))   # with a gallery the folders are a disk cache
    if db is not None:
        err = db.ping()
        print(f"[+] Gallery store: Supabase {'OK' if err is None else 'UNAVAILABLE: ' + err}")
    psutil.cpu_percent(None)       # first call primes the counter

    @app.exception_handler(SupabaseError)
    async def supabase_down(request: Request, exc: SupabaseError):
        return JSONResponse({"detail": f"Gallery storage (Supabase) unavailable: {str(exc)[:200]}"}, status_code=503)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        ip, path = client_ip(request), request.url.path
        if monitor.is_blocked(ip) and not path.startswith(("/admin", "/api/admin", "/static/")):
            return JSONResponse({"detail": "This IP address was blocked by the administrator."}, status_code=403)
        response = await call_next(request)
        if path.startswith("/static/"):
            # Local demo: always revalidate UI files (cheap 304s) so edits show up without a hard reload
            response.headers["Cache-Control"] = "no-cache"
        elif not path.startswith(("/api/admin", "/admin")):       # the admin is not one of the users
            monitor.request(ip, request.method, path, request.headers.get("user-agent", ""))
        return response

    # ------------------------------------------------------------------ helpers
    def get_video(vid: str) -> VideoEntry:
        v = videos.get(vid)
        if v is None:
            raise HTTPException(404, "Video not found")
        videos.move_to_end(vid)
        return v

    def get_analysis(aid: str) -> Analysis:
        a = analyses.get(aid)
        if a is None:
            raise HTTPException(404, "Analysis not found (expired). Please analyse the video again.")
        analyses.move_to_end(aid)
        return a

    def get_job(job_id: str):
        return jobs.get(job_id) or net_jobs.get(job_id)

    def evict() -> None:
        while len(videos) > MAX_VIDEOS:
            vid, v = videos.popitem(last=False)
            for key, aid in list(analysis_keys.items()):
                if key[0] == vid:
                    analysis_keys.pop(key)
                    analyses.pop(aid, None)
            if gallery is None:
                shutil.rmtree(v.folder, ignore_errors=True)
        while len(analyses) > MAX_ANALYSES:
            aid, _ = analyses.popitem(last=False)
            for key, val in list(analysis_keys.items()):
                if val == aid:
                    analysis_keys.pop(key)
        if gallery is not None:
            cache.trim(in_use=set(videos))

    def prepare(v: VideoEntry, progress) -> None:
        if v.ready:
            return
        if not v.proxy.exists():
            if v.source is None:
                raise ValueError("The video is still downloading from the gallery")
            tmp = v.folder / "proxy.tmp.mp4"
            video_io.make_proxy(v.source, tmp, lambda f: progress("prepare", f * 100, 100))
            tmp.replace(v.proxy)
        v.info = video_io.probe(v.proxy)
        progress("prepare", 100, 100)

    def save_video_to_gallery(v: VideoEntry) -> None:
        """Background (net_jobs): proxy + thumbnail -> Supabase, unless already there."""
        if gallery is None or not v.save or v.gallery == "saved":
            return
        v.gallery = "pending"

        def run(job):
            try:
                gallery.save_video(v.id, v.name, v.proxy, v.info or {}, video_io.first_frame_jpeg(v.proxy), v.ip)
                v.gallery = "saved"
                gallery_cache["t"] = 0.0
                monitor.event(v.ip, "gallery_save", v.id, name=v.name)
            except SupabaseError as e:
                v.gallery = "full" if "full" in str(e) else "error"
                monitor.event(v.ip, "gallery_save_failed", v.id, name=v.name, error=str(e)[:200])
        net_jobs.submit("save_video", {}, run, {"ip": v.ip, "video_id": v.id, "video_name": v.name})

    def save_analysis_to_gallery(v: VideoEntry, a: Analysis) -> None:
        if gallery is None or not v.save:
            return

        def run(job):
            if v.gallery != "saved":        # the video itself could not be stored
                return
            try:
                row = gallery.save_analysis(v.id, a)
                analysis_meta[a.id] = {"from_gallery": False, "stored_at": time.time(), "saved": True}
                gallery_cache["t"] = 0.0
                monitor.event(v.ip, "analysis_saved", v.id, name=v.name, mb=round(row["blob_bytes"] / 2**20, 2))
            except SupabaseError as e:
                if "full" in str(e):
                    v.gallery = "full"
                monitor.event(v.ip, "analysis_save_failed", v.id, name=v.name, error=str(e)[:200])
        net_jobs.submit("save_analysis", {}, run, {"ip": v.ip, "video_id": v.id, "video_name": v.name})

    def register(vid: str, name: str, source: Optional[Path], folder: Path, ip: str, save: bool) -> VideoEntry:
        with registry_lock:
            v = videos.get(vid)
            if v is not None:
                videos.move_to_end(vid)
                v.save = v.save or save
                prev = get_job(v.prepare_job) if v.prepare_job else None
                if v.ready or (prev is not None and prev.state in ("queued", "running")):
                    if v.ready:
                        save_video_to_gallery(v)
                    return v
            else:
                v = VideoEntry(vid, name, source, folder, ip, save)
                videos[vid] = v
                evict()

        def run(job):
            prepare(v, job.progress)
            save_video_to_gallery(v)
        job = jobs.submit("prepare", {"prepare": (0, 100)}, run, {"ip": ip, "video_id": vid, "video_name": name})
        v.prepare_job = job.id
        return v

    def gallery_rows(max_age: float = 10.0):
        if gallery is None:
            return []
        if time.time() - gallery_cache["t"] > max_age:
            try:
                gallery_cache["rows"], gallery_cache["error"] = gallery.list(), None
            except Exception as e:
                gallery_cache["rows"], gallery_cache["error"] = [], str(e)[:300]
            gallery_cache["t"] = time.time()
        return gallery_cache["rows"]

    def check_queue(ip: str) -> None:
        if any(j.meta.get("ip") == ip for j in jobs.active(("analyze",))):
            raise HTTPException(429, "You already have a video being analysed or waiting. Wait for it to finish.")
        if len(jobs.active()) >= cfg.max_queue:
            raise HTTPException(429, "The server is busy (queue full). Try again in a few minutes.")

    def start_analysis(v: VideoEntry, params: AnalyzeParams, ip: str) -> Dict[str, Any]:
        clip = params.clip and engine.clip_model is not None
        run_params = {"max_frames": params.max_frames, "stride": params.stride, "conf": params.conf, "clip": clip,
                      "tracker": TRACKER_VERSION}
        key = (v.id, params.max_frames, params.stride, round(params.conf, 2), clip, TRACKER_VERSION)
        if not params.reanalyse and key in analysis_keys and analysis_keys[key] in analyses:
            aid = analysis_keys[key]
            return {"analysis_id": aid, "cached": True, "job_id": None, **analysis_meta.get(aid, {})}

        if not params.reanalyse and gallery is not None:
            try:
                arow = gallery.find_analysis(v.id, params_key(run_params))
            except Exception as e:
                arow = None
                print(f"[!] gallery lookup failed: {e}")
            if arow is not None:
                def load(job):
                    job.progress("download", 0, 1)
                    a = gallery.fetch_analysis(v.id, arow, v.folder, v.proxy)
                    with registry_lock:
                        analyses[a.id] = a
                        analysis_keys[key] = a.id
                        analysis_meta[a.id] = {"from_gallery": True, "stored_at": arow["created_at"]}
                        evict()
                    monitor.event(ip, "analysis_loaded", v.id, name=v.name, tracks=len(a.tracks))
                    return {"analysis_id": a.id, "from_gallery": True, "stored_at": arow["created_at"]}
                job = net_jobs.submit("load", {"download": (0, 100)}, load, {"ip": ip, "video_id": v.id, "video_name": v.name})
                return {"analysis_id": None, "cached": False, "job_id": job.id}

        check_queue(ip)
        aid = uuid.uuid4().hex[:12]

        def run(job):
            t0 = time.time()
            prepare(v, job.progress)
            a = engine.analyze(v.proxy, v.id, aid, run_params, job.progress)
            with registry_lock:
                for k, old in list(analysis_keys.items()):
                    if k == key:
                        analyses.pop(old, None)
                analyses[aid] = a
                analysis_keys[key] = aid
                analysis_meta[aid] = {"from_gallery": False}
                evict()
            job.meta["tracks"] = len(a.tracks)          # shown on the admin page
            monitor.event(ip, "analysis_done", v.id, name=v.name, tracks=len(a.tracks), seconds=round(time.time() - t0))
            save_analysis_to_gallery(v, a)
            return {"analysis_id": aid, "from_gallery": False}

        job = jobs.submit("analyze", ANALYZE_STAGES, run, {"ip": ip, "video_id": v.id, "video_name": v.name,
                                                             "reanalyse": params.reanalyse})
        monitor.event(ip, "analysis_start", v.id, name=v.name, reanalyse=params.reanalyse)
        return {"analysis_id": None, "cached": False, "job_id": job.id}

    def require_admin(request: Request) -> None:
        if not admin.valid(request.cookies.get(COOKIE)):
            raise HTTPException(401, "Admin login required")

    # ------------------------------------------------------------------ status / checkpoints
    @app.get("/api/status")
    def status():
        info = engine.checkpoint_info
        return {
            "device": str(engine.device),
            "detector": engine.detector_backend,
            "checkpoint": info,
            "checkpoints": engine.scan_checkpoints(),
            "ffmpeg": video_io.FFMPEG_EXE is not None,
            "busy": jobs.busy(),
            "palette": [hex_ for _, hex_ in RANK_COLORS],
            "default_weights": engine.default_weights,
            "max_upload_mb": max_upload_mb,
            "attributes": engine.attr_model is not None,
            "clip": engine.clip_model is not None,
            "clip_weight": CLIP_WEIGHT,
            "gallery": gallery is not None,
        }

    @app.post("/api/checkpoint")
    def switch_checkpoint(body: CheckpointBody):
        if jobs.busy():
            raise HTTPException(409, "A job is running. Wait for it to finish before switching checkpoints.")
        try:
            return {"checkpoint": engine.load_checkpoint(body.name)}
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))

    # ------------------------------------------------------------------ videos
    @app.post("/api/videos")
    def upload_video(request: Request, file: UploadFile = File(...), save: bool = Form(True)):
        ip = client_ip(request)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in video_io.VIDEO_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(video_io.VIDEO_EXTENSIONS))}")
        if not monitor.allow_upload(ip):
            raise HTTPException(429, f"Too many uploads from this address (limit {cfg.uploads_per_hour} per hour).")
        tmp = store / f"incoming_{uuid.uuid4().hex}{ext}"
        sha = hashlib.sha1()
        size, limit = 0, max_upload_mb * 1024 * 1024
        try:
            with tmp.open("wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File larger than {max_upload_mb} MB")
                    sha.update(chunk)
                    out.write(chunk)
            vid = sha.hexdigest()[:16]
            name = Path(file.filename).name
            monitor.event(ip, "upload", vid, name=name, mb=round(size / 2**20, 1), save=save)
            if vid in videos:
                tmp.unlink(missing_ok=True)
                v = get_video(vid)
                v.save = v.save or save
                if v.ready:
                    save_video_to_gallery(v)
                return v.to_dict()
            try:
                video_io.probe(tmp)
            except ValueError as e:
                raise HTTPException(422, str(e))
            folder = cache.folder(vid)
            source = folder / f"original{ext}"
            tmp.replace(source)
        finally:
            tmp.unlink(missing_ok=True)
        return register(vid, name, source, folder, ip, save).to_dict()

    @app.get("/api/videos/{vid}")
    def video_info(vid: str):
        return get_video(vid).to_dict()

    @app.get("/api/videos/{vid}/stream")
    def stream(vid: str):
        v = get_video(vid)
        if not v.ready:
            raise HTTPException(409, "Video is still being prepared")
        return FileResponse(v.proxy, media_type="video/mp4")

    # ------------------------------------------------------------------ gallery (shared, no login)
    @app.get("/api/gallery")
    def list_gallery():
        out = []
        for r in gallery_rows():
            an = r.get("analyses") or []
            out.append({"video_id": r["id"], "name": r["name"], "duration": r.get("duration"),
                        "width": r.get("width"), "height": r.get("height"), "created_at": r["created_at"],
                        "n_tracks": an[0]["n_tracks"] if an else None, "analysed_at": an[0]["created_at"] if an else None,
                        "thumb_url": f"/api/gallery/{r['id']}/thumb"})
        return {"enabled": gallery is not None, "error": gallery_cache["error"], "videos": out}

    @app.get("/api/gallery/{vid}/thumb")
    def gallery_thumb(vid: str):
        if gallery is None:
            raise HTTPException(404, "No gallery")
        try:
            data = gallery.fetch_thumb(vid, cache.folder(vid))
        except SupabaseError:
            raise HTTPException(404, "No thumbnail")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/gallery/{vid}/open")
    def open_gallery_video(vid: str, request: Request):
        if gallery is None:
            raise HTTPException(404, "No gallery")
        ip = client_ip(request)
        if vid in videos:
            v = get_video(vid)
            monitor.event(ip, "gallery_open", vid, name=v.name)
            return v.to_dict()
        row = gallery.get(vid)
        if row is None:
            raise HTTPException(404, "Video not in the gallery")
        v = VideoEntry(vid, row["name"], None, cache.folder(vid), row.get("uploader_ip") or "", save=True)
        v.gallery = "saved"
        with registry_lock:
            videos[vid] = v
            evict()
        monitor.event(ip, "gallery_open", vid, name=v.name)

        def fetch(job):
            job.progress("download", 0, 1)
            gallery.fetch_proxy(row, v.folder)
            v.info = video_io.probe(v.proxy)
            try:
                gallery.touch(vid, (row.get("opens") or 0) + 1)
            except SupabaseError:
                pass
        if v.proxy.exists() and v.proxy.stat().st_size == row["proxy_bytes"]:
            v.info = video_io.probe(v.proxy)
            threading.Thread(target=lambda: gallery.touch(vid, (row.get("opens") or 0) + 1), daemon=True).start()
        else:
            job = net_jobs.submit("fetch", {"download": (0, 100)}, fetch, {"ip": ip, "video_id": vid, "video_name": v.name})
            v.prepare_job = job.id
        return v.to_dict()

    # ------------------------------------------------------------------ analysis
    @app.post("/api/videos/{vid}/analyze")
    def analyze(vid: str, params: AnalyzeParams, request: Request):
        return start_analysis(get_video(vid), params, client_ip(request))

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request):
        job = get_job(job_id)
        if job is not None and job.meta.get("ip") not in (None, client_ip(request)):
            raise HTTPException(403, "Only the person who started this job can cancel it")
        return {"cancelled": jobs.cancel(job_id) or net_jobs.cancel(job_id)}

    @app.get("/api/analyses/{aid}")
    def analysis_info(aid: str):
        a = get_analysis(aid)
        payload = engine.tracks_payload(a)
        payload["video_name"] = videos[a.video_id].name if a.video_id in videos else ""
        payload.update(analysis_meta.get(aid, {}))
        return payload

    @app.post("/api/analyses/{aid}/search")
    def search(aid: str, body: SearchBody, request: Request):
        a = get_analysis(aid)
        t0 = time.time()
        res = engine.search(a, body.prompt)
        res["elapsed_ms"] = int((time.time() - t0) * 1000)
        res["checkpoint"] = engine.ckpt_key
        monitor.event(client_ip(request), "search", a.video_id, prompt=body.prompt[:200])
        return res

    # ------------------------------------------------------------------ admin
    @app.post("/api/admin/login")
    def admin_login(body: PasswordBody, request: Request):
        ip = client_ip(request)
        if not admin.configured():
            raise HTTPException(503, "No admin password is set (ADMIN_PASSWORD_HASH in .env)")
        if admin.locked(ip):
            raise HTTPException(429, "Too many wrong passwords. Try again in 10 minutes.")
        token = admin.login(ip, body.password)
        monitor.event(ip, "admin_login" if token else "admin_login_failed")
        if token is None:
            raise HTTPException(401, "Wrong password")
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, token, max_age=SESSION_S, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
        return resp

    @app.post("/api/admin/logout")
    def admin_logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    @app.get("/api/admin/overview")
    def admin_overview(request: Request):
        require_admin(request)
        vm, disk = psutil.virtual_memory(), shutil.disk_usage(store)
        all_jobs = sorted(jobs.list() + net_jobs.list(), key=lambda j: j.created, reverse=True)
        gal = None
        if gallery is not None:
            rows = gallery_rows(5.0)
            used = sum((r["proxy_bytes"] or 0) + (r.get("thumb_bytes") or 0)
                       + sum(x["blob_bytes"] or 0 for x in r.get("analyses") or []) for r in rows)
            gal = {"videos": len(rows), "used_bytes": used, "limit_bytes": gallery.limit_bytes,
                   "error": gallery_cache["error"]}
        return {
            "system": {"cpu_pct": psutil.cpu_percent(None), "cpus": psutil.cpu_count(),
                       "ram_used": vm.used, "ram_total": vm.total, "proc_rss": psutil.Process().memory_info().rss,
                       "disk_free": disk.free, "disk_total": disk.total, "uptime_s": round(time.time() - monitor.started),
                       "checkpoint": engine.checkpoint_info.get("name"), "device": str(engine.device)},
            "jobs": [j.admin_dict() for j in all_jobs[:60]],
            "online": monitor.online(),
            "blocked": [{"ip": ip, "reason": r} for ip, r in monitor.blocked.items()],
            "gallery": gal,
            "cache_bytes": cache.size() if gallery is not None else None,
            "limits": {"max_queue": cfg.max_queue, "uploads_per_hour": cfg.uploads_per_hour},
            "in_memory": {"videos": len(videos), "analyses": len(analyses)},
        }

    @app.post("/api/admin/jobs/{job_id}/cancel")
    def admin_cancel(job_id: str, request: Request):
        require_admin(request)
        ok = jobs.cancel(job_id) or net_jobs.cancel(job_id)
        monitor.event(client_ip(request), "admin_cancel_job", None, job=job_id, ok=ok)
        return {"cancelled": ok}

    @app.post("/api/admin/block")
    def admin_block(body: BlockBody, request: Request):
        require_admin(request)
        if body.ip == client_ip(request):
            raise HTTPException(400, "That is your own address")
        monitor.block(body.ip, body.reason)
        for j in jobs.active() + net_jobs.active():      # stop what the blocked address queued
            if j.meta.get("ip") == body.ip:
                jobs.cancel(j.id) or net_jobs.cancel(j.id)
        monitor.event(client_ip(request), "admin_block", None, target=body.ip, reason=body.reason)
        return {"ok": True}

    @app.delete("/api/admin/block/{ip}")
    def admin_unblock(ip: str, request: Request):
        require_admin(request)
        monitor.unblock(ip)
        monitor.event(client_ip(request), "admin_unblock", None, target=ip)
        return {"ok": True}

    @app.get("/api/admin/gallery")
    def admin_gallery(request: Request):
        require_admin(request)
        if gallery is None:
            return {"enabled": False, "videos": []}
        return {"enabled": True, "videos": gallery_rows(0.0), "error": gallery_cache["error"]}

    @app.delete("/api/admin/gallery/{vid}")
    def admin_delete_video(vid: str, request: Request):
        require_admin(request)
        if gallery is None:
            raise HTTPException(404, "No gallery")
        if any(j.meta.get("video_id") == vid for j in jobs.active() + net_jobs.active()):
            raise HTTPException(409, "A job is using this video; cancel it first")
        gallery.delete_video(vid)
        with registry_lock:
            v = videos.pop(vid, None)
            for key, aid in list(analysis_keys.items()):
                if key[0] == vid:
                    analysis_keys.pop(key)
                    analyses.pop(aid, None)
        shutil.rmtree(store / vid, ignore_errors=True)
        gallery_cache["t"] = 0.0
        monitor.event(client_ip(request), "admin_delete_video", vid, name=v.name if v else "")
        return {"ok": True}

    @app.post("/api/admin/gallery/{vid}/reanalyse")
    def admin_reanalyse(vid: str, request: Request):
        require_admin(request)
        if gallery is None:
            raise HTTPException(404, "No gallery")
        row = gallery.get(vid)
        if row is None:
            raise HTTPException(404, "Video not in the gallery")
        ip = client_ip(request)
        an = row.get("analyses") or []
        p = an[0]["params"] if an else {"max_frames": 0, "stride": 1, "conf": 0.35, "clip": True}
        v = videos.get(vid)
        if v is None:
            v = VideoEntry(vid, row["name"], None, cache.folder(vid), row.get("uploader_ip") or "", save=True)
            v.gallery = "saved"
            with registry_lock:
                videos[vid] = v
                evict()
        if not v.ready:
            gallery.fetch_proxy(row, v.folder)
            v.info = video_io.probe(v.proxy)
        params = AnalyzeParams(max_frames=p.get("max_frames", 0), stride=p.get("stride", 1), conf=p.get("conf", 0.35),
                               clip=p.get("clip", True), reanalyse=True)
        if len(jobs.active()) >= cfg.max_queue:
            raise HTTPException(429, "The queue is full")
        return start_analysis(v, params, ip)

    @app.post("/api/admin/password")
    def admin_password(body: NewPasswordBody, request: Request):
        require_admin(request)
        if not admin.login(client_ip(request), body.current):
            raise HTTPException(401, "Current password is wrong")
        try:
            admin.change_password(body.new)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        monitor.event(client_ip(request), "admin_password_changed")
        return {"ok": True}

    # ------------------------------------------------------------------ UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/admin", include_in_schema=False)
    def admin_page():
        return FileResponse(STATIC_DIR / "admin.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
