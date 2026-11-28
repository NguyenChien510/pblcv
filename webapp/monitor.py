"""
Who is using the demo: people online (last request per IP), per-IP limits and blocked IPs, for the /admin page.

The client IP comes from uvicorn's proxy-header handling (web_demo.py trusts X-Forwarded-For from Caddy, the only
thing that can reach the app port). Blocked IPs are kept in Supabase `blocked_ips` when the gallery store is on.
Events (uploads, analyses, searches...) are kept in a small in-memory ring for diagnostics only; the admin page
shows analysis jobs straight from the job managers instead of a log.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from .supabase_client import SupabaseClient

ONLINE_WINDOW_S = 300


class Monitor:
    def __init__(self, db: Optional[SupabaseClient], uploads_per_hour: int):
        self.db = db
        self.uploads_per_hour = uploads_per_hour
        self.started = time.time()
        self.lock = threading.Lock()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.seen: Dict[str, Dict[str, Any]] = {}            # ip -> last_seen, user_agent, last request
        self.uploads: Dict[str, Deque[float]] = defaultdict(deque)
        self.blocked: Dict[str, str] = {}                    # ip -> reason
        if db is not None:
            try:
                self.blocked = {r["ip"]: r.get("reason") or "" for r in db.select("blocked_ips", {"select": "ip,reason"})}
            except Exception as e:
                print(f"[!] Could not read blocked IPs from Supabase: {e}")

    # ------------------------------------------------------------------ requests and events
    def request(self, ip: str, method: str, path: str, ua: str) -> None:
        with self.lock:
            s = self.seen.setdefault(ip, {"ip": ip, "first_seen": time.time()})
            s["last_seen"] = time.time()
            s["user_agent"] = ua[:200]
            # polling (job progress, video readiness) says nothing about what the person is doing
            if not path.startswith(("/static/", "/api/jobs/")) and not (method == "GET" and path.startswith("/api/videos/")
                                                                       and path.count("/") == 3):
                s["last_method"], s["last_path"] = method, path

    def event(self, ip: str, action: str, video_id: Optional[str] = None, **detail: Any) -> None:
        with self.lock:
            self.events.append({"ts": time.time(), "ip": ip, "action": action, "video_id": video_id, "detail": detail})

    def online(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self.lock:
            return sorted((dict(s) for s in self.seen.values() if now - s["last_seen"] <= ONLINE_WINDOW_S),
                          key=lambda s: -s["last_seen"])

    # ------------------------------------------------------------------ limits
    def allow_upload(self, ip: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.uploads[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.uploads_per_hour:
                return False
            q.append(now)
            return True

    def is_blocked(self, ip: str) -> bool:
        return ip in self.blocked

    def block(self, ip: str, reason: str) -> None:
        self.blocked[ip] = reason
        if self.db is not None:
            self.db.insert("blocked_ips", {"ip": ip, "reason": reason}, upsert=True)

    def unblock(self, ip: str) -> None:
        self.blocked.pop(ip, None)
        if self.db is not None:
            self.db.delete("blocked_ips", {"ip": f"eq.{ip}"})
