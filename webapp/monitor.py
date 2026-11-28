"""
Who is using the demo: request log per IP, people online, activity events, per-IP limits and blocked IPs.

The client IP comes from uvicorn's proxy-header handling (web_demo.py trusts X-Forwarded-For from Caddy, the only
thing that can reach the app port). Events are kept in memory and, with the gallery store, appended to Supabase
`activity_log` by a background thread (kept 30 days) so the admin page still has them after a restart.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from .supabase_client import SupabaseClient

ONLINE_WINDOW_S = 300
LOG_KEEP_DAYS = 30


class Monitor:
    def __init__(self, db: Optional[SupabaseClient], uploads_per_hour: int):
        self.db = db
        self.uploads_per_hour = uploads_per_hour
        self.started = time.time()
        self.lock = threading.Lock()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=3000)
        self.seen: Dict[str, Dict[str, Any]] = {}            # ip -> last_seen, requests, user_agent, last_path
        self.uploads: Dict[str, Deque[float]] = defaultdict(deque)
        self.blocked: Dict[str, str] = {}                    # ip -> reason
        self._pending: List[Dict[str, Any]] = []
        if db is not None:
            try:
                self.blocked = {r["ip"]: r.get("reason") or "" for r in db.select("blocked_ips", {"select": "ip,reason"})}
            except Exception as e:
                print(f"[!] Could not read blocked IPs from Supabase: {e}")
            threading.Thread(target=self._flush_loop, daemon=True, name="activity-log").start()

    # ------------------------------------------------------------------ requests and events
    def request(self, ip: str, path: str, status: int, ua: str) -> None:
        with self.lock:
            s = self.seen.setdefault(ip, {"ip": ip, "requests": 0, "first_seen": time.time()})
            s["requests"] += 1
            s["last_seen"] = time.time()
            s["user_agent"] = ua[:200]
            if not path.startswith(("/static/", "/api/jobs/")):
                s["last_path"] = path
                s["last_status"] = status

    def event(self, ip: str, action: str, video_id: Optional[str] = None, **detail: Any) -> None:
        e = {"ts": time.time(), "ip": ip, "action": action, "video_id": video_id, "detail": detail}
        with self.lock:
            self.events.append(e)
            if self.db is not None:
                self._pending.append(e)

    def online(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self.lock:
            return sorted((dict(s) for s in self.seen.values() if now - s["last_seen"] <= ONLINE_WINDOW_S),
                          key=lambda s: -s["last_seen"])

    def recent(self, ip: str = "", action: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        with self.lock:
            ev = [e for e in reversed(self.events) if (not ip or e["ip"] == ip) and (not action or e["action"] == action)]
        return ev[:limit]

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

    # ------------------------------------------------------------------ persistence
    def flush(self) -> None:
        """Pending events -> Supabase (every 5 s, and at shutdown)."""
        if self.db is None:
            return
        with self.lock:
            batch, self._pending = self._pending, []
        if batch:
            try:
                self.db.insert("activity_log", [{
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e["ts"])), "ip": e["ip"],
                    "action": e["action"], "video_id": e["video_id"], "detail": e["detail"]} for e in batch])
            except Exception as ex:
                print(f"[!] activity log not saved ({len(batch)} events): {ex}")

    def _flush_loop(self) -> None:
        last_prune = 0.0
        while True:
            time.sleep(5)
            self.flush()
            if time.time() - last_prune > 86400:
                try:
                    cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - LOG_KEEP_DAYS * 86400))
                    self.db.delete("activity_log", {"ts": f"lt.{cutoff}"})
                    last_prune = time.time()
                except Exception as ex:
                    print(f"[!] activity log prune failed: {ex}")

    def history(self, ip: str = "", action: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        """Older events from Supabase (after a restart the in-memory log is empty)."""
        if self.db is None:
            return []
        params = {"select": "*", "order": "ts.desc", "limit": str(limit)}
        if ip:
            params["ip"] = f"eq.{ip}"
        if action:
            params["action"] = f"eq.{action}"
        return self.db.select("activity_log", params)
