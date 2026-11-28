"""
Admin login for /admin: one password, stored as a scrypt hash (Python standard library, no extra package).

The hash lives in Supabase `app_settings` (key admin_password_hash, so the admin page can change it) and falls back
to ADMIN_PASSWORD_HASH in .env. A successful login sets an HMAC-signed session cookie (12 h). Five wrong passwords
from one IP within 10 minutes lock that IP out of the admin login for 10 minutes.

Make a hash:   python -m webapp.admin hash-password
"""
import base64
import getpass
import hashlib
import hmac
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

SESSION_S = 12 * 3600
COOKIE = "admin_session"
FAIL_LIMIT, FAIL_WINDOW_S = 5, 600
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    b = lambda x: base64.b64encode(x).decode()
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b(salt)}${b(h)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, h = stored.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(h))
    except (ValueError, TypeError):
        return False


class AdminAuth:
    def __init__(self, secret: str, env_hash: str, db=None):
        self.secret = (secret or secrets.token_hex(32)).encode()   # random: admins log in again after a restart
        self.env_hash = env_hash.strip().strip("'\"")
        self.db = db
        self.fails: Dict[str, Deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def stored_hash(self) -> str:
        if self.db is not None:
            try:
                rows = self.db.select("app_settings", {"select": "value", "key": "eq.admin_password_hash"})
                if rows and rows[0]["value"]:
                    return rows[0]["value"]
            except Exception as e:
                print(f"[!] admin password from Supabase unavailable, using ADMIN_PASSWORD_HASH: {e}")
        return self.env_hash

    def configured(self) -> bool:
        return bool(self.stored_hash())

    def locked(self, ip: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.fails[ip]
            while q and now - q[0] > FAIL_WINDOW_S:
                q.popleft()
            return len(q) >= FAIL_LIMIT

    def login(self, ip: str, password: str) -> Optional[str]:
        """Session token, or None (wrong password)."""
        if verify_password(password, self.stored_hash()):
            with self.lock:
                self.fails.pop(ip, None)
            exp = str(int(time.time()) + SESSION_S)
            return f"{exp}.{hmac.new(self.secret, exp.encode(), 'sha256').hexdigest()}"
        with self.lock:
            self.fails[ip].append(time.time())
        return None

    def valid(self, token: Optional[str]) -> bool:
        if not token or "." not in token:
            return False
        exp, sig = token.split(".", 1)
        good = hmac.new(self.secret, exp.encode(), "sha256").hexdigest()
        return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()

    def change_password(self, new: str) -> None:
        if self.db is None:
            raise RuntimeError("No Supabase store: set ADMIN_PASSWORD_HASH in .env instead")
        self.db.insert("app_settings", {"key": "admin_password_hash", "value": hash_password(new)}, upsert=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["hash-password"]:
        pw = getpass.getpass("Admin password: ")
        if not pw or pw != getpass.getpass("Again: "):
            sys.exit("Passwords empty or different.")
        print(hash_password(pw))
    else:
        sys.exit("usage: python -m webapp.admin hash-password")
