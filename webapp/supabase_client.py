"""
Minimal Supabase client over REST (PostgREST + Storage), server side only, with the project's secret key.

The browser never talks to Supabase: the secret key bypasses row level security and lives in the server's .env
(SUPABASE_URL, SUPABASE_SECRET_KEY). Tables have RLS on and no policies (deploy/supabase/schema.sql), so the
publishable/anon key cannot read them either. New keys (sb_secret_...) go in the `apikey` header only; a legacy
service_role JWT is also sent as the bearer token.
"""
import re
from typing import Any, Dict, List, Optional

import httpx


class SupabaseError(RuntimeError):
    pass


class SupabaseClient:
    def __init__(self, url: str, key: str, timeout: float = 60.0):
        headers = {"apikey": key}
        if key.startswith("eyJ"):                 # legacy service_role JWT
            headers["Authorization"] = f"Bearer {key}"
        # the dashboard's Data API page shows the URL with /rest/v1/ appended; the project root is what we need
        self.url = re.sub(r"/(rest|storage|auth)/v1$", "", url.strip().rstrip("/"))
        self.http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        """Every call: network errors and HTTP errors both become SupabaseError (e.g. a paused project)."""
        try:
            r = self.http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise SupabaseError(f"{method} {path}: {e}") from e
        if r.status_code >= 400:
            raise SupabaseError(f"{method} {path}: HTTP {r.status_code} {r.text[:300]}")
        return r

    # ------------------------------------------------------------------ PostgREST
    def select(self, table: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
        return self._req("GET", f"/rest/v1/{table}", params=params).json()

    def insert(self, table: str, row: Any, upsert: bool = False) -> None:
        prefer = "return=minimal" + (",resolution=merge-duplicates" if upsert else "")
        self._req("POST", f"/rest/v1/{table}", json=row, headers={"Prefer": prefer})

    def update(self, table: str, match: Dict[str, str], values: Dict[str, Any]) -> None:
        self._req("PATCH", f"/rest/v1/{table}", params=match, json=values, headers={"Prefer": "return=minimal"})

    def delete(self, table: str, match: Dict[str, str]) -> None:
        self._req("DELETE", f"/rest/v1/{table}", params=match, headers={"Prefer": "return=minimal"})

    # ------------------------------------------------------------------ Storage
    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        self._req("POST", f"/storage/v1/object/{bucket}/{path}", content=data,
                  headers={"Content-Type": content_type, "x-upsert": "true"})

    def download(self, bucket: str, path: str) -> bytes:
        return self._req("GET", f"/storage/v1/object/{bucket}/{path}").content

    def remove(self, bucket: str, paths: List[str]) -> None:
        self._req("DELETE", f"/storage/v1/object/{bucket}", json={"prefixes": paths})

    def list_objects(self, bucket: str, prefix: str, limit: int = 1000) -> List[Dict[str, Any]]:
        return self._req("POST", f"/storage/v1/object/list/{bucket}", json={"prefix": prefix, "limit": limit}).json()

    def ping(self) -> Optional[str]:
        """None when the project answers, else the error (e.g. paused after a week without activity)."""
        try:
            self.select("app_settings", {"select": "key", "limit": "1"})
            return None
        except SupabaseError as e:
            return str(e)[:300]
