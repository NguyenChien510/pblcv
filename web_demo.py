#!/usr/bin/env python3
"""
SpaceTime-DSCA web demo: find a person in any video from a text description.

  python web_demo.py                       # http://127.0.0.1:8000
  python web_demo.py --port 8080 --no-browser

The app lives in webapp/ (engine, jobs, API) and webapp/static/ (UI).
Settings come from the environment or a .env file next to this script (never committed):
  SUPABASE_URL, SUPABASE_SECRET_KEY   shared gallery of analysed videos (webapp/store.py); without them: in-memory
  ADMIN_PASSWORD_HASH                 /admin login (python -m webapp.admin hash-password)
  TRUST_PROXY=1                       behind Caddy: take the client IP from X-Forwarded-For (deploy/docker-compose.yml)
"""

import argparse
import os
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_env_file(path: Path) -> None:
    """KEY=VALUE lines into os.environ (real environment variables win)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main():
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description="SpaceTime-DSCA video person search web demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--no_clip", action="store_true", help="Do not load the CLIP scorer (saves ~600 MB RAM)")
    parser.add_argument("--max_upload_mb", type=int, default=1024, help="Upload size limit in MB")
    args = parser.parse_args()

    import uvicorn
    from webapp.engine import SearchEngine
    from webapp.server import ServerConfig, create_app

    engine = SearchEngine(ROOT, use_clip=not args.no_clip)
    app = create_app(engine, ROOT / "uploads", args.max_upload_mb, ServerConfig.from_env())

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 64)
    print(f"  SpaceTime-DSCA Person Search  ->  {url}")
    print(f"  Device     : {str(engine.device).upper()}")
    print(f"  Detector   : {engine.detector_backend}")
    print(f"  Checkpoint : {engine.checkpoint_info.get('name')}")
    print(f"  Gallery    : {'Supabase' if os.environ.get('SUPABASE_URL') else 'off (no SUPABASE_URL)'}")
    print("=" * 64 + "\n")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open_new_tab(url)), daemon=True).start()

    # Behind Caddy the app port is reachable only from Caddy, so its X-Forwarded-For is the real client IP
    # (/admin shows it, per-IP limits use it). Elsewhere trust only localhost (uvicorn's default).
    trust = "*" if os.environ.get("TRUST_PROXY") == "1" else "127.0.0.1"
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", proxy_headers=True, forwarded_allow_ips=trust)


if __name__ == "__main__":
    main()
