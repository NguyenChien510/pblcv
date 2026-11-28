#!/usr/bin/env bash
# EC2 idle stop for the web demo (INSTALL.md section 6I). Run every minute by pblcv-autostop.timer (as root).
# Stops this instance when
#   - the admin pressed "Tắt server" on /admin, or
#   - nobody has used the app for AUTOSTOP_IDLE_MIN minutes (deploy/.env, default 30; 0 = never) and no video is
#     being analysed or waiting.
# The app reports both at /api/power (webapp/server.py). If the app does not answer (deploy, restart, crash) nothing
# happens: the CloudWatch CPU alarm from INSTALL.md 6C stays as the backstop. "shutdown" stops an EBS-backed instance
# as long as its shutdown behaviour is "Stop" (the EC2 default; `deploy/ec2-power.sh status` shows it).
set -uo pipefail

[ -f /etc/pblcv-autostop.env ] && . /etc/pblcv-autostop.env
PBLCV_DIR="${PBLCV_DIR:-/home/ubuntu/pblcv}"
SHUTDOWN="${SHUTDOWN:-shutdown -h now}"          # overridable for testing
cd "$PBLCV_DIR/deploy" 2>/dev/null || exit 0

idle_min=$(grep -E '^AUTOSTOP_IDLE_MIN=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -dc '0-9')
idle_min="${idle_min:-30}"
up_s=$(cut -d. -f1 /proc/uptime)

json=$(docker compose exec -T app python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/power', timeout=5).read().decode())" \
  2>/dev/null) || exit 0
read -r idle_s busy requested < <(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
print(int(d.get("idle_s", 0)), int(bool(d.get("busy"))), int(bool(d.get("shutdown_requested"))))' "$json" 2>/dev/null) || exit 0

if [ "$requested" = "1" ]; then
  logger -t pblcv-autostop "stop requested from /admin: stopping the instance"
  $SHUTDOWN
# idle stop: not when switched off (0), and never in the first 10 minutes after boot (the app is still loading)
elif [ "$idle_min" -gt 0 ] && [ "$up_s" -ge "${MIN_UPTIME_S:-600}" ] && [ "$busy" = "0" ] && [ "$idle_s" -ge $((idle_min * 60)) ]; then
  logger -t pblcv-autostop "no users for $((idle_s / 60)) min (limit $idle_min) and nothing running: stopping the instance"
  $SHUTDOWN
fi
exit 0
