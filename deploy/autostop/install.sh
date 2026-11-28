#!/usr/bin/env bash
# Installs (or updates) the EC2 idle stop: /usr/local/bin/pblcv-autostop + a systemd timer running it every minute.
# Called by deploy/remote-deploy.sh on every deploy and by deploy/ec2-setup.sh; safe to run again.
# Usage: sudo bash deploy/autostop/install.sh /home/ubuntu/pblcv
set -euo pipefail
dir="${1:?repo folder, e.g. /home/ubuntu/pblcv}"
install -m 755 "$dir/deploy/autostop/pblcv-autostop.sh" /usr/local/bin/pblcv-autostop
printf 'PBLCV_DIR=%s\n' "$dir" > /etc/pblcv-autostop.env
cat > /etc/systemd/system/pblcv-autostop.service <<'UNIT'
[Unit]
Description=Stop this EC2 instance when the web demo is idle or the admin asked for it

[Service]
Type=oneshot
ExecStart=/usr/local/bin/pblcv-autostop
UNIT
cat > /etc/systemd/system/pblcv-autostop.timer <<'UNIT'
[Unit]
Description=Check every minute whether the web demo is idle

[Timer]
OnBootSec=5min
OnUnitActiveSec=1min

[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now pblcv-autostop.timer >/dev/null 2>&1
echo "Idle stop installed (pblcv-autostop.timer, every minute)."
