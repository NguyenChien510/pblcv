#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04/24.04 EC2 instance for the web demo (INSTALL.md section 6).
# Run it on the instance as the default user:
#   curl -fsSL https://raw.githubusercontent.com/NguyenChien510/pblcv/main/deploy/ec2-setup.sh -o ec2-setup.sh
#   bash ec2-setup.sh <name>.duckdns.org      # or your own domain; no argument = <public-ip>.sslip.io
# It asks for the demo password and, for a DuckDNS name, the DuckDNS token (neither is echoed or put on a
# command line). DEMO_PASSWORD / DUCKDNS_TOKEN in the environment skip the prompts.
# Steps: Docker, 4 GB swap, repo clone, model files (~960 MB), DuckDNS updater (boot + every 5 min, so the name
# follows the new public IP after each stop/start), deploy/.env for Caddy, and the SSH key GitHub Actions deploys
# with. Re-running is safe (finished steps are skipped); run it again to change the password.
# The app itself is started by the first CI/CD run (GitHub -> Actions -> "Build and deploy web demo" -> Run workflow).
set -euo pipefail

DOMAIN="${1:-}"
REPO_URL="${REPO_URL:-https://github.com/NguyenChien510/pblcv.git}"
DIR="$HOME/pblcv"

imds() {   # EC2 instance metadata (IMDSv2)
  local tok
  tok=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  curl -fsS -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"
}

if [ -z "${DEMO_PASSWORD:-}" ]; then
  read -rsp "Password for the demo login 'demo': " DEMO_PASSWORD; echo
fi
[ -n "$DEMO_PASSWORD" ] || { echo "The password must not be empty."; exit 1; }
case "$DOMAIN" in
  *.duckdns.org)
    if [ -z "${DUCKDNS_TOKEN:-}" ]; then
      read -rsp "DuckDNS token (top of https://www.duckdns.org after signing in): " DUCKDNS_TOKEN; echo
    fi
    [ -n "$DUCKDNS_TOKEN" ] || { echo "A DuckDNS name needs its token."; exit 1; }
    ;;
esac

echo "== Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"

echo "== Swap (4 GB, guards against out-of-memory kills during analysis)"
if ! swapon --show | grep -q .; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== Code"
if [ ! -d "$DIR/.git" ]; then
  git clone --depth 1 "$REPO_URL" "$DIR"
fi

echo "== Model files (not in git, not in the image)"
dl() {   # url dest: skip files already downloaded
  if [ ! -s "$2" ]; then
    echo "   $2"
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
}
CK="$DIR/checkpoints"
mkdir -p "$CK/reid" "$CK/attributes"
dl https://github.com/NguyenChien510/pblcv/releases/download/v2.0/best.pth "$CK/best.pth"
dl https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt "$DIR/yolov8n.pt"
dl https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt "$CK/yolov8n-pose.pt"
OMZ=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1
if [ ! -s "$CK/attributes/person_attribute_infer/inference.pdmodel" ]; then   # PA-100K attributes (PaddleClas PULC)
  dl https://paddleclas.bj.bcebos.com/models/PULC/inference/person_attribute_infer.tar "$CK/attributes/par.tar"
  tar -xf "$CK/attributes/par.tar" -C "$CK/attributes" && rm "$CK/attributes/par.tar"
fi
for m in reid:person-reidentification-retail-0277 attributes:person-attributes-recognition-crossroad-0230; do
  folder="${m%%:*}"; name="${m#*:}"
  for ext in xml bin; do
    dl "$OMZ/$name/FP32/$name.$ext" "$CK/$folder/$name.$ext"
  done
done

if [ -n "${DUCKDNS_TOKEN:-}" ]; then
  echo "== DuckDNS: $DOMAIN follows this instance's public IP (at boot and every 5 minutes)"
  printf 'DUCKDNS_DOMAIN=%s\nDUCKDNS_TOKEN=%s\n' "${DOMAIN%.duckdns.org}" "$DUCKDNS_TOKEN" \
    | sudo tee /etc/duckdns.env >/dev/null
  sudo chmod 600 /etc/duckdns.env
  sudo tee /usr/local/bin/duckdns-update >/dev/null <<'EOF'
#!/usr/bin/env bash
# Points the DuckDNS name at this instance's current public IPv4 (EC2 gives a new one after every stop/start).
set -euo pipefail
. /etc/duckdns.env
tok=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
ip=$(curl -fsS -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/meta-data/public-ipv4)
# URL via curl's config on stdin so the token never appears in the process list
res=$(printf 'url = "https://www.duckdns.org/update?domains=%s&token=%s&ip=%s"\n' \
        "$DUCKDNS_DOMAIN" "$DUCKDNS_TOKEN" "$ip" | curl -fsS --retry 3 -K -)
echo "duckdns $DUCKDNS_DOMAIN -> $ip: $res"
[ "$res" = "OK" ]
EOF
  sudo chmod 755 /usr/local/bin/duckdns-update
  sudo tee /etc/systemd/system/duckdns.service >/dev/null <<'EOF'
[Unit]
Description=Point the DuckDNS name at this instance's public IP
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/duckdns-update
EOF
  sudo tee /etc/systemd/system/duckdns.timer >/dev/null <<'EOF'
[Unit]
Description=DuckDNS update at boot and every 5 minutes

[Timer]
OnBootSec=15s
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now duckdns.timer >/dev/null 2>&1
  if ! sudo systemctl start duckdns.service; then
    sudo journalctl -u duckdns.service -n 5 --no-pager || true
    echo "DuckDNS update failed: check the name ($DOMAIN) and the token, then run this script again."
    exit 1
  fi
  sudo journalctl -u duckdns.service -n 1 --no-pager -o cat
fi

echo "== deploy/.env (domain + shared login)"
if [ -z "$DOMAIN" ]; then
  DOMAIN="$(imds public-ipv4).sslip.io"
fi
hash=$(sudo docker run --rm caddy:2 caddy hash-password --plaintext "$DEMO_PASSWORD")
env_file="$DIR/deploy/.env"
keep=$(grep -E '^APP_IMAGE=' "$env_file" 2>/dev/null || true)   # keep the deployed image on a re-run
cat > "$env_file" <<EOF
DOMAIN=$DOMAIN
HTTP_PORT=80
HTTPS_PORT=443
AUTH_USER=demo
AUTH_HASH='$hash'
EOF
[ -n "$keep" ] && echo "$keep" >> "$env_file"
chmod 600 "$env_file"

echo "== SSH key for GitHub Actions"
key="$HOME/.ssh/github_deploy"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
ssh-keygen -q -t ed25519 -N '' -C "github-actions-deploy" -f "$key" <<<y >/dev/null
cat "$key.pub" >> "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"

case "$DOMAIN" in
  *.sslip.io) host="${DOMAIN%.sslip.io}"; host_note="the public IP: attach an Elastic IP, it changes on stop/start" ;;
  *)          host="$DOMAIN";             host_note="follows the instance's IP" ;;
esac
cat <<EOF

=====================================================================
 Done. Site (after the first deploy): https://$DOMAIN   login: demo
=====================================================================
In GitHub: repo Settings -> Secrets and variables -> Actions
  Variables tab: EC2_HOST = $host   ($host_note)
                 EC2_USER = $USER        (only needed if not "ubuntu")
  Secrets tab:   EC2_SSH_KEY = the whole private key below, including the BEGIN/END lines

EOF
cat "$key"
rm -f "$key"
cat <<EOF

The private key above was deleted from this server; if you lose it, run this script again for a new one.
Then: GitHub -> Actions -> "Build and deploy web demo" -> Run workflow (or push to main).
EOF
