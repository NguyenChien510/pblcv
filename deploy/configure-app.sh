#!/usr/bin/env bash
# Sets the web demo's own settings in deploy/.env and restarts the app (INSTALL.md section 6):
#   SUPABASE_URL, SUPABASE_SECRET_KEY   shared gallery of analysed videos (deploy/supabase/schema.sql must have run)
#   ADMIN_PASSWORD_HASH                 password of https://<domain>/admin (scrypt, same format as webapp/admin.py)
# Run on the server, again whenever you want to change them:   bash ~/pblcv/deploy/configure-app.sh
# The key and the password are read without echo and never put on a command line.
set -euo pipefail
cd "$(dirname "$0")"
env_file=.env
[ -f "$env_file" ] || { echo "deploy/.env not found: run deploy/ec2-setup.sh first."; exit 1; }

read -rp "Supabase Project URL (https://<ref>.supabase.co; empty = no gallery): " SUPABASE_URL
SUPABASE_URL="${SUPABASE_URL%/}"
SUPABASE_URL="${SUPABASE_URL%/rest/v1}"        # the dashboard's Data API page shows it with /rest/v1 appended
SUPABASE_SECRET_KEY=""
if [ -n "$SUPABASE_URL" ]; then
  read -rsp "Supabase secret key (sb_secret_..., not shown): " SUPABASE_SECRET_KEY; echo
  [ -n "$SUPABASE_SECRET_KEY" ] || { echo "A Supabase URL needs its secret key."; exit 1; }
fi
read -rsp "Password for https://.../admin (not shown): " PW; echo
read -rsp "Same password again: " PW2; echo
[ -n "$PW" ] && [ "$PW" = "$PW2" ] || { echo "The passwords are empty or different."; exit 1; }

HASH=$(PW="$PW" python3 -c '
import base64, hashlib, os, secrets
salt = secrets.token_bytes(16)
h = hashlib.scrypt(os.environ["PW"].encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
b = lambda x: base64.b64encode(x).decode()
print(f"scrypt$16384$8$1${b(salt)}${b(h)}")')
unset PW PW2

tmp=$(mktemp)
grep -vE '^(SUPABASE_URL|SUPABASE_SECRET_KEY|ADMIN_PASSWORD_HASH)=' "$env_file" > "$tmp" || true
{
  cat "$tmp"
  echo "SUPABASE_URL=$SUPABASE_URL"
  echo "SUPABASE_SECRET_KEY=$SUPABASE_SECRET_KEY"
  echo "ADMIN_PASSWORD_HASH='$HASH'"
} > "$env_file"
rm -f "$tmp"
chmod 600 "$env_file"
echo "Saved to deploy/.env"

if [ -n "$SUPABASE_URL" ]; then
  # key via curl's config on stdin, so it does not show up in the process list
  code=$(printf 'header = "apikey: %s"\n' "$SUPABASE_SECRET_KEY" \
         | curl -s -o /dev/null -w '%{http_code}' -K - "$SUPABASE_URL/rest/v1/app_settings?select=key&limit=1" || true)
  if [ "$code" = "200" ]; then
    echo "Supabase: OK"
  else
    echo "Supabase answered HTTP $code: check the URL, the secret key and that deploy/supabase/schema.sql was run."
  fi
fi

if sudo docker compose ps -q app 2>/dev/null | grep -q .; then
  sudo docker compose up -d app >/dev/null
  echo "App restarted with the new settings (it needs about a minute to load the models)."
else
  echo "The app is not running yet; the first CI/CD deploy will start it with these settings."
fi
