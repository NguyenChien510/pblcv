#!/usr/bin/env bash
# Runs on the EC2 host, fed over SSH by .github/workflows/deploy.yml, which exports IMAGE (ghcr.io/...:<sha>),
# GHCR_USER, GHCR_TOKEN (the job's short-lived GITHUB_TOKEN) and REF (the commit) first.
# Pulls the new image, restarts the stack, waits for the app to answer and rolls back to the previous image if
# it does not. Keeps only the current and the previous image so the 30 GB disk does not fill up.
set -euo pipefail

cd "$HOME/pblcv"
git fetch --quiet --depth 1 origin "$REF"
git reset --quiet --hard FETCH_HEAD            # compose file and Caddyfile follow the deployed commit
cd deploy
touch .env

prev=$(grep -E '^APP_IMAGE=' .env | cut -d= -f2- || true)
set_image() { sed -i '/^APP_IMAGE=/d' .env; echo "APP_IMAGE=$1" >> .env; }

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null
trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT
echo "Pulling $IMAGE"
docker pull --quiet "$IMAGE"

set_image "$IMAGE"
docker compose up -d --no-build --remove-orphans

# The app loads BERT, CLIP and the checkpoint before it answers (about a minute on 2 vCPU).
for i in $(seq 1 60); do
  if docker compose exec -T app python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5)" >/dev/null 2>&1; then
    echo "App is up after ~$((i * 5)) s: $IMAGE"
    repo="${IMAGE%:*}"
    docker images --format '{{.Repository}}:{{.Tag}}' "$repo" | grep -vxF -e "$IMAGE" -e "${prev:-none}" \
      | xargs -r docker rmi >/dev/null 2>&1 || true
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 5
done

echo "The app did not answer within 5 minutes. Last log lines:"
docker compose logs --tail 80 app || true
if [ -n "$prev" ] && [ "$prev" != "$IMAGE" ]; then
  set_image "$prev"
  docker compose up -d --no-build
  echo "Rolled back to $prev"
fi
exit 1
