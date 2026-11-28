#!/usr/bin/env bash
# Start, stop or check the EC2 host from a terminal, without logging in to AWS (INSTALL.md section 6I).
# Runs the GitHub workflow .github/workflows/ec2-power.yml through the GitHub CLI and waits for it:
#   deploy/ec2-power.sh start     # waits until the website answers (2-3 minutes)
#   deploy/ec2-power.sh stop
#   deploy/ec2-power.sh status
set -euo pipefail
action="${1:-status}"
case "$action" in start|stop|status) ;; *) echo "usage: $0 start|stop|status"; exit 1 ;; esac
command -v gh >/dev/null || { echo "Needs the GitHub CLI (gh), logged in: gh auth login"; exit 1; }

before=$(gh run list --workflow ec2-power.yml --limit 1 --json databaseId --jq '.[0].databaseId // 0')
gh workflow run ec2-power.yml -f action="$action"
echo "Requested '$action'; waiting for the workflow to start..."
for _ in $(seq 1 30); do
  id=$(gh run list --workflow ec2-power.yml --limit 1 --json databaseId --jq '.[0].databaseId // 0')
  [ "$id" != "$before" ] && break
  sleep 2
done
[ "$id" != "$before" ] || { echo "The workflow did not start; see GitHub -> Actions -> EC2 power."; exit 1; }
status=0
gh run watch "$id" --exit-status --interval 5 > /dev/null || status=$?
gh run view "$id" --log | grep -o '>> .*' | sed 's/^>> //' || true
exit $status
