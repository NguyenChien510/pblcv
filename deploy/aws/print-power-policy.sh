#!/usr/bin/env bash
# Run on the EC2 host: prints this instance's id, region and account, and the IAM policy for the "EC2 power"
# GitHub workflow filled in for it (INSTALL.md section 6I). Reads the instance metadata (IMDSv2); changes nothing.
#   bash ~/pblcv/deploy/aws/print-power-policy.sh
set -euo pipefail
tok=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
doc=$(curl -fsS -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/dynamic/instance-identity/document)
read -r instance region account < <(python3 -c '
import json, sys
d = json.loads(sys.argv[1]); print(d["instanceId"], d["region"], d["accountId"])' "$doc")
policy="$(dirname "$0")/ec2-power-policy.json"
cat <<INFO
GitHub -> Settings -> Secrets and variables -> Actions -> Variables:
  EC2_INSTANCE_ID = $instance
  AWS_REGION      = $region

IAM policy for the user that runs the workflow (paste into IAM -> Policies -> Create policy -> JSON):
INFO
sed -e "s/REGION/$region/" -e "s/ACCOUNT_ID/$account/" -e "s/INSTANCE_ID/$instance/" "$policy"
