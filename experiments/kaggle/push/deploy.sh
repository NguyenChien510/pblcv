#!/usr/bin/env bash
# Deploy recipe m3 (phase m3_s1) to Kaggle. Needs ~/.kaggle/kaggle.json of account dorisnguyen510.
set -euo pipefail
cd "$(dirname "$0")"
K=../../../.venv/bin/kaggle
$K datasets version -p code -m "recipe m3 (phase m3_s1)"
echo "Waiting for the dataset version to be processed..."
for i in $(seq 1 40); do
  s=$($K datasets status dorisnguyen510/pblcv-code 2>&1 || true); echo "  $s"
  [[ "$s" == *ready* ]] && break; sleep 15
done
$K kernels push -p kernel
echo "Started. Status: $K kernels status dorisnguyen510/spacetime-dsca-m3"
