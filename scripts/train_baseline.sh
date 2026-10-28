#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
A2_DEVICE="${1:-cuda}"
if [ "$#" -gt 0 ]; then shift; fi
exec "${A2_PYTHON:-python}" -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device "$A2_DEVICE" "$@"
