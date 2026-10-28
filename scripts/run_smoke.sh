#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
A2_DEVICE="${1:-cpu}"
"${A2_PYTHON:-python}" -m src.train --config configs/smoke.yaml --out_dir outputs/smoke --device "$A2_DEVICE"
"${A2_PYTHON:-python}" -m src.sample --checkpoint outputs/smoke/best.pt --out outputs/smoke/samples.json --device "$A2_DEVICE" --max-new-tokens 16
"${A2_PYTHON:-python}" -m src.evaluate --checkpoint outputs/smoke/best.pt --tokens data/val.bin --out outputs/smoke/evaluation.json --device "$A2_DEVICE" --max-tokens 1024
