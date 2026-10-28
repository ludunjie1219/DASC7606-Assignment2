#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; archive arguments stay relative to that directory.
exec "${A2_PYTHON:-python}" scripts/validate_submission.py "$@"
