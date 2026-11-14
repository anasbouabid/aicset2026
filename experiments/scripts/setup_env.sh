#!/usr/bin/env bash
# Create or update the conda env `synthfraud` from environment.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. Install Miniforge, then:" >&2
  echo "  conda env create -f $ROOT/environment.yml" >&2
  echo "  conda activate synthfraud" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx synthfraud; then
  conda env update -n synthfraud -f environment.yml --prune
else
  conda env create -f environment.yml
fi
echo "activate with: conda activate synthfraud"
