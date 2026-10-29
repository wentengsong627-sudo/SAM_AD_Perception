#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
SAM_AD_PYTHON="${SAM_AD_PYTHON:-python3.12}"
backend="${1:-both}"
"$SAM_AD_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Server requires Python 3.12"'
if [[ ! -x .venv/bin/python ]]; then "$SAM_AD_PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install --index-url https://pypi.org/simple --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-server.txt
.venv/bin/python -B prepare_assets.py --backend "$backend"
if [[ "$backend" == both ]]; then
  .venv/bin/python -B -m sam_ad.doctor
else
  .venv/bin/python -B -m sam_ad.doctor --backend "$backend"
fi
printf '%s\n' 'Ready. Run bash start_A.sh or bash start_D.sh in a separate terminal.'
