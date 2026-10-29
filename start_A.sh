#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/python -B -m sam_ad.server --backend A --port "${SAM_AD_PORT:-8850}" --bind "${SAM_AD_BIND:-127.0.0.1}"
