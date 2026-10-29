#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/python -B -m sam_ad.server --backend D --port "${SAM_AD_PORT:-8851}" --bind "${SAM_AD_BIND:-127.0.0.1}"
