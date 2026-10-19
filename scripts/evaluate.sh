#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${ASTRA_PYTHON:-python}" -B -m tactalent eval "$@"
