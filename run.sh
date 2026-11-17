#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec python src/main.py "$@"
