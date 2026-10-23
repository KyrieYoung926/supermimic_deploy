#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CYCLONEDDS_HOME="$PWD/cyclonedds/install"
exec .venv/bin/python deploy_real.py "$@"
