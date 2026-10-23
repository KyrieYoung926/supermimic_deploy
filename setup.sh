#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sdk_commit=814556d15970dd2ecf1c9984e845ca02ab07e206
if [[ ! -e unitree_sdk2_python/.git ]]; then
  echo "Clone the Unitree SDK yourself before running setup:" >&2
  echo "  git clone https://github.com/unitreerobotics/unitree_sdk2_python.git" >&2
  echo "  git -C unitree_sdk2_python checkout $sdk_commit" >&2
  exit 1
fi
if [[ "$(git -C unitree_sdk2_python rev-parse HEAD)" != "$sdk_commit" ]]; then
  echo "Expected Unitree SDK commit $sdk_commit. Run:" >&2
  echo "  git -C unitree_sdk2_python checkout $sdk_commit" >&2
  exit 1
fi
uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" ]]; then
  uv_bin="$PWD/.tools/uv/bin/uv"
  if [[ ! -x "$uv_bin" ]]; then
    echo "uv was not found; installing it into $PWD/.tools/uv"
    python3 -m pip install --target "$PWD/.tools/uv" uv
  fi
fi
if [[ ! -d cyclonedds/.git ]]; then
  git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git
fi
cmake -S cyclonedds -B "cyclonedds/build-$(uname -m)" -DCMAKE_INSTALL_PREFIX="$PWD/cyclonedds/install" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build "cyclonedds/build-$(uname -m)" --parallel 4
cmake --install "cyclonedds/build-$(uname -m)"
export CYCLONEDDS_HOME="$PWD/cyclonedds/install"
"$uv_bin" venv --allow-existing .venv --python 3.12
"$uv_bin" pip install --python .venv/bin/python -r requirements.txt -e unitree_sdk2_python
