#!/usr/bin/env bash
# Запуск fastopencv-bench против указанной установки OpenCV
# Использование: ./scripts/run-bench.sh [dispatch|stock|native]
set -euo pipefail
VARIANT="${1:-dispatch}"
ROOT="/home/roman/workspace/FastOpemCV"
export OpenCV_DIR="$ROOT/install/linux-$VARIANT/lib/cmake/opencv4"
export OPENCV_INCLUDE_PATHS="+/opt/cuda/include"
export LD_LIBRARY_PATH="$ROOT/install/linux-$VARIANT/lib:/opt/cuda/lib64"
cd "$ROOT/rust/fastopencv-bench"
cargo build --release -q
exec ./target/release/fastopencv-bench
