#!/usr/bin/env bash
# Сборка библиотеки fastcv (статическая) под текущую машину
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
OCV="$ROOT/install/linux-dispatch"
OUT="$ROOT/build/fastcv"
mkdir -p "$OUT"

CXXFLAGS="-O3 -march=native -fPIC -I$OCV/include/opencv4 -I/usr/include/openblas -I$ROOT/fastcv/include"
for f in "$ROOT"/fastcv/src/*.cpp; do
  obj="$OUT/$(basename "${f%.cpp}").o"
  g++ $CXXFLAGS -c "$f" -o "$obj"
done
ar rcs "$OUT/libfastcv.a" "$OUT"/*.o
echo "== $OUT/libfastcv.a собрана"
