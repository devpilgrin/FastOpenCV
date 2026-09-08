#!/usr/bin/env bash
# FastOpenCV: сборка и запуск регрессионного тест-сьюта fastcv
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OCV="$ROOT/install/linux-dispatch"

g++ -O2 "$ROOT/fastcv/tests/fastcv_test.cpp" "$ROOT/build/fastcv/libfastcv.a" \
    -I"$ROOT/fastcv/include" -I"$OCV/include/opencv4" \
    -L"$OCV/lib" -lopencv_calib3d -lopencv_imgproc -lopencv_flann -lopencv_features2d -lopencv_core \
    -lopenblas -lfftw3f -lfftw3f_threads \
    -Wl,-rpath,"$OCV/lib" -o "$ROOT/build/fastcv/fastcv_test"

"$ROOT/build/fastcv/fastcv_test"
