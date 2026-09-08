#!/usr/bin/env bash
# FastOpenCV: оптимизированная сборка OpenCV 4.14.0 (Linux x86-64, CPU+CUDA)
# Использование: ./scripts/build-linux-opt.sh [native]
#   native - вариант с CPU_BASELINE=NATIVE (только для этой машины, без диспатчинга)
set -euo pipefail

# CUDA toolkit живёт в /opt/cuda и не попадает в PATH автоматически
export PATH="/opt/cuda/bin:$PATH"

# NB: bash 5.3 не переваривает "${BASH_SOURCE[0]}" с хвостом /.. внутри "$()" - разбито на два шага
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SRC="$ROOT/third_party/opencv"
CONTRIB="$ROOT/third_party/opencv_contrib/modules"
VARIANT="${1:-dispatch}"
BUILD_DIR="$ROOT/build/linux-$VARIANT"
INSTALL_DIR="$ROOT/install/linux-$VARIANT"
# Число потоков: JOBS=8 ./scripts/build-linux-opt.sh — по умолчанию все ядра
JOBS="${JOBS:-$(nproc)}"

GENERATOR="Unix Makefiles"
command -v ninja >/dev/null && GENERATOR="Ninja"

# Базовый SIMD-профиль: бинарник работает на любом AVX2-CPU,
# AVX-512 подключается рантайм-диспатчингом
BASELINE="AVX2"
DISPATCH="AVX512_SKX"
EXTRA_OPT=( -DENABLE_FASTMATH=ON )
if [[ "$VARIANT" == "native" ]]; then
  BASELINE="NATIVE"
  DISPATCH=""
fi
if [[ "$VARIANT" == "stock" ]]; then
  # Контрольный вариант: дефолтные настройки OpenCV, как у большинства дистрибутивов
  BASELINE=""
  DISPATCH=""
  EXTRA_OPT=()
fi

# LTO + CUDA: решено патчем OpenCVDetectCUDAUtils.cmake (фильтрация -flto из
# host-флагов nvcc; fatbinData больше не попадает в LTO-IR). LTO включён всегда.
CUDA_FLAGS=()
LTO_FLAGS=( -DENABLE_LTO=ON )
if command -v nvcc >/dev/null; then
  CUDA_FLAGS=(
    -DWITH_CUDA=ON
    -DCUDA_ARCH_BIN=8.9
    -DCUDA_ARCH_PTX=
    -DCUDA_FAST_MATH=ON
    -DWITH_CUBLAS=ON
    -DWITH_CUFFT=ON
    -DOPENCV_DNN_CUDA=ON
    -DBUILD_opencv_cudafeatures2d=ON
  )
  pacman -Q cudnn >/dev/null 2>&1 && CUDA_FLAGS+=( -DWITH_CUDNN=ON )
else
  CUDA_FLAGS+=( -DWITH_CUDA=OFF )
  echo "!! nvcc не найден - сборка без CUDA"
fi

cmake -S "$SRC" -B "$BUILD_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  "${LTO_FLAGS[@]}" \
  "${EXTRA_OPT[@]}" \
  ${BASELINE:+-DCPU_BASELINE="$BASELINE"} \
  ${DISPATCH:+-DCPU_DISPATCH="$DISPATCH"} \
  -DWITH_TBB=ON -DWITH_OPENMP=OFF -DWITH_PTHREADS_PF=ON \
  -DWITH_OPENCL=ON -DWITH_VULKAN=ON \
  -DWITH_LAPACK=ON -DLAPACK_IMPL=OpenBLAS \
  -DLAPACK_LIBRARIES=/usr/lib/libopenblas.so \
  -DLAPACK_INCLUDE_DIR=/usr/include/openblas \
  -DLAPACK_CBLAS_H=cblas.h -DLAPACK_LAPACKE_H=lapacke.h \
  -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_VTK=OFF \
  -DWITH_FFMPEG=ON -DWITH_V4L=ON \
  -DWITH_IPP=OFF \
  -DOPENCV_EXTRA_MODULES_PATH="$CONTRIB" \
  -DBUILD_LIST="" \
  -DBUILD_opencv_world=OFF \
  -DBUILD_opencv_python3=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_js=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
  -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON \
  -DBUILD_PROTOBUF=ON -DPROTOBUF_UPDATE_FILES=OFF \
  -DWITH_OBSENSOR=OFF \
  "${CUDA_FLAGS[@]}"

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"
echo "== Установлено в $INSTALL_DIR"
