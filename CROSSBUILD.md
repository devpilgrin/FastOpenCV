# Кросс-сборки FastOpenCV

## Windows (MinGW-w64)

Требуется тулчейн (нужен sudo):

```bash
sudo pacman -S --needed mingw-w64-gcc mingw-w64-cmake
```

Далее (после установки):

```bash
mkdir -p build/win64
cmake -S third_party/opencv -B build/win64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$PWD/scripts/toolchain-mingw64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc,imgcodecs \
  -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_FFMPEG=OFF -DWITH_IPP=OFF \
  -DWITH_LAPACK=OFF -DWITH_OPENEXR=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_EXAMPLES=OFF \
  -DCPU_BASELINE=AVX2 -DCPU_DISPATCH=AVX512_SKX
cmake --build build/win64 --parallel 8
```

Особенности:
- CUDA/cuDNN/FFmpeg/LAPACK в кросс-сборке отключены (MinGW + nvcc несовместимы без MSVC;
  для CUDA под Windows - нативная сборка MSVC с теми же флагами)
- Патчи `patches/` применимы как обычно (git apply), кроме специфики toolchain
- fastcv кросс-компилируется тем же тулчейном (universal intrinsics портируемы);
  исключение: `dft_fftw.cpp` требует кросс-FFTW (отключить при необходимости)

## ARM (aarch64, на будущее)

```bash
sudo pacman -S --needed aarch64-linux-gnu-gcc
# toolchain-файл аналогично; CPU_BASELINE=NEON, диспатчинг AVX-512 неприменим.
# fastcv: universal intrinsics имеют NEON-бэкенд; VBMI-специфика (lut_8u) потребует
# адаптации (на NEON - vqtbl4q_u8).
```

Статус: путь задокументирован, фактическая кросс-сборка не верифицирована (нет тулчейна
в системе на момент записи). После установки mingw-w64-gcc - прогнать сборку и regression
сьют под wine/qemu.
