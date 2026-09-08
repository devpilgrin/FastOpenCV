# FastOpenCV

[![Релиз](https://img.shields.io/github/v/release/devpilgrin/FastOpenCV?display_name=tag&sort=semver)](https://github.com/devpilgrin/FastOpenCV/releases)
[![Лицензия](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)](fastcv/)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia)](scripts/build-linux-opt.sh)
[![AVX-512](https://img.shields.io/badge/SIMD-AVX--512%20%2B%20VBMI-orange)](fastcv/)
[![Платформа](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey?logo=linux)](CROSSBUILD.md)

[English](README.md) | **Русский**

Оптимизированная сборка OpenCV 4.14.0 и библиотека кернелов `fastcv`, переписывающая медленные/устаревшие кодовые пути OpenCV. Разработано и измерено на AMD Zen 5 (AVX-512 + VBMI, 32 потока) + NVIDIA RTX 4090, Arch Linux.

## Что внутри

- **Сборочная система** (`scripts/`) - воспроизводимые сборки OpenCV 4.14.0 + contrib: `linux-dispatch` (CUDA 13.3 sm_89, cuDNN, TBB, LAPACK/OpenBLAS, AVX2 + диспатчинг AVX512_SKX, LTO) и контрольная `linux-stock`.
- **Патчи** (`patches/`) - хирургические git-патчи к OpenCV: параллельный DFT/DCT (в стоке однопоточный), совместимость CUDA+LTO, бэкпорты для FFmpeg 9.
- **Библиотека fastcv** (`fastcv/`) - 30+ кернелов-замен с тем же API `cv::Mat`. Каждый кернел снабжён бенчмарком до/после и проверкой точности (bit-exact или честно помеченная аппроксимация).
- **Бенчмарки** (`bench/`) - аудит-сьют, воспроизводящий каждое число ниже.
- **Rust-биндинги** (`rust/`) - бенч-харнес opencv-rust + FFI-крейт `fastcv-sys` (проверен cargo-тестами).
- **Регрессионный сьют** (`fastcv/tests/`) - 14 проверок точности и порогов производительности (`scripts/run-fastcv-tests.sh`).

## Результаты (1080p 8UC1, мс, стоковый OpenCV -> fastcv)

### Фильтры изображений

| Операция | OpenCV сток | fastcv | Ускорение | Точность |
|---|---|---|---|---|
| medianBlur k=9 | 12.9 | 0.9 | **14-17x** | побитово |
| Sobel k=5 / k=7 | 1.89 / 3.91 | 0.50 / 0.59 | **3.8x / 6.7x** | побитово |
| Laplacian k=5 | 4.02 | 2.10 | **1.9x** | побитово |
| filter2D 8u->32f | 7.2 | 0.23 | **17-32x** | maxDiff 4e-7 |
| sepFilter2D generic | 0.96-2.35 | 0.48-0.74 | **2.0-3.2x** | maxDiff <= 1 |
| adaptiveThreshold GAUSSIAN | 2.1-3.5 | 0.23-0.67 | **5.3-9.1x** | 99.95% пикселей |
| sqrBoxFilter | 0.93-0.98 | 0.45-0.71 | **1.3-2.2x** | побитово |
| cornerHarris | 4.8 | 0.5 | **8.5-12x** | maxDiff 4e-9 |

### Геометрия и камера

| Операция | OpenCV сток | fastcv | Ускорение |
|---|---|---|---|
| resize CUBIC | 3.4 | 0.8 | **4.1x** (побитово) |
| resize AREA x4 | 0.92 | 0.02-0.25 | **3.5-18.7x** |
| **undistort** | 9.9 | 0.87 | **11.4x** (побитово) |
| undistort RT (карты заранее) | 9.9 | 0.52 | **19x** |
| medianBlur 8UC3 k=9 | 105.7 | 6.4 | **16.6x** (побитово) |
| edgePreservingFilter RECURS / NC | 174 / 496 | 14.0 / 21.8 | **12.5x / 22.7x** |
| detailEnhance | 138 | 23.5 | **5.9x** |

`cv::undistort` режет кадр на полосы по 2 строки (4096/cols) и вызывает `initUndistortRectifyMap`+`remap` ~540 раз на кадр. FastOpenCV строит карты один раз.

Фильтры photo-модуля используют собственный скалярный `Domain_Filter` (per-pixel `img.at<float>()`, код 2011 года) вместо параллельного `ximgproc::dtFilter` - fastcv перенаправляет их на dtFilter.

### Математика ядра

| Операция | OpenCV сток | fastcv | Ускорение |
|---|---|---|---|
| gemm 1024^3 64F | 27.6 GFLOP/s | ~2600 GFLOP/s | **до 74x** |
| eigenSym 256 | 249 мс | 6.8 мс | **36.6x** |
| SVD 256 | 95 мс | 7.6 мс | **12.5x** |
| DFT/DCT (патч параллелизма) | - | - | **5-12x** |
| DFT через FFTW (r2c) | - | - | **1.9-2.9x** поверх патча |
| pow 32F | - | - | **19-22x** |
| sortIdx по строкам | 16.6 | 0.71 | **23.5x** |
| sum 8UC3 1080p | 0.81 | 0.03 | **26.7x** |
| transform 32FC3 | 7.9 | 0.74 | **10.7x** |
| phase / cartToPolar | 0.46/0.56 | 0.10/0.17 | **4.8x/3.3x** |
| calcHist | 0.33 | 0.06 | **5.0x** |

### Главные находки аудита

1. **gemm в OpenCV однопоточный** - ветка cblas закомментирована в `matmul.simd.hpp` (`/*if(`). 27 GFLOP/s на CPU, способном на 2600
2. **cv::undistort вызывает remap 540 раз на кадр** (полосы по 2 строки)
3. **adaptiveThreshold GAUSSIAN**: convertTo(32F) -> GaussianBlur на float (мимо быстрого fixed-point пути) -> convertTo(8U) -> скалярный однопоточный LUT
4. **Sobel/Laplacian**: ручной SIMD только для k=3, большие ядра проваливаются в generic float
5. **GPU не универсален**: gaussian и resize быстрее на CPU (3x и 2x), fastcv medianBlur на CPU быстрее RTX 4090 в 3 раза; GPU оправдан только на тяжёлых фильтрах (bilateral 22x)
6. **OpenCV dnn CUDA-бэкенд хуже CPU** (7.3 мс против 2.9) и уступает TensorRT 11.2 FP16 (**0.117 мс, 8241 fps**) в 63 раза

### Честные отказы (с замерами)

boxFilter 8U/32F (паритет - GCC хорошо автовекторизует скалярные циклы OpenCV), guidedFilter (реализация ximgproc конкурентна), fastNlMeansDenoising (уже параллелен и SIMD), integral sqsum (memory-bound), морфология van Herk (O(k) SIMD OpenCV конкурентен до k~63), cv::sum на малых матрицах (оверхед параллелизма ~13 мкс; есть диспетчер `fastcv::sum8uSmart`).

### Сборка

```bash
# Зависимости (Arch): ninja ccache cuda cudnn openblas onetbb ffmpeg fftw
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git third_party/opencv
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv_contrib.git third_party/opencv_contrib
# наложить patches/ на third_party/opencv, затем:
JOBS=8 ./scripts/build-linux-opt.sh dispatch    # JOBS ограничивает параллелизм
./scripts/build-fastcv.sh                        # библиотека fastcv
./scripts/run-fastcv-tests.sh                    # регрессионный сьют
```

Регрессия: целевой сьют `opencv_test_core` 8100/8100 PASS на обеих сборках; полные core/imgproc сьюты зелёные с testdata opencv_extra@4.14.0.

### Лицензия

Apache 2.0 (как у OpenCV).
