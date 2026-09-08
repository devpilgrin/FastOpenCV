# FastOpenCV

[![Release](https://img.shields.io/github/v/release/devpilgrin/FastOpenCV?display_name=tag&sort=semver)](https://github.com/devpilgrin/FastOpenCV/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)](fastcv/)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia)](scripts/build-linux-opt.sh)
[![AVX-512](https://img.shields.io/badge/SIMD-AVX--512%20%2B%20VBMI-orange)](fastcv/)
[![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey?logo=linux)](CROSSBUILD.md)

**English** | [Русский](README.ru.md)

Optimized OpenCV 4.14.0 build and a companion kernel library (`fastcv`) that rewrites slow/legacy OpenCV code paths. Developed and benchmarked on AMD Zen 5 (AVX-512 + VBMI, 32 threads) + NVIDIA RTX 4090 (sm_89), Arch Linux.

## What this is

- **Build system** (`scripts/`) - reproducible optimized OpenCV 4.14.0 + contrib builds: `linux-dispatch` (CUDA 13.3 sm_89, cuDNN, TBB, LAPACK/OpenBLAS, AVX2 baseline + AVX512_SKX dispatch, LTO) and `linux-stock` control build.
- **Patches** (`patches/`) - surgical git patches to OpenCV: parallel DFT/DCT (single-threaded in stock OpenCV), CUDA+LTO compatibility fix, FFmpeg 9 backports.
- **fastcv library** (`fastcv/`) - 30+ drop-in replacement kernels with the same `cv::Mat` API. Every kernel ships with before/after benchmarks and accuracy verification (bit-exact or honestly labeled approximation).
- **Benchmarks** (`bench/`) - the audit suite that produced every number below.
- **Rust bindings** (`rust/`) - opencv-rust bench harness + `fastcv-sys` FFI crate (cargo-tested).
- **Regression suite** (`fastcv/tests/`) - 14 accuracy + performance-threshold checks (`scripts/run-fastcv-tests.sh`).

## Results (1080p 8UC1 image, ms, stock OpenCV -> fastcv)

### Image filters

| Operation | Stock OpenCV | fastcv | Speedup | Accuracy |
|---|---|---|---|---|
| medianBlur k=9 | 12.9 | 0.9 | **14-17x** | bit-exact |
| Sobel k=5 | 1.89 | 0.50 | **3.8x** | bit-exact |
| Sobel k=7 | 3.91 | 0.59 | **6.7x** | bit-exact |
| Laplacian k=5 | 4.02 | 2.10 | **1.9x** | bit-exact |
| filter2D 8u->32f (7x7) | 7.2 | 0.23 | **17-32x** | maxDiff 4e-7 |
| sepFilter2D generic float | 0.96-2.35 | 0.48-0.74 | **2.0-3.2x** | maxDiff <= 1 |
| adaptiveThreshold GAUSSIAN | 2.1-3.5 | 0.23-0.67 | **5.3-9.1x** | 99.95% px identical |
| sqrBoxFilter | 0.93-0.98 | 0.45-0.71 | **1.3-2.2x** | bit-exact |
| cornerHarris | 4.8 | 0.5 | **8.5-12x** | maxDiff 4e-9 |
| bilateralFilter d=15+ | - | - | up to **13x** | approx (grid) |

### Geometry & camera

| Operation | Stock OpenCV | fastcv | Speedup | Accuracy |
|---|---|---|---|---|
| resize INTER_CUBIC | 3.4 | 0.8 | **4.1x** | bit-exact |
| resize INTER_AREA x4 | 0.92 | 0.02-0.25 | **3.5-18.7x** | +/-1 on 0.2% px |
| **undistort** (one-shot) | 9.9 | 0.87 | **11.4x** | bit-exact |
| undistort RT (precomputed maps) | 9.9 | 0.52 | **19x** | bit-exact |
| medianBlur 8UC3 k=9 | 105.7 | 6.4 | **16.6x** | bit-exact |
| Sobel 8UC3 k=5 | 5.7 | 1.4 | **3.9x** | bit-exact |
| edgePreservingFilter RECURS / NC | 174 / 496 | 14.0 / 21.8 | **12.5x / 22.7x** | maxDiff 1 |
| detailEnhance | 138 | 23.5 | **5.9x** | maxDiff 1 |

`cv::undistort` slices the frame into 2-row stripes (4096/cols) and calls `initUndistortRectifyMap`+`remap` ~540 times per frame. FastOpenCV computes maps once.

Photo-module filters (edgePreserving, detailEnhance, pencilSketch) use their own legacy scalar `Domain_Filter` (per-pixel `img.at<float>()`, 2011-era code) instead of the parallel `ximgproc::dtFilter` - fastcv re-routes them through dtFilter.

### Core math

| Operation | Stock OpenCV | fastcv | Speedup | Notes |
|---|---|---|---|---|
| gemm 1024^3 64F | 27.6 GFLOP/s | ~2600 GFLOP/s | **up to 74x** | BLAS (stock gemm is single-threaded - the cblas branch is commented out in `matmul.simd.hpp`) |
| eigenSym 256 | 249 ms | 6.8 ms | **36.6x** | LAPACK dsyev |
| SVD 256 | 95 ms | 7.6 ms | **12.5x** | LAPACK |
| DFT/DCT (native path) | - | - | **5-12x** | OpenCV patch: parallel_for per rows |
| DFT via FFTW (r2c) | - | - | **1.9-2.9x** | on top of the parallel patch |
| pow 32F (gamma) | - | - | **19-22x** | relErr <= 1e-5 |
| sortIdx by rows | 16.6 | 0.71 | **23.5x** | stock sortIdx is single-threaded |
| sum 8UC3 1080p | 0.81 | 0.03 | **26.7x** | AVX-512 vpsadbw + parallel |
| transform 32FC3 | 7.9 | 0.74 | **10.7x** | bit-exact |
| phase / cartToPolar | 0.46/0.56 | 0.10/0.17 | **4.8x/3.3x** | more accurate than cv LUT (0.004 deg vs libm) |
| LUT (VBMI vpermb) | - | - | **1.6-2.1x** | Zen 5 has VBMI; OpenCV dispatch does not enable it |
| calcHist 1D 256 | 0.33 | 0.06 | **5.0x** | bit-exact, parallel per-worker histograms |

### GPU findings (RTX 4090, resident, no transfers)

GPU is not universally faster:

| Operation | GPU | CPU (stock) | CPU (fastcv) | Winner |
|---|---|---|---|---|
| GaussianBlur 5x5 | 0.24 | 0.08 | - | **CPU 3x** |
| resize 2x | 0.027 | 0.014 | - | **CPU 2x** |
| medianBlur 9 | 7.2 | 39.1 | 2.5 | **fastcv CPU 3x vs GPU** |
| boxFilter 15x15 | 0.26 | 0.56 | - | GPU 2x |
| bilateralFilter | 0.11 | 2.36 | - | **GPU 22x** |

Rule: cheap filters -> CPU; heavy filters (bilateral, large box) -> GPU. Upload+download transfers add 0.2-2 ms.

### DNN (MobileNetV2 224x224)

| Backend | ms |
|---|---|
| OpenCV dnn CUDA FP16 | **7.34 (worse than CPU)** |
| OpenCV dnn CPU | 2.87 |
| onnxruntime-cpu | 2.46 |
| **TensorRT 11.2 FP32/TF32** | **0.166 (17x vs cv CPU)** |
| **TensorRT 11.2 FP16** | **0.117 (25x vs cv CPU, 8241 fps)** |

OpenCV's dnn CUDA backend should not be used; for GPU inference use TensorRT
(`trtexec`; TRT 11 removed `--fp16` - strongly typed networks require an FP16 ONNX,
see `scripts/onnx_to_fp16.py`).

## Honest rejections (measured, do not revisit)

boxFilter 8U/32F (parity - GCC auto-vectorizes OpenCV's scalar loops well), guidedFilter (ximgproc's fused implementation is competitive, ours was 0.8x), fastNlMeansDenoising (already parallel+SIMD), integral sqsum (memory-bound), van Herk morphology (OpenCV's O(k) SIMD is competitive up to k~63), cv::sum on small matrices (parallel overhead ~13 us; use `fastcv::sum8uSmart` dispatcher, crossover at ~2M px).

## Repository layout

```
fastcv/            fast kernel library (30+ kernels, libfastcv.a) + regression tests
scripts/           build-linux-opt.sh, build-fastcv.sh, run-bench.sh, onnx_to_fp16.py
patches/           git patches for OpenCV (DFT/DCT parallelism, CUDA+LTO, FFmpeg 9)
bench/             audit & verification benches (every number is reproducible)
rust/              opencv-rust bench harness + fastcv-sys FFI crate
PLAN.md            full work log with all measurements
QUEUE.md           analysis queue: statuses, rejects, open items
CROSSBUILD.md      Windows (MinGW) / ARM build paths
```

## Build

```bash
# Dependencies (Arch): ninja ccache cuda cudnn openblas onetbb ffmpeg fftw
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git third_party/opencv
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv_contrib.git third_party/opencv_contrib
# apply patches/ to third_party/opencv, then:
JOBS=8 ./scripts/build-linux-opt.sh dispatch    # JOBS limits parallelism
./scripts/build-fastcv.sh                        # fastcv kernel library
./scripts/run-fastcv-tests.sh                    # regression suite
```

Regression: `opencv_test_core` targeted suite 8100/8100 PASS on both dispatch and stock builds; full imgproc/core suites green with opencv_extra@4.14.0 testdata.

## License

Apache 2.0 (same as OpenCV).
