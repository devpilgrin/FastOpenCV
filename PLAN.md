# FastOpenCV — план работ

Цель: максимально оптимизированная сборка OpenCV 4.14.0 (CPU+GPU) + Rust-обёртка.
Платформы: Linux x86-64 (основная), Windows, ARM/embedded (кросс-сборки позже).

## Этап 0. Окружение
- [x] Разведка: Ryzen 9 9950X3D (AVX-512), RTX 4090 (CUDA 13.3, sm_89), 128 ГБ RAM
- [x] ninja 1.13.2, ccache 4.14, cuda 13.3, cudnn 9.25.1 установлены
- [x] Клонирование opencv 4.14.0 + opencv_contrib 4.14.0 (shallow)
- [x] Скрипт scripts/build-linux-opt.sh (bash 5.3: нельзя "${BASH_SOURCE[0]}/.." внутри "$()")
- [x] Сборка linux-dispatch: УСПЕХ (CUDA 13.3+cuDNN 9.25.1+TBB+AVX512_SKX dispatch; LTO off из-за fatbin)
- [x] FFmpeg 9.0.1 несовместим с 4.14 - наложены патчи PR #29662 (supported_framerates) и hw.hpp из PR #29533 (pix_fmts); лежат в patches/
- [x] opencv_perf_core и opencv_perf_cudaarithm - работают, GPU отвечает
- [x] Rust-crate fastopencv-bench: собран, CPU+CUDA+DNN бенчи работают из Rust

## Результаты бенчмарков (07.09.2026, эта машина)
| Операция | stock | dispatch (opt) | CUDA |
|---|---|---|---|
| GaussianBlur 9x9 1080p | 0.353 | 0.32 | 0.16-1.1 (шумно) |
| DFT 2048x2048 32F | 28.1 | 29.1 | - |
| GEMM 1024x1024 64F | 68.4 | 68-76 | - |
| Resize 4K->720p AREA | 0.87 | 0.89 | - |
| cvtColor 4K BGR2GRAY | 0.112 | 0.10-0.13 | - |
| Sobel 3x3 1080p | 0.73 | 0.74 | - |
| DNN MobileNetV2 CPU | 2.94 (dispatch) | - | FP32 2.44 / FP16 1.58 |

ВЫВОД: stock уже включает диспатчинг AVX-512 => флаги дают ±5% (шум). Реальные рычаги:
1) алгоритмическая перепись кернелов (подтверждает историю про 40x),
2) DNN: OpenCV CUDA-backend слаб (1.9x) - для RT-инференса смотреть TensorRT/свои кернелы,
3) узкие места I/O и пайплайна.
- [!] Грабля: GCC LTO + CUDA несовместимы - fatbinData дублируется при слиянии LTO-модулей. LTO включается только в сборках без CUDA
  РЕШЕНО (этап 6b): патч cmake/OpenCVDetectCUDAUtils.cmake (patches/fastopencv-cuda-lto.patch) -
  -flto вырезается из host-флагов nvcc в ocv_cuda_filter_options. LTO теперь включён и с CUDA.
  Верифицировано: сборка+install успешны, nvcc-команды без -flto, C++ с -flto=auto,
  lena_fastcv OK, cudaarithm 192/192, test_core targeted 610/610.

## Этап 1. Оптимизированная C++ сборка (Linux, эта машина)
Ключевые флаги CMake:
- CMAKE_BUILD_TYPE=Release, ENABLE_LTO=ON (thin), ENABLE_FASTMATH=ON
- CPU_BASELINE=AVX2, CPU_DISPATCH=AVX512_SKX (бинарник переносимый, диспатчинг в рантайме)
- WITH_TBB=ON (уже стоит onetbb), WITH_OPENMP=OFF (TBB предпочтительнее)
- WITH_CUDA=ON, CUDA_ARCH_BIN=8.9, WITH_CUDNN=ON, OPENCV_DNN_CUDA=ON
- WITH_OPENCL=ON, WITH_VULKAN=ON
- BUILD_opencv_world=OFF (модульно — быстрее линковка), BUILD_TESTS/PERF_TESTS=ON (для бенчей)
- ccache + ninja
- Отдельный вариант native: CPU_BASELINE=NATIVE для сравнения

## Этап 2. Бенчмарки
- opencv_perf_core / opencv_perf_imgproc / opencv_perf_dnn: stock vs optimized
- Собственный микро-бенч: типовой RT-пайплайн (decode -> resize -> blur -> dnn infer)

## Этап 3. Rust-обёртка
- Проверить совместимость crate `opencv` (twistedfall) с 4.14; иначе — свой генератор биндингов
- crate fastopencv-sys (линковка с нашей сборкой) + идиоматичный safe-слой
- cargo bench против C++ бенчей (overhead FFI)

## Этап 4. Кросс-платформенность
- Windows: toolchain-файл, MSVC/Clang-cl
- ARM/embedded: Linux aarch64 toolchain, NEON, урезанные модули по необходимости

## Заметки
- Ожидание "40x": реалистично 1.2-3x от флагов компилятора; большие кратности - от CUDA-диспатчинга DNN, алгоритмических замен и устранения I/O-узких мест. Измеряем, не верим на слово.
- Подтверждение от сэра: у знакомых 40x дал именно пересмотр алгоритмов/вычислений в старом коде. Поэтому добавляем:

## Этап 5. Алгоритмическая оптимизация - результаты (fastcv/)
| Кернел | OpenCV | fastcv | Speedup | Точность |
|---|---|---|---|---|
| cornerHarris 8UC1 b=3/5/7 | 4.0/5.6/6.4 ms | 0.48/0.61/0.53 ms | 8.5-12x | exact (1e-9) |
| medianBlur 8UC1 k=7/9/15/25 | 37/38/33/29 ms | 2.2-2.4 ms | 14-17x | bit-exact |
| bilateralFilter 8UC1 d=9 | 2.3 ms | 2.8 ms | 0.83x | ТОЧНЕЕ OpenCV (у него квантование, maxDiff 24 vs эталон) |

Ключевые приёмы:
- cornerHarris: fusion Sobel->cov->boxSum->R по кольцевому буферу (без промежуточных матриц), границы boxSum - отражение ЗНАЧЕНИЙ cov (не пикселей!)
- medianBlur: путь O(1) у OpenCV ОДНОПОТОЧНЫЙ - добавлен parallel_for по полосам + их же column-histogram machinery; раскладка h_fine [bin][col][lo]
- bilateral: gather-LUT быстрее poly-exp (2x), но unrolled-симметрия OpenCV сильна; дальше - только bilateral grid (O(1) в d)

Грабли universal intrinsics для external-кода:
- #include <opencv2/core/simd_intrinsics.hpp> ПЕРВЫМ (иначе 128-бит эмуляция молча)
- vx_load_expand/vx_load для wide-типов; v_int16: v_add_wrap/v_sub_wrap; operator* на v_float32 ненадежен - использовать v_mul/v_add
- v_lut(float[256], idx) - gather, работает

Следующие кандидаты: filter2D 7x7 (7.2ms), resize INTER_CUBIC (3.4ms), warpAffine (3.0ms), bilateral через grid.

## Этап 5b. Аудит core (core_audit.cpp, 1080p)
| Операция | OpenCV | Вывод |
|---|---|---|
| sum 8UC3 | 0.806 ms (7.7 GB/s) | ОДНОПОТОЧНЫЙ -> fastcv 26.7x (0.030 ms), exact |
| LUT 8UC1/C3 | 0.039/0.101 ms | CV_LUT8U_SIMD=0 на x86 без VBMI; VBMI есть на Zen5, но нет в dispatch 4.14 -> fastcv vpermb 1.6-2.1x |
| add/absdiff/convert/divide 130-157 GB/s | у памяти | ок |
| exp/log 32F | ~26 GB/s | compute-bound, умеренный запас |
| split/merge | ~53 GB/s | ~2x запас (deinterleave) |
| minMaxLoc | 66 GB/s | ~1.5x запас |
| equalizeHist | 10 GB/s | уже parallel (2 фазы), не трогаем |

Новые кернелы: fastcv/src/sum_8u.cpp (SAD vpsadbw + parallel), fastcv/src/lut_8u.cpp (VBMI 4x vpermb + blend).
fastcv собирается как libfastcv.a: scripts/build-fastcv.sh, заголовок fastcv/include/fastcv/fastcv.hpp.

## Этап 5c. Матричные алгоритмы (matrix_audit.cpp) - подозрение сэра подтверждено
OpenCV 4.14 БЕЗ LAPACK (исходное состояние):
- gemm: ~26 GFLOP/s полка (однопоточный blocked 128x128; cblas-ветка для мелких закомментирована)
- eigen(sym) n=1024: 30.8 СЕК (внутренний Jacobi; LAPACK syev в OpenCV отсутствует, только Eigen-библиотека)
- SVD n=1024: 4.75с; invert 1.65с

ПОСЛЕ -DWITH_LAPACK=ON + OpenBLAS 0.3.34 (LAPACK_IMPL=OpenBLAS + явные LAPACK_LIBRARIES/INCLUDE - автодетект ломается на OpenMP:: импортах):
- gemm 2048 32F: 624ms -> 9.2ms (1871 GFLOP/s, 74x) - теперь cv::gemm ходит в BLAS сам
- invert(LU) 1024: 1654 -> 45.5ms (36x); SVD 1024: 4750 -> 577ms (8.2x)
- eigen: LAPACK-пути НЕТ -> fastcv::eigenSym (dsyev): 10.6x/51x/60x (128/512/1024), residual 1e-15
- регрессия мелких gemm (256³: 1.3 -> 3.2ms из-за thread sync) -> fastcv::gemm адаптивные потоки (<6e7 FLOP: 1 поток, <4e8: 8): 256³ = 0.12ms
- OpenBLAS dsyev мелкий: 117ms -> 3.7ms в 1 поток (n=128) - то же лечение

## Этап 5d. Вторая волна аудита core (core_audit2.cpp)
Проверены: reduce, min/compare/inRange, phase/magnitude/cartToPolar/polarToCart, dct, patchNaNs,
transform, mulSpectrums, матричные выражения, determinant, trace, invert SVD/CHOLESKY, solve QR,
eigenNonSymmetric, solvePoly, PCA, calcCovarMatrix, kmeans.

Аномалии и действия:
- transform 32FC3 4x4: 8.8ms однопоточный (NAryMatIterator, нет parallel_for) -> fastcv::transform32f: 0.74ms, 10.7x, bit-exact
- invert DECOMP_SVD 64x64: 1.58ms/вызов - это синхронизация потоков OpenBLAS в gesdd; мелкие LAPACK-вызовы требуют адаптивных потоков (как eigenSym)
- dct 1024x1024: 14.6ms - умеренный запас (~3-5x), свой DCT
- polarToCart: 1.0ms - скалярный sincos, ~3x запас
- остальное в норме: reduce 0.056ms, compare 0.12ms, PCA 4.3ms, kmeans ок, determinant 9мкс

## Этап 5e. imgproc-кандидаты + КРИТИЧЕСКАЯ находка setNumThreads(0)
ЛОВУШКА: setNumThreads(0) в связке OpenCV 4.14+TBB даёт 4x ЗАМЕДЛЕНИЕ параллельных
операций (default/-1/16/32 - быстрые). Первый kernel_audit был снят в этом режиме -
таблица переизмерена. Бенчи fastcv всегда шли в default-режиме, сравнения честные.

Исправленная таблица кандидатов (default-режим, 1080p):
- warpAffine 0.39ms, remap 0.22ms, GaussianBlur 25x25 0.61ms, pyrDown 0.19ms - НОРМА (сняты с очереди)
- resize INTER_CUBIC 0.88ms -> fastcv::resizeCubic8u (двухфазный, gather для C1): downscale 1.5-1.9x, bit-exact; upscale 0.7-0.9x (cv лучше - не использовать для upscale)
- filter2D 8UC1->32F: FilterNoVec! k=7: 7.28 -> 0.42ms (17.6x), k=11: 18.7 -> 0.58ms (32x), rel 1e-7 (FMA vs mul+add)
- dilate ellipse 15x15: 1.65ms - несепарабельный ядро; запас требует rect-декомпозиции (approx)
- bilateral: ждёт bilateral grid

Итого fastcv: 10 кернелов. libfastcv.a актуальна.

## Этап 5g. Третья волна core (core_audit3.cpp)
Проверены: sort, sortIdx, dft/idft (1024/2048, real/complex), dot, norm+маска, hconcat/vconcat,
repeat, rotate, copyMakeBorder, clone, setTo, pow (спец/общий), cubeRoot, max, bitwise.

Результаты:
- sortIdx rows: 16.6ms однопоточный -> fastcv::sortIdxRows32f 0.71ms (23.5x), значения совпадают
- sort rows: cv уже параллелен (2.4ms) - parity, наш вариант не нужен
- DFT: native-путь однопоточный (parallel_for только в IPP-ветках, IPP=off). dft 1024 c2c 20ms,
  2048 r2c 32ms. БОЛЬШАЯ работа: параллелить проходы строк/столбцов (~4-8x) или FFTW-class rewrite
- pow общий показатель (1.7): 1.54ms - ~3x запас через векторный exp2/log2
- hconcat/vconcat/repeat/rotate/clone/setTo/bitwise - у памяти, ок
- norm с маской, dot - ок

## Этап 5h. DFT/DCT/pow/bilateral-grid/ellipse (по приказу сэра)
DFT/DCT - патч исходников OpenCV (patches/fastopencv-dxt-parallel.patch, бэкап dxt.cpp.orig):
- rowDft/colDft параллелизированы (per-worker буферы); DCT-цикл аналогично
- КРИТИЧНО: RealDFT/CCSIDFT мутировали shared c.factors[0] (>>=1 / <<=1) - race при параллелизме;
  исправлено локальными копиями factors. Без этого - heap corruption (поймано coredump)
- Результаты: dft 1024 r2c 6.5->0.62ms (10.5x), c2c 20.4->2.05ms (10x), 2048 r2c 32.4->6.0ms (5.4x),
  idft 13.6->1.78ms (7.6x), dct 1024 14.6->1.16ms (12.6x). Выводы ПОБИТОВО идентичны (dft_ref diff)
- pow общий показатель: fastcv::pow32f 19-22x, rel err ~1e-5 (atanh-log2 + exp2 полиномы)
- bilateral grid: АППРОКСИМАЦИЯ. splat параллелен (приватные сетки+слияние). d=15: 1.8x, d=25: 4.5x,
  meanDiff ~5.4 при sigma=50. Рекомендация: для больших d/препроцессинга; точный - bilateralFilter8u
- ellipse-морфология: зонотоп 4-сегментный. Прямой van Herk по диагоналям - cache-hostile (отклонён),
  shear-вариант выигрывает только при k>=31 (1.6x), ниже - проигрыш точному. Честно помечено как approx.

## Этап 5i. Систематический разбор фильтров (filter_audit2.cpp)
Все "провалы" imgproc-тестов (sepFilter2D, StackBlur, GaussianBlur_Bitexact) - отсутствие
testdata (baboon/fruits/lena), НЕ баги. opencv_extra скачан sparse; с тегом 4.14.0 все
13 бывших провалов (core+imgproc) стали PASSED (base64-ошибка была рассинхроном
testdata master vs код 4.14). РЕГРЕССИЙ НЕТ, окружение тестов починено.

Скоростная карта (1080p, default-режим):
- ЗДОРОВЫ: boxFilter (ColumnSum O(1); GCC автовекторизует - моя перепись дала parity/хуже,
  ОТКЛОНЕНА как кандидат), GaussianBlur все пути, stackBlur, Sobel 3x3/Scharr, Laplacian k1
- АНОМАЛИИ ИСПРАВЛЕНЫ: Sobel 5x5/7x7 (1.9/2.6ms; спецпуть только 3x3) -> fastcv::sobel8u16sLarge
  3.8x/6.7x bit-exact; Laplacian k5 (4.0ms) -> fastcv::laplacian8u16sLarge 1.9x bit-exact
  (k=7 у cv 32F-путь - не трогаем); adaptiveThreshold GAUSS -> fastcv::adaptiveThreshold8u
  5.3-9.1x (cv: convertTo(32F)+float-GaussianBlur+скалярный LUT; наш: fixed-point + parallel LUT;
  99.92-99.97% пикселей совпадают - boundary-флипы +-1, честная approx)
- АНОМАЛИИ ОТКРЫТЫ: sepFilter2D generic float 9x9 (0.88ms), sqrBoxFilter (0.92ms - отдельный путь),
  adaptiveThreshold GAUSS 25 (3.6ms - наследует boxFilter+порог), guidedFilter (20ms - 4+ boxFilter),
  fastNlMeansDenoising (75ms), edgePreservingFilter/detailEnhance (138-166ms, photo-модуль, ниша)
- Урок: GB/s-метрика аудита вводит в заблуждение для фильтров с реальной работой; GCC -O3
  автовекторизует простые скалярные циклы OpenCV неплохо. Проверять каждого кандидата замером.

## Этап 5j. Исполнение QUEUE.md (все семейства)
ГОТОВО (новые кернелы fastcv):
- sepFilter2D_8u: 2.0-3.2x (8U maxDiff<=1 FMA, 32F ~1e-4)
- sqrBoxFilter8u: 1.3-2.2x bit-exact
- calcHist8u: 5.0x bit-exact (параллельные per-worker гистограммы)
- resizeAreaInt8u: x4 18.7x (C1) / 3.5x (C3), maxDiff 1 на 0.2% px; x2 - cv быстрее
- phase32f/cartToPolar32f: 4.8x/3.3x, точность vs libm 0.004 град (cv сам LUT-approx)
- sum8uSmart: адаптивный диспетчер cv::sum (<2M px) / fastcv (>=2M; crossover замерен:
  parallel-оверхед ~13 мкс, cv C1 SIMD до 1024^2 быстрее)
- laplacian8u16sLarge k=5: 1.9x bit-exact (k=7 у cv float-путь - не трогаем)
- adaptiveThreshold8u GAUSS: 5.3-9.1x (99.92-99.97% px совпадают)

АУДИТЫ БЕЗ НАХОДОК (cv здоров): warpAffine/warpPerspective/remap/rotate (0.24-0.73 мс),
cvtColor все (0.03-0.29 мс, до 278 GB/s), demosaic, split/merge, integral plain (80 GB/s),
CLAHE, resize AREA x2/LINEAR.

ОТКЛОНЕНО С ЗАМЕРАМИ: boxFilter 8U (паритет), boxFilter 32F (паритет), guidedFilter (0.8x -
ximgproc слитая реализация хороша), NLM (parallel+SIMD уже), integral sqsum (memory-bound:
double выход 8B/px), van Herk морфология (cv O(k) SIMD конкурентен до k~63; моя 0.1-0.3x).

CUDA АУДИТ (1080p 8UC1, RTX 4090, resident): gaussian5 GPU 0.241 vs CPU 0.076 (CPU!),
resize2x GPU 0.027 vs CPU 0.014 (CPU!), median9 GPU 7.2 vs cv 39 vs FASTCV 2.5 (CPU быстрее
RTX 4090!), box15 GPU 0.26 vs CPU 0.56 (GPU), bilateral GPU 0.107 vs CPU 2.36 (GPU 22x).
Правило: дешёвые фильтры -> CPU, тяжёлые (bilateral, большие box) -> GPU; end-to-end
трансферы upload+download добавляют ~0.2-2 мс.

Открыто: DNN CUDA-бэкенд (1.9x - TensorRT-путь, исследование); P3-ниши по запросу.
fastcv: 23 кернела в libfastcv.a.

## Этап 5k. P3-ниши + DNN-исследование
UNDISTORT: cv::undistort stripe_size0 = 4096/cols = 2 строки на 1080p -> initUndistortRectifyMap+remap
вызываются 540 раз на кадр (9.9 мс). fastcv::undistort8u one-shot 0.87 мс (11.4x bit-exact),
RT с прекомпьютом карт 0.52 мс (19x). Это самый частый паттерн в видеопайплайнах с камерой.
PHOTO: edgePreserving 164/475 мс, detailEnhance 136, pencilSketch 126, stylization 501 -
тяжёлы по природе (domain transform внутри); путь улучшения = оптимизация DTF (1-2 дня, 2-4x),
отложено как ниша. inpaint Telea 5.2 мс - ок. YUV/YCrCb/XYZ конверсии здоровы (0.04-0.42 мс).
DNN (MobileNetV2 224x224): OpenCV CPU 2.87 мс, OpenCV CUDA FP16 7.3 мс (хуже CPU!),
onnxruntime-cpu 1.29 2.46 мс. TensorRT в Arch нет (AUR/NVIDIA repo); ожидаемый TRT-выигрыш
на RTX 4090 ~0.3-0.5 мс. Вывод: OpenCV dnn CUDA-бэкенд не использовать; для GPU-инференса -
onnxruntime+TensorRT EP или TRT напрямую (требует установки libnvinfer).
fastcv: 24 кернела.

## Этап 5l. Мультиканальность, FFTW, FFI, photo, регрессия
- C3: medianBlur8uC3 16.6x bit-exact (split+3xC1; НЕ параллелить по каналам - вложенный
  parallel_for сериализуется!); sepFilter2D_8u C3 2.0x (плоский проход W*cn, тапы шагом cn);
  sobel8u16sLarge C3 3.9x bit-exact (тот же трюк)
- DFT через FFTW (fastcv/src/dft_fftw.cpp): r2c 1.9x (1024) / 2.9x (2048) к нашему патчу;
  интерьер maxDiff 1.5e-4. ГРАБЛЯ: FFTW_MEASURE перезаписывает данные планируемых массивов -
  планировать на scratch + FFTW_UNALIGNED + new-array execute; планировщик не thread-safe (мьютекс)
- photo-стилизации (photo_stylize_fast.cpp): photo-модуль использует собственный legacy
  Domain_Filter (npr.hpp, img.at<float>() скалярно, 2011 г.) вместо параллельного
  ximgproc::dtFilter! edgePreserving RECURS 174->14.0 мс (12.5x), NC 496->21.8 (22.7x),
  detailEnhance 138->23.5 (5.9x), везде maxDiff 1
- C API (fastcv_c.h/fastcv_c.cpp) + rust/fastcv-sys: FFI-биндинги, 4/4 cargo test PASSED
- Регрессионный сьют fastcv/tests/fastcv_test.cpp (14 проверок, включая порог perf): ALL PASS
  (scripts/run-fastcv-tests.sh)
- CROSSBUILD.md: пути Windows (mingw-w64, CUDA off) и ARM (NEON, lut_8u требует адаптации)

## Этап 6. Регрессионная верификация (bench/test_*.txt)
- opencv_test_core ЦЕЛЕВОЙ (DFT/DCT/Mat/Gemm/LU/SVD/Eigen): 8100/8100 PASS на dispatch И stock
- opencv_test_core ПОЛНЫЙ: 12415 OK (dispatch) / 12405 OK (stock), провалы ИДЕНТИЧНЫ: 7 шт.,
  все - отсутствие opencv_extra testdata (filestorage base64, globbing) - не наши патчи
- opencv_test_imgproc (Filter/Resize/Smooth/Median/Morph/Blur/Hist): 7019 OK на обеих,
  6 провалов идентичны (StackBlur, sepFilter2D anchor/delta/identity/shift/zeroPadding,
  GaussianBlur_Bitexact) - pre-existing в стоке 4.14.0 (GCC 16?)
- Заметка: тесты на dispatch шли 107с vs 10с stock - вероятная oversubscription
  OpenBLAS(OpenMP) x TBB в LAPACK-тестах. Для продакшена: OPENBLAS_NUM_THREADS или
  адаптивные потоки fastcv

## Этап 5f. Аудит памяти (memory_bench.cpp, memory_frag.cpp)
Статика (alloc.cpp, matrix.cpp, bufferpool.impl.hpp):
- StdMatAllocator: каждый Mat::create = fastMalloc, release = fastFree + delete UMatData. Пулов НЕТ -
  BufferPoolController по умолчанию Dummy (реальный пул только у CUDA host mem / OpenCL)
- fastMalloc на glibc: aligned-alloc ОТКЛЮЧЁН по умолчанию (issue #15526), ручное выравнивание через
  malloc+alignPtr - корректно и дёшево
- refcount: атомики CV_XADD, корректно

Эмпирика:
- Mat create/release: 0.03-0.06 мкс (даже 24MB) - glibc tcache/arena справляется
- параллельный churn 1MB x 32 потока: 2.4 мкс/оп - норма
- mixed-size churn 20k: RSS стабилен, память возвращается ОС, фрагментации нет
- RT-паттерн (fresh vs reused buffers, blur+convert 1080p): разница всего 0.004 мс/кадр
- единственная реальная цена: first-touch page faults на больших свежих буферах (~45мкс/6MB)

Вывод: подсистема здорова. Для строгого RT - паттерн переиспользования буферов или
свой pooling MatAllocator (setDefaultAllocator API есть). Внедрять глобально не стали -
измеренный выигрыш 4 мкс/кадр не оправдывает риск.
GpuMat: отдельный cuda::BufferPool существует (per-stream) - для GPU-путей использовать его.
