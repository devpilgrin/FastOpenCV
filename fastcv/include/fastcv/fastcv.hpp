// FastOpenCV: оптимизированные кернели - единый заголовок
#pragma once
#include <opencv2/core.hpp>

namespace fastcv {

// Морфология/фильтры
void medianBlur8u(const cv::Mat& src, cv::Mat& dst, int ksize);        // 14-17x, bit-exact, k>5 (k<=5 проксируйте в cv::)
void bilateralFilter8u(const cv::Mat& src, cv::Mat& dst, int d,        // точнее cv:: (без квантования), ~parity
                       double sigmaColor, double sigmaSpace);

// Детекторы
void cornerHarris8u(const cv::Mat& src, cv::Mat& dst,                  // 8.5-12x, exact, ksize=3, REFLECT_101
                    int blockSize, double k);

// Core
void sum8u(const cv::Mat& src, double* sums);                          // 8UC3: 26x; cn 1..4
void lut8u(const cv::Mat& src, const uchar* lut, cv::Mat& dst);        // VBMI vpermb: 1.6-2.1x, любое cn
void transform32f(const cv::Mat& src, cv::Mat& dst, const cv::Mat& m); // 10.7x, bit-exact (32F, scn/dcn<=4)
void resizeCubic8u(const cv::Mat& src, cv::Mat& dst, cv::Size dsize);  // downscale 1.5-1.9x, bit-exact (C1/C3)
void filter2D_8u32f(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel, // 6-32x (8UC1->32F; у OpenCV FilterNoVec),
                    cv::Point anchor, double delta);                          // rel err ~1e-7 (FMA)
void sortIdxRows32f(const cv::Mat& src, cv::Mat& dst, bool ascending); // 23.5x (cv::sortIdx однопоточный)
void pow32f(const cv::Mat& src, cv::Mat& dst, double power);           // 19-22x, rel err ~1e-5 (векторный exp2/log2)
void bilateralGrid8u(const cv::Mat& src, cv::Mat& dst, int d,          // АППРОКСИМАЦИЯ (Paris-Durand grid):
                     double sigmaColor, double sigmaSpace);            // выигрыш от d>=15 (1.8x) до d=25 (4.5x)
void morphEllipseApprox8u(const cv::Mat& src, cv::Mat& dst,            // АППРОКСИМАЦИЯ (зонотоп 4 сегмента):
                          int ksize, int op);                          // имеет смысл только k>=31 (1.6x)
void sobel8u16sLarge(const cv::Mat& src, cv::Mat& dst,                 // 3.8x (k=5) / 6.7x (k=7), bit-exact;
                     int dx, int dy, int ksize);                       // для k=3 используйте cv::Sobel
void laplacian8u16sLarge(const cv::Mat& src, cv::Mat& dst,             // 1.9x, bit-exact; ТОЛЬКО k=5
                         int ksize);                                   // (k=7 у cv float-путь - не трогаем)
void adaptiveThreshold8u(const cv::Mat& src, cv::Mat& dst,             // GAUSS: 5.3-9.1x (99.95% px совпадают,
                         double maxValue, int method, int type,        // fixed-point vs float gaussian);
                         int blockSize, double delta);                 // MEAN: паритет (cv там и так boxFilter)
void sepFilter2D_8u(const cv::Mat& src, cv::Mat& dst, int ddepth,      // 2.0-3.2x; 8U: maxDiff<=1 (FMA),
                    const cv::Mat& kx, const cv::Mat& ky);             // 32F: maxDiff ~1e-4
void sqrBoxFilter8u(const cv::Mat& src, cv::Mat& dst,                  // 1.3-2.2x, bit-exact (8U->32F)
                    cv::Size ksize, bool normalize);
void calcHist8u(const cv::Mat& src, cv::Mat& hist);                    // 5.0x, bit-exact (256-bin 32F)
void resizeAreaInt8u(const cv::Mat& src, cv::Mat& dst, int factor);    // x4: 18.7x (C1) / 3.5x (C3),
                                                                     // maxDiff 1 на 0.2% px; x2 - берите cv::resize
void phase32f(const cv::Mat& x, const cv::Mat& y, cv::Mat& dst,        // 4.8x; vs libm 0.004 град
              bool angleInDegrees = true);                             // (точнее cv::phase LUT; на границе 0/360 отличается)
void cartToPolar32f(const cv::Mat& x, const cv::Mat& y,                // 3.3x, mag relDiff 3e-7
                    cv::Mat& mag, cv::Mat& ang, bool angleInDegrees = true);
cv::Scalar sum8uSmart(const cv::Mat& src);                             // адаптивно: cv::sum (<2M px) / fastcv (>=2M)

// Калибровка камеры (undistort): у cv stripe=2 строки на 1080p -> 540 вызовов remap
void buildUndistortMaps(const cv::Mat& K, const cv::Mat& dist,         // карты один раз (0.28 мс)
                        const cv::Mat& newK, cv::Size size, cv::Mat& map1, cv::Mat& map2);
void undistortPrecomputed(const cv::Mat& src, cv::Mat& dst,            // RT: 0.52 мс (19x к cv::undistort)
                          const cv::Mat& map1, const cv::Mat& map2);
void undistort8u(const cv::Mat& src, cv::Mat& dst, const cv::Mat& K,   // one-shot 11.4x, bit-exact
                 const cv::Mat& dist, const cv::Mat& newK = cv::Mat());

// Линейная алгебра (требуют линковки -lopenblas)
void gemm(const cv::Mat& A, const cv::Mat& B, double alpha,            // BLAS с адаптивным числом потоков;
          const cv::Mat& C, double beta, cv::Mat& D, int flags);       // мелкие матрицы - до 25x быстрее cv::gemm
bool eigenSym(const cv::Mat& src, cv::Mat& evals, cv::Mat& evecs);     // LAPACK syev: 10-60x быстрее cv::eigen

} // namespace fastcv
