// FastOpenCV: GEMM через OpenBLAS (cv::gemm в 4.14 однопоточный, BLAS-вызов
// в matmul.simd.hpp закомментирован). Поддержка флагов GEMM_1_T/2_T/3_T.
#include <opencv2/core.hpp>
#include <cblas.h>

namespace fastcv {

using namespace cv;

template <typename T>
static void gemmT(const Mat& A, const Mat& B, double alpha, const Mat& C,
                  double beta, Mat& D, int flags, int depth);

void gemm(const Mat& A, const Mat& B, double alpha, const Mat& C,
          double beta, Mat& D, int flags) {
    CV_Assert(A.depth() == B.depth() && (A.depth() == CV_32F || A.depth() == CV_64F));
    CV_Assert(A.channels() == 1 && B.channels() == 1 && (C.empty() || C.channels() == 1));
    int m = (flags & GEMM_1_T) ? A.cols : A.rows;
    int n = (flags & GEMM_2_T) ? B.rows : B.cols;
    int k = (flags & GEMM_1_T) ? A.rows : A.cols;
    CV_Assert(((flags & GEMM_2_T) ? B.cols : B.rows) == k);
    D.create(m, n, A.type());
    if (A.depth() == CV_32F)
        gemmT<float>(A, B, alpha, C, beta, D, flags, CV_32F);
    else
        gemmT<double>(A, B, alpha, C, beta, D, flags, CV_64F);
}

template <typename T>
static void gemmT(const Mat& A, const Mat& B, double alpha, const Mat& C,
                  double beta, Mat& D, int flags, int depth) {
    CBLAS_TRANSPOSE ta = (flags & GEMM_1_T) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (flags & GEMM_2_T) ? CblasTrans : CblasNoTrans;
    int m = D.rows, n = D.cols;
    int k = (flags & GEMM_1_T) ? A.rows : A.cols;
    int lda = (int)(A.step / sizeof(T));
    int ldb = (int)(B.step / sizeof(T));
    int ldd = (int)(D.step / sizeof(T));

    // D = beta*C (с копией, т.к. cblas пишет in-place в D)
    if (!C.empty()) {
        if (C.data != D.data) C.copyTo(D);
    }
    T a = (T)alpha, b = C.empty() ? (T)0 : (T)beta;

    // мелкие матрицы: многопоточность OpenBLAS даёт 3мс оверхеда на синхронизацию.
    // Замеры на 9950X3D: <6e7 FLOP - 1 поток, <4e8 - 8 потоков, далее - все.
    double work = 2.0 * m * n * k;
    int prev = 0;
    if (work < 4e8) {
        prev = openblas_get_num_threads();
        openblas_set_num_threads_local(work < 6e7 ? 1 : 8);
    }

    if (depth == CV_32F)
        cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, (float)a,
                    (const float*)A.data, lda, (const float*)B.data, ldb,
                    (float)b, (float*)D.data, ldd);
    else
        cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, (double)a,
                    (const double*)A.data, lda, (const double*)B.data, ldb,
                    (double)b, (double*)D.data, ldd);
    if (work < 4e8) openblas_set_num_threads_local(prev);
}

} // namespace fastcv
