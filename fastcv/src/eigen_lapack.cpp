// FastOpenCV: eigen для симметричных матриц через LAPACK syev
// (cv::eigen в 4.14 знает только внутренний Jacobi или Eigen-библиотеку;
//  LAPACK syev недоступен - а это 30+ секунд для n=1024 против ~0.3с у dsyev)
#include <opencv2/core.hpp>
#include <cblas.h>
#include <lapacke.h>
#include <algorithm>

namespace fastcv {

using namespace cv;

// evals - в порядке УБЫВАНИЯ (как cv::eigen), evecs - строки-собственные векторы
bool eigenSym(const Mat& src, Mat& evals, Mat& evecs) {
    CV_Assert(src.rows == src.cols && (src.type() == CV_32FC1 || src.type() == CV_64FC1));
    int n = src.rows;
    bool is64 = src.type() == CV_64FC1;

    evecs.create(n, n, src.type());
    evals.create(n, 1, src.type());
    src.copyTo(evecs); // syev пишет векторы поверх матрицы (столбцы LAPACK = строки у нас)

    int info;
    // OpenBLAS: на малых n синхронизация потоков дороже работы (117мс -> 3.7мс при n=128)
    int prev = 0;
    bool small = (double)n * n * n < 3e8;
    if (small) {
        prev = openblas_get_num_threads();
        openblas_set_num_threads_local(1);
    }
    if (is64) {
        double* a = (double*)evecs.data;
        double* w = (double*)evals.data;
        info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, a, n, w);
    } else {
        float* a = (float*)evecs.data;
        float* w = (float*)evals.data;
        info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', n, a, n, w);
    }
    if (small) openblas_set_num_threads_local(prev);
    if (info != 0) return false;

    // LAPACK хранит векторы по столбцам; cv::eigen - по строкам, порядок descending.
    // Транспонируем и разворачиваем (порядок evals тоже разворачиваем).
    flip(evals, evals, 0);
    Mat tmp = evecs.t();
    flip(tmp, evecs, 0);
    return true;
}

} // namespace fastcv
