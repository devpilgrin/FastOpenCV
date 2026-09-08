// Бенч: cv::eigen vs fastcv::eigenSym (LAPACK dsyev) + мелкий GEMM (потоки OpenBLAS)
#include <opencv2/core.hpp>
#include <cblas.h>
#include <chrono>
#include <cstdio>

namespace fastcv {
bool eigenSym(const cv::Mat&, cv::Mat&, cv::Mat&);
void gemm(const cv::Mat&, const cv::Mat&, double, const cv::Mat&, double, cv::Mat&, int);
}

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double timeit(int iters, F&& f) {
    f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

int main() {
    RNG rng(7);
    printf("=== eigen sym 64F ===\n");
    for (int n : {128, 512, 1024}) {
        Mat a(n, n, CV_64F);
        rng.fill(a, RNG::UNIFORM, -1, 1);
        a = a * a.t() + Mat::eye(n, n, CV_64F);
        Mat w1, v1, w2, v2;
        double t_ref = timeit(3, [&] { eigen(a, w1, v1); });
        double t_fast = timeit(3, [&] { fastcv::eigenSym(a, w2, v2); });
        // проверка: A*v = lambda*v для первых 3 собственных векторов
        double maxerr = 0;
        for (int i = 0; i < 3; i++) {
            Mat v = v2.row(i).t();
            Mat err = a * v - w2.at<double>(i) * v;
            maxerr = std::max(maxerr, norm(err, NORM_INF) / norm(a, NORM_INF));
        }
        double werr = norm(w1, w2, NORM_INF) / norm(w1, NORM_INF);
        printf("n=%4d: cv %9.1f ms | fastcv %8.1f ms | %6.1fx | residual %.1e | evalsRelErr %.1e\n",
               n, t_ref, t_fast, t_ref / t_fast, maxerr, werr);
    }

    printf("=== мелкий GEMM: потоки OpenBLAS ===\n");
    for (int n : {128, 256, 384, 512}) {
        Mat a(n, n, CV_32F), b(n, n, CV_32F), c;
        rng.fill(a, RNG::UNIFORM, -1, 1);
        rng.fill(b, RNG::UNIFORM, -1, 1);
        openblas_set_num_threads(32);
        double t32 = timeit(30, [&] { fastcv::gemm(a, b, 1.0, Mat(), 0, c, 0); });
        openblas_set_num_threads(8);
        double t8 = timeit(30, [&] { fastcv::gemm(a, b, 1.0, Mat(), 0, c, 0); });
        openblas_set_num_threads(1);
        double t1 = timeit(30, [&] { fastcv::gemm(a, b, 1.0, Mat(), 0, c, 0); });
        openblas_set_num_threads(32);
        printf("n=%4d: 32t %7.3f ms | 8t %7.3f ms | 1t %7.3f ms\n", n, t32, t8, t1);
    }
    return 0;
}
