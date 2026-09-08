// Бенч: cv::gemm vs fastcv::gemm (OpenBLAS)
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv { void gemm(const cv::Mat&, const cv::Mat&, double, const cv::Mat&, double, cv::Mat&, int); }

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
    for (int n : {512, 1024, 2048}) {
        for (int depth : {CV_32F, CV_64F}) {
            Mat a(n, n, depth), b(n, n, depth), c1, c2;
            rng.fill(a, RNG::UNIFORM, -1, 1);
            rng.fill(b, RNG::UNIFORM, -1, 1);
            gemm(a, b, 1.0, noArray(), 0, c1);
            fastcv::gemm(a, b, 1.0, Mat(), 0, c2, 0);
            double maxrel = norm(c1, c2, NORM_INF) / std::max(1.0, norm(c1, NORM_INF));
            int iters = n <= 1024 ? 10 : 3;
            double t_ref = timeit(iters, [&] { gemm(a, b, 1.0, noArray(), 0, c1); });
            double t_fast = timeit(iters, [&] { fastcv::gemm(a, b, 1.0, Mat(), 0, c2, 0); });
            double gf = 2.0 * n * n * n;
            printf("GEMM %4d %s: cv %8.1f ms (%6.1f GF) | fastcv %7.1f ms (%7.1f GF) | %5.1fx | relErr %.1e\n",
                   n, depth == CV_32F ? "32F" : "64F", t_ref, gf / (t_ref * 1e6),
                   t_fast, gf / (t_fast * 1e6), t_ref / t_fast, maxrel);
        }
    }
    return 0;
}
