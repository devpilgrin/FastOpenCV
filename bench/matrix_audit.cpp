// Аудит матричных операций OpenCV: GEMM, SVD, invert, solve, eigen
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double timeit(int iters, F&& f) {
    f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

int main() {
    // setNumThreads(0) - ЛОВУШКА: в этой связке OpenCV+TBB даёт 4x затормаживание
    RNG rng(7);

    printf("=== GEMM (GFLOP/s) ===\n");
    for (int n : {256, 512, 1024, 2048}) {
        for (int depth : {CV_32F, CV_64F}) {
            Mat a(n, n, depth), b(n, n, depth), c;
            rng.fill(a, RNG::UNIFORM, -1, 1);
            rng.fill(b, RNG::UNIFORM, -1, 1);
            int iters = n <= 512 ? 20 : 5;
            double ms = timeit(iters, [&] { gemm(a, b, 1.0, noArray(), 0, c); });
            double gflop = 2.0 * n * n * n / (ms * 1e6);
            printf("GEMM %4dx%4d %s: %9.3f ms  %8.1f GFLOP/s\n",
                   n, n, depth == CV_32F ? "32F" : "64F", ms, gflop);
        }
    }

    printf("=== разложения (64F) ===\n");
    for (int n : {128, 512, 1024}) {
        Mat a(n, n, CV_64F);
        rng.fill(a, RNG::UNIFORM, -1, 1);
        a = a * a.t() + Mat::eye(n, n, CV_64F); // SPD для честности

        Mat w, u, vt;
        double ms_svd = timeit(3, [&] { SVD::compute(a, w, u, vt); });
        Mat inv;
        double ms_inv = timeit(5, [&] { invert(a, inv, DECOMP_LU); });
        Mat x, b = Mat::ones(n, 1, CV_64F);
        double ms_solve = timeit(5, [&] { solve(a, b, x, DECOMP_LU); });
        Mat evals, evecs;
        double ms_eig = timeit(3, [&] { eigen(a, evals, evecs); });
        printf("n=%4d: SVD %8.1f ms | invert(LU) %8.3f ms | solve(LU) %8.3f ms | eigen(sym) %8.1f ms\n",
               n, ms_svd, ms_inv, ms_solve, ms_eig);
    }
    return 0;
}
