// Сравнение cv::bilateralFilter vs fastcv::bilateralFilter8u
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv { void bilateralFilter8u(const cv::Mat&, cv::Mat&, int, double, double); }

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 2; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);

    for (int d : {5, 9, 15}) {
        Mat ref, fast;
        bilateralFilter(img, ref, d, 50, 50);
        fastcv::bilateralFilter8u(img, fast, d, 50, 50);

        Mat diff;
        absdiff(ref, fast, diff);
        double maxd;
        minMaxLoc(diff, nullptr, &maxd);
        double meanErr = cv::mean(diff)[0];

        double t_ref = timeit(10, [&] { bilateralFilter(img, ref, d, 50, 50); });
        double t_fast = timeit(10, [&] { fastcv::bilateralFilter8u(img, fast, d, 50, 50); });
        printf("d=%2d: OpenCV %8.3f ms | fastcv %8.3f ms | speedup %5.2fx | maxDiff %.0f meanErr %.4f\n",
               d, t_ref, t_fast, t_ref / t_fast, maxd, meanErr);
    }
    return 0;
}
