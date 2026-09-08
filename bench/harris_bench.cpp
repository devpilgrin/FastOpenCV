// Сравнение cv::cornerHarris vs fastcv::cornerHarris8u
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv { void cornerHarris8u(const cv::Mat&, cv::Mat&, int, double); }

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

    for (int bs : {3, 5, 7}) {
        Mat ref, fast;
        cornerHarris(img, ref, bs, 3, 0.04);
        fastcv::cornerHarris8u(img, fast, bs, 0.04);

        Mat diff;
        absdiff(ref, fast, diff);
        double maxd, maxref;
        minMaxLoc(diff, nullptr, &maxd);
        minMaxLoc(ref, nullptr, &maxref);
        double rel = maxref > 0 ? maxd / maxref : 0;

        double t_ref = timeit(10, [&] { cornerHarris(img, ref, bs, 3, 0.04); });
        double t_fast = timeit(10, [&] { fastcv::cornerHarris8u(img, fast, bs, 0.04); });
        printf("block=%d: OpenCV %8.3f ms | fastcv %8.3f ms | speedup %5.2fx | maxAbsDiff %.3g (rel %.2e)\n",
               bs, t_ref, t_fast, t_ref / t_fast, maxd, rel);
    }
    return 0;
}
