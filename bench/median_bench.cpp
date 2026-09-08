// Сравнение cv::medianBlur vs fastcv::medianBlur8u (k > 5, 8UC1)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv { void medianBlur8u(const cv::Mat&, cv::Mat&, int); }

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

    for (int k : {7, 9, 15, 25}) {
        Mat ref, fast;
        medianBlur(img, ref, k);
        fastcv::medianBlur8u(img, fast, k);

        Mat diff;
        absdiff(ref, fast, diff);
        double maxd;
        minMaxLoc(diff, nullptr, &maxd);
        int nz = countNonZero(diff);

        double t_ref = timeit(7, [&] { medianBlur(img, ref, k); });
        double t_fast = timeit(7, [&] { fastcv::medianBlur8u(img, fast, k); });
        printf("k=%2d: OpenCV %9.3f ms | fastcv %8.3f ms | speedup %6.2fx | maxDiff %.0f nzDiff %d\n",
               k, t_ref, t_fast, t_ref / t_fast, maxd, nz);
    }
    return 0;
}
