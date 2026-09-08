#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void filter2D_8u32f(const cv::Mat&, cv::Mat&, const cv::Mat&, cv::Point, double); }
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
    for (int k : {3, 5, 7, 11}) {
        Mat kern(k, k, CV_32F);
        rng.fill(kern, RNG::UNIFORM, -1, 1);
        Mat ref, fast;
        filter2D(img, ref, CV_32F, kern);
        fastcv::filter2D_8u32f(img, fast, kern, Point(-1, -1), 0);
        Mat diff; absdiff(ref, fast, diff);
        double maxd, maxref; minMaxLoc(diff, nullptr, &maxd); minMaxLoc(ref, nullptr, &maxref);
        double t_ref = timeit(15, [&] { filter2D(img, ref, CV_32F, kern); });
        double t_fast = timeit(15, [&] { fastcv::filter2D_8u32f(img, fast, kern, Point(-1,-1), 0); });
        printf("k=%2d: cv %8.3f ms | fastcv %7.3f ms | %5.1fx | maxDiff %.3g (rel %.1e)\n",
               k, t_ref, t_fast, t_ref / t_fast, maxd, maxd / std::max(1.0, maxref));
    }
    return 0;
}
