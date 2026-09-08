#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void adaptiveThreshold8u(const cv::Mat&, cv::Mat&, double, int, int, int, double); }
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    for (int bs : {11, 25}) {
        for (int m : {ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C}) {
            Mat a, b;
            adaptiveThreshold(img, a, 255, m, THRESH_BINARY, bs, 5);
            fastcv::adaptiveThreshold8u(img, b, 255, m, THRESH_BINARY, bs, 5);
            Mat d; absdiff(a, b, d);
            double md; minMaxLoc(d, nullptr, &md);
            double t1 = timeit(30, [&] { adaptiveThreshold(img, a, 255, m, THRESH_BINARY, bs, 5); });
            double t2 = timeit(30, [&] { fastcv::adaptiveThreshold8u(img, b, 255, m, THRESH_BINARY, bs, 5); });
            printf("adaptiveThr bs=%d %s: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %4.0f\n",
                   bs, m == ADAPTIVE_THRESH_MEAN_C ? "MEAN " : "GAUSS", t1, t2, t1 / t2, md);
        }
    }
    return 0;
}
