#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void guidedFilterGray8u(const cv::Mat&, const cv::Mat&, cv::Mat&, int, double); }
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
    for (int r : {4, 8}) {
        Mat a, b, d;
        ximgproc::guidedFilter(img, img, a, r, 1e-4);
        fastcv::guidedFilterGray8u(img, img, b, r, 1e-4);
        absdiff(a, b, d);
        double md; minMaxLoc(d, nullptr, &md);
        int nd = countNonZero(d);
        double t1 = timeit(15, [&]{ ximgproc::guidedFilter(img, img, a, r, 1e-4); });
        double t2 = timeit(15, [&]{ fastcv::guidedFilterGray8u(img, img, b, r, 1e-4); });
        printf("guided r=%d: cv %8.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f (%d px)\n", r, t1, t2, t1/t2, md, nd);
    }
    return 0;
}
