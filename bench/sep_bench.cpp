#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void sepFilter2D_8u(const cv::Mat&, cv::Mat&, int, const cv::Mat&, const cv::Mat&); }
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
    for (int k : {9, 15, 25}) {
        Mat k1d = getGaussianKernel(k, -1, CV_32F);
        // 8U -> 8U
        Mat a, b, d;
        sepFilter2D(img, a, -1, k1d, k1d);
        fastcv::sepFilter2D_8u(img, b, -1, k1d, k1d);
        absdiff(a, b, d);
        double md; minMaxLoc(d, nullptr, &md);
        double t1 = timeit(30, [&] { sepFilter2D(img, a, -1, k1d, k1d); });
        double t2 = timeit(30, [&] { fastcv::sepFilter2D_8u(img, b, -1, k1d, k1d); });
        printf("sepFilter 8U->8U  k=%2d: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f\n", k, t1, t2, t1/t2, md);
        // 8U -> 32F
        Mat a32, b32, d32;
        sepFilter2D(img, a32, CV_32F, k1d, k1d);
        fastcv::sepFilter2D_8u(img, b32, CV_32F, k1d, k1d);
        absdiff(a32, b32, d32);
        minMaxLoc(d32, nullptr, &md);
        t1 = timeit(30, [&] { sepFilter2D(img, a32, CV_32F, k1d, k1d); });
        t2 = timeit(30, [&] { fastcv::sepFilter2D_8u(img, b32, CV_32F, k1d, k1d); });
        printf("sepFilter 8U->32F k=%2d: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.3g\n", k, t1, t2, t1/t2, md);
    }
    return 0;
}
