#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void erodeRect8u(const cv::Mat&, cv::Mat&, int); void dilateRect8u(const cv::Mat&, cv::Mat&, int); }
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
    for (int k : {5, 15, 31, 63}) {
        Mat kern = getStructuringElement(MORPH_RECT, {k, k});
        Mat a, b, d;
        erode(img, a, kern);
        fastcv::erodeRect8u(img, b, k);
        absdiff(a, b, d);
        double md; minMaxLoc(d, nullptr, &md);
        double t1 = timeit(30, [&]{ erode(img, a, kern); });
        double t2 = timeit(30, [&]{ fastcv::erodeRect8u(img, b, k); });
        printf("erode k=%2d: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f\n", k, t1, t2, t1/t2, md);
        dilate(img, a, kern);
        fastcv::dilateRect8u(img, b, k);
        absdiff(a, b, d); minMaxLoc(d, nullptr, &md);
        t1 = timeit(30, [&]{ dilate(img, a, kern); });
        t2 = timeit(30, [&]{ fastcv::dilateRect8u(img, b, k); });
        printf("dilate k=%2d: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f\n", k, t1, t2, t1/t2, md);
    }
    return 0;
}
