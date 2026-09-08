#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void sobel8u16sLarge(const cv::Mat&, cv::Mat&, int, int, int); }
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
    for (int k : {3, 5, 7}) {
        Mat a, b;
        Sobel(img, a, CV_16S, 1, 0, k);
        fastcv::sobel8u16sLarge(img, b, 1, 0, k);
        Mat d; absdiff(a, b, d);
        double md; minMaxLoc(d, nullptr, &md);
        double t1 = timeit(30, [&] { Sobel(img, a, CV_16S, 1, 0, k); });
        double t2 = timeit(30, [&] { fastcv::sobel8u16sLarge(img, b, 1, 0, k); });
        printf("Sobel k=%d X: cv %7.3f ms | fastcv %7.3f ms | %5.1fx | maxDiff %.0f\n", k, t1, t2, t1 / t2, md);
        Sobel(img, a, CV_16S, 0, 1, k);
        fastcv::sobel8u16sLarge(img, b, 0, 1, k);
        absdiff(a, b, d); minMaxLoc(d, nullptr, &md);
        printf("Sobel k=%d Y:                                    maxDiff %.0f\n", k, md);
    }
    return 0;
}
