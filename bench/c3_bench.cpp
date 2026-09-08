#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
namespace fastcv {
void medianBlur8uC3(const cv::Mat&, cv::Mat&, int);
void sepFilter2D_8u(const cv::Mat&, cv::Mat&, int, const cv::Mat&, const cv::Mat&);
void sobel8u16sLarge(const cv::Mat&, cv::Mat&, int, int, int);
}
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
    Mat img3(1080, 1920, CV_8UC3);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    Mat a, b, d;
    double md;
    medianBlur(img3, a, 9);
    fastcv::medianBlur8uC3(img3, b, 9);
    absdiff(a, b, d); minMaxLoc(d.reshape(1), nullptr, &md);
    printf("medianBlur C3 k=9:  cv %8.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f\n",
           timeit(20, [&]{ medianBlur(img3, a, 9); }),
           timeit(20, [&]{ fastcv::medianBlur8uC3(img3, b, 9); }),
           0.0, md);
    Mat k1d = getGaussianKernel(15, -1, CV_32F);
    sepFilter2D(img3, a, -1, k1d, k1d);
    fastcv::sepFilter2D_8u(img3, b, -1, k1d, k1d);
    absdiff(a, b, d); minMaxLoc(d.reshape(1), nullptr, &md);
    printf("sepFilter C3 k=15:  cv %8.3f | fastcv %7.3f | maxDiff %.0f\n",
           timeit(20, [&]{ sepFilter2D(img3, a, -1, k1d, k1d); }),
           timeit(20, [&]{ fastcv::sepFilter2D_8u(img3, b, -1, k1d, k1d); }), md);
    Sobel(img3, a, CV_16S, 1, 0, 5);
    Mat b16;
    fastcv::sobel8u16sLarge(img3, b16, 1, 0, 5);
    absdiff(a, b16, d); minMaxLoc(d.reshape(1), nullptr, &md);
    printf("Sobel C3 k=5:       cv %8.3f | fastcv %7.3f | maxDiff %.0f\n",
           timeit(20, [&]{ Sobel(img3, a, CV_16S, 1, 0, 5); }),
           timeit(20, [&]{ fastcv::sobel8u16sLarge(img3, b16, 1, 0, 5); }), md);
    return 0;
}
