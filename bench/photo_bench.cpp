#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void edgePreservingFast(const cv::Mat&, cv::Mat&, int, float, float);
void detailEnhanceFast(const cv::Mat&, cv::Mat&, float, float); }
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
    edgePreservingFilter(img3, a, 1, 60, 0.4f);
    fastcv::edgePreservingFast(img3, b, 1, 60, 0.4f);
    absdiff(a, b, d); minMaxLoc(d.reshape(1), nullptr, &md);
    printf("edgePreserving RECURS: cv %8.2f | fastcv %7.2f | maxDiff %.0f\n",
           timeit(5, [&]{ edgePreservingFilter(img3, a, 1, 60, 0.4f); }),
           timeit(10, [&]{ fastcv::edgePreservingFast(img3, b, 1, 60, 0.4f); }), md);
    edgePreservingFilter(img3, a, 2, 60, 0.4f);
    fastcv::edgePreservingFast(img3, b, 2, 60, 0.4f);
    absdiff(a, b, d); minMaxLoc(d.reshape(1), nullptr, &md);
    printf("edgePreserving NC:     cv %8.2f | fastcv %7.2f | maxDiff %.0f\n",
           timeit(5, [&]{ edgePreservingFilter(img3, a, 2, 60, 0.4f); }),
           timeit(10, [&]{ fastcv::edgePreservingFast(img3, b, 2, 60, 0.4f); }), md);
    detailEnhance(img3, a);
    fastcv::detailEnhanceFast(img3, b, 10, 0.15f);
    absdiff(a, b, d); minMaxLoc(d.reshape(1), nullptr, &md);
    printf("detailEnhance:         cv %8.2f | fastcv %7.2f | maxDiff %.0f\n",
           timeit(5, [&]{ detailEnhance(img3, a); }),
           timeit(10, [&]{ fastcv::detailEnhanceFast(img3, b, 10, 0.15f); }), md);
    return 0;
}
