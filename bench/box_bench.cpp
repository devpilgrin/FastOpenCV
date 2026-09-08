#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void boxFilter8u(const cv::Mat&, cv::Mat&, cv::Size, bool); }
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
    for (int k : {5, 15, 31}) {
        for (int norm : {1, 0}) {
            Mat a, b;
            boxFilter(img, a, -1, {k, k}, Point(-1,-1), norm != 0);
            fastcv::boxFilter8u(img, b, {k, k}, norm != 0);
            Mat d; absdiff(a, b, d);
            double md; minMaxLoc(d, nullptr, &md);
            double t1 = timeit(30, [&] { boxFilter(img, a, -1, {k, k}, Point(-1,-1), norm != 0); });
            double t2 = timeit(30, [&] { fastcv::boxFilter8u(img, b, {k, k}, norm != 0); });
            printf("box %2dx%d norm=%d: cv %7.3f ms | fastcv %7.3f ms | %5.1fx | maxDiff %.0f\n",
                   k, k, norm, t1, t2, t1 / t2, md);
        }
    }
    return 0;
}
