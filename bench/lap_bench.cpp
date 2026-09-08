#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void laplacian8u16sLarge(const cv::Mat&, cv::Mat&, int); }
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
    for (int k : {5, 7}) {
        Mat a, b, d;
        Laplacian(img, a, CV_16S, k);
        fastcv::laplacian8u16sLarge(img, b, k);
        absdiff(a, b, d);
        double md; minMaxLoc(d, nullptr, &md);
        int ndiff = countNonZero(d);
        double t1 = timeit(30, [&] { Laplacian(img, a, CV_16S, k); });
        double t2 = timeit(30, [&] { fastcv::laplacian8u16sLarge(img, b, k); });
        printf("Laplacian k=%d: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f (%d px)\n",
               k, t1, t2, t1 / t2, md, ndiff);
    }
    return 0;
}
