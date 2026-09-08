#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void transform32f(const cv::Mat&, cv::Mat&, const cv::Mat&); }
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 2; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
int main() {
    RNG rng(7);
    Mat img(1080, 1920, CV_32FC3);
    rng.fill(img, RNG::UNIFORM, 0, 1);
    Mat M = (Mat_<double>(4, 4) <<
        1.1, 0.05, -0.02, 0.1,
        -0.03, 0.9, 0.04, 0.2,
        0.01, -0.05, 1.2, 0.05,
        0, 0, 0, 1);
    Mat ref, fast;
    transform(img, ref, M);
    fastcv::transform32f(img, fast, M);
    Mat diff; absdiff(ref, fast, diff);
    double maxd; minMaxLoc(diff, nullptr, &maxd);
    double t_ref = timeit(50, [&] { transform(img, ref, M); });
    double t_fast = timeit(50, [&] { fastcv::transform32f(img, fast, M); });
    printf("transform 32FC3 4x4: cv %7.3f ms | fastcv %7.3f ms | %5.1fx | maxDiff %.2e\n",
           t_ref, t_fast, t_ref / t_fast, maxd);
    return 0;
}
