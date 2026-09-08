#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void sum8u(const cv::Mat&, double*); }
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iters;
}
int main() {
    RNG rng(42);
    double s_[4];
    for (int n : {128, 256, 384, 512, 768, 1024, 1920}) {
        Mat img(n, n, CV_8UC1);
        rng.fill(img, RNG::UNIFORM, 0, 256);
        double t1 = timeit(200, [&]{ Scalar s = sum(img); (void)s[0]; });
        double t2 = timeit(200, [&]{ fastcv::sum8u(img, s_); (void)s_[0]; });
        printf("n=%5d: cv %8.2f us | fastcv %8.2f us | %6.2fx\n", n, t1, t2, t1/t2);
    }
    return 0;
}
