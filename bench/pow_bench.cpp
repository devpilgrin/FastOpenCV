#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void pow32f(const cv::Mat&, cv::Mat&, double); }
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
    Mat img(1080, 1920, CV_32FC1);
    rng.fill(img, RNG::UNIFORM, 0.01, 10.0);
    for (double p : {1.7, -0.3}) {
        Mat ref, fast;
        pow(img, p, ref);
        fastcv::pow32f(img, fast, p);
        Mat rel; absdiff(ref, fast, rel); divide(rel, ref, rel);
        double maxrel; minMaxLoc(rel, nullptr, &maxrel);
        double t_ref = timeit(20, [&] { pow(img, p, ref); });
        double t_fast = timeit(20, [&] { fastcv::pow32f(img, fast, p); });
        printf("pow ^%4.1f: cv %7.3f ms | fastcv %7.3f ms | %5.1fx | maxRelErr %.2e\n",
               p, t_ref, t_fast, t_ref / t_fast, maxrel);
    }
    return 0;
}
