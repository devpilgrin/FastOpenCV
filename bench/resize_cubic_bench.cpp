// cv::resize INTER_CUBIC vs fastcv::resizeCubic8u
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv { void resizeCubic8u(const cv::Mat&, cv::Mat&, cv::Size); }

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
    struct Case { int sw, sh, dw, dh, cn; };
    Case cases[] = {
        {1920, 1080, 960, 540, 1},   // 2x down C1
        {1920, 1080, 960, 540, 3},   // 2x down C3
        {1920, 1080, 1280, 720, 1},  // 1.5x down C1
        {1920, 1080, 1280, 720, 3},  // 1.5x down C3
        {960, 540, 1920, 1080, 1},   // 2x up C1
        {960, 540, 1920, 1080, 3},   // 2x up C3
        {1920, 1080, 640, 480, 3},   // 3x/2.25x down C3
    };
    for (auto& cs : cases) {
        Mat img(cs.sh, cs.sw, CV_MAKETYPE(CV_8U, cs.cn));
        rng.fill(img, RNG::UNIFORM, 0, 256);
        Mat ref, fast;
        resize(img, ref, Size(cs.dw, cs.dh), 0, 0, INTER_CUBIC);
        fastcv::resizeCubic8u(img, fast, Size(cs.dw, cs.dh));
        Mat diff; absdiff(ref, fast, diff);
        double maxd; minMaxLoc(diff, nullptr, &maxd);
        int nz = countNonZero(diff.reshape(1));
        double t_ref = timeit(20, [&] { resize(img, ref, Size(cs.dw, cs.dh), 0, 0, INTER_CUBIC); });
        double t_fast = timeit(20, [&] { fastcv::resizeCubic8u(img, fast, Size(cs.dw, cs.dh)); });
        printf("%4dx%-4d C%d -> %4dx%-4d: cv %7.3f ms | fastcv %7.3f ms | %5.1fx | maxDiff %2.0f nz %d\n",
               cs.sw, cs.sh, cs.cn, cs.dw, cs.dh, t_ref, t_fast, t_ref / t_fast, maxd, nz);
    }
    return 0;
}
