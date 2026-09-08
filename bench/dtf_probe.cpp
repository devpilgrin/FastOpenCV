#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <chrono>
#include <cstdio>
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
    Mat dst;
    printf("dtFilter RF NC  iter=3: %8.2f ms\n", timeit(5, [&]{ ximgproc::dtFilter(img3, img3, dst, 60, 30, ximgproc::DTF_NC, 3); }));
    printf("dtFilter RF IC  iter=3: %8.2f ms\n", timeit(5, [&]{ ximgproc::dtFilter(img3, img3, dst, 60, 30, ximgproc::DTF_IC, 3); }));
    printf("dtFilter NC     iter=1: %8.2f ms\n", timeit(5, [&]{ ximgproc::dtFilter(img3, img3, dst, 60, 30, ximgproc::DTF_NC, 1); }));
    return 0;
}
