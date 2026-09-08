#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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
    Mat d;
    printf("cv medianBlur 8UC3 k=9:   %8.3f ms\n", timeit(20, [&]{ medianBlur(img3, d, 9); }));
    Mat k1d = getGaussianKernel(15, -1, CV_32F);
    printf("cv sepFilter2D 8UC3 k=15: %8.3f ms\n", timeit(20, [&]{ sepFilter2D(img3, d, -1, k1d, k1d); }));
    printf("cv Sobel 8UC3 k=5:        %8.3f ms\n", timeit(20, [&]{ Sobel(img3, d, CV_16S, 1, 0, 5); }));
    return 0;
}
