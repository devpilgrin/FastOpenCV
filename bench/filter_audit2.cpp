// Расширенный аудит фильтров imgproc/photo/ximgproc (1080p, default-режим потоков)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/ximgproc.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double bench(const char* name, double bytes, int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
    printf("%-46s %9.3f ms  %8.2f GB/s\n", name, ms, bytes / (ms * 1e6));
    return ms;
}

int main() {
    const int W = 1920, H = 1080;
    const double PX = (double)W * H;
    Mat u8(H, W, CV_8UC1, Scalar(128));
    Mat u8c3(H, W, CV_8UC3, Scalar(128, 100, 80));
    Mat f32(H, W, CV_32FC1, Scalar(0.5f));
    Mat dst, dstf;
    printf("=== линейные фильтры ===\n");
    bench("boxFilter 5x5 8UC1", PX * 2, 100, [&] { boxFilter(u8, dst, -1, {5, 5}); });
    bench("boxFilter 15x15 8UC1", PX * 2, 100, [&] { boxFilter(u8, dst, -1, {15, 15}); });
    bench("boxFilter 15x15 unnorm", PX * 2, 100, [&] { boxFilter(u8, dst, -1, {15, 15}, Point(-1,-1), false); });
    bench("sqrBoxFilter 5x5 8UC1", PX * 2, 50, [&] { sqrBoxFilter(u8, dstf, -1, {5, 5}); });
    bench("GaussianBlur 5x5 8UC1", PX * 2, 100, [&] { GaussianBlur(u8, dst, {5, 5}, 1.0); });
    bench("GaussianBlur 5x5 32F", PX * 8, 100, [&] { GaussianBlur(f32, dstf, {5, 5}, 1.0); });
    bench("GaussianBlur 15x15 8UC3", PX * 6, 50, [&] { GaussianBlur(u8c3, dst, {15, 15}, 3.0); });
    bench("sepFilter2D 9x9 8UC1", PX * 2, 50, [&] {
        Mat k = Mat::ones(9, 1, CV_32F) / 9.f;
        sepFilter2D(u8, dst, -1, k, k);
    });
    bench("stackBlur r=10 8UC1", PX * 2, 50, [&] { stackBlur(u8, dst, Size(21, 21)); });

    printf("=== производные ===\n");
    bench("Sobel 5x5 8UC1->16S", PX * 3, 100, [&] { Sobel(u8, dst, CV_16S, 1, 0, 5); });
    bench("Sobel 7x7 8UC1->16S", PX * 3, 100, [&] { Sobel(u8, dst, CV_16S, 1, 0, 7); });
    bench("Scharr 8UC1->16S", PX * 3, 100, [&] { Scharr(u8, dst, CV_16S, 1, 0); });
    bench("Laplacian 8UC1->16S k5", PX * 3, 50, [&] { Laplacian(u8, dst, CV_16S, 5); });
    bench("Laplacian 8UC1->8U k1", PX * 2, 50, [&] { Laplacian(u8, dst, CV_8U, 1); });

    printf("=== нелинейные/краевые ===\n");
    bench("adaptiveThreshold GAUSS 25", PX * 2, 30, [&] { adaptiveThreshold(u8, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5); });
    bench("guidedFilter r=8 8UC1", PX * 2, 10, [&] { ximgproc::guidedFilter(u8, u8, dst, 8, 100); });
    bench("fastNlMeansDenoising h=10 (1080p)", PX * 2, 2, [&] { fastNlMeansDenoising(u8, dst, 10, 7, 21); });
    bench("edgePreservingFilter NORMCONV 8UC3", PX * 6, 3, [&] { edgePreservingFilter(u8c3, dst, 1, 60, 0.4f); });
    bench("detailEnhance 8UC3", PX * 6, 3, [&] { detailEnhance(u8c3, dst); });
    return 0;
}
