// FastOpenCV: аудит базовых операций модуля core (и несколько imgproc)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F>
double bench(const char* name, double bytes, int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
    printf("%-44s %9.3f ms  %8.2f GB/s\n", name, ms, bytes / (ms * 1e6));
    return ms;
}

int main() {
    // setNumThreads(0) - ЛОВУШКА: в этой связке OpenCV+TBB даёт 4x затормаживание
    const int W = 1920, H = 1080;
    const double PX = (double)W * H;

    Mat u8a(H, W, CV_8UC1, Scalar(100)), u8b(H, W, CV_8UC1, Scalar(50));
    Mat f32a(H, W, CV_32FC1, Scalar(0.5f)), f32b(H, W, CV_32FC1, Scalar(0.25f));
    Mat u8c3a(H, W, CV_8UC3, Scalar(100, 150, 200));
    Mat dst8, dst32, dstc3;
    std::vector<Mat> chans;

    printf("=== арифметика ===\n");
    bench("add 8UC1", PX * 3, 100, [&] { add(u8a, u8b, dst8); });
    bench("add 32FC1", PX * 4 * 3, 100, [&] { add(f32a, f32b, dst32); });
    bench("multiply 32FC1", PX * 4 * 3, 100, [&] { multiply(f32a, f32b, dst32); });
    bench("addWeighted 8UC1", PX * 3, 100, [&] { addWeighted(u8a, 0.7, u8b, 0.3, 0, dst8); });
    bench("convertTo 8U->32F", PX * 5, 100, [&] { u8a.convertTo(dst32, CV_32F, 1.0 / 255); });
    bench("convertScaleAbs 32F->8U", PX * 5, 100, [&] { convertScaleAbs(f32a, dst8, 255, 0); });
    bench("absdiff 8UC1", PX * 3, 100, [&] { absdiff(u8a, u8b, dst8); });
    bench("sqrt 32FC1", PX * 8, 100, [&] { sqrt(f32a, dst32); });
    bench("exp 32FC1", PX * 8, 30, [&] { exp(f32a, dst32); });
    bench("log 32FC1", PX * 8, 30, [&] { log(f32a, dst32); });
    bench("pow 32FC1 ^0.5", PX * 8, 30, [&] { pow(f32a, 0.5, dst32); });
    bench("divide 32FC1", PX * 12, 100, [&] { divide(f32a, f32b, dst32); });

    printf("=== редукции ===\n");
    bench("countNonZero 8UC1", PX, 100, [&] { countNonZero(u8a); });
    bench("minMaxLoc 8UC1", PX, 100, [&] { minMaxLoc(u8a, nullptr, nullptr); });
    bench("norm L2 32FC1", PX * 4, 100, [&] { norm(f32a, NORM_L2); });
    bench("meanStdDev 8UC1", PX, 100, [&] { Scalar m, s; meanStdDev(u8a, m, s); });
    bench("sum 8UC3", PX * 3, 100, [&] { sum(u8c3a); });

    printf("=== память/каналы ===\n");
    bench("split 8UC3", PX * 3 * 2, 100, [&] { split(u8c3a, chans); });
    bench("merge 3x8UC1", PX * 3 * 2, 100, [&] { merge(chans, dstc3); });
    bench("transpose 8UC1", PX * 2, 50, [&] { transpose(u8a, dst8); });
    bench("flip 8UC1 (гориз.)", PX * 2, 100, [&] { flip(u8a, dst8, 1); });
    bench("LUT 8UC1", PX * 2, 100, [&] {
        static Mat lut(1, 256, CV_8UC1);
        if (!lut.data) { for (int i = 0; i < 256; i++) lut.at<uchar>(i) = (uchar)(255 - i); }
        LUT(u8a, lut, dst8);
    });
    bench("normalize MINMAX 8U->32F", PX * 5, 50, [&] { normalize(u8a, dst32, 0, 1, NORM_MINMAX, CV_32F); });

    printf("=== сортировки/матрицы ===\n");
    Mat sq(1024, 1024, CV_32FC1, Scalar(0.5f));
    bench("invert 64x64 64F (x100)", 64.0 * 64 * 8 * 3 * 100, 20, [&] {
        Mat m64 = sq(Rect(0, 0, 64, 64));
        for (int i = 0; i < 100; i++) { Mat inv; invert(m64, inv, DECOMP_LU); }
    });
    bench("solve 6x6 (x1000)", 6.0 * 6 * 8 * 2 * 1000, 20, [&] {
        Mat A = (Mat_<double>(6, 6) << 4,1,0,0,0,0, 1,4,1,0,0,0, 0,1,4,1,0,0, 0,0,1,4,1,0, 0,0,0,1,4,1, 0,0,0,0,1,4);
        Mat b = (Mat_<double>(6, 1) << 1, 2, 3, 4, 5, 6);
        for (int i = 0; i < 1000; i++) { Mat x; solve(A, b, x, DECOMP_LU); }
    });
    return 0;
}
