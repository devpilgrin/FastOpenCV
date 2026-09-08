// Бенч: bilateral grid и ellipse-морфология
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv {
void bilateralGrid8u(const cv::Mat&, cv::Mat&, int, double, double);
void morphEllipseApprox8u(const cv::Mat&, cv::Mat&, int, int);
}

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
    // реалистичное изображение: градиенты + шум + края
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    GaussianBlur(img, img, {21, 21}, 5); // крупномасштабная структура
    Mat noise(1080, 1920, CV_8UC1);
    rng.fill(noise, RNG::UNIFORM, 0, 40);
    add(img, noise, img);

    printf("=== bilateral ===\n");
    for (int d : {9, 15, 25}) {
        Mat ref, grid;
        bilateralFilter(img, ref, d, 50, 50);
        fastcv::bilateralGrid8u(img, grid, d, 50, 50);
        Mat diff; absdiff(ref, grid, diff);
        double maxd; minMaxLoc(diff, nullptr, &maxd);
        double meand = mean(diff)[0];
        double t_ref = timeit(7, [&] { bilateralFilter(img, ref, d, 50, 50); });
        double t_grid = timeit(7, [&] { fastcv::bilateralGrid8u(img, grid, d, 50, 50); });
        printf("d=%2d: exact %8.3f ms | grid %7.3f ms | %5.1fx | maxDiff %3.0f meanDiff %.3f\n",
               d, t_ref, t_grid, t_ref / t_grid, maxd, meand);
    }

    printf("=== morphology ellipse (dilate) ===\n");
    for (int k : {9, 15, 31}) {
        Mat se = getStructuringElement(MORPH_ELLIPSE, {k, k});
        Mat ref, approx;
        dilate(img, ref, se);
        fastcv::morphEllipseApprox8u(img, approx, k, MORPH_DILATE);
        Mat diff; absdiff(ref, approx, diff);
        double maxd; minMaxLoc(diff, nullptr, &maxd);
        double meand = mean(diff)[0];
        int nz = countNonZero(diff);
        double t_ref = timeit(10, [&] { dilate(img, ref, se); });
        double t_ap = timeit(10, [&] { fastcv::morphEllipseApprox8u(img, approx, k, MORPH_DILATE); });
        printf("k=%2d: exact %8.3f ms | approx %7.3f ms | %5.1fx | maxDiff %3.0f meanDiff %.3f nz %d\n",
               k, t_ref, t_ap, t_ref / t_ap, maxd, meand, nz);
    }
    return 0;
}
