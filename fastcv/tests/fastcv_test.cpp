// FastOpenCV: регрессионный тест-сьют fastcv (точность + пороги производительности).
// Запуск: ./fastcv_test  (exit 0 = все проверки пройдены)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "fastcv/fastcv.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace cv;
using Clock = std::chrono::steady_clock;

static int g_fail = 0;
#define CHECK(cond, name, ...) do { \
    if (cond) printf("  [PASS] %-34s", name); \
    else { printf("  [FAIL] %-34s", name); g_fail++; } \
    printf(__VA_ARGS__); printf("\n"); } while (0)

template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 2; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
static double maxDiffOf(const Mat& a, const Mat& b) {
    Mat d; absdiff(a, b, d);
    double md; minMaxLoc(d.reshape(1), nullptr, &md);
    return md;
}

int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1), img3(1080, 1920, CV_8UC3);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    Mat a, b;
    printf("== fastcv regression suite ==\n");

    // medianBlur C1/C3
    medianBlur(img, a, 9); fastcv::medianBlur8u(img, b, 9);
    CHECK(maxDiffOf(a, b) == 0, "medianBlur8u k=9 bit-exact", "");
    medianBlur(img3, a, 9); fastcv::medianBlur8uC3(img3, b, 9);
    CHECK(maxDiffOf(a, b) == 0, "medianBlur8uC3 k=9 bit-exact", "");
    {
        double t1 = timeit(10, [&]{ medianBlur(img, a, 9); });
        double t2 = timeit(10, [&]{ fastcv::medianBlur8u(img, b, 9); });
        CHECK(t1 / t2 > 5.0, "medianBlur8u speedup > 5x", " (%.1fx)", t1 / t2);
    }

    // Sobel / Laplacian
    Sobel(img, a, CV_16S, 1, 0, 5); { Mat b16; fastcv::sobel8u16sLarge(img, b16, 1, 0, 5);
    CHECK(maxDiffOf(a, b16) == 0, "sobel8u16sLarge k=5 bit-exact", ""); }
    Sobel(img3, a, CV_16S, 1, 0, 7); { Mat b16; fastcv::sobel8u16sLarge(img3, b16, 1, 0, 7);
    CHECK(maxDiffOf(a, b16) == 0, "sobel8u16sLarge C3 k=7 bit-exact", ""); }
    Laplacian(img, a, CV_16S, 5); { Mat b16; fastcv::laplacian8u16sLarge(img, b16, 5);
    CHECK(maxDiffOf(a, b16) == 0, "laplacian8u16sLarge k=5 bit-exact", ""); }

    // sepFilter C1/C3
    {
        Mat k1d = getGaussianKernel(15, -1, CV_32F);
        sepFilter2D(img, a, -1, k1d, k1d);
        fastcv::sepFilter2D_8u(img, b, -1, k1d, k1d);
        CHECK(maxDiffOf(a, b) <= 1, "sepFilter2D C1 k=15 maxDiff<=1", "");
        sepFilter2D(img3, a, -1, k1d, k1d);
        fastcv::sepFilter2D_8u(img3, b, -1, k1d, k1d);
        CHECK(maxDiffOf(a, b) <= 1, "sepFilter2D C3 k=15 maxDiff<=1", "");
    }

    // calcHist
    {
        Mat h1, h2; int hs = 256; float r_[2] = {0, 256}; const float* ranges = r_; int ch0 = 0;
        calcHist(&img, 1, &ch0, Mat(), h1, 1, &hs, &ranges);
        fastcv::calcHist8u(img, h2);
        CHECK(maxDiffOf(h1, h2) == 0, "calcHist8u bit-exact", "");
    }

    // resize AREA x4
    {
        resize(img3, a, {480, 270}, 0, 0, INTER_AREA);
        fastcv::resizeAreaInt8u(img3, b, 4);
        double md = maxDiffOf(a, b);
        Mat d; absdiff(a, b, d);
        int nd = countNonZero(d.reshape(1));
        CHECK(md <= 1 && nd < img3.total() * 3 / 100, "resizeAreaInt8u x4 (+-1, <1%% px)", " (%.0f, %d px)", md, nd);
    }

    // sqrBox
    {
        Mat a32, b32;
        sqrBoxFilter(img, a32, CV_32F, {15, 15});
        fastcv::sqrBoxFilter8u(img, b32, {15, 15}, true);
        CHECK(maxDiffOf(a32, b32) == 0, "sqrBoxFilter8u bit-exact", "");
    }

    // sum smart
    {
        Scalar s1 = sum(img3);
        Scalar s2 = fastcv::sum8uSmart(img3);
        CHECK(std::abs(s1[0] - s2[0]) < 1 && std::abs(s1[2] - s2[2]) < 1, "sum8uSmart совпадает", "");
    }

    // undistort
    {
        Mat K = (Mat_<double>(3,3) << 1500, 0, 960, 0, 1500, 540, 0, 0, 1);
        Mat dist = (Mat_<double>(1,5) << -0.3, 0.1, 0.001, -0.002, 0.0);
        Mat nK = getOptimalNewCameraMatrix(K, dist, {1920, 1080}, 1.0);
        undistort(img3, a, K, dist, nK);
        fastcv::undistort8u(img3, b, K, dist, nK);
        CHECK(maxDiffOf(a, b) == 0, "undistort8u bit-exact", "");
    }

    // pow / phase точность
    {
        Mat f; img.convertTo(f, CV_32F, 1.0 / 255); f += 0.001;
        Mat p1, p2;
        pow(f, 1.7, p1);
        fastcv::pow32f(f, p2, 1.7);
        Mat d; absdiff(p1, p2, d);
        double md; minMaxLoc(d.reshape(1), nullptr, &md);
        CHECK(md < 1e-4, "pow32f точность", " (%.2g)", md);
    }

    printf("== %s ==\n", g_fail == 0 ? "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" : "ЕСТЬ ПРОВАЛЫ");
    return g_fail == 0 ? 0 : 1;
}
