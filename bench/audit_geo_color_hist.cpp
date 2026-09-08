#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <chrono>
#include <cstdio>
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
static void rep(const char* name, double ms, double bytes) {
    printf("%-36s %8.3f ms  %7.2f GB/s\n", name, ms, bytes / ms / 1e6);
}
int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1), img3(1080, 1920, CV_8UC3), fimg(1080, 1920, CV_32FC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    img.convertTo(fimg, CV_32F);
    const double MB1 = 1080.0 * 1920;
    Mat dst, dst3, f32;
    Mat M = getRotationMatrix2D({960, 540}, 10, 1.0);

    printf("== guidedFilter компоненты ==\n");
    rep("boxFilter 32FC1 15x15", timeit(20, [&]{ boxFilter(fimg, f32, -1, {15,15}); }), MB1*4);
    rep("guidedFilter r=8 eps=1e-4", timeit(10, [&]{ ximgproc::guidedFilter(img3, img3, dst3, 8, 1e-4); }), MB1*6);

    printf("\n== Геометрия ==\n");
    rep("warpAffine LINEAR 8UC1", timeit(30, [&]{ warpAffine(img, dst, M, {1920,1080}); }), MB1*2);
    rep("warpAffine LINEAR 8UC3", timeit(30, [&]{ warpAffine(img3, dst3, M, {1920,1080}); }), MB1*6);
    { Mat M3 = Mat::eye(3,3,CV_64F); M.copyTo(M3(Rect(0,0,3,2))); M3.at<double>(2,2)=1; rep("warpPerspective 8UC1", timeit(30, [&]{ warpPerspective(img, dst, M3, {1920,1080}); }), MB1*2); }
    rep("resize AREA 2x down", timeit(30, [&]{ resize(img3, dst3, {960,540}, 0,0, INTER_AREA); }), MB1*3);
    rep("resize AREA 4x down", timeit(30, [&]{ resize(img3, dst3, {480,270}, 0,0, INTER_AREA); }), MB1*3);
    rep("resize LINEAR 2x down 8UC3", timeit(30, [&]{ resize(img3, dst3, {960,540}, 0,0, INTER_LINEAR); }), MB1*3);
    rep("resize LANCZOS4 2x down", timeit(30, [&]{ resize(img3, dst3, {960,540}, 0,0, INTER_LANCZOS4); }), MB1*3);
    rep("resize 2x up LINEAR 8UC3", timeit(30, [&]{ resize(img3, dst3, {3840,2160}, 0,0, INTER_LINEAR); }), MB1*12);
    rep("rotate90 8UC3", timeit(30, [&]{ rotate(img3, dst3, ROTATE_90_CLOCKWISE); }), MB1*6);
    Mat map1, map2;
    {
        Mat mr(1080, 1920, CV_32F), mc(1080, 1920, CV_32F);
        for (int y = 0; y < 1080; y++) for (int x = 0; x < 1920; x++) {
            mr.at<float>(y,x) = std::min(1079.f, y * 1.0f + 3.f);
            mc.at<float>(y,x) = std::min(1919.f, x * 0.5f + 100.f);
        }
        map1 = mc; map2 = mr;
    }
    rep("remap bilinear 8UC1", timeit(30, [&]{ remap(img, dst, map1, map2, INTER_LINEAR); }), MB1*2);
    rep("remap bilinear 8UC3", timeit(30, [&]{ remap(img3, dst3, map1, map2, INTER_LINEAR); }), MB1*6);

    printf("\n== Цвет ==\n");
    rep("cvtColor BGR2GRAY", timeit(30, [&]{ cvtColor(img3, dst, COLOR_BGR2GRAY); }), MB1*4);
    rep("cvtColor BGR2HSV", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_BGR2HSV); }), MB1*6);
    rep("cvtColor BGR2Lab", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_BGR2Lab); }), MB1*6);
    rep("cvtColor BGR2YUV", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_BGR2YUV); }), MB1*6);
    rep("demosaic BGGR2BGR", timeit(30, [&]{ demosaicing(img, dst3, COLOR_BayerBGGR2BGR); }), MB1*4);
    {
        std::vector<Mat> ch;
        rep("split 8UC3", timeit(30, [&]{ split(img3, ch); }), MB1*6);
        rep("merge 8UC3", timeit(30, [&]{ merge(ch, dst3); }), MB1*6);
    }

    printf("\n== Гистограммы/статистика ==\n");
    {
        Mat hist; int hs = 256; float r_[2] = {0, 256}; const float* ranges = r_; int ch0 = 0;
        rep("calcHist 1D 256", timeit(30, [&]{ calcHist(&img, 1, &ch0, Mat(), hist, 1, &hs, &ranges); }), MB1);
    }
    rep("integral 8U->32S", timeit(30, [&]{ integral(img, dst); }), MB1*5);
    {
        Mat sq; rep("integral sqsum", timeit(30, [&]{ integral(img, dst, sq); }), MB1*13);
    }
    rep("CLAHE 8x8", timeit(10, [&]{ Ptr<CLAHE> c = createCLAHE(2.0, {8,8}); c->apply(img, dst); }), MB1*2);
    return 0;
}
