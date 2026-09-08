// Бенч: cv::sum vs fastcv::sum8u, cv::LUT vs fastcv::lut8u
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>

namespace fastcv {
void sum8u(const cv::Mat&, double*);
void lut8u(const cv::Mat&, const uchar*, cv::Mat&);
}

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

int main() {
    RNG rng(42);
    Mat img1(1080, 1920, CV_8UC1), img3(1080, 1920, CV_8UC3);
    rng.fill(img1, RNG::UNIFORM, 0, 256);
    rng.fill(img3, RNG::UNIFORM, 0, 256);

    // --- sum ---
    for (const Mat& m : {img1, img3}) {
        int cn = m.channels();
        Scalar ref = sum(m);
        double fast[4] = {0};
        fastcv::sum8u(m, fast);
        double t_ref = timeit(50, [&] { sum(m); });
        double t_fast = timeit(50, [&] { fastcv::sum8u(m, fast); });
        double err = 0;
        for (int c = 0; c < cn; c++) err = std::max(err, std::abs(ref[c] - fast[c]));
        printf("sum 8UC%d: OpenCV %7.3f ms | fastcv %7.3f ms | speedup %5.1fx | err %.0f\n",
               cn, t_ref, t_fast, t_ref / t_fast, err);
    }

    // --- LUT ---
    Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) lut.at<uchar>(i) = (uchar)((i * 7 + 13) & 0xFF);
    for (const Mat& m : {img1, img3}) {
        int cn = m.channels();
        Mat ref, fast;
        LUT(m, lut, ref);
        fastcv::lut8u(m, lut.ptr(), fast);
        Mat diff; absdiff(ref, fast, diff);
        double maxd; minMaxLoc(diff, nullptr, &maxd);
        double t_ref = timeit(50, [&] { LUT(m, lut, ref); });
        double t_fast = timeit(50, [&] { fastcv::lut8u(m, lut.ptr(), fast); });
        printf("LUT 8UC%d: OpenCV %7.3f ms | fastcv %7.3f ms | speedup %5.1fx | maxDiff %.0f\n",
               cn, t_ref, t_fast, t_ref / t_fast, maxd);
    }
    return 0;
}
