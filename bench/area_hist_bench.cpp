#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv {
void calcHist8u(const cv::Mat&, cv::Mat&);
void resizeAreaInt8u(const cv::Mat&, cv::Mat&, int);
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
    Mat img(1080, 1920, CV_8UC1), img3(1080, 1920, CV_8UC3);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    // calcHist
    {
        Mat h1, h2; int hs = 256; float r_[2] = {0, 256}; const float* ranges = r_; int ch0 = 0;
        calcHist(&img, 1, &ch0, Mat(), h1, 1, &hs, &ranges);
        fastcv::calcHist8u(img, h2);
        Mat d; absdiff(h1, h2, d);
        double md; minMaxLoc(d, nullptr, &md);
        double t1 = timeit(30, [&]{ calcHist(&img, 1, &ch0, Mat(), h1, 1, &hs, &ranges); });
        double t2 = timeit(30, [&]{ fastcv::calcHist8u(img, h2); });
        printf("calcHist 1D: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f\n", t1, t2, t1/t2, md);
    }
    // resize AREA int factors
    for (int f : {2, 4}) {
        for (int c : {1, 3}) {
            Mat src = c == 1 ? img : img3, a, b, d;
            resize(src, a, {src.cols / f, src.rows / f}, 0, 0, INTER_AREA);
            fastcv::resizeAreaInt8u(src, b, f);
            absdiff(a, b, d);
            double md; minMaxLoc(d, nullptr, &md);
            int nd = countNonZero(d.reshape(1));
            double t1 = timeit(30, [&]{ resize(src, a, {src.cols / f, src.rows / f}, 0, 0, INTER_AREA); });
            double t2 = timeit(30, [&]{ fastcv::resizeAreaInt8u(src, b, f); });
            printf("resize AREA x%d C%d: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f (%d px)\n",
                   f, c, t1, t2, t1/t2, md, nd);
        }
    }
    return 0;
}
