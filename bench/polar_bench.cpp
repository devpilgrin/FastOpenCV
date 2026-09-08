#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
namespace fastcv { void phase32f(const cv::Mat&, const cv::Mat&, cv::Mat&, bool);
void cartToPolar32f(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, bool); }
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
    Mat x(1080, 1920, CV_32F), y(1080, 1920, CV_32F);
    rng.fill(x, RNG::NORMAL, 0, 100);
    rng.fill(y, RNG::NORMAL, 0, 100);
    Mat a, b, d;
    phase(x, y, a, true);
    fastcv::phase32f(x, y, b, true);
    absdiff(a, b, d);
    double md; minMaxLoc(d, nullptr, &md);
    // точность относительно libm atan2
    Mat ref(x.size(), CV_32F);
    for (int r = 0; r < 1080; r += 53)
        for (int i = 0; i < 1920; i++)
            ref.at<float>(r, i) = (float)(std::atan2(y.at<float>(r, i), x.at<float>(r, i)) * 180.0 / CV_PI);
    Mat bsamp, rsamp, df;
    {   // выборка строк для сравнения с libm
        std::vector<Mat> rows_b, rows_r;
        for (int r = 0; r < 1080; r += 53) { rows_b.push_back(b.row(r)); rows_r.push_back(ref.row(r)); }
        vconcat(rows_b, bsamp); vconcat(rows_r, rsamp);
    }
    absdiff(bsamp, rsamp, df);
    double mdref; minMaxLoc(df, nullptr, &mdref);
    double t1 = timeit(30, [&]{ phase(x, y, a, true); });
    double t2 = timeit(30, [&]{ fastcv::phase32f(x, y, b, true); });
    printf("phase: cv %7.3f | fastcv %7.3f | %5.1fx | vs cv(LUT) maxDiff %.2f град | vs libm %.2g град\n",
           t1, t2, t1/t2, md, mdref);
    Mat m1, a1, m2, a2;
    cartToPolar(x, y, m1, a1, true);
    fastcv::cartToPolar32f(x, y, m2, a2, true);
    absdiff(a1, a2, d); minMaxLoc(d, nullptr, &md);
    absdiff(m1, m2, d); double mdm; minMaxLoc(d, nullptr, &mdm);
    t1 = timeit(30, [&]{ cartToPolar(x, y, m1, a1, true); });
    t2 = timeit(30, [&]{ fastcv::cartToPolar32f(x, y, m2, a2, true); });
    printf("cartToPolar: cv %7.3f | fastcv %7.3f | %5.1fx | angle maxDiff %.2f | mag relDiff %.2g\n",
           t1, t2, t1/t2, md, mdm / 100.0);
    return 0;
}
