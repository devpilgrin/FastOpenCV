#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv {
void sortRows32f(const cv::Mat&, cv::Mat&, bool);
void sortIdxRows32f(const cv::Mat&, cv::Mat&, bool);
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
    RNG rng(7);
    Mat img(1080, 1920, CV_32FC1);
    rng.fill(img, RNG::UNIFORM, 0, 1);
    Mat ref, fast;
    sort(img, ref, SORT_EVERY_ROW | SORT_ASCENDING);
    fastcv::sortRows32f(img, fast, true);
    double d = norm(ref, fast, NORM_INF);
    double t_ref = timeit(10, [&] { sort(img, ref, SORT_EVERY_ROW | SORT_ASCENDING); });
    double t_fast = timeit(10, [&] { fastcv::sortRows32f(img, fast, true); });
    printf("sort rows:    cv %7.3f ms | fastcv %7.3f ms | %5.1fx | diff %.1e\n", t_ref, t_fast, t_ref / t_fast, d);

    Mat small = img(Rect(0, 0, 1920, 256)).clone();
    Mat refi, fasti;
    sortIdx(small, refi, SORT_EVERY_ROW | SORT_ASCENDING);
    fastcv::sortIdxRows32f(small, fasti, true);
    // значения по индексам должны совпадать (порядок равных элементов может отличаться)
    double maxd = 0;
    for (int y = 0; y < small.rows; y++)
        for (int x = 0; x < small.cols; x++)
            maxd = std::max(maxd, (double)std::abs(small.at<float>(y, refi.at<int>(y, x)) - small.at<float>(y, fasti.at<int>(y, x))));
    t_ref = timeit(10, [&] { sortIdx(small, refi, SORT_EVERY_ROW | SORT_ASCENDING); });
    t_fast = timeit(10, [&] { fastcv::sortIdxRows32f(small, fasti, true); });
    printf("sortIdx rows: cv %7.3f ms | fastcv %7.3f ms | %5.1fx | valDiff %.1e\n", t_ref, t_fast, t_ref / t_fast, maxd);
    return 0;
}
