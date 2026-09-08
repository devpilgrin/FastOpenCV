#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void undistort8u(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&);
void buildUndistortMaps(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Size, cv::Mat&, cv::Mat&);
void undistortPrecomputed(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&); }
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
    Mat img3(1080, 1920, CV_8UC3);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    Mat K = (Mat_<double>(3,3) << 1500, 0, 960, 0, 1500, 540, 0, 0, 1);
    Mat dist = (Mat_<double>(1,5) << -0.3, 0.1, 0.001, -0.002, 0.0);
    Mat nK = getOptimalNewCameraMatrix(K, dist, {1920,1080}, 1.0);
    Mat a, b, d;
    undistort(img3, a, K, dist, nK);
    fastcv::undistort8u(img3, b, K, dist, nK);
    absdiff(a, b, d);
    double md; minMaxLoc(d.reshape(1), nullptr, &md);
    double t1 = timeit(30, [&]{ undistort(img3, a, K, dist, nK); });
    double t2 = timeit(30, [&]{ fastcv::undistort8u(img3, b, K, dist, nK); });
    printf("undistort one-shot: cv %7.3f | fastcv %7.3f | %5.1fx | maxDiff %.0f\n", t1, t2, t1/t2, md);
    Mat m1, m2;
    fastcv::buildUndistortMaps(K, dist, nK, {1920,1080}, m1, m2);
    double t3 = timeit(30, [&]{ fastcv::undistortPrecomputed(img3, b, m1, m2); });
    printf("undistort RT (precomputed maps): fastcv %7.3f ms | vs cv one-shot %5.1fx\n", t3, t1/t3);
    return 0;
}
