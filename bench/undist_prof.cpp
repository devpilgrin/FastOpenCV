#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cstdio>
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
    Mat img3(1080, 1920, CV_8UC3);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    Mat K = (Mat_<double>(3,3) << 1500, 0, 960, 0, 1500, 540, 0, 0, 1);
    Mat dist = (Mat_<double>(1,5) << -0.3, 0.1, 0.001, -0.002, 0.0);
    Mat nK = getOptimalNewCameraMatrix(K, dist, {1920,1080}, 1.0);
    Mat m1f, m2f, m1s, m2s, dst;
    printf("initUndistortRectifyMap 32F:  %8.3f ms\n",
           timeit(30, [&]{ initUndistortRectifyMap(K, dist, Mat(), nK, {1920,1080}, CV_32FC1, m1f, m2f); }));
    printf("initUndistortRectifyMap 16SC2:%8.3f ms\n",
           timeit(30, [&]{ initUndistortRectifyMap(K, dist, Mat(), nK, {1920,1080}, CV_16SC2, m1s, m2s); }));
    printf("convertMaps 32F->16SC2:       %8.3f ms\n",
           timeit(30, [&]{ convertMaps(m1f, m2f, m1s, m2s, CV_16SC2); }));
    printf("remap fixed 16SC2 8UC3:       %8.3f ms\n",
           timeit(30, [&]{ remap(img3, dst, m1s, m2s, INTER_LINEAR); }));
    printf("remap float 32F 8UC3:         %8.3f ms\n",
           timeit(30, [&]{ remap(img3, dst, m1f, m2f, INTER_LINEAR); }));
    printf("undistort total:              %8.3f ms\n",
           timeit(30, [&]{ undistort(img3, dst, K, dist, nK); }));
    return 0;
}
