#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
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
    Mat img(1080, 1920, CV_8UC1), img3(1080, 1920, CV_8UC3);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    rng.fill(img3, RNG::UNIFORM, 0, 256);
    Mat dst, dst3;
    printf("== photo ==\n");
    printf("edgePreserving NORMCONV: %8.2f ms\n", timeit(5, [&]{ edgePreservingFilter(img3, dst3, 1, 60, 0.4f); }));
    printf("edgePreserving RECURS:   %8.2f ms\n", timeit(5, [&]{ edgePreservingFilter(img3, dst3, 2, 60, 0.4f); }));
    printf("detailEnhance:           %8.2f ms\n", timeit(5, [&]{ detailEnhance(img3, dst3); }));
    printf("pencilSketch:            %8.2f ms\n", timeit(5, [&]{ Mat g, c; pencilSketch(img3, g, c); }));
    printf("stylization:             %8.2f ms\n", timeit(5, [&]{ stylization(img3, dst3); }));
    printf("inpaint (Telea 1%px):    %8.2f ms\n", timeit(5, [&]{ Mat m = Mat::zeros(1080, 1920, CV_8U); m(Rect(400,300,100,100)) = 255; inpaint(img3, m, dst3, 3, INPAINT_TELEA); }));
    printf("== undistort ==\n");
    {
        Mat K = (Mat_<double>(3,3) << 1500, 0, 960, 0, 1500, 540, 0, 0, 1);
        Mat dist = (Mat_<double>(1,5) << -0.3, 0.1, 0.001, -0.002, 0.0);
        Mat nK = getOptimalNewCameraMatrix(K, dist, {1920,1080}, 1.0);
        printf("undistort (map+remap): %8.2f ms\n", timeit(20, [&]{ undistort(img3, dst3, K, dist, nK); }));
        Mat m1, m2;
        printf("initUndistortRectifyMap: %8.2f ms\n", timeit(20, [&]{ initUndistortRectifyMap(K, dist, Mat(), nK, {1920,1080}, CV_32FC1, m1, m2); }));
    }
    printf("== YUV/YCrCb ==\n");
    printf("cvtColor BGR2YCrCb:      %8.3f ms\n", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_BGR2YCrCb); }));
    printf("cvtColor YCrCb2BGR:      %8.3f ms\n", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_YCrCb2BGR); }));
    printf("cvtColor BGR2YUV_I420:   %8.3f ms\n", timeit(30, [&]{ cvtColor(img3, dst, COLOR_BGR2YUV_I420); }));
    printf("cvtColor BGR2Luv:        %8.3f ms\n", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_BGR2Luv); }));
    printf("cvtColor BGR2XYZ:        %8.3f ms\n", timeit(30, [&]{ cvtColor(img3, dst3, COLOR_BGR2XYZ); }));
    return 0;
}
