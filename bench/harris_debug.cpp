// Локализация расхождения fastcv::cornerHarris8u vs cv::cornerHarris
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>

namespace fastcv { void cornerHarris8u(const cv::Mat&, cv::Mat&, int, double); }
using namespace cv;

int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    int bs = 3;
    Mat ref, fast;
    cornerHarris(img, ref, bs, 3, 0.04);
    fastcv::cornerHarris8u(img, fast, bs, 0.04);
    Mat diff; absdiff(ref, fast, diff);

    int b = 8; // border zone
    Mat centerDiff = diff(Rect(b, b, diff.cols - 2 * b, diff.rows - 2 * b));
    double maxAll, maxCenter;
    minMaxLoc(diff, nullptr, &maxAll);
    minMaxLoc(centerDiff, nullptr, &maxCenter);
    printf("maxDiff all=%.6g center=%.6g\n", maxAll, maxCenter);

    // где максимум
    Point p; minMaxLoc(diff, nullptr, nullptr, nullptr, &p);
    printf("max at (%d,%d): ref=%.6g fast=%.6g\n", p.x, p.y,
           ref.at<float>(p), fast.at<float>(p));
    // и в центре
    minMaxLoc(centerDiff, nullptr, nullptr, nullptr, &p);
    p.x += b; p.y += b;
    printf("center max at (%d,%d): ref=%.6g fast=%.6g\n", p.x, p.y,
           ref.at<float>(p), fast.at<float>(p));
    return 0;
}
