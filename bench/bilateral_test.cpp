// Отладка корректности fastcv::bilateralFilter8u против наивной double-реализации
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cmath>

namespace fastcv { void bilateralFilter8u(const cv::Mat&, cv::Mat&, int, double, double); }

using namespace cv;

static Mat naive(const Mat& src, int d, double sc, double ss) {
    int r = d / 2;
    Mat padded, dst(src.size(), CV_8UC1);
    copyMakeBorder(src, padded, r, r, r, r, BORDER_REPLICATE);
    std::vector<double> sw(d * d);
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            sw[(dy + r) * d + dx + r] = std::exp(-(dx * dx + dy * dy) / (2.0 * ss * ss));
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            double c = padded.at<uchar>(y + r, x + r), sum = 0, ws = 0;
            for (int dy = -r; dy <= r; dy++)
                for (int dx = -r; dx <= r; dx++) {
                    double v = padded.at<uchar>(y + r + dy, x + r + dx);
                    double w = sw[(dy + r) * d + dx + r] * std::exp(-(v - c) * (v - c) / (2.0 * sc * sc));
                    sum += w * v; ws += w;
                }
            dst.at<uchar>(y, x) = saturate_cast<uchar>(sum / ws);
        }
    return dst;
}

static void report(const char* name, const Mat& a, const Mat& b) {
    Mat diff; absdiff(a, b, diff);
    double maxd; minMaxLoc(diff, nullptr, &maxd);
    printf("%-24s maxDiff %4.0f mean %6.4f\n", name, maxd, mean(diff)[0]);
}

int main() {
    RNG rng(42);
    Mat img(64, 64, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    int d = 9; double sc = 50, ss = 50;

    Mat ref_cv, fast, ref_naive;
    bilateralFilter(img, ref_cv, d, sc, ss);
    fastcv::bilateralFilter8u(img, fast, d, sc, ss);
    ref_naive = naive(img, d, sc, ss);

    report("fastcv vs naive", fast, ref_naive);
    report("fastcv vs opencv", fast, ref_cv);
    report("opencv vs naive", ref_cv, ref_naive);
    return 0;
}
