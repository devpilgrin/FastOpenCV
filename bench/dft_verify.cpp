#include <opencv2/core.hpp>
#include <cstdio>
#include <cmath>
namespace fastcv { void dftR2C(const cv::Mat&, cv::Mat&); }
using namespace cv;
int main() {
    RNG rng(42);
    Mat img(1024, 1024, CV_32F);
    rng.fill(img, RNG::UNIFORM, 0, 1);
    Mat cvOut, fOut;
    dft(img, cvOut, DFT_COMPLEX_OUTPUT);
    fastcv::dftR2C(img, fOut);
    // интерьер: y в [1..H/2], x в [1..W/2-1] - CCS и FFTW layout совпадают
    double md = 0;
    for (int y = 1; y <= 512; y += 37)
        for (int x = 1; x < 511; x += 13) {
            Vec2f a = cvOut.at<Vec2f>(y, x), b = fOut.at<Vec2f>(y, x);
            md = std::max(md, (double)std::abs(a[0] - b[0]));
            md = std::max(md, (double)std::abs(a[1] - b[1]));
        }
    printf("r2c interior maxDiff: %.3g (ожидание ~1e-4 для float)\n", md);
    return 0;
}
