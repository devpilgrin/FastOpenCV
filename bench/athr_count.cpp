#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
namespace fastcv { void adaptiveThreshold8u(const cv::Mat&, cv::Mat&, double, int, int, int, double); }
using namespace cv;
int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    Mat photo = imread("../data/lena.jpg", IMREAD_GRAYSCALE);
    resize(photo, photo, {1920, 1080});
    for (Mat im : {img, photo}) {
        for (int bs : {11, 25}) {
            Mat a, b, d;
            adaptiveThreshold(im, a, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bs, 5);
            fastcv::adaptiveThreshold8u(im, b, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bs, 5);
            absdiff(a, b, d);
            int n = countNonZero(d);
            printf("bs=%d %s: отличаются %d из %d пикселей (%.3f%%)\n", bs,
                   im.rows == 1080 && mean(im)[0] > 120 ? "noise" : "lena ", n, im.total(), 100.0 * n / im.total());
        }
    }
    return 0;
}
