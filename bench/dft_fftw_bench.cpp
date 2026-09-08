#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
namespace fastcv { void dftR2C(const cv::Mat&, cv::Mat&); void dftC2C(const cv::Mat&, cv::Mat&, bool); }
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    f(); // первый вызов - план FFTW
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
int main() {
    RNG rng(42);
    for (int n : {1024, 2048}) {
        Mat img(n, n, CV_32F);
        rng.fill(img, RNG::UNIFORM, 0, 1);
        Mat cvOut, fOut;
        double tcv = timeit(20, [&]{ dft(img, cvOut, DFT_COMPLEX_OUTPUT); });
        double tf = timeit(20, [&]{ fastcv::dftR2C(img, fOut); });
        // верификация: cv CCS vs FFTW layout. Сравним магнитуды нулевого столбца/строки отдельно,
        // а для внутренней области - напрямую (CCS-упаковка совпадает с FFTW для [1..H-1][1..W/2])
        // Простая проверка: энергия спектра
        double e1 = 0, e2 = 0;
        Mat mag1, mag2;
        std::vector<Mat> ch;
        split(cvOut, ch); magnitude(ch[0], ch[1], mag1);
        split(fOut, ch); magnitude(ch[0], ch[1], mag2);
        e1 = sum(mag1)[0]; e2 = sum(mag2)[0];
        printf("dft r2c %5d: cv %7.3f ms | fftw %7.3f ms | %5.1fx | энергии cv=%.4g fftw=%.4g (CCS vs FFTW layout)\n",
               n, tcv, tf, tcv / tf, e1, e2);
    }
    // c2c
    for (int n : {1024}) {
        Mat img(n, n, CV_32F), imgc;
        rng.fill(img, RNG::UNIFORM, 0, 1);
        Mat ch[2] = {img, Mat::zeros(n, n, CV_32F)};
        merge(ch, 2, imgc);
        Mat cvOut, fOut;
        double tcv = timeit(20, [&]{ dft(imgc, cvOut); });
        double tf = timeit(20, [&]{ fastcv::dftC2C(imgc, fOut, false); });
        Mat d; absdiff(cvOut, fOut, d);
        double md; minMaxLoc(d.reshape(1), nullptr, &md);
        printf("dft c2c %5d: cv %7.3f ms | fftw %7.3f ms | %5.1fx | maxDiff %.3g\n", n, tcv, tf, tcv / tf, md);
    }
    return 0;
}
