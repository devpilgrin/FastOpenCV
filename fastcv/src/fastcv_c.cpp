// FastOpenCV: C API - реализация поверх C++ кернелов.
#include "fastcv/fastcv_c.h"
#include "fastcv/fastcv.hpp"
#include <opencv2/imgproc.hpp>

using namespace cv;

extern "C" {

void fastcv_median_blur_8u(const uint8_t* src, uint8_t* dst, int rows, int cols, int ksize) {
    Mat s(rows, cols, CV_8UC1, (void*)src), d(rows, cols, CV_8UC1, dst);
    if (ksize <= 5) { Mat t; cv::medianBlur(s, t, ksize); t.copyTo(d); }
    else fastcv::medianBlur8u(s, d, ksize);
}

void fastcv_median_blur_8uc3(const uint8_t* src, uint8_t* dst, int rows, int cols, int ksize) {
    Mat s(rows, cols, CV_8UC3, (void*)src), d(rows, cols, CV_8UC3, dst);
    if (ksize <= 5) { Mat t; cv::medianBlur(s, t, ksize); t.copyTo(d); }
    else fastcv::medianBlur8uC3(s, d, ksize);
}

void fastcv_sobel_8u16s(const uint8_t* src, int16_t* dst, int rows, int cols, int channels,
                        int dx, int dy, int ksize) {
    Mat s(rows, cols, CV_MAKETYPE(CV_8U, channels), (void*)src);
    Mat d(rows, cols, CV_MAKETYPE(CV_16S, channels), dst);
    if (ksize == 3) { Mat t; cv::Sobel(s, t, CV_16S, dx, dy, 3); t.copyTo(d); }
    else fastcv::sobel8u16sLarge(s, d, dx, dy, ksize);
}

void fastcv_sep_filter_8u(const uint8_t* src, uint8_t* dst, int rows, int cols, int channels,
                          const float* kx, int kw, const float* ky, int kh) {
    Mat s(rows, cols, CV_MAKETYPE(CV_8U, channels), (void*)src);
    Mat d(rows, cols, CV_MAKETYPE(CV_8U, channels), dst);
    Mat kxm(kw, 1, CV_32F, (void*)kx), kym(kh, 1, CV_32F, (void*)ky);
    fastcv::sepFilter2D_8u(s, d, -1, kxm, kym);
}

void fastcv_calc_hist_8u(const uint8_t* src, int rows, int cols, float* hist256) {
    Mat s(rows, cols, CV_8UC1, (void*)src), h;
    fastcv::calcHist8u(s, h);
    memcpy(hist256, h.ptr<float>(), 256 * sizeof(float));
}

void fastcv_resize_area_int_8u(const uint8_t* src, uint8_t* dst, int rows, int cols, int channels, int factor) {
    Mat s(rows, cols, CV_MAKETYPE(CV_8U, channels), (void*)src);
    Mat d(rows / factor, cols / factor, CV_MAKETYPE(CV_8U, channels), dst);
    fastcv::resizeAreaInt8u(s, d, factor);
}

double fastcv_sum_8u_c1(const uint8_t* src, int rows, int cols) {
    Mat s(rows, cols, CV_8UC1, (void*)src);
    return fastcv::sum8uSmart(s)[0];
}

void fastcv_dft_r2c_32f(const float* src, float* dst_interleaved, int rows, int cols) {
    Mat s(rows, cols, CV_32F, (void*)src), d;
    fastcv::dftR2C(s, d);
    memcpy(dst_interleaved, d.data, (size_t)rows * (cols / 2 + 1) * 2 * sizeof(float));
}

} // extern "C"
