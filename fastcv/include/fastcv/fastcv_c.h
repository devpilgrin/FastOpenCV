// FastOpenCV: C API для FFI (Rust и др.). Плоские указатели, без cv::Mat.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Все функции: src/dst - непрерывные буферы rows x cols x channels.
// dst должен быть выделен вызывающим (размер как у src, кроме sobel: int16).
void fastcv_median_blur_8u(const uint8_t* src, uint8_t* dst, int rows, int cols, int ksize);
void fastcv_median_blur_8uc3(const uint8_t* src, uint8_t* dst, int rows, int cols, int ksize);
void fastcv_sobel_8u16s(const uint8_t* src, int16_t* dst, int rows, int cols, int channels,
                        int dx, int dy, int ksize); // ksize>=3; ksize==3 -> вызов cv::Sobel внутри не делаем, only large path
void fastcv_sep_filter_8u(const uint8_t* src, uint8_t* dst, int rows, int cols, int channels,
                          const float* kx, int kw, const float* ky, int kh);
void fastcv_calc_hist_8u(const uint8_t* src, int rows, int cols, float* hist256);
void fastcv_resize_area_int_8u(const uint8_t* src, uint8_t* dst, int rows, int cols, int channels, int factor);
double fastcv_sum_8u_c1(const uint8_t* src, int rows, int cols); // адаптивно (smart)
void fastcv_dft_r2c_32f(const float* src, float* dst_interleaved, int rows, int cols); // dst: rows x (cols/2+1) x 2 float (FFTW layout)

#ifdef __cplusplus
}
#endif
