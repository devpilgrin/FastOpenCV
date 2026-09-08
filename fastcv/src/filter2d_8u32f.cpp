// FastOpenCV: filter2D 8UC1 -> 32FC1 (float kernel). У OpenCV эта комбинация
// типов явно направлена в FilterNoVec (скаляр). Здесь: SIMD по пикселям
// (u8 -> f32, muladd на тап) + parallel_for по строкам. Порядок суммирования
// по тапам как у OpenCV (ky, kx по возрастанию) -> побитовая близость.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void filter2D_8u32f(const Mat& src, Mat& dst, const Mat& kernel, Point anchor, double delta) {
    CV_Assert(src.type() == CV_8UC1 && kernel.type() == CV_32FC1);
    const int kw = kernel.cols, kh = kernel.rows;
    if (anchor.x < 0) anchor.x = kw / 2;
    if (anchor.y < 0) anchor.y = kh / 2;
    dst.create(src.size(), CV_32FC1);

    // паддинг BORDER_REFLECT_101 (BORDER_DEFAULT у filter2D)
    Mat padded;
    copyMakeBorder(src, padded, anchor.y, kh - 1 - anchor.y, anchor.x, kw - 1 - anchor.x,
                   BORDER_REFLECT_101);

    parallel_for_(Range(0, src.rows), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            float* dp = dst.ptr<float>(y);
            const int W = src.cols;
            int x = 0;
#if CV_SIMD
            const int CN = v_uint16::nlanes; // 32 px за итерацию на AVX-512
            for (; x <= W - CN; x += CN) {
                v_float32 acc0 = vx_setzero_f32(), acc1 = vx_setzero_f32();
                for (int ky = 0; ky < kh; ky++) {
                    const uchar* srow = padded.ptr<uchar>(y + ky) + x;
                    const float* krow = kernel.ptr<float>(ky);
                    for (int kx = 0; kx < kw; kx++) {
                        v_uint16 w = vx_load_expand(srow + kx); // u8 -> u16, CN линий
                        v_uint32 q0, q1;
                        v_expand(w, q0, q1);
                        v_float32 kk = vx_setall_f32(krow[kx]);
                        acc0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(q0)), kk, acc0);
                        acc1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(q1)), kk, acc1);
                    }
                }
                v_float32 vd = vx_setall_f32((float)delta);
                v_store(dp + x, v_add(acc0, vd));
                v_store(dp + x + v_float32::nlanes, v_add(acc1, vd));
            }
#endif
            for (; x < W; x++) {
                float s = 0;
                for (int ky = 0; ky < kh; ky++) {
                    const uchar* srow = padded.ptr<uchar>(y + ky) + x;
                    const float* krow = kernel.ptr<float>(ky);
                    for (int kx = 0; kx < kw; kx++)
                        s += srow[kx] * krow[kx];
                }
                dp[x] = s + (float)delta;
            }
        }
    });
}

} // namespace fastcv
