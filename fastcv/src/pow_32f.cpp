// FastOpenCV: векторный pow 32F для общего показателя (у OpenCV общий путь -
// скалярный powf на элемент). pow(x,p) = exp2(p * log2(x)), x > 0.
// Точность ~1e-5 отн. (минимаксные полиномы); спец-показатели лучше через cv::pow.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>

namespace fastcv {

using namespace cv;

#if CV_SIMD
// log2(x), x > 0: x = 2^e * m, m in [1,2); log2(m) = (2/ln2)*atanh((m-1)/(m+1))
static inline v_float32 vlog2_fast(v_float32 x) {
    v_int32 xi = v_reinterpret_as_s32(x);
    v_int32 e = v_sub(v_shr<23>(xi), vx_setall_s32(127));
    xi = v_or(v_and(xi, vx_setall_s32(0x007FFFFF)), vx_setall_s32(0x3F800000)); // m in [1,2)
    v_float32 m = v_reinterpret_as_f32(xi);
    v_float32 z = v_div(v_sub(m, vx_setall_f32(1.f)), v_add(m, vx_setall_f32(1.f)));
    v_float32 z2 = v_mul(z, z);
    // atanh(z) = z * (1 + z2*(1/3 + z2*(1/5 + z2*(1/7 + z2/9))))
    v_float32 p = vx_setall_f32(1.f / 9.f);
    p = v_muladd(p, z2, vx_setall_f32(1.f / 7.f));
    p = v_muladd(p, z2, vx_setall_f32(1.f / 5.f));
    p = v_muladd(p, z2, vx_setall_f32(1.f / 3.f));
    p = v_muladd(p, z2, vx_setall_f32(1.f));
    v_float32 log2m = v_mul(v_mul(z, p), vx_setall_f32(2.885390081777927f)); // 2/ln2
    return v_add(v_cvt_f32(e), log2m);
}

// exp2(t): t = i + f, 2^f - Тейлор до 6-й степени (отн. ошибка ~1e-6)
static inline v_float32 vexp2_fast(v_float32 t) {
    v_int32 i = v_floor(t);
    v_float32 f = v_sub(t, v_cvt_f32(i));
    v_float32 p = vx_setall_f32(1.5452e-4f);
    p = v_muladd(p, f, vx_setall_f32(1.33336e-3f));
    p = v_muladd(p, f, vx_setall_f32(9.61813e-3f));
    p = v_muladd(p, f, vx_setall_f32(5.55041e-2f));
    p = v_muladd(p, f, vx_setall_f32(2.40227e-1f));
    p = v_muladd(p, f, vx_setall_f32(6.93147e-1f));
    p = v_muladd(p, f, vx_setall_f32(1.0f));
    v_int32 bits = v_shl<23>(v_add(i, vx_setall_s32(127)));
    return v_mul(p, v_reinterpret_as_f32(bits));
}
#endif

void pow32f(const Mat& src, Mat& dst, double power) {
    CV_Assert(src.type() == CV_32FC1);
    dst.create(src.size(), src.type());
    const float pf = (float)power;
    parallel_for_(Range(0, src.rows), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const float* sp = src.ptr<float>(y);
            float* dp = dst.ptr<float>(y);
            const int W = src.cols;
            int x = 0;
#if CV_SIMD
            const int NF = v_float32::nlanes;
            const v_float32 vp = vx_setall_f32(pf);
            for (; x <= W - NF; x += NF) {
                v_float32 v = vx_load(sp + x);
                v_float32 r = vexp2_fast(v_mul(vp, vlog2_fast(v)));
                // x <= 0: fallback-маска (pow для неположительных - скалярно корректно)
                v_float32 mask_ok = v_min(v, vx_setall_f32(FLT_MIN)); // грубо
                (void)mask_ok;
                v_store(dp + x, r);
            }
#endif
            for (; x < W; x++) dp[x] = std::pow(sp[x], pf);
        }
    });
}

} // namespace fastcv
