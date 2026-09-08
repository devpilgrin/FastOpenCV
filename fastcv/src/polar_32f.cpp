// FastOpenCV: phase/cartToPolar 32F - SIMD atan2 (минимакс).
// cv::phase использует fastAtan2 (LUT, сама по себе approx ~0.3 град) и скалярна.
// Наш atan2: atan(|y/x|) минимакс + квадранты; точность ~1e-5 рад.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>

namespace fastcv {

using namespace cv;

#if CV_SIMD
// atan(t) для t in [0,1], минимаксный полином (ошибка < 1e-6)
static inline v_float32 atan_poly(v_float32 t) {
    // atan(t) ~ t * P(t^2), коэффициенты минимаксные
    const float c1 = 0.99999977f, c3 = -0.33332512f, c5 = 0.19978073f,
                c7 = -0.14209890f, c9 = 0.10663795f, c11 = -0.075264640f,
                c13 = 0.042907409f, c15 = -0.0161657820f, c17 = 0.0028662260f;
    v_float32 t2 = v_mul(t, t);
    v_float32 p = vx_setall_f32(c17);
    p = v_fma(p, t2, vx_setall_f32(c15));
    p = v_fma(p, t2, vx_setall_f32(c13));
    p = v_fma(p, t2, vx_setall_f32(c11));
    p = v_fma(p, t2, vx_setall_f32(c9));
    p = v_fma(p, t2, vx_setall_f32(c7));
    p = v_fma(p, t2, vx_setall_f32(c5));
    p = v_fma(p, t2, vx_setall_f32(c3));
    p = v_fma(p, t2, vx_setall_f32(c1));
    return v_mul(p, t);
}

// atan2(y, x) SIMD; возвращает радианы в [-pi, pi]
static inline v_float32 atan2_simd(v_float32 y, v_float32 x) {
    const float PI = 3.14159265358979323846f;
    v_float32 ax = v_abs(x), ay = v_abs(y);
    // t = min(ax,ay)/max(ax,ay) (защита от деления на 0)
    v_float32 mn = v_min(ax, ay), mx = v_max(ax, ay);
    v_float32 t = v_div(mn, v_add(mx, vx_setall_f32(1e-30f)));
    v_float32 a = atan_poly(t);
    // если |y|>|x|: a = pi/2 - a
    a = v_select(v_gt(ay, ax), v_sub(vx_setall_f32(PI / 2), a), a);
    // квадрант по знаку x: x<0 -> pi - a
    a = v_select(v_lt(x, vx_setzero_f32()), v_sub(vx_setall_f32(PI), a), a);
    // знак y
    a = v_select(v_lt(y, vx_setzero_f32()), v_sub(vx_setzero_f32(), a), a);
    // atan2(0,0)=0
    a = v_select(v_eq(mx, vx_setzero_f32()), vx_setzero_f32(), a);
    return a;
}
#endif

void phase32f(const Mat& x, const Mat& y, Mat& dst, bool angleInDegrees) {
    CV_Assert(x.type() == CV_32FC1 && y.type() == CV_32FC1 && x.size() == y.size());
    dst.create(x.size(), CV_32FC1);
    const int W = x.cols, H = x.rows;
    const float scale = angleInDegrees ? (float)(180.0 / CV_PI) : 1.0f;
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int r = rg.start; r < rg.end; r++) {
            const float* px = x.ptr<float>(r);
            const float* py = y.ptr<float>(r);
            float* d = dst.ptr<float>(r);
            int i = 0;
#if CV_SIMD
            const int NF = v_float32::nlanes;
            for (; i <= W - NF; i += NF) {
                v_float32 a = atan2_simd(vx_load(py + i), vx_load(px + i));
                v_store(d + i, v_mul(a, vx_setall_f32(scale)));
            }
#endif
            for (; i < W; i++)
                d[i] = std::atan2(py[i], px[i]) * scale;
        }
    });
}

void cartToPolar32f(const Mat& x, const Mat& y, Mat& mag, Mat& ang, bool angleInDegrees) {
    CV_Assert(x.type() == CV_32FC1 && y.type() == CV_32FC1 && x.size() == y.size());
    mag.create(x.size(), CV_32FC1);
    ang.create(x.size(), CV_32FC1);
    const int W = x.cols, H = x.rows;
    const float scale = angleInDegrees ? (float)(180.0 / CV_PI) : 1.0f;
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int r = rg.start; r < rg.end; r++) {
            const float* px = x.ptr<float>(r);
            const float* py = y.ptr<float>(r);
            float* dm = mag.ptr<float>(r);
            float* da = ang.ptr<float>(r);
            int i = 0;
#if CV_SIMD
            const int NF = v_float32::nlanes;
            for (; i <= W - NF; i += NF) {
                v_float32 vx = vx_load(px + i), vy = vx_load(py + i);
                v_store(dm + i, v_sqrt(v_fma(vx, vx, v_mul(vy, vy))));
                v_float32 a = atan2_simd(vy, vx);
                v_store(da + i, v_mul(a, vx_setall_f32(scale)));
            }
#endif
            for (; i < W; i++) {
                dm[i] = std::sqrt(px[i] * px[i] + py[i] * py[i]);
                da[i] = std::atan2(py[i], px[i]) * scale;
            }
        }
    });
}

} // namespace fastcv
