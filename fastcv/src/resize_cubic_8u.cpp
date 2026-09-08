// FastOpenCV: resize INTER_CUBIC для 8UC1/8UC3 - двухфазный конвейер:
// фаза 1 (parallel по строкам src): горизонтальный cubic -> int-буфер [sh x dw*cn]
// фаза 2 (parallel по строкам dst): вертикальный cubic SIMD float -> u8
// Каждая строка ресайзится ровно один раз (нет дублирования по полосам).
// Арифметика как у OpenCV: A=-0.75, фикс. точка 11 бит, вертикаль float/2048^2.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>
#include <cmath>

#define FCV_RCS 2048 // INTER_RESIZE_COEF_SCALE

namespace fastcv {

using namespace cv;

static inline void cubicCoeffs(float x, float* c) {
    const float A = -0.75f;
    c[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    c[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    c[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    c[3] = 1.f - c[0] - c[1] - c[2];
}

void resizeCubic8u(const Mat& src, Mat& dst, Size dsize) {
    CV_Assert(src.depth() == CV_8U && (src.channels() == 1 || src.channels() == 3));
    dst.create(dsize, src.type());
    const int sw = src.cols, sh = src.rows, cn = src.channels();
    const int dw = dsize.width, dh = dsize.height;
    const double scaleX = (double)sw / dw, scaleY = (double)sh / dh;
    const int rowW = dw * cn;

    // --- предвычисление X (общее для всех строк) ---
    std::vector<int> xofs(dw * 4);
    std::vector<short> a0(dw), a1(dw), a2(dw), a3(dw);
    for (int dx = 0; dx < dw; dx++) {
        float fx = (float)((dx + 0.5) * scaleX - 0.5);
        int sx = cvFloor(fx);
        fx -= sx;
        float c[4];
        cubicCoeffs(fx, c);
        short ia[4];
        for (int j = 0; j < 4; j++) {
            xofs[dx * 4 + j] = std::min(std::max(sx - 1 + j, 0), sw - 1) * cn;
            ia[j] = saturate_cast<short>(c[j] * FCV_RCS);
        }
        a0[dx] = ia[0]; a1[dx] = ia[1]; a2[dx] = ia[2]; a3[dx] = ia[3];
    }
    // интерьер для gather-пути (cn=1): где тапы не зажаты
    int lo = 0, hi = dw;
    while (lo < dw && xofs[lo * 4 + 3] - xofs[lo * 4] != 3) lo++;
    while (hi > lo && xofs[(hi - 1) * 4 + 3] - xofs[(hi - 1) * 4] != 3) hi--;

    // --- фаза 1: горизонтальный проход всех строк ---
    Mat hbuf(sh, rowW, CV_32SC1);
    parallel_for_(Range(0, sh), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* S = src.ptr(y);
            int* D = hbuf.ptr<int>(y);
            int dx = 0;
            if (cn == 1) {
                for (; dx < lo; dx++)
                    D[dx] = S[xofs[dx * 4]] * a0[dx] + S[xofs[dx * 4 + 1]] * a1[dx] +
                            S[xofs[dx * 4 + 2]] * a2[dx] + S[xofs[dx * 4 + 3]] * a3[dx];
#if defined(__AVX512F__)
                alignas(64) int idxbuf[16];
                for (; dx <= hi - 16; dx += 16) {
                    for (int t = 0; t < 16; t++) idxbuf[t] = xofs[(dx + t) * 4];
                    __m512i packed = _mm512_i32gather_epi32(_mm512_load_si512(idxbuf), S, 1);
                    __m512i b0 = _mm512_and_si512(packed, _mm512_set1_epi32(0xFF));
                    __m512i b1 = _mm512_and_si512(_mm512_srli_epi32(packed, 8), _mm512_set1_epi32(0xFF));
                    __m512i b2 = _mm512_and_si512(_mm512_srli_epi32(packed, 16), _mm512_set1_epi32(0xFF));
                    __m512i b3 = _mm512_srli_epi32(packed, 24);
                    __m512i va0 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(a0.data() + dx)));
                    __m512i va1 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(a1.data() + dx)));
                    __m512i va2 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(a2.data() + dx)));
                    __m512i va3 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(a3.data() + dx)));
                    __m512i sum = _mm512_add_epi32(
                        _mm512_add_epi32(_mm512_mullo_epi32(b0, va0), _mm512_mullo_epi32(b1, va1)),
                        _mm512_add_epi32(_mm512_mullo_epi32(b2, va2), _mm512_mullo_epi32(b3, va3)));
                    _mm512_storeu_si512(D + dx, sum);
                }
#endif
                for (; dx < dw; dx++)
                    D[dx] = S[xofs[dx * 4]] * a0[dx] + S[xofs[dx * 4 + 1]] * a1[dx] +
                            S[xofs[dx * 4 + 2]] * a2[dx] + S[xofs[dx * 4 + 3]] * a3[dx];
            } else {
                for (int dxe = 0; dxe < dw; dxe++)
                    for (int c = 0; c < cn; c++)
                        D[dxe * cn + c] = S[xofs[dxe * 4] + c] * a0[dxe] +
                                          S[xofs[dxe * 4 + 1] + c] * a1[dxe] +
                                          S[xofs[dxe * 4 + 2] + c] * a2[dxe] +
                                          S[xofs[dxe * 4 + 3] + c] * a3[dxe];
            }
        }
    });

    // --- фаза 2: вертикальный проход ---
    const float vscale = 1.f / (FCV_RCS * FCV_RCS);
    parallel_for_(Range(0, dh), [&](const Range& rg) {
        for (int dy = rg.start; dy < rg.end; dy++) {
            float fy = (float)((dy + 0.5) * scaleY - 0.5);
            int sy = cvFloor(fy);
            fy -= sy;
            float cb[4];
            cubicCoeffs(fy, cb);
            short ib[4];
            for (int k = 0; k < 4; k++) ib[k] = saturate_cast<short>(cb[k] * FCV_RCS);
            const int* rows[4];
            for (int k = 0; k < 4; k++)
                rows[k] = hbuf.ptr<int>(std::min(std::max(sy - 1 + k, 0), sh - 1));
            v_float32 b0 = vx_setall_f32(ib[0] * vscale), b1 = vx_setall_f32(ib[1] * vscale),
                      b2 = vx_setall_f32(ib[2] * vscale), b3 = vx_setall_f32(ib[3] * vscale);
            uchar* dp = dst.ptr(dy);
            int x = 0;
#if CV_SIMD
            const int N16 = v_int16::nlanes;
            const int NF = v_float32::nlanes;
            for (; x <= rowW - N16; x += N16) {
                v_pack_u_store(dp + x, v_pack(
                    v_round(v_muladd(v_cvt_f32(vx_load(rows[0] + x)), b0,
                            v_muladd(v_cvt_f32(vx_load(rows[1] + x)), b1,
                            v_muladd(v_cvt_f32(vx_load(rows[2] + x)), b2,
                                     v_mul(v_cvt_f32(vx_load(rows[3] + x)), b3))))),
                    v_round(v_muladd(v_cvt_f32(vx_load(rows[0] + x + NF)), b0,
                            v_muladd(v_cvt_f32(vx_load(rows[1] + x + NF)), b1,
                            v_muladd(v_cvt_f32(vx_load(rows[2] + x + NF)), b2,
                                     v_mul(v_cvt_f32(vx_load(rows[3] + x + NF)), b3)))))));
            }
#endif
            for (; x < rowW; x++) {
                float v = rows[0][x] * (ib[0] * vscale) + rows[1][x] * (ib[1] * vscale) +
                          rows[2][x] * (ib[2] * vscale) + rows[3][x] * (ib[3] * vscale);
                dp[x] = saturate_cast<uchar>(v);
            }
        }
    });
}

} // namespace fastcv
