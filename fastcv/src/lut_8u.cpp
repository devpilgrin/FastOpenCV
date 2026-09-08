// FastOpenCV: LUT для 8U (lutcn=1, любое cn) через AVX512-VBMI vpermb.
// У OpenCV LUT8u на x86 без VBMI - скалярный (gather медленнее скаляра,
// а в dispatch-списке 4.14 VBMI-варианта нет - даже на Zen5/CNL+ железе).
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define FASTCV_CPUID_OK 1
#endif

namespace fastcv {

using namespace cv;

static bool hasVBMI() {
#if defined(FASTCV_CPUID_OK)
    static const bool v = __builtin_cpu_supports("avx512vbmi") != 0;
    return v;
#else
    return false;
#endif
}

// VBMI-ядро: таблица 256 байт = 4 zmm; vpermb берёт младшие 6 бит индекса,
// выбор регистра - по битам 6-7.
#if defined(__AVX512VBMI__)
static void lut8u_vbmi_row(const uchar* src, uchar* dst, int total,
                           __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    const __m512i m3 = _mm512_set1_epi8(3);
    int i = 0;
    for (; i <= total - 64; i += 64) {
        __m512i idx = _mm512_loadu_si512(src + i);
        __m512i sel = _mm512_and_si512(_mm512_srli_epi16(idx, 6), m3);
        __m512i r = _mm512_permutexvar_epi8(idx, t0);
        r = _mm512_mask_permutexvar_epi8(r, _mm512_cmpeq_epi8_mask(sel, _mm512_set1_epi8(1)), idx, t1);
        r = _mm512_mask_permutexvar_epi8(r, _mm512_cmpeq_epi8_mask(sel, _mm512_set1_epi8(2)), idx, t2);
        r = _mm512_mask_permutexvar_epi8(r, _mm512_cmpeq_epi8_mask(sel, _mm512_set1_epi8(3)), idx, t3);
        _mm512_storeu_si512(dst + i, r);
    }
    // хвост (total % 64) обрабатывает вызывающий скалярно
}
#endif

void lut8u(const Mat& src, const uchar* lut, Mat& dst) {
    CV_Assert(src.depth() == CV_8U);
    dst.create(src.dims, src.size, src.type());

#if defined(__AVX512VBMI__)
    if (hasVBMI() && src.dims <= 2) {
        __m512i t0 = _mm512_loadu_si512(lut);
        __m512i t1 = _mm512_loadu_si512(lut + 64);
        __m512i t2 = _mm512_loadu_si512(lut + 128);
        __m512i t3 = _mm512_loadu_si512(lut + 192);
        const int W = src.cols * src.channels();
        parallel_for_(Range(0, dst.rows), [&](const Range& rg) {
            for (int y = rg.start; y < rg.end; y++) {
                const uchar* sp = src.ptr(y);
                uchar* dp = dst.ptr(y);
                lut8u_vbmi_row(sp, dp, W, t0, t1, t2, t3);
                int i = W & ~63;
                for (; i < W; i++) dp[i] = lut[sp[i]];
            }
        });
        return;
    }
#endif
    // fallback: скаляр, но параллельный
    parallel_for_(Range(0, dst.rows), [&](const Range& rg) {
        const int W = src.cols * src.channels();
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* sp = src.ptr(y);
            uchar* dp = dst.ptr(y);
            for (int i = 0; i < W; i++) dp[i] = lut[sp[i]];
        }
    });
}

} // namespace fastcv
