// FastOpenCV: быстрый сумматор 8U (cv::sum у OpenCV однопоточный).
// cn=1: AVX-512 SAD (vpsadbw) - предел по памяти; cn>1: параллельный скаляр по каналам.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void sum8u(const Mat& src, double* sums) {
    const int cn = src.channels();
    CV_Assert(src.depth() == CV_8U && cn >= 1 && cn <= 4);
    const int W = src.cols, H = src.rows;
    const size_t step = src.step;

    std::vector<uint64> partial((size_t)getNumThreads() * 4, 0);

    parallel_for_(Range(0, H), [&](const Range& rg) {
        uint64 acc[4] = {0, 0, 0, 0};
        int tid = getThreadNum();
        if (tid < 0 || tid >= getNumThreads()) tid = 0;
        if (cn == 1) {
            for (int y = rg.start; y < rg.end; y++) {
                const uchar* p = src.ptr(y);
                int x = 0;
#if defined(__AVX512BW__)
                __m512i acc64 = _mm512_setzero_si512();
                for (; x <= W - 64; x += 64)
                    acc64 = _mm512_add_epi64(acc64, _mm512_sad_epu8(_mm512_loadu_si512(p + x), _mm512_setzero_si512()));
                acc[0] += (uint64)_mm512_reduce_add_epi64(acc64);
#elif defined(__SSE2__)
                __m128i acc64 = _mm_setzero_si128();
                for (; x <= W - 16; x += 16)
                    acc64 = _mm_add_epi64(acc64, _mm_sad_epu8(_mm_loadu_si128((const __m128i*)(p + x)), _mm_setzero_si128()));
                alignas(16) uint64 t[2];
                _mm_store_si128((__m128i*)t, acc64);
                acc[0] += t[0] + t[1];
#endif
                for (; x < W; x++) acc[0] += p[x];
            }
        } else {
            for (int y = rg.start; y < rg.end; y++) {
                const uchar* p = src.ptr(y);
                if (cn == 2) {
                    for (int x = 0; x < W * 2; x += 2) { acc[0] += p[x]; acc[1] += p[x + 1]; }
                } else if (cn == 3) {
                    for (int x = 0; x < W * 3; x += 3) { acc[0] += p[x]; acc[1] += p[x + 1]; acc[2] += p[x + 2]; }
                } else {
                    for (int x = 0; x < W * 4; x += 4) { acc[0] += p[x]; acc[1] += p[x + 1]; acc[2] += p[x + 2]; acc[3] += p[x + 3]; }
                }
            }
        }
        uint64* dstp = &partial[(size_t)tid * 4];
        for (int c = 0; c < cn; c++) dstp[c] += acc[c];
    });
    (void)step;

    for (int c = 0; c < cn; c++) {
        uint64 s = 0;
        for (int t = 0; t < getNumThreads(); t++) s += partial[(size_t)t * 4 + c];
        sums[c] = (double)s;
    }
}

} // namespace fastcv
