// FastOpenCV: sqrBoxFilter 8UC1 -> 32F (normalize=true, BORDER_REFLECT_101).
// Паттерн boxFilter: скользящие суммы квадратов. int32 аккумулятор
// (255^2 * 32^2 < 2^31 - безопасно для k<=31).
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void sqrBoxFilter8u(const Mat& src, Mat& dst, Size ksize, bool normalize) {
    CV_Assert(src.type() == CV_8UC1);
    const int kw = ksize.width, kh = ksize.height;
    const int rx = kw / 2, ry = kh / 2;
    const int W = src.cols, H = src.rows;
    dst.create(src.size(), CV_32FC1);

    Mat padded;
    copyMakeBorder(src, padded, ry, kh - 1 - ry, rx, kw - 1 - rx, BORDER_REFLECT_101);

    Mat hbuf(padded.rows, W, CV_32SC1);
    parallel_for_(Range(0, padded.rows), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* S = padded.ptr<uchar>(y);
            int* D = hbuf.ptr<int>(y);
            int s = 0;
            for (int i = 0; i < kw; i++) { int v = S[i]; s += v * v; }
            D[0] = s;
            for (int x = 1; x < W; x++) {
                int va = S[x + kw - 1], vs = S[x - 1];
                s += va * va - vs * vs;
                D[x] = s;
            }
        }
    });

    const float fscale = (float)(1.0 / ((double)kw * kh));
    const int bands = std::max(1, getNumThreads() * 2);
    const int bh = (H + bands - 1) / bands;
    parallel_for_(Range(0, bands), [&](const Range& rg) {
        for (int b = rg.start; b < rg.end; b++) {
            int y0 = b * bh, y1 = std::min(H, y0 + bh);
            if (y0 >= y1) continue;
            std::vector<int> SUM((size_t)W);
            for (int i = 0; i < kh; i++) {
                const int* pr = hbuf.ptr<int>(y0 + i);
                int x = 0;
#if CV_SIMD
                const int NF0 = v_int32::nlanes;
                for (; x <= W - NF0; x += NF0)
                    v_store(SUM.data() + x, v_add(vx_load(SUM.data() + x), vx_load(pr + x)));
#endif
                for (; x < W; x++) SUM[x] += pr[x];
            }
            for (int y = y0; y < y1; y++) {
                float* dp = dst.ptr<float>(y);
                int x = 0;
#if CV_SIMD
                const int NF = v_int32::nlanes;
                for (; x <= W - NF; x += NF) {
                    v_float32 vf = v_cvt_f32(vx_load(SUM.data() + x));
                    if (normalize) vf = v_mul(vf, vx_setall_f32(fscale));
                    v_store(dp + x, vf);
                }
#endif
                for (; x < W; x++)
                    dp[x] = normalize ? SUM[x] * fscale : (float)SUM[x];
                if (y + 1 < y1) {
                    const int* padd = hbuf.ptr<int>(y + kh);
                    const int* psub = hbuf.ptr<int>(y);
                    int xx = 0;
#if CV_SIMD
                    for (; xx <= W - NF; xx += NF)
                        v_store(SUM.data() + xx,
                                v_sub(v_add(vx_load(SUM.data() + xx), vx_load(padd + xx)),
                                      vx_load(psub + xx)));
#endif
                    for (; xx < W; xx++) SUM[xx] += padd[xx] - psub[xx];
                }
            }
        }
    });
}

} // namespace fastcv
