// FastOpenCV: sepFilter2D generic float 8UC1 -> 8U/32F.
// У OpenCV generic-путь (RowFilter/ColumnFilter float) без специализации для
// произвольных float-ядер заметно уступает спецпутям (GaussianBlur и пр.).
// Двухпроходный: строки u8*float -> f32 буфер, столбцы f32*float -> выход.
// BORDER_REFLECT_101 (умолчание OpenCV).
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void sepFilter2D_8u(const Mat& src, Mat& dst, int ddepth,
                    const Mat& kernelX, const Mat& kernelY) {
    CV_Assert(src.type() == CV_8UC1);
    Mat kx_, ky_;
    kernelX.convertTo(kx_, CV_32F);
    kernelY.convertTo(ky_, CV_32F);
    const int kw = kx_.total() == 1 ? kx_.cols : (int)kx_.total();
    const int kh = ky_.total() == 1 ? ky_.cols : (int)ky_.total();
    const float* kx = kx_.ptr<float>();
    const float* ky = ky_.ptr<float>();
    const int rx = kw / 2, ry = kh / 2;
    const int W = src.cols, H = src.rows;
    const int outDepth = ddepth < 0 ? CV_8U : ddepth;
    CV_Assert(outDepth == CV_8U || outDepth == CV_32F);
    dst.create(src.size(), CV_MAKETYPE(outDepth, 1));

    Mat padded;
    copyMakeBorder(src, padded, ry, kh - 1 - ry, rx, kw - 1 - rx, BORDER_REFLECT_101);

    // фаза 1: горизонталь -> f32
    Mat hbuf(padded.rows, W, CV_32FC1);
    parallel_for_(Range(0, padded.rows), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* S = padded.ptr<uchar>(y);
            float* D = hbuf.ptr<float>(y);
            int x = 0;
#if CV_SIMD
            const int NF = v_float32::nlanes;
            for (; x <= W - NF; x += NF) {
                v_float32 acc = vx_setzero_f32();
                for (int k = 0; k < kw; k++) {
                    v_float32 v = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(S + x + k)));
                    acc = v_fma(v, vx_setall_f32(kx[k]), acc);
                }
                v_store(D + x, acc);
            }
#endif
            for (; x < W; x++) {
                float s = 0;
                for (int k = 0; k < kw; k++) s += S[x + k] * kx[k];
                D[x] = s;
            }
        }
    });

    // фаза 2: вертикаль, полосы строк, строчный доступ
    const int bands = std::max(1, getNumThreads() * 2);
    const int bh = (H + bands - 1) / bands;
    const bool toU8 = outDepth == CV_8U;
    parallel_for_(Range(0, bands), [&](const Range& rg) {
        for (int b = rg.start; b < rg.end; b++) {
            int y0 = b * bh, y1 = std::min(H, y0 + bh);
            if (y0 >= y1) continue;
            for (int y = y0; y < y1; y++) {
                int x = 0;
#if CV_SIMD
                const int NF = v_float32::nlanes;
                for (; x <= W - NF; x += NF) {
                    v_float32 acc = vx_setzero_f32();
                    for (int k = 0; k < kh; k++) {
                        const float* pr = hbuf.ptr<float>(y + k);
                        acc = v_fma(vx_load(pr + x), vx_setall_f32(ky[k]), acc);
                    }
                    if (toU8) {
                        // упаковка f32 -> u8 через i32 с округлением
                        alignas(64) int tmp[64];
                        v_store(tmp, v_round(acc));
                        uchar* dp = dst.ptr<uchar>(y);
                        for (int t = 0; t < NF; t++) dp[x + t] = saturate_cast<uchar>(tmp[t]);
                    } else {
                        v_store(dst.ptr<float>(y) + x, acc);
                    }
                }
#endif
                if (toU8) {
                    uchar* dp = dst.ptr<uchar>(y);
                    for (; x < W; x++) {
                        float s = 0;
                        for (int k = 0; k < kh; k++) s += hbuf.ptr<float>(y + k)[x] * ky[k];
                        dp[x] = saturate_cast<uchar>(s);
                    }
                } else {
                    float* dp = dst.ptr<float>(y);
                    for (; x < W; x++) {
                        float s = 0;
                        for (int k = 0; k < kh; k++) s += hbuf.ptr<float>(y + k)[x] * ky[k];
                        dp[x] = s;
                    }
                }
            }
        }
    });
}

} // namespace fastcv
