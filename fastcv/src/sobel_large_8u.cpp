// FastOpenCV: Sobel/Scharr 8UC1->16S для больших ядер (5,7,...) -
// у OpenCV специализированный SIMD-путь есть только для 3x3/Scharr,
// большие ядра идут через generic float sepFilter2D.
// Фиксированная точка: коэффициенты getDerivKernels целочисленны; i32 аккумуляторы.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void sobel8u16sLarge(const Mat& src, Mat& dst, int dx, int dy, int ksize) {
    const int cn = src.channels();
    CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) && ksize >= 3 && dx + dy > 0 && dx < ksize && dy < ksize);
    dst.create(src.size(), CV_MAKETYPE(CV_16S, cn));
    const int W = src.cols, H = src.rows;
    const int Wc = W * cn;
    const int r = ksize / 2;

    // точные коэффициенты OpenCV (целые значения в float-векторах)
    Mat kxf, kyf;
    getDerivKernels(kxf, kyf, dx, dy, ksize, false, CV_32F);
    const float* kx = kxf.ptr<float>();
    const float* ky = kyf.ptr<float>();

    Mat padded;
    copyMakeBorder(src, padded, r, ksize - 1 - r, r, ksize - 1 - r, BORDER_REFLECT_101);

    // фаза 1: горизонталь (kernel X) -> int32 буфер
    Mat hbuf(padded.rows, Wc, CV_32SC1);
    parallel_for_(Range(0, padded.rows), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* S = padded.ptr<uchar>(y);
            int* D = hbuf.ptr<int>(y);
            int x = 0;
#if CV_SIMD
            const int CN = v_uint16::nlanes;
            for (; x <= Wc - CN; x += CN) {
                v_int32 a0 = vx_setzero_s32(), a1 = vx_setzero_s32();
                for (int k = 0; k < ksize; k++) {
                    v_uint16 w = vx_load_expand(S + x + k * cn);
                    v_uint32 q0, q1;
                    v_expand(w, q0, q1);
                    v_int32 kk = vx_setall_s32((int)kx[k]);
                    a0 = v_add(a0, v_mul(v_reinterpret_as_s32(q0), kk));
                    a1 = v_add(a1, v_mul(v_reinterpret_as_s32(q1), kk));
                }
                v_store(D + x, a0);
                v_store(D + x + v_int32::nlanes, a1);
            }
#endif
            for (; x < Wc; x++) {
                int s = 0;
                for (int k = 0; k < ksize; k++) s += S[x + k * cn] * (int)kx[k];
                D[x] = s;
            }
        }
    });

    // фаза 2: вертикаль (kernel Y) -> 16S, полосы строк, строчный доступ
    const int bands = std::max(1, getNumThreads() * 2);
    const int bh = (H + bands - 1) / bands;
    parallel_for_(Range(0, bands), [&](const Range& rg) {
        for (int b = rg.start; b < rg.end; b++) {
            int y0 = b * bh, y1 = std::min(H, y0 + bh);
            if (y0 >= y1) continue;
            for (int y = y0; y < y1; y++) {
                short* dps = dst.ptr<short>(y);
                int x = 0;
#if CV_SIMD
                const int NF = v_int32::nlanes;
                for (; x <= Wc - 2 * NF; x += 2 * NF) {
                    v_int32 a0 = vx_setzero_s32(), a1 = vx_setzero_s32();
                    for (int k = 0; k < ksize; k++) {
                        const int* pr = hbuf.ptr<int>(y + k);
                        v_int32 kk = vx_setall_s32((int)ky[k]);
                        a0 = v_add(a0, v_mul(vx_load(pr + x), kk));
                        a1 = v_add(a1, v_mul(vx_load(pr + x + NF), kk));
                    }
                    v_store(dps + x, v_pack(a0, a1)); // saturating pack i32->i16
                }
#endif
                for (; x < Wc; x++) {
                    int s = 0;
                    for (int k = 0; k < ksize; k++) s += hbuf.ptr<int>(y + k)[x] * (int)ky[k];
                    dps[x] = saturate_cast<short>(s);
                }
            }
        }
    });
}

} // namespace fastcv
