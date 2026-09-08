// FastOpenCV: Laplacian 8UC1->16S для ksize>3 (у OpenCV - два generic float-FilterEngine).
// Семантика cv: каждый компонент (d2x, d2y) сатурируется к 16S, затем saturating add.
// Точно bit-exact для k=5 (cv использует wdepth=16S); для k=7 cv идёт через 32F -
// возможны расхождения +/-1 (помечено).
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

namespace fastcv {

using namespace cv;

void sobel8u16sLarge(const Mat& src, Mat& dst, int dx, int dy, int ksize);

void laplacian8u16sLarge(const Mat& src, Mat& dst, int ksize) {
    // Только k=5: cv для k=5 использует wdepth=16S (наша семантика совпадает точно),
    // для k=7 cv уходит в 32F float-путь - наше fixed-point там не bit-exact.
    CV_Assert(src.type() == CV_8UC1 && ksize == 5);
    Mat d2x, d2y;
    sobel8u16sLarge(src, d2x, 2, 0, ksize);
    sobel8u16sLarge(src, d2y, 0, 2, ksize);
    dst.create(src.size(), CV_16SC1);
    const int H = src.rows, W = src.cols;
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const short* a = d2x.ptr<short>(y);
            const short* b = d2y.ptr<short>(y);
            short* d = dst.ptr<short>(y);
            int x = 0;
#if CV_SIMD
            const int NN = v_int16::nlanes;
            const int NF32 = v_int32::nlanes;
            for (; x <= W - NN; x += NN) {
                v_int16 va = vx_load(a + x), vb = vx_load(b + x);
                // расширяемся до i32, складываем, пакуем с насыщением
                v_int32 lo_a, hi_a, lo_b, hi_b;
                v_expand(va, lo_a, hi_a);
                v_expand(vb, lo_b, hi_b);
                v_int16 r0 = v_pack(v_add(lo_a, lo_b), v_add(hi_a, hi_b));
                v_store(d + x, r0);
                (void)NF32;
            }
#endif
            for (; x < W; x++)
                d[x] = saturate_cast<short>((int)a[x] + b[x]);
        }
    });
}

} // namespace fastcv
