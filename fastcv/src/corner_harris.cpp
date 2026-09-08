// FastOpenCV: слитый (fused) cornerHarris для 8UC1, ksize=3, BORDER_REFLECT_101.
// Пайплайн Sobel3x3 -> ковариации -> boxSum(blockSize) -> Harris R по строкам
// с кольцевым буфером, без промежуточных полноразмерных матриц.
// Границы точно как у OpenCV: Sobel - отражение пикселей, boxSum - отражение
// значений ковариаций (cov за пределами = копия отражённой, БЕЗ смены знака).
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>
#include <cstring>

namespace fastcv {

using namespace cv;

class CornerHarrisBody : public ParallelLoopBody {
public:
    CornerHarrisBody(const Mat& padded, Mat& dst, int blockSize, float scale, double k)
        : padded_(padded), dst_(dst), b_(blockSize), r_(blockSize / 2),
          scale_(scale), k_((float)k) {}

    // REFLECT_101 для индекса строки
    static int reflect101(int p, int len) {
        while (p < 0 || p >= len)
            p = p < 0 ? -p : 2 * len - 2 - p;
        return p;
    }

    void operator()(const Range& range) const override {
        const int W = dst_.cols;
        const int H = dst_.rows;

        // кольцо: b_ слотов x 3 плоскости (xx, xy, yy), строка шириной W (только валидные x)
        std::vector<float> ring((size_t)b_ * 3 * W);
        auto plane = [&](int slot, int ch) -> float* {
            return ring.data() + ((size_t)slot * 3 + ch) * W;
        };

        // cov строки ВАЛИДНОЙ выходной строки y (0 <= y < H), x в [0, W)
        auto covRow = [&](int y, int slot) {
            // padded имеет pad=1 (только для Sobel): пиксель (x,y) -> padded(y+1, x+1)
            const uchar* r0 = padded_.ptr<uchar>(y);
            const uchar* r1 = padded_.ptr<uchar>(y + 1);
            const uchar* r2 = padded_.ptr<uchar>(y + 2);
            float* oxx = plane(slot, 0);
            float* oxy = plane(slot, 1);
            float* oyy = plane(slot, 2);
            int x = 0;
#if CV_SIMD
            const int N16 = v_int16::nlanes;
            const v_float32 vs = vx_setall_f32(scale_);
            for (; x <= W - N16; x += N16) {
                v_int16 a0 = v_reinterpret_as_s16(vx_load_expand(r0 + x));
                v_int16 a1 = v_reinterpret_as_s16(vx_load_expand(r0 + x + 1));
                v_int16 a2 = v_reinterpret_as_s16(vx_load_expand(r0 + x + 2));
                v_int16 b0 = v_reinterpret_as_s16(vx_load_expand(r1 + x));
                v_int16 b2 = v_reinterpret_as_s16(vx_load_expand(r1 + x + 2));
                v_int16 c0 = v_reinterpret_as_s16(vx_load_expand(r2 + x));
                v_int16 c1 = v_reinterpret_as_s16(vx_load_expand(r2 + x + 1));
                v_int16 c2 = v_reinterpret_as_s16(vx_load_expand(r2 + x + 2));
                v_int16 dx16 = v_sub_wrap(v_add_wrap(v_add_wrap(a2, v_add_wrap(b2, b2)), c2),
                                          v_add_wrap(v_add_wrap(a0, v_add_wrap(b0, b0)), c0));
                v_int16 dy16 = v_sub_wrap(v_add_wrap(v_add_wrap(c0, v_add_wrap(c1, c1)), c2),
                                          v_add_wrap(v_add_wrap(a0, v_add_wrap(a1, a1)), a2));
                v_int32 e[4];
                v_expand(dx16, e[0], e[1]);
                v_expand(dy16, e[2], e[3]);
                for (int g = 0; g < 2; g++) {
                    v_float32 fx = v_mul(v_cvt_f32(e[g]), vs);
                    v_float32 fy = v_mul(v_cvt_f32(e[2 + g]), vs);
                    int off = x + g * v_float32::nlanes;
                    v_store(oxx + off, v_mul(fx, fx));
                    v_store(oxy + off, v_mul(fx, fy));
                    v_store(oyy + off, v_mul(fy, fy));
                }
            }
#endif
            for (; x < W; x++) {
                int a0 = r0[x], a1 = r0[x + 1], a2 = r0[x + 2];
                int b0 = r1[x], b2 = r1[x + 2];
                int c0 = r2[x], c1 = r2[x + 1], c2 = r2[x + 2];
                float dx = float((a2 + 2 * b2 + c2) - (a0 + 2 * b0 + c0)) * scale_;
                float dy = float((c0 + 2 * c1 + c2) - (a0 + 2 * a1 + a2)) * scale_;
                oxx[x] = dx * dx; oxy[x] = dx * dy; oyy[x] = dy * dy;
            }
        };

        // заполнение слота строкой y' с отражением за пределами [0, H)
        auto fillSlot = [&](int yp, int slot) {
            covRow(reflect101(yp, H), slot);
        };

        for (int i = 0; i < b_; i++)
            fillSlot(range.start - r_ + i, i);

        // расширенные горизонтальные суммы (с отражением): ширина W + 2*r_
        const int ew = W + 2 * r_;
        std::vector<float> ex((size_t)ew), exy((size_t)ew), ey((size_t)ew);

        for (int y = range.start; y < range.end; y++) {
            // вертикальный boxSum по кольцу в центр расширенного массива (x' = j - r_)
            for (int ch = 0; ch < 3; ch++) {
                float* e = ch == 0 ? ex.data() : ch == 1 ? exy.data() : ey.data();
                int x = 0;
#if CV_SIMD
                const int NF = v_float32::nlanes;
                for (; x <= W - NF; x += NF) {
                    v_float32 a = vx_setzero_f32();
                    for (int i = 0; i < b_; i++)
                        a = v_add(a, vx_load(plane(i, ch) + x));
                    v_store(e + r_ + x, a);
                }
#endif
                for (; x < W; x++) {
                    float a = 0;
                    for (int i = 0; i < b_; i++)
                        a += plane(i, ch)[x];
                    e[r_ + x] = a;
                }
                // отражение границ: слева x'=-p -> p, справа W+k -> W-2-k
                for (int p = 1; p <= r_; p++) {
                    e[r_ - p] = e[r_ + p];                 // x' = -p -> p
                    e[r_ + W - 1 + p] = e[r_ + W - 1 - p]; // x' = W-1+p -> W-1-p
                }
            }
            // горизонтальный boxSum + Harris R
            float* dp = dst_.ptr<float>(y);
            int x = 0;
#if CV_SIMD
            const v_float32 vk = vx_setall_f32(k_);
            const int NF2 = v_float32::nlanes;
            for (; x <= W - NF2; x += NF2) {
                v_float32 a = vx_setzero_f32(), b = vx_setzero_f32(), c = vx_setzero_f32();
                for (int dxx = 0; dxx < b_; dxx++) {
                    a = v_add(a, vx_load(ex.data() + x + dxx));
                    b = v_add(b, vx_load(exy.data() + x + dxx));
                    c = v_add(c, vx_load(ey.data() + x + dxx));
                }
                v_float32 tr = v_add(a, c);
                v_float32 det = v_sub(v_mul(a, c), v_mul(b, b));
                v_store(dp + x, v_sub(det, v_mul(vk, v_mul(tr, tr))));
            }
#endif
            for (; x < W; x++) {
                float a = 0, b = 0, c = 0;
                for (int dxx = 0; dxx < b_; dxx++) {
                    a += ex[x + dxx]; b += exy[x + dxx]; c += ey[x + dxx];
                }
                float tr = a + c;
                dp[x] = (a * c - b * b) - k_ * (tr * tr);
            }
            // сдвиг кольца
            if (y + 1 < range.end) {
                int slot = (y - range.start) % b_;
                fillSlot(y + r_ + 1, slot);
            }
        }
    }

private:
    const Mat& padded_;
    Mat& dst_;
    int b_, r_;
    float scale_, k_;
};

void cornerHarris8u(const Mat& src, Mat& dst, int blockSize, double k) {
    CV_Assert(src.type() == CV_8UC1 && blockSize >= 3 && blockSize % 2 == 1);
    dst.create(src.size(), CV_32FC1);

    Mat padded; // pad=1 - только для границы Sobel (REFLECT_101)
    copyMakeBorder(src, padded, 1, 1, 1, 1, BORDER_REFLECT_101);

    // масштаб как в cv::cornerEigenValsVecs для ksize=3, 8U
    float scale = 1.0f / ((1 << 2) * blockSize * 255.0f);
    parallel_for_(Range(0, src.rows), CornerHarrisBody(padded, dst, blockSize, scale, k));
}

} // namespace fastcv
