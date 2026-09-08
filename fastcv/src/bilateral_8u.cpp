// FastOpenCV: SIMD-оптимизированный bilateralFilter для 8UC1
// Ключевые отличия от baseline:
//  - векторизация по пикселям через universal intrinsics (AVX2/AVX-512)
//  - abs diff на u8 (1 инструкция) вместо float-вычитания
//  - вес цвета через v_lut (gather по 256-entry таблице) вместо exp
//  - parallel_for_ по строкам (TBB)
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>
#include <vector>

namespace fastcv {

using namespace cv;

class Bilateral8uBody : public ParallelLoopBody {
public:
    Bilateral8uBody(const Mat& padded, Mat& dst, int r,
                    const float* spaceW, int spaceStep, const float* colorW)
        : padded_(padded), dst_(dst), r_(r), spaceW_(spaceW),
          spaceStep_(spaceStep), colorW_(colorW) {}

    void operator()(const Range& range) const override {
        const int w = dst_.cols;
        const int sstep = (int)padded_.step;
#if CV_SIMD
        const int NF = v_float32::nlanes;      // 8 (AVX2) / 16 (AVX-512)
        const int CN = v_uint8::nlanes;        // 32 / 64 байт за итерацию
        const int G = CN / NF;                 // групп float-векторов
#endif
        for (int y = range.start; y < range.end; y++) {
            // dst(x,y) соответствует padded(x + r, y + r); центр окна - sp[x + r]
            const uchar* sp = padded_.ptr<uchar>(y + r_);
            uchar* dp = dst_.ptr<uchar>(y);
            int x = 0;
#if CV_SIMD
            for (; x <= w - CN; x += CN) {
                v_uint8 c8 = vx_load(sp + x + r_);
                v_float32 sum[G], ws[G];
                for (int g = 0; g < G; g++) {
                    sum[g] = vx_setzero_f32();
                    ws[g] = vx_setzero_f32();
                }
                for (int dy = -r_; dy <= r_; dy++) {
                    const uchar* nrow = sp + dy * sstep + x + r_;
                    const float* sw = spaceW_ + (dy + r_) * spaceStep_ + r_;
                    for (int dx = -r_; dx <= r_; dx++) {
                        v_float32 s = vx_setall_f32(sw[dx]);
                        v_uint8 n8 = vx_load(nrow + dx);
                        v_uint8 d8 = v_absdiff(n8, c8);
                        // expand absdiff и значения в G групп по NF float
                        v_uint16 d16a, d16b, n16a, n16b;
                        v_expand(d8, d16a, d16b);
                        v_expand(n8, n16a, n16b);
                        v_uint32 dq[G], nq[G];
                        v_expand(d16a, dq[0], dq[1]);
                        v_expand(n16a, nq[0], nq[1]);
                        if (G == 4) {
                            v_expand(d16b, dq[2], dq[3]);
                            v_expand(n16b, nq[2], nq[3]);
                        }
                        for (int g = 0; g < G; g++) {
                            v_float32 wg = v_mul(s, v_lut(colorW_, v_reinterpret_as_s32(dq[g])));
                            v_float32 nf = v_cvt_f32(v_reinterpret_as_s32(nq[g]));
                            sum[g] = v_muladd(wg, nf, sum[g]);
                            ws[g] = v_add(ws[g], wg);
                        }
                    }
                }
                // округление и упаковка
                v_int32 ri[G];
                for (int g = 0; g < G; g++)
                    ri[g] = v_round(v_div(sum[g], ws[g]));
                v_int16 p16a = v_pack(ri[0], ri[1]);
                if (G == 4) {
                    v_int16 p16b = v_pack(ri[2], ri[3]);
                    v_uint8 res = v_pack_u(p16a, p16b);
                    vx_store(dp + x, res);
                } else {
                    v_store_low(dp + x, v_pack_u(p16a, p16a));
                }
            }
#endif
            for (; x < w; x++) { // скалярный хвост
                int c = sp[x + r_];
                float sum = 0.f, ws = 0.f;
                for (int dy = -r_; dy <= r_; dy++) {
                    const uchar* nrow = sp + dy * sstep + x + r_;
                    const float* sw = spaceW_ + (dy + r_) * spaceStep_ + r_;
                    for (int dx = -r_; dx <= r_; dx++) {
                        int diff = std::abs(int(nrow[dx]) - c);
                        float wt = sw[dx] * colorW_[diff];
                        sum += wt * nrow[dx];
                        ws += wt;
                    }
                }
                dp[x] = saturate_cast<uchar>(sum / ws);
            }
        }
    }

private:
    const Mat& padded_;
    Mat& dst_;
    int r_;
    const float* spaceW_;
    int spaceStep_;
    const float* colorW_;
};

void bilateralFilter8u(const Mat& src, Mat& dst, int d, double sigmaColor, double sigmaSpace) {
    CV_Assert(src.type() == CV_8UC1 && d > 0 && d % 2 == 1);
    const int r = d / 2;
    dst.create(src.size(), src.type());

    Mat padded;
    copyMakeBorder(src, padded, r, r, r, r, BORDER_REPLICATE);

    std::vector<float> spaceW(d * d);
    double ks = -1.0 / (2.0 * sigmaSpace * sigmaSpace);
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            spaceW[(dy + r) * d + (dx + r)] = (float)std::exp(ks * (dx * dx + dy * dy));

    // таблица весов цвета для |diff| в [0,255]
    std::vector<float> colorW(256);
    double kc = -1.0 / (2.0 * sigmaColor * sigmaColor);
    for (int i = 0; i < 256; i++)
        colorW[i] = (float)std::exp(kc * i * i);

    parallel_for_(Range(0, src.rows),
                  Bilateral8uBody(padded, dst, r, spaceW.data(), d, colorW.data()));
}

} // namespace fastcv
