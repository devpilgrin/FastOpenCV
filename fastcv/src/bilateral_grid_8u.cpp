// FastOpenCV: быстрый bilateral для 8UC1 через bilateral grid (Paris & Durand).
// АППРОКСИМАЦИЯ точного bilateral (O(1) в d): splat -> blur grid -> slice.
// Точность зависит от сигм; для типичных RT-параметров (sigmaSpace 3-20) ошибка
// мала. Для точного результата используйте fastcv::bilateralFilter8u.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>
#include <cmath>

namespace fastcv {

using namespace cv;

void bilateralGrid8u(const Mat& src, Mat& dst, int d, double sigmaColor, double sigmaSpace) {
    CV_Assert(src.type() == CV_8UC1);
    dst.create(src.size(), src.type());
    const int W = src.cols, H = src.rows;

    // шаги сетки (эвристика Paris-Durand)
    const double ss = std::max(1.0, sigmaSpace / 2.0);      // пикселей на ячейку
    const double sr = std::max(4.0, sigmaColor / 4.0);      // интенсивности на ячейку
    const int gw = (int)std::ceil(W / ss) + 2;
    const int gh = (int)std::ceil(H / ss) + 2;
    const int gz = (int)std::ceil(256.0 / sr) + 2;

    // сетка: [z][y][x] два массива - значения и веса
    std::vector<float> gval((size_t)gz * gh * gw, 0.f), gwgt((size_t)gz * gh * gw, 0.f);
    auto idx = [&](int z, int y, int x) { return ((size_t)z * gh + y) * gw + x; };

    // splat: параллельно по полосам с приватными сетками + слияние
    const int nbands = std::max(1, std::min(H / 8, getNumThreads()));
    const int bh = (H + nbands - 1) / nbands;
    std::vector<std::vector<float>> pval(nbands), pwgt(nbands);
    parallel_for_(Range(0, nbands), [&](const Range& rg) {
        for (int b = rg.start; b < rg.end; b++) {
            auto& lv = pval[b]; auto& lw = pwgt[b];
            lv.assign((size_t)gz * gh * gw, 0.f);
            lw.assign((size_t)gz * gh * gw, 0.f);
            int y_end = std::min(H, (b + 1) * bh);
            for (int y = b * bh; y < y_end; y++) {
                const uchar* sp = src.ptr<uchar>(y);
                float fy = y / (float)ss;
                int y0 = (int)fy;
                float wy1 = fy - y0, wy0 = 1.f - wy1;
                for (int x = 0; x < W; x++) {
                    float fx = x / (float)ss;
                    int x0 = (int)fx;
                    float wx1 = fx - x0, wx0 = 1.f - wx1;
                    float v = sp[x];
                    float fz = v / (float)sr;
                    int z0 = (int)fz;
                    float wz1 = fz - z0, wz0 = 1.f - wz1;
                    for (int dz = 0; dz <= 1; dz++) {
                        float wz = dz ? wz1 : wz0;
                        for (int dy = 0; dy <= 1; dy++) {
                            float wzy = wz * (dy ? wy1 : wy0);
                            for (int dxx = 0; dxx <= 1; dxx++) {
                                float w = wzy * (dxx ? wx1 : wx0);
                                size_t i = idx(z0 + dz, y0 + dy, x0 + dxx);
                                lw[i] += w;
                                lv[i] += w * v;
                            }
                        }
                    }
                }
            }
        }
    });
    // слияние приватных сеток (parallel по z)
    parallel_for_(Range(0, gz * gh), [&](const Range& rg) {
        for (int zy = rg.start; zy < rg.end; zy++) {
            size_t base = (size_t)zy * gw;
            for (int b = 0; b < nbands; b++) {
                if (pval[b].empty()) continue;
                for (int x = 0; x < gw; x++) {
                    gval[base + x] += pval[b][base + x];
                    gwgt[base + x] += pwgt[b][base + x];
                }
            }
        }
    });

    // blur сетки: сепарабельный [1,2,1]/4 по x, y, z (parallel по z-слайсам)
    auto blurAxis = [&](std::vector<float>& g, int axis) {
        std::vector<float> tmp = g;
        parallel_for_(Range(0, gz), [&](const Range& rg) {
            for (int z = rg.start; z < rg.end; z++)
                for (int y = 0; y < gh; y++)
                    for (int x = 0; x < gw; x++) {
                        int xm = axis == 0 ? std::max(x - 1, 0) : x;
                        int xp = axis == 0 ? std::min(x + 1, gw - 1) : x;
                        int ym = axis == 1 ? std::max(y - 1, 0) : y;
                        int yp = axis == 1 ? std::min(y + 1, gh - 1) : y;
                        int zm = axis == 2 ? std::max(z - 1, 0) : z;
                        int zp = axis == 2 ? std::min(z + 1, gz - 1) : z;
                        tmp[idx(z, y, x)] = (g[idx(zm, ym, xm)] + 2 * g[idx(z, y, x)] +
                                             g[idx(zp, yp, xp)]) * 0.25f;
                    }
        });
        g.swap(tmp);
    };
    blurAxis(gval, 0); blurAxis(gval, 1); blurAxis(gval, 2);
    blurAxis(gwgt, 0); blurAxis(gwgt, 1); blurAxis(gwgt, 2);

    // slice (parallel по строкам)
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* sp = src.ptr<uchar>(y);
            uchar* dp = dst.ptr<uchar>(y);
            float fy = y / (float)ss;
            int y0 = std::min((int)fy, gh - 2);
            float wy1 = fy - y0, wy0 = 1.f - wy1;
            for (int x = 0; x < W; x++) {
                float fx = x / (float)ss;
                int x0 = std::min((int)fx, gw - 2);
                float wx1 = fx - x0, wx0 = 1.f - wx1;
                float v = sp[x];
                float fz = v / (float)sr;
                int z0 = std::min((int)fz, gz - 2);
                float wz1 = fz - z0, wz0 = 1.f - wz1;
                float val = 0, wgt = 0;
                for (int dz = 0; dz <= 1; dz++) {
                    float wz = dz ? wz1 : wz0;
                    for (int dy = 0; dy <= 1; dy++) {
                        float wzy = wz * (dy ? wy1 : wy0);
                        for (int dxx = 0; dxx <= 1; dxx++) {
                            float w = wzy * (dxx ? wx1 : wx0);
                            size_t i = idx(z0 + dz, y0 + dy, x0 + dxx);
                            val += w * gval[i];
                            wgt += w * gwgt[i];
                        }
                    }
                }
                dp[x] = saturate_cast<uchar>(wgt > 1e-6f ? val / wgt : v);
            }
        }
    });
}

} // namespace fastcv
