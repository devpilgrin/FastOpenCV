// FastOpenCV: medianBlur 8UC1 для ksize > 5 - параллельная версия O(1)-алгоритма
// с двухуровневой гистограммой (16 coarse + 16x16 fine, ushort).
// Отличия от cv::medianBlur_8u_O1:
//  - parallel_for_ по горизонтальным полосам (у OpenCV этот путь однопоточный!)
//  - cn=1 only
// Семантика: exact median, граница BORDER_REPLICATE - побитово как у OpenCV.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>
#include <cstring>

namespace fastcv {

using namespace cv;

typedef ushort HT; // как в OpenCV

// полоса строк [y0, y1), полная ширина, страйпы по столбцам
static void medianBand8u(const Mat& src, Mat& dst, int r, int y0, int y1) {
    const int W = dst.cols, H = dst.rows;
    const int ksize = 2 * r + 1;
    const size_t sstep = src.step;
    const int STRIPE = 512;

    std::vector<HT> hbuf_c, hbuf_f;

    for (int x0 = 0; x0 < W; x0 += STRIPE) {
        const int sw = std::min(W - x0, STRIPE);
        const int n = sw + 2 * r; // столбцы страйпа с окантовкой
        hbuf_c.assign(16 * n, 0);
        hbuf_f.assign(256 * n, 0);
        HT* h_coarse = hbuf_c.data();
        HT* h_fine = hbuf_f.data();
        // столбец jj страйпа соответствует столбцу изображения clamp(x0 - r + jj)
        auto colOf = [&](int jj) { return std::min(std::max(x0 - r + jj, 0), W - 1); };
        auto rowOf = [&](int yy) { return std::min(std::max(yy, 0), H - 1); };

#define COP(jj, x, op) \
        h_coarse[16 * (jj) + ((x) >> 4)] op, \
        h_fine[16 * n * ((x) >> 4) + 16 * (jj) + ((x) & 0xF)] op

        // инициализация: гистограммы столбцов содержат строки [y0-r-1 .. y0+r-1]
        // (цикл по строкам ниже первым делом сдвигает окно к [i-r .. i+r])
        for (int yy = y0 - r - 1; yy <= y0 + r - 1; yy++) {
            const uchar* p = src.ptr(rowOf(yy)) + 0;
            for (int jj = 0; jj < n; jj++) {
                COP(jj, p[colOf(jj)], ++);
            }
        }

        for (int i = y0; i < y1; i++) {
            // сдвиг по вертикали: убрать строку i-r-1, добавить i+r
            {
                const uchar* p0 = src.ptr(rowOf(i - r - 1));
                const uchar* p1 = src.ptr(rowOf(i + r));
                for (int jj = 0; jj < n; jj++) {
                    COP(jj, p0[colOf(jj)], --);
                    COP(jj, p1[colOf(jj)], ++);
                }
            }

            // рабочая гистограмма окна
            HT H_coarse[16];
            alignas(64) HT H_fine[16][16];
            HT luc[16];
            std::memset(H_coarse, 0, sizeof(H_coarse));
            std::memset(H_fine, 0, sizeof(H_fine));
            std::memset(luc, 0, sizeof(luc));

            // v_coarse: бегущая сумма coarse-гистограмм столбцов окна.
            // Первый выходной столбец j=r (изображение x0): окно столбцов 0..2r (кламп)
#if CV_SIMD256
            v_uint16x16 v_coarse = v256_setzero_u16();
            for (int jj = 0; jj < 2 * r; jj++)
                v_coarse = v_add(v_coarse, v256_load(h_coarse + 16 * std::min(jj, n - 1)));
#elif CV_SIMD128
            v_uint16x8 v_coarsel = v_setzero_u16(), v_coarseh = v_setzero_u16();
            for (int jj = 0; jj < 2 * r; jj++) {
                v_coarsel = v_add(v_coarsel, v_load(h_coarse + 16 * std::min(jj, n - 1)));
                v_coarseh = v_add(v_coarseh, v_load(h_coarse + 16 * std::min(jj, n - 1) + 8));
            }
#endif

            uchar* drow = dst.ptr(i) + x0;
            const int t = 2 * r * r + 2 * r; // ранг медианы-1: ksize^2/2 = 2r(r+1)

            for (int j = r; j < n - r; j++) {
                // добавить правый столбец окна в coarse
                int jr = std::min(j + r, n - 1);
                int jl = std::max(j - r, 0);
#if CV_SIMD256
                v_coarse = v_add(v_coarse, v256_load(h_coarse + 16 * jr));
                v_store(H_coarse, v_coarse);
#elif CV_SIMD128
                v_coarsel = v_add(v_coarsel, v_load(h_coarse + 16 * jr));
                v_coarseh = v_add(v_coarseh, v_load(h_coarse + 16 * jr + 8));
                v_store(H_coarse, v_coarsel);
                v_store(H_coarse + 8, v_coarseh);
#else
                for (int ind = 0; ind < 16; ind++) {
                    H_coarse[ind] += h_coarse[16 * jr + ind];
                }
#endif

                // поиск coarse-бина медианы
                int k = 0, sum = 0;
                for (k = 0; k < 16; k++) {
                    sum += H_coarse[k];
                    if (sum > t) { sum -= H_coarse[k]; break; }
                }

                // fine-сегмент бина k: догнать накопление до окна [j-r .. j+r]
#if CV_SIMD256
                v_uint16x16 v_fine;
#elif CV_SIMD128
                v_uint16x8 v_finel, v_fineh;
#endif
                if (luc[k] <= j - r) {
#if CV_SIMD256
                    v_fine = v256_setzero_u16();
#elif CV_SIMD128
                    v_finel = v_setzero_u16();
                    v_fineh = v_setzero_u16();
#else
                    std::memset(H_fine[k], 0, 16 * sizeof(HT));
#endif
                    const HT* segbase = h_fine + 16 * n * k;
                    for (luc[k] = (HT)std::max(j - r, 0); luc[k] < std::min(j + r + 1, n); ++luc[k]) {
                        const HT* sxp = segbase + 16 * luc[k];
#if CV_SIMD256
                        v_fine = v_add(v_fine, v256_load(sxp));
#elif CV_SIMD128
                        v_finel = v_add(v_finel, v_load(sxp));
                        v_fineh = v_add(v_fineh, v_load(sxp + 8));
#else
                        for (int ind = 0; ind < 16; ind++) H_fine[k][ind] += sxp[ind];
#endif
                    }
                } else {
#if CV_SIMD256
                    v_fine = v256_load(H_fine[k]);
#elif CV_SIMD128
                    v_finel = v_load(H_fine[k]);
                    v_fineh = v_load(H_fine[k] + 8);
#endif
                    const HT* segbase = h_fine + 16 * n * k;
                    for (; luc[k] < j + r + 1; ++luc[k]) {
                        const HT* padd = segbase + 16 * std::min((int)luc[k], n - 1);
                        const HT* psub = segbase + 16 * std::max((int)luc[k] - 2 * r - 1, 0);
#if CV_SIMD256
                        v_fine = v_sub(v_add(v_fine, v256_load(padd)), v256_load(psub));
#elif CV_SIMD128
                        v_finel = v_sub(v_add(v_finel, v_load(padd)), v_load(psub));
                        v_fineh = v_sub(v_add(v_fineh, v_load(padd + 8)), v_load(psub + 8));
#else
                        for (int ind = 0; ind < 16; ind++) H_fine[k][ind] += padd[ind] - psub[ind];
#endif
                    }
                }
#if CV_SIMD256
                v_store(H_fine[k], v_fine);
                v_coarse = v_sub(v_coarse, v256_load(h_coarse + 16 * jl));
#elif CV_SIMD128
                v_store(H_fine[k], v_finel);
                v_store(H_fine[k] + 8, v_fineh);
                v_coarsel = v_sub(v_coarsel, v_load(h_coarse + 16 * jl));
                v_coarseh = v_sub(v_coarseh, v_load(h_coarse + 16 * jl + 8));
#else
                for (int ind = 0; ind < 16; ind++) H_coarse[ind] -= h_coarse[16 * jl + ind];
#endif

                // медиана в fine-сегменте
                const HT* segment = H_fine[k];
                int b = 0;
                for (b = 0; b < 16; b++) {
                    sum += segment[b];
                    if (sum > t) break;
                }
                drow[j - r] = (uchar)(16 * k + b);
            }
        }
#undef COP
    }
}

class MedianBandBody : public ParallelLoopBody {
public:
    MedianBandBody(const Mat& src, Mat& dst, int r) : src_(src), dst_(dst), r_(r) {}
    void operator()(const Range& range) const override {
        medianBand8u(src_, dst_, r_, range.start, range.end);
    }
private:
    const Mat& src_;
    Mat& dst_;
    int r_;
};

void medianBlur8u(const Mat& src, Mat& dst, int ksize) {
    CV_Assert(src.type() == CV_8UC1 && ksize >= 3 && ksize % 2 == 1);
    int r = ksize / 2;
    dst.create(src.size(), src.type());
    // полосы по ~32 строки: достаточно для 32 потоков, накладные расходы малы
    int bands = std::max(1, std::min(src.rows, getNumThreads() * 2));
    int bh = (src.rows + bands - 1) / bands;
    parallel_for_(Range(0, bands), [&](const Range& rg) {
        for (int b = rg.start; b < rg.end; b++) {
            int y0 = b * bh, y1 = std::min(src.rows, y0 + bh);
            if (y0 < y1) medianBand8u(src, dst, r, y0, y1);
        }
    });
}

} // namespace fastcv
