// FastOpenCV: DFT через FFTW (float, многопоточный). Кэш планов по (H,W).
// Семантика: r2c 2D, выход - стандартный полуспектр FFTW [H x (W/2+1)] complex64.
// ВНИМАНИЕ: layout отличается от cv::DFT CCS-упаковки (конвертер ниже).
// Для c2c: отдельная функция. Планировщик FFTW не thread-safe - мьютекс.
#include <opencv2/core.hpp>
#include <fftw3.h>
#include <map>
#include <mutex>
#include <vector>

namespace fastcv {

using namespace cv;

namespace {
struct PlanKey {
    int h, w;
    bool operator<(const PlanKey& o) const { return h < o.h || (h == o.h && w < o.w); }
};
std::map<PlanKey, fftwf_plan> g_r2c, g_c2r, g_c2c;
std::mutex g_fftMtx;
bool g_threadsInit = false;

static void ensureThreads() {
    if (!g_threadsInit) {
        fftwf_init_threads();
        fftwf_plan_with_nthreads(std::max(1, getNumThreads()));
        g_threadsInit = true;
    }
}

// вспомогательные буферы для планировщика: FFTW_MEASURE ПЕРЕЗАПИСЫВАЕТ данные
// в массивах, на которых строится план - планируем на scratch, исполняем на
// пользовательских указателях (new-array execute + FFTW_UNALIGNED).
static fftwf_plan getR2C(int h, int w) {
    std::lock_guard<std::mutex> lk(g_fftMtx);
    ensureThreads();
    PlanKey k{h, w};
    auto it = g_r2c.find(k);
    if (it != g_r2c.end()) return it->second;
    std::vector<float> in((size_t)h * w);
    std::vector<fftwf_complex> out((size_t)h * (w / 2 + 1));
    fftwf_plan p = fftwf_plan_dft_r2c_2d(h, w, in.data(), out.data(),
                                         FFTW_MEASURE | FFTW_UNALIGNED);
    g_r2c[k] = p;
    return p;
}
static fftwf_plan getC2C(int h, int w, int sign) {
    std::lock_guard<std::mutex> lk(g_fftMtx);
    ensureThreads();
    PlanKey k{h, w * (sign == FFTW_BACKWARD ? -1 : 1)};
    auto it = g_c2c.find(k);
    if (it != g_c2c.end()) return it->second;
    std::vector<fftwf_complex> in((size_t)h * w), out((size_t)h * w);
    fftwf_plan p = fftwf_plan_dft_2d(h, w, in.data(), out.data(), sign,
                                     FFTW_MEASURE | FFTW_UNALIGNED);
    g_c2c[k] = p;
    return p;
}
} // namespace

// r2c: src 32F [H x W] -> dst 32FC2 [H x (W/2+1)] (layout FFTW)
void dftR2C(const Mat& src, Mat& dst) {
    CV_Assert(src.type() == CV_32FC1 && src.isContinuous());
    const int H = src.rows, W = src.cols;
    dst.create(H, W / 2 + 1, CV_32FC2);
    float* in = (float*)src.data;
    fftwf_complex* out = (fftwf_complex*)dst.data;
    fftwf_plan p = getR2C(H, W);
    fftwf_execute_dft_r2c(p, in, out);
}

// c2c: src 32FC2 [H x W] -> dst 32FC2 [H x W]; inverse=false -> forward
void dftC2C(const Mat& src, Mat& dst, bool inverse) {
    CV_Assert(src.type() == CV_32FC2 && src.isContinuous());
    const int H = src.rows, W = src.cols;
    dst.create(H, W, CV_32FC2);
    fftwf_complex* in = (fftwf_complex*)src.data;
    fftwf_complex* out = (fftwf_complex*)dst.data;
    fftwf_plan p = getC2C(H, W, inverse ? FFTW_BACKWARD : FFTW_FORWARD);
    fftwf_execute_dft(p, in, out);
    if (inverse) dst /= (double)(H * W); // масштаб как у cv::idft
}

} // namespace fastcv
