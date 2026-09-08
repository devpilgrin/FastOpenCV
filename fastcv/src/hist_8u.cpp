// FastOpenCV: calcHist 8UC1 256-bin - параллельные per-worker гистограммы + merge.
// cv::calcHist скалярен с gather-зависимостью (~0.33ms на 1080p).
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void calcHist8u(const Mat& src, Mat& hist /* 256x1 32F */) {
    CV_Assert(src.type() == CV_8UC1);
    const int H = src.rows, W = src.cols;
    const int nt = std::max(1, getNumThreads());
    std::vector<std::vector<int>> partial(nt, std::vector<int>(256, 0));
    parallel_for_(Range(0, H), [&](const Range& rg) {
        int tid = getThreadNum();
        if (tid < 0 || tid >= nt) tid = 0;
        int* h = partial[tid].data();
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* p = src.ptr<uchar>(y);
            for (int x = 0; x < W; x++) h[p[x]]++;
        }
    });
    hist.create(256, 1, CV_32F);
    float* hp = hist.ptr<float>();
    for (int i = 0; i < 256; i++) {
        int s = 0;
        for (int t = 0; t < nt; t++) s += partial[t][i];
        hp[i] = (float)s;
    }
}

} // namespace fastcv
