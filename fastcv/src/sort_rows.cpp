// FastOpenCV: параллельные sort/sortIdx по строкам (cv::sort/sortIdx однопоточны).
// Семантика: SORT_EVERY_ROW | SORT_ASCENDING/DESCENDING, 32F.
#include <opencv2/core.hpp>
#include <algorithm>
#include <vector>

namespace fastcv {

using namespace cv;

void sortRows32f(const Mat& src, Mat& dst, bool ascending) {
    CV_Assert(src.type() == CV_32FC1);
    dst.create(src.size(), src.type());
    parallel_for_(Range(0, src.rows), [&](const Range& rg) {
        std::vector<float> buf((size_t)src.cols);
        for (int y = rg.start; y < rg.end; y++) {
            const float* sp = src.ptr<float>(y);
            std::copy(sp, sp + src.cols, buf.begin());
            if (ascending)
                std::sort(buf.begin(), buf.end());
            else
                std::sort(buf.begin(), buf.end(), std::greater<float>());
            std::copy(buf.begin(), buf.end(), dst.ptr<float>(y));
        }
    });
}

void sortIdxRows32f(const Mat& src, Mat& dst, bool ascending) {
    CV_Assert(src.type() == CV_32FC1);
    dst.create(src.size(), CV_32SC1);
    parallel_for_(Range(0, src.rows), [&](const Range& rg) {
        std::vector<int> idx((size_t)src.cols);
        for (int y = rg.start; y < rg.end; y++) {
            const float* sp = src.ptr<float>(y);
            int* di = dst.ptr<int>(y);
            for (int x = 0; x < src.cols; x++) idx[x] = x;
            if (ascending)
                std::sort(idx.begin(), idx.end(), [&](int a, int b) { return sp[a] < sp[b]; });
            else
                std::sort(idx.begin(), idx.end(), [&](int a, int b) { return sp[a] > sp[b]; });
            std::copy(idx.begin(), idx.end(), di);
        }
    });
}

} // namespace fastcv
