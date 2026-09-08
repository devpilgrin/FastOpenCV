// FastOpenCV: adaptiveThreshold 8UC1.
// OpenCV GAUSSIAN_C: convertTo(32F) -> GaussianBlur(32F, медленный float-путь)
// -> convertTo(8U) + скалярный ОДНОПОТОЧНЫЙ LUT-порог по всему изображению.
// У нас: 8U fixed-point GaussianBlur + параллельный LUT-порог.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

namespace fastcv {

using namespace cv;

void adaptiveThreshold8u(const Mat& src, Mat& dst, double maxValue,
                         int method, int type, int blockSize, double delta) {
    CV_Assert(src.type() == CV_8UC1 && blockSize % 2 == 1 && blockSize > 1);
    dst.create(src.size(), src.type());
    if (maxValue < 0) { dst.setTo(0); return; }

    Mat mean;
    if (method == ADAPTIVE_THRESH_MEAN_C)
        boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
                  Point(-1, -1), true, BORDER_REPLICATE | BORDER_ISOLATED);
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
        GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0,
                     BORDER_REPLICATE | BORDER_ISOLATED);
    else
        CV_Error(Error::StsBadFlag, "unknown method");

    const uchar imaxval = saturate_cast<uchar>(maxValue);
    const int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
    uchar tab[768];
    for (int i = 0; i < 768; i++) {
        if (type == THRESH_BINARY) tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
        else                       tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
    }
    const int H = src.rows, W = src.cols;
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const uchar* s = src.ptr<uchar>(y);
            const uchar* m = mean.ptr<uchar>(y);
            uchar* d = dst.ptr<uchar>(y);
            for (int x = 0; x < W; x++)
                d[x] = tab[s[x] - m[x] + 255];
        }
    });
}

} // namespace fastcv
