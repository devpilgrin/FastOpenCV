// FastOpenCV: адаптивный sum 8U - cv::sum на малых (кэш-резидентных),
// fastcv::sum8u на больших. Порог ~2M пикселей (замеренная точка пересечения).
#include <opencv2/core.hpp>

namespace fastcv {

using namespace cv;

void sum8u(const Mat& src, double* sums);

Scalar sum8uSmart(const Mat& src) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    if (src.total() < 2000000)
        return cv::sum(src);
    double s[4] = {0, 0, 0, 0};
    sum8u(src, s);
    return Scalar(s[0], s[1], s[2], s[3]);
}

} // namespace fastcv
