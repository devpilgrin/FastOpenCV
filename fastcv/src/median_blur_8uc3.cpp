// FastOpenCV: medianBlur 8UC3 - split + 3x параллельный C1 O(1)-median + merge.
// cv::medianBlur 8UC3 k>5 ~106 мс на 1080p (однопоточный per-channel путь).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

void medianBlur8u(const Mat& src, Mat& dst, int ksize);

void medianBlur8uC3(const Mat& src, Mat& dst, int ksize) {
    CV_Assert(src.type() == CV_8UC3);
    std::vector<Mat> ch(3), out(3);
    split(src, ch);
    // НЕ параллелить по каналам: вложенный parallel_for_ сериализуется.
    // medianBlur8u сам параллелен - вызываем последовательно.
    for (int c = 0; c < 3; c++)
        medianBlur8u(ch[c], out[c], ksize);
    merge(out, dst);
}

} // namespace fastcv
