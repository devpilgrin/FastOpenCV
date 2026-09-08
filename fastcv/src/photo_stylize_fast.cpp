// FastOpenCV: edgePreservingFilter / detailEnhance через ximgproc::dtFilter.
// photo-модуль использует СОБСТВЕННЫЙ legacy Domain_Filter (npr.hpp) с per-pixel
// img.at<float>() - полностью скалярный код 2011 года (edgePreserving RECURS: 475 мс).
// ximgproc::dtFilter - тот же domain transform, но параллельный (13.7 мс).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/ximgproc.hpp>

namespace fastcv {

using namespace cv;

// flags: cv::RECURS_FILTER (1) или cv::NORMCONV_FILTER (2)
void edgePreservingFast(const Mat& src, Mat& dst, int flags, float sigma_s, float sigma_r) {
    CV_Assert(src.type() == CV_8UC3);
    Mat imgf, resf;
    src.convertTo(imgf, CV_32FC3, 1.0 / 255.0);
    int mode = flags == 2 ? ximgproc::DTF_NC : ximgproc::DTF_RF;
    ximgproc::dtFilter(imgf, imgf, resf, sigma_s, sigma_r, mode, 3);
    convertScaleAbs(resf, dst, 255.0, 0);
}

void detailEnhanceFast(const Mat& src, Mat& dst, float sigma_s, float sigma_r) {
    CV_Assert(src.type() == CV_8UC3);
    Mat imgf;
    src.convertTo(imgf, CV_32FC3, 1.0 / 255.0);
    Mat lab;
    cvtColor(imgf, lab, COLOR_BGR2Lab);
    std::vector<Mat> ch;
    split(lab, ch);
    Mat L;
    ch[0].convertTo(L, CV_32F, 1.0 / 255.0);
    Mat res;
    // как в photo: RECURS-фильтр по L
    ximgproc::dtFilter(L, L, res, sigma_s, sigma_r, ximgproc::DTF_RF, 3);
    Mat detail = L - res;
    multiply(detail, 3.0f, detail);
    L = res + detail;
    L.convertTo(ch[0], ch[0].depth(), 255.0);
    merge(ch, lab);
    Mat out;
    cvtColor(lab, out, COLOR_Lab2BGR);
    out.convertTo(dst, CV_8UC3, 255.0);
}

} // namespace fastcv
