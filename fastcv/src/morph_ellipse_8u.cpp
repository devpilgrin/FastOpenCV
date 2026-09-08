// FastOpenCV: морфология с эллиптическим SE через декомпозицию в зонотоп
// (4 сегмента: 0/45/90/135 градусов). Эллипс в OpenCV - несепарабельный O(k^2);
// здесь - композиция линейных SE (rect - van Herk O(1) у OpenCV; диагонали -
// через сдвиг строк). АППРОКСИМАЦИЯ: форма SE - зонотоп, не точный эллипс.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace fastcv {

using namespace cv;

// дилатация/эрозия диагональным сегментом через сдвиг строк (shear trick):
// strided-доступ по диагоналям оказался cache-hostile (каждая выборка - промах),
// shear+линейный доступ быстрее несмотря на две копии.
// slope=+1: направление (1,1); slope=-1: (1,-1)
static void morphDiagLine8u(const Mat& src, Mat& dst, int len, int slope, int op) {
    const int H = src.rows, W = src.cols;
    const int Ws = W + H;
    uchar fill = (op == MORPH_ERODE) ? 255 : 0;
    Mat sheared(H, Ws, CV_8UC1, Scalar(fill));
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            int shift = slope > 0 ? y : (H - 1 - y);
            src.row(y).copyTo(sheared.row(y).colRange(shift, shift + W));
        }
    });
    Mat tmp;
    if (op == MORPH_DILATE)
        dilate(sheared, tmp, getStructuringElement(MORPH_RECT, Size(len, 1)));
    else
        erode(sheared, tmp, getStructuringElement(MORPH_RECT, Size(len, 1)));
    parallel_for_(Range(0, H), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            int shift = slope > 0 ? y : (H - 1 - y);
            tmp.row(y).colRange(shift, shift + W).copyTo(dst.row(y));
        }
    });
}

void morphEllipseApprox8u(const Mat& src, Mat& dst, int ksize, int op) {
    CV_Assert(src.type() == CV_8UC1 && ksize >= 3 && ksize % 2 == 1);
    CV_Assert(op == MORPH_ERODE || op == MORPH_DILATE);
    const int r = ksize / 2;
    // 4 сегмента зонотопа, полудлина l = r*pi/(2*4) (сходимость опорной функции к r)
    int l = std::max(1, cvRound(r * CV_PI / 8.0));
    int len = 2 * l + 1;

    Mat cur = src.clone(), tmp;
    // 0° и 90° - rect-линиями у OpenCV (van Herk O(1))
    Mat seH = getStructuringElement(MORPH_RECT, Size(len, 1));
    Mat seV = getStructuringElement(MORPH_RECT, Size(1, len));
    if (op == MORPH_DILATE) { dilate(cur, tmp, seH); dilate(tmp, cur, seV); }
    else { erode(cur, tmp, seH); erode(tmp, cur, seV); }
    // диагонали
    morphDiagLine8u(cur, tmp, len, +1, op);
    morphDiagLine8u(tmp, cur, len, -1, op);
    cur.copyTo(dst);
}

} // namespace fastcv
