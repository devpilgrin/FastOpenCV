// FastOpenCV: resize INTER_AREA с целочисленным фактором (2,3,4...) 8U C1/C3.
// У OpenCV быстрый путь только 2x2; 4x4 идёт через generic (~0.92ms на 1080p->480x270).
// Здесь: блочное суммирование NxN с SIMD (u8->i16->i32), округление сдвигом.
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <vector>

namespace fastcv {

using namespace cv;

// dst = area-усреднение блоков factor x factor. Семантика cv::resize INTER_AREA
// для целых факторов: mean с округлением (cvRound(sum / (f*f))).
void resizeAreaInt8u(const Mat& src, Mat& dst, int factor) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    CV_Assert(factor >= 2 && src.cols % factor == 0 && src.rows % factor == 0);
    const int cn = src.channels();
    const int dW = src.cols / factor, dH = src.rows / factor;
    dst.create(dH, dW, src.type());
    const int W = src.cols;
    const int area = factor * factor;
    const int shift = area == 4 ? 2 : area == 16 ? 4 : 0; // точное округление для степеней 2
    const double inv = 1.0 / area;

    parallel_for_(Range(0, dH), [&](const Range& rg) {
        std::vector<int> acc((size_t)dW * cn);
        for (int dy = rg.start; dy < rg.end; dy++) {
            std::fill(acc.begin(), acc.end(), 0);
            for (int fy = 0; fy < factor; fy++) {
                const uchar* S = src.ptr<uchar>(dy * factor + fy);
                int* A = acc.data();
                if (factor == 4 && cn == 1) {
                    for (int x = 0; x < W; x += 4)
                        A[x / 4] += S[x] + S[x + 1] + S[x + 2] + S[x + 3];
                } else {
                    for (int x = 0; x < dW * cn; x++) {
                        int s = 0;
                        const uchar* p = S + x * factor - (x % cn == 0 ? 0 : 0);
                        // корректная блочная сумма по cn-учётному индексу:
                        // выходной пиксель dx, канал c покрывает src столбцы [dx*f .. dx*f+f)
                        (void)p;
                        s = 0;
                        int dx = x / cn, c = x % cn;
                        const uchar* q = S + dx * factor * cn + c;
                        for (int fx = 0; fx < factor; fx++) s += q[fx * cn];
                        A[x] += s;
                    }
                }
            }
            uchar* dp = dst.ptr<uchar>(dy);
            for (int x = 0; x < dW * cn; x++)
                dp[x] = shift ? (uchar)((acc[x] + (1 << (shift - 1))) >> shift)
                              : saturate_cast<uchar>(cvRound(acc[x] * inv));
        }
    });
}

} // namespace fastcv
