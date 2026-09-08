#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
using namespace cv;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    const int k = 31, r = k / 2;
    const int W = img.cols, H = img.rows;
    for (int it = 0; it < 3; it++) {
        auto t0 = Clock::now();
        Mat padded;
        copyMakeBorder(img, padded, r, k - 1 - r, r, k - 1 - r, BORDER_REFLECT_101);
        auto t1 = Clock::now();
        const int PW = padded.cols;
        Mat tmp(padded.rows, W, CV_8UC1);
        parallel_for_(Range(0, padded.rows), [&](const Range& rg) {
            std::vector<uchar> gl(PW), hl(PW);
            uchar* G = gl.data(); uchar* Hh = hl.data();
            for (int y = rg.start; y < rg.end; y++) {
                const uchar* S = padded.ptr<uchar>(y);
                G[0] = S[0];
                int cnt = 1;
                for (int i = 1; i < PW; i++) {
                    if (cnt == k) cnt = 0;
                    G[i] = (cnt == 0) ? S[i] : std::max(S[i], G[i - 1]);
                    cnt++;
                }
                Hh[PW - 1] = S[PW - 1];
                cnt = 1;
                for (int i = PW - 2; i >= 0; i--) {
                    if (cnt == k) cnt = 0;
                    Hh[i] = (cnt == 0) ? S[i] : std::max(S[i], Hh[i + 1]);
                    cnt++;
                }
                uchar* D = tmp.ptr<uchar>(y);
                for (int x = 0; x < W; x++) D[x] = std::max(G[x + k - 1], Hh[x]);
            }
        });
        auto t2 = Clock::now();
        printf("iter %d: border %.3f ms | horiz %.3f ms\n", it, ms(t0, t1), ms(t1, t2));
    }
    return 0;
}
