// FastOpenCV: аудит кернелов imgproc/core - ищем операции с аномально низкой
// эффективной пропускной способностью (кандидаты на алгоритмическую оптимизацию)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace cv;
using Clock = std::chrono::steady_clock;

struct Result { std::string name; double ms; double gbps; };

template <typename F>
Result bench(const std::string& name, double bytes_per_op, int iters, F&& f) {
    for (int i = 0; i < 3; i++) f(); // прогрев
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
    return {name, ms, bytes_per_op / (ms * 1e6)}; // GB/s
}

int main() {
    // setNumThreads(0) - ЛОВУШКА: в этой связке OpenCV+TBB даёт 4x затормаживание // все ядра
    std::vector<Result> out;
    const int W = 1920, H = 1080;
    const double PX = (double)W * H;

    Mat u8c1(H, W, CV_8UC1, Scalar(128));
    Mat u8c3(H, W, CV_8UC3, Scalar(128, 64, 32));
    Mat f32c1(H, W, CV_32FC1, Scalar(0.5f));
    Mat dst;

    auto rep = [&](const Result& r) {
        out.push_back(r);
        printf("%-42s %9.3f ms  %8.2f GB/s\n", r.name.c_str(), r.ms, r.gbps);
    };

    // --- эталоны полосы памяти ---
    rep(bench("copyTo 8UC1 (memory roof)", PX * 2, 100, [&] { u8c1.copyTo(dst); }));
    rep(bench("threshold 8UC1", PX * 2, 100, [&] { threshold(u8c1, dst, 127, 255, THRESH_BINARY); }));

    // --- морфология ---
    rep(bench("erode rect 3x3 8UC1", PX * 2, 100, [&] { erode(u8c1, dst, getStructuringElement(MORPH_RECT, {3, 3})); }));
    rep(bench("erode rect 15x15 8UC1", PX * 2, 50, [&] { erode(u8c1, dst, getStructuringElement(MORPH_RECT, {15, 15})); }));
    rep(bench("dilate ellipse 15x15 8UC1 (несепар.)", PX * 2, 50, [&] { dilate(u8c1, dst, getStructuringElement(MORPH_ELLIPSE, {15, 15})); }));
    rep(bench("morphologyEx BLACKHAT 31x31", PX * 2, 20, [&] { morphologyEx(u8c1, dst, MORPH_BLACKHAT, getStructuringElement(MORPH_RECT, {31, 31})); }));

    // --- фильтры ---
    rep(bench("blur 15x15 8UC1", PX * 2, 50, [&] { blur(u8c1, dst, {15, 15}); }));
    rep(bench("medianBlur 5 8UC1", PX * 2, 50, [&] { medianBlur(u8c1, dst, 5); }));
    rep(bench("medianBlur 9 8UC1", PX * 2, 20, [&] { medianBlur(u8c1, dst, 9); }));
    rep(bench("filter2D 7x7 8UC1->32F", PX * (1 + 4), 30, [&] { filter2D(u8c1, dst, CV_32F, Mat::ones(7, 7, CV_32F) / 49.0); }));
    rep(bench("GaussianBlur 25x25 8UC1", PX * 2, 30, [&] { GaussianBlur(u8c1, dst, {25, 25}, 5); }));
    rep(bench("bilateralFilter 9 8UC1", PX * 2, 10, [&] { bilateralFilter(u8c1, dst, 9, 50, 50); }));

    // --- геометрия ---
    rep(bench("resize 2x down INTER_LINEAR 8UC3", PX * 3 + PX * 3 / 4, 50, [&] { resize(u8c3, dst, {W / 2, H / 2}, 0, 0, INTER_LINEAR); }));
    rep(bench("resize 2x down INTER_AREA 8UC3", PX * 3 + PX * 3 / 4, 50, [&] { resize(u8c3, dst, {W / 2, H / 2}, 0, 0, INTER_AREA); }));
    rep(bench("resize 2x down INTER_CUBIC 8UC3", PX * 3 + PX * 3 / 4, 50, [&] { resize(u8c3, dst, {W / 2, H / 2}, 0, 0, INTER_CUBIC); }));
    rep(bench("warpAffine 8UC3 INTER_LINEAR", PX * 3 * 2, 50, [&] {
        Mat M = (Mat_<double>(2, 3) << 1, 0.05, 10, -0.05, 1, 5);
        warpAffine(u8c3, dst, M, {W, H}, INTER_LINEAR);
    }));
    rep(bench("remap 8UC1 (fixed maps)", PX * 2 * 2, 50, [&] {
        static Mat mx, my;
        if (mx.empty()) { mx.create(H, W, CV_32F); my.create(H, W, CV_32F);
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { mx.at<float>(y, x) = (float)x; my.at<float>(y, x) = (float)y; } }
        remap(u8c1, dst, mx, my, INTER_LINEAR);
    }));

    // --- прочее ---
    rep(bench("integral 8UC1->32S", PX * (1 + 4), 30, [&] { integral(u8c1, dst, CV_32S); }));
    rep(bench("Sobel 3x3 8UC1->16S", PX * (1 + 2), 100, [&] { Sobel(u8c1, dst, CV_16S, 1, 0); }));
    rep(bench("Sobel 3x3 32FC1->32F", PX * 4 * 2, 100, [&] { Sobel(f32c1, dst, CV_32F, 1, 0); }));
    rep(bench("equalizeHist 8UC1", PX * 2, 30, [&] { equalizeHist(u8c1, dst); }));
    rep(bench("cornerHarris 8UC1", PX * (1 + 4), 10, [&] { cornerHarris(u8c1, dst, 3, 3, 0.04); }));
    rep(bench("pyrDown 8UC3", PX * 3 + PX * 3 / 4, 50, [&] { pyrDown(u8c3, dst); }));
    rep(bench("cvtColor BGR2HSV 8UC3", PX * 3 * 2, 50, [&] { cvtColor(u8c3, dst, COLOR_BGR2HSV); }));
    rep(bench("Laplacian 8UC1->16S k3", PX * 3, 50, [&] { Laplacian(u8c1, dst, CV_16S, 3); }));

    printf("\n=== Сводка: кандидаты (< 3 GB/s при размере >= 2 МП) ===\n");
    for (auto& r : out)
        if (r.gbps < 3.0 && r.name.find("roof") == std::string::npos)
            printf("%-42s %9.3f ms  %8.2f GB/s\n", r.name.c_str(), r.ms, r.gbps);
    return 0;
}
