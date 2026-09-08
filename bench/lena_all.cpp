// Lena: все оптимизированные фильтры, сток vs патченная+fastcv
// Компилируется дважды: -DUSE_FASTCV=1 (dispatch) и без (stock).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#ifdef USE_FASTCV
#include "fastcv/fastcv.hpp"
#endif

using namespace cv;
using Clock = std::chrono::steady_clock;

struct Row { std::string op; double ms; double maxdiff; };

template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

static void saveGray(const char* path, const Mat& m) {
    Mat v; m.convertTo(v, CV_8U); imwrite(path, v);
}

int main() {
    Mat lenaC = imread("/home/roman/workspace/FastOpemCV/data/lena.jpg");
    if (lenaC.empty()) { printf("NO LENA\n"); return 1; }
    Mat lena;
    cvtColor(lenaC, lena, COLOR_BGR2GRAY);
    printf("lena: %dx%d\n", lena.cols, lena.rows);
    const int N = 50; // итераций (512x512 маленькая)

    std::vector<Row> rows;
    auto add = [&](const char* op, double ms, double maxdiff) { rows.push_back({op, ms, maxdiff}); };

#ifdef USE_FASTCV
    // ---------- fastcv / patched ----------
    { Mat a, b; medianBlur(lena, a, 9); fastcv::medianBlur8u(lena, b, 9);
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("medianBlur k=9", timeit(N, [&] { fastcv::medianBlur8u(lena, b, 9); }), md);
      imwrite("/tmp/lena_median_fastcv.png", b); }
    { Mat a, b; bilateralFilter(lena, a, 9, 50, 50); fastcv::bilateralFilter8u(lena, b, 9, 50, 50);
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("bilateralFilter d=9", timeit(N, [&] { fastcv::bilateralFilter8u(lena, b, 9, 50, 50); }), md); }
    { Mat b; fastcv::bilateralGrid8u(lena, b, 15, 50, 50);
      add("bilateralGrid d=15 (approx)", timeit(N, [&] { fastcv::bilateralGrid8u(lena, b, 15, 50, 50); }), -1); }
    { Mat a, b; cornerHarris(lena, a, 3, 3, 0.04); fastcv::cornerHarris8u(lena, b, 3, 0.04);
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("cornerHarris b=3", timeit(N, [&] { fastcv::cornerHarris8u(lena, b, 3, 0.04); }), md); }
    { Mat a, b; resize(lena, a, {256, 256}, 0, 0, INTER_CUBIC); fastcv::resizeCubic8u(lena, b, {256, 256});
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("resize CUBIC 512->256", timeit(N, [&] { fastcv::resizeCubic8u(lena, b, {256, 256}); }), md);
      imwrite("/tmp/lena_resize_fastcv.png", b); }
    { Mat k = Mat::ones(7, 7, CV_32F) / 49.f; Mat a, b;
      filter2D(lena, a, CV_32F, k); fastcv::filter2D_8u32f(lena, b, k, Point(-1, -1), 0);
      Mat d; absdiff(a, b, d); double md, mr; minMaxLoc(d, nullptr, &md); minMaxLoc(a, nullptr, &mr);
      add("filter2D 7x7 ->32F", timeit(N, [&] { fastcv::filter2D_8u32f(lena, b, k, Point(-1, -1), 0); }), md / mr); }
    { Mat lut(1, 256, CV_8UC1); for (int i = 0; i < 256; i++) lut.at<uchar>(i) = 255 - i;
      Mat a, b; LUT(lena, lut, a); fastcv::lut8u(lena, lut.ptr(), b);
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("LUT инверсия", timeit(N, [&] { fastcv::lut8u(lena, lut.ptr(), b); }), md);
      imwrite("/tmp/lena_lut_fastcv.png", b); }
    { double s[1]; fastcv::sum8u(lena, s);
      add("sum", timeit(N, [&] { fastcv::sum8u(lena, s); }), 0); }
    { Mat f; lena.convertTo(f, CV_32F); Mat m = Mat::eye(4, 4, CV_64F); Mat c3; cvtColor(f, c3, COLOR_GRAY2BGR);
      Mat a, b; transform(c3, a, m); fastcv::transform32f(c3, b, m);
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("transform 3ch", timeit(N, [&] { fastcv::transform32f(c3, b, m); }), md); }
    { Mat f; lena.convertTo(f, CV_32F, 1.0 / 255); f += 0.01f; Mat a, b;
      pow(f, 1.7, a); fastcv::pow32f(f, b, 1.7);
      Mat d; absdiff(a, b, d); divide(d, a, d); double md; minMaxLoc(d, nullptr, &md);
      add("pow ^1.7 (rel)", timeit(N, [&] { fastcv::pow32f(f, b, 1.7); }), md); }
    { Mat f; lena.convertTo(f, CV_32F); Mat d;
      add("dft 512 (патч)", timeit(N, [&] { dft(f, d); }), 0); }
    { Mat f; lena.convertTo(f, CV_32F); Mat d;
      add("dct 512 (патч)", timeit(N, [&] { dct(f, d); }), 0); }
    { Mat se = getStructuringElement(MORPH_ELLIPSE, {31, 31}); Mat a, b;
      dilate(lena, a, se); fastcv::morphEllipseApprox8u(lena, b, 31, MORPH_DILATE);
      Mat d; absdiff(a, b, d); double md; minMaxLoc(d, nullptr, &md);
      add("dilate ellipse k=31 (approx)", timeit(N, [&] { fastcv::morphEllipseApprox8u(lena, b, 31, MORPH_DILATE); }), md); }
    { Mat a; sortIdx(lena, a, SORT_EVERY_ROW); Mat f; lena.convertTo(f, CV_32F); Mat b; fastcv::sortIdxRows32f(f, b, true);
      add("sortIdx rows (32F)", timeit(N, [&] { fastcv::sortIdxRows32f(f, b, true); }), -1); }
    // lapack-уровень
    { Mat a(512, 512, CV_64F); lena.convertTo(a, CV_64F); Mat b;
      add("gemm 512 (BLAS)", timeit(N, [&] { gemm(a, a, 1.0, noArray(), 0, b); }), 0); }
    { Mat a(256, 256, CV_64F); lena(Rect(0, 0, 256, 256)).convertTo(a, CV_64F); a = a * a.t() + Mat::eye(256, 256, CV_64F);
      Mat w, u, vt; add("SVD 256 (LAPACK)", timeit(20, [&] { SVD::compute(a, w, u, vt); }), 0); }
    { Mat a(256, 256, CV_64F); lena(Rect(0, 0, 256, 256)).convertTo(a, CV_64F); a = a * a.t() + Mat::eye(256, 256, CV_64F);
      Mat w, v; add("eigenSym 256 (fastcv)", timeit(20, [&] { fastcv::eigenSym(a, w, v); }), 0); }
#else
    // ---------- stock ----------
    { Mat a; add("medianBlur k=9", timeit(N, [&] { medianBlur(lena, a, 9); }), 0); }
    { Mat a; add("bilateralFilter d=9", timeit(N, [&] { bilateralFilter(lena, a, 9, 50, 50); }), 0); }
    { Mat a; add("bilateralGrid d=15 (approx)", 0, 0); }
    { Mat a; add("cornerHarris b=3", timeit(N, [&] { cornerHarris(lena, a, 3, 3, 0.04); }), 0); }
    { Mat a; add("resize CUBIC 512->256", timeit(N, [&] { resize(lena, a, {256, 256}, 0, 0, INTER_CUBIC); }), 0);
      imwrite("/tmp/lena_resize_stock.png", a); }
    { Mat k = Mat::ones(7, 7, CV_32F) / 49.f; Mat a;
      add("filter2D 7x7 ->32F", timeit(N, [&] { filter2D(lena, a, CV_32F, k); }), 0); }
    { Mat lut(1, 256, CV_8UC1); for (int i = 0; i < 256; i++) lut.at<uchar>(i) = 255 - i;
      Mat a; add("LUT инверсия", timeit(N, [&] { LUT(lena, lut, a); }), 0);
      imwrite("/tmp/lena_lut_stock.png", a); }
    { add("sum", timeit(N, [&] { volatile Scalar s = sum(lena); }), 0); }
    { Mat f; lena.convertTo(f, CV_32F); Mat m = Mat::eye(4, 4, CV_64F); Mat c3; cvtColor(f, c3, COLOR_GRAY2BGR);
      Mat a; add("transform 3ch", timeit(N, [&] { transform(c3, a, m); }), 0); }
    { Mat f; lena.convertTo(f, CV_32F, 1.0 / 255); f += 0.01f; Mat a;
      add("pow ^1.7 (rel)", timeit(N, [&] { pow(f, 1.7, a); }), 0); }
    { Mat f; lena.convertTo(f, CV_32F); Mat d;
      add("dft 512 (патч)", timeit(N, [&] { dft(f, d); }), 0); }
    { Mat f; lena.convertTo(f, CV_32F); Mat d;
      add("dct 512 (патч)", timeit(N, [&] { dct(f, d); }), 0); }
    { Mat se = getStructuringElement(MORPH_ELLIPSE, {31, 31}); Mat a;
      add("dilate ellipse k=31 (approx)", timeit(N, [&] { dilate(lena, a, se); }), 0); }
    { Mat a; Mat f; lena.convertTo(f, CV_32F);
      add("sortIdx rows (32F)", timeit(N, [&] { sortIdx(f, a, SORT_EVERY_ROW); }), 0); }
    { Mat a(512, 512, CV_64F); lena.convertTo(a, CV_64F); Mat b;
      add("gemm 512 (BLAS)", timeit(N, [&] { gemm(a, a, 1.0, noArray(), 0, b); }), 0); }
    { Mat a(256, 256, CV_64F); lena(Rect(0, 0, 256, 256)).convertTo(a, CV_64F); a = a * a.t() + Mat::eye(256, 256, CV_64F);
      Mat w, u, vt; add("SVD 256 (LAPACK)", timeit(20, [&] { SVD::compute(a, w, u, vt); }), 0); }
    { Mat a(256, 256, CV_64F); lena(Rect(0, 0, 256, 256)).convertTo(a, CV_64F); a = a * a.t() + Mat::eye(256, 256, CV_64F);
      Mat w, v; add("eigenSym 256 (fastcv)", timeit(20, [&] { eigen(a, w, v); }), 0); }
#endif

    printf("%-30s %12s %12s\n", "operation", "ms", "maxDiff");
    for (auto& r : rows) {
        if (r.ms <= 0) { printf("%-30s %12s %12s\n", r.op.c_str(), "-", "-"); continue; }
        if (r.maxdiff < 0) printf("%-30s %12.4f %12s\n", r.op.c_str(), r.ms, "n/a");
        else printf("%-30s %12.4f %12.4g\n", r.op.c_str(), r.ms, r.maxdiff);
    }
    return 0;
}
