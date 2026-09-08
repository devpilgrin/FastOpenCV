// Аудит оставшихся математических операций ядра OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 2; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
#define B(name, iters, ...) printf("%-42s %10.3f ms\n", name, timeit(iters, [&] { __VA_ARGS__; }));

int main() {
    // setNumThreads(0) - ЛОВУШКА: в этой связке OpenCV+TBB даёт 4x затормаживание
    RNG rng(7);
    const int W = 1920, H = 1080;
    Mat f32a(H, W, CV_32FC1), f32b(H, W, CV_32FC1);
    rng.fill(f32a, RNG::UNIFORM, 0.1, 2.0);
    rng.fill(f32b, RNG::UNIFORM, 0.1, 2.0);
    Mat dst, dst2;

    printf("=== большие 32F 1080p ===\n");
    B("reduce rows SUM", 50, reduce(f32a, dst, 0, REDUCE_SUM));
    B("reduce cols AVG", 50, reduce(f32a, dst, 1, REDUCE_AVG));
    B("min (elemwise)", 50, min(f32a, f32b, dst));
    B("compare GT", 50, compare(f32a, f32b, dst, CMP_GT));
    B("inRange 32F", 50, inRange(f32a, 0.5, 1.5, dst));
    B("phase", 20, phase(f32a, f32b, dst));
    B("magnitude", 50, magnitude(f32a, f32b, dst));
    B("cartToPolar", 20, cartToPolar(f32a, f32b, dst, dst2));
    B("polarToCart", 50, polarToCart(f32a, f32b, dst, dst2));
    B("fastAtan2 loop x100", 10, { volatile float s = 0; for (int i = 0; i < 100; i++) s += fastAtan2(0.7f, 0.3f); });
    B("dct 1024x1024", 10, { Mat d; dct(f32a(Rect(0, 0, 1024, 1024)), d); });
    B("patchNaNs", 50, { Mat t = f32a.clone(); t.at<float>(500, 500) = NAN; patchNaNs(t); });
    B("transform 4x4 (3ch)", 50, {
        Mat c3; cvtColor(f32a, c3, COLOR_GRAY2BGR);
        Mat M = Mat::eye(4, 4, CV_64F);
        transform(c3, dst2, M);
    });
    B("mulSpectrums 1024", 10, {
        Mat d; dft(f32a(Rect(0, 0, 1024, 1024)), d, DFT_COMPLEX_OUTPUT);
        mulSpectrums(d, d, d, 0);
    });

    printf("=== матричные выражения 512x512 64F ===\n");
    Mat m512(512, 512, CV_64F), n512(512, 512, CV_64F);
    rng.fill(m512, RNG::UNIFORM, -1, 1);
    rng.fill(n512, RNG::UNIFORM, -1, 1);
    B("A*B (выражение)", 5, { Mat r = m512 * n512; });
    B("A*B + A - B (цепочка)", 5, { Mat r = m512 * n512 + m512 - n512; });
    B("A.t()*B", 5, { Mat r = m512.t() * n512; });

    printf("=== малые матрицы 64F ===\n");
    Mat m64(64, 64, CV_64F);
    rng.fill(m64, RNG::UNIFORM, -1, 1);
    m64 = m64 * m64.t() + Mat::eye(64, 64, CV_64F) * 10;
    B("determinant 64x64 x1000", 5, { volatile double s = 0; for (int i = 0; i < 1000; i++) s += determinant(m64); });
    B("trace 512x512 x1000", 10, { volatile double s = 0; for (int i = 0; i < 1000; i++) s += trace(m512)[0]; });
    B("invert DECOMP_SVD 64x64 x100", 5, { for (int i = 0; i < 100; i++) { Mat inv; invert(m64, inv, DECOMP_SVD); } });
    B("invert DECOMP_CHOLESKY 64x64 x100", 5, { for (int i = 0; i < 100; i++) { Mat inv; invert(m64, inv, DECOMP_CHOLESKY); } });
    B("solve DECOMP_QR 64x64 x100", 5, {
        Mat b64 = Mat::ones(64, 1, CV_64F);
        for (int i = 0; i < 100; i++) { Mat x; solve(m64, b64, x, DECOMP_QR); }
    });
    B("eigenNonSymmetric 64x64 x10", 5, { for (int i = 0; i < 10; i++) { Mat e, ev; eigenNonSymmetric(m64, e, ev); } });
    B("solvePoly кубика x10000", 5, {
        Mat coeffs = (Mat_<double>(1, 4) << 1, -6, 11, -6);
        for (int i = 0; i < 10000; i++) { Mat roots; solvePoly(coeffs, roots); }
    });

    printf("=== PCA / ковариация ===\n");
    Mat data(5000, 64, CV_64F);
    rng.fill(data, RNG::UNIFORM, -1, 1);
    B("PCA 5000x64 (10 комп.)", 5, { PCA pca(data, Mat(), PCA::DATA_AS_ROW, 10); });
    B("calcCovarMatrix 5000x64", 5, {
        Mat cov, mean;
        calcCovarMatrix(data, cov, mean, COVAR_NORMAL | COVAR_ROWS | COVAR_SCALE);
    });
    B("kmeans 10k x 32f, 8 кластеров", 3, {
        Mat pts(10000, 32, CV_32F); rng.fill(pts, RNG::UNIFORM, 0, 1);
        Mat labels, centers;
        kmeans(pts, 8, labels, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 20, 0.1),
               3, KMEANS_PP_CENTERS, centers);
    });
    return 0;
}
