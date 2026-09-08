// Третья волна аудита core: sort, dot, DFT-варианты, память-операции
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>

using namespace cv;
using Clock = std::chrono::steady_clock;

template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 2; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
#define B(name, iters, ...) printf("%-44s %10.3f ms\n", name, timeit(iters, [&] { __VA_ARGS__; }));

int main() {
    RNG rng(7);
    const int W = 1920, H = 1080;
    Mat f32(H, W, CV_32FC1), f32b(H, W, CV_32FC1);
    rng.fill(f32, RNG::UNIFORM, 0, 1);
    rng.fill(f32b, RNG::UNIFORM, 0, 1);
    Mat dst, dsti;

    printf("=== сортировки ===\n");
    B("sort rows ASC 1080x1920 32F", 10, sort(f32, dst, SORT_EVERY_ROW | SORT_ASCENDING));
    B("sortIdx rows 256x1920 32F", 10, { Mat small = f32(Rect(0, 0, W, 256)); sortIdx(small, dsti, SORT_EVERY_ROW); });

    printf("=== DFT ===\n");
    Mat sq(1024, 1024, CV_32FC1);
    rng.fill(sq, RNG::UNIFORM, 0, 1);
    B("dft 1024 real->complex", 15, dft(sq, dst));
    B("dft 1024 complex->complex", 15, { Mat cplx; dft(sq, cplx, DFT_COMPLEX_OUTPUT); dft(cplx, dst, DFT_COMPLEX_INPUT); });
    B("dft 2048 real->complex", 5, { Mat big(2048, 2048, CV_32FC1); rng.fill(big, RNG::UNIFORM, 0, 1); dft(big, dst); });
    B("idft 1024 (inverse+scale)", 15, { Mat cplx; dft(sq, cplx, DFT_COMPLEX_OUTPUT); idft(cplx, dst, DFT_SCALE | DFT_REAL_OUTPUT); });

    printf("=== dot / скалярные ===\n");
    B("Mat::dot 1080p 32F", 50, { volatile double s = f32.dot(f32b); });
    B("norm L2 с маской", 50, { Mat mask; compare(f32, f32b, mask, CMP_GT); norm(f32, f32b, NORM_L2, mask); });

    printf("=== память-операции ===\n");
    B("hconcat 2x1080p 32F", 50, hconcat(f32, f32b, dst));
    B("vconcat 2x1080p 32F", 50, vconcat(f32, f32b, dst));
    B("repeat 960x540 -> 1080p", 50, { Mat small = f32(Rect(0, 0, 960, 540)); repeat(small, 2, 2, dst); });
    B("rotate ROTATE_90 1080p", 50, rotate(f32, dst, ROTATE_90_CLOCKWISE));
    B("copyMakeBorder +64 1080p", 50, copyMakeBorder(f32, dst, 64, 64, 64, 64, BORDER_REPLICATE));
    B("Mat::clone 1080p 32F", 50, { Mat c = f32.clone(); });
    B("setTo 1080p 32F", 50, dst.setTo(1.5));

    printf("=== прочее ===\n");
    B("pow 32F ^2 (спец.путь)", 50, pow(f32, 2.0, dst));
    B("pow 32F ^1.7 (общий)", 20, pow(f32, 1.7, dst));
    B("cubeRoot", 20, { Mat t = f32.clone(); for (int i = 0; i < 100; i++) t.at<float>(i % H, i % W) = cubeRoot(t.at<float>(i % H, i % W)); });
    B("max 32F elemwise", 50, max(f32, f32b, dst));
    B("bitwise_and 8UC1", 50, { Mat a(H, W, CV_8UC1, Scalar(5)), b(H, W, CV_8UC1, Scalar(3)), d; bitwise_and(a, b, d); });
    return 0;
}
