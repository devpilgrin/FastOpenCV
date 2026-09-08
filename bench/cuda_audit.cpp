#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
namespace fastcv { void medianBlur8u(const cv::Mat&, cv::Mat&, int); }
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
template <typename F> double timeitGPU(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    cuda::Stream s;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) { f(); }
    cuda::DeviceInfo().name();
    cuda::Stream::Null().waitForCompletion();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
int main() {
    RNG rng(42);
    Mat img(1080, 1920, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    cuda::GpuMat g_src, g_dst;
    g_src.upload(img);

    // без учёта трансферов (resident)
    auto gf = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, {5,5}, 1.0);
    printf("gaussian 5x5  GPU-resident: %.3f ms | CPU cv: ", timeitGPU(50, [&]{ gf->apply(g_src, g_dst); }));
    Mat cpu; printf("%.3f ms\n", timeit(30, [&]{ GaussianBlur(img, cpu, {5,5}, 1.0); }));

    printf("resize 2x     GPU-resident: %.3f ms | CPU cv: ", timeitGPU(50, [&]{ cuda::resize(g_src, g_dst, {960,540}); }));
    printf("%.3f ms\n", timeit(30, [&]{ resize(img, cpu, {960,540}); }));

    auto mf = cuda::createMedianFilter(CV_8UC1, 9);
    printf("median 9      GPU-resident: %.3f ms | CPU cv: ", timeitGPU(50, [&]{ mf->apply(g_src, g_dst); }));
    printf("%.3f ms | fastcv: ", timeit(30, [&]{ medianBlur(img, cpu, 9); }));
    Mat fb; printf("%.3f ms\n", timeit(30, [&]{ fastcv::medianBlur8u(img, fb, 9); }));

    auto bf = cuda::createBoxFilter(CV_8UC1, CV_8UC1, {15,15});
    printf("box 15x15     GPU-resident: %.3f ms | CPU cv: ", timeitGPU(50, [&]{ bf->apply(g_src, g_dst); }));
    printf("%.3f ms\n", timeit(30, [&]{ boxFilter(img, cpu, -1, {15,15}); }));

    printf("bilateral     GPU-resident: %.3f ms | CPU cv: ",
           timeitGPU(30, [&]{ cuda::bilateralFilter(g_src, g_dst, 9, 50, 10); }));
    printf("%.3f ms\n", timeit(20, [&]{ bilateralFilter(img, cpu, 9, 50, 10); }));

    // с трансферами (upload+download) - честный end-to-end
    printf("\n== end-to-end (upload+download) ==\n");
    printf("gaussian e2e:  %.3f ms\n", timeit(20, [&]{
        g_src.upload(img); gf->apply(g_src, g_dst); g_dst.download(cpu); }));
    printf("median9  e2e:  %.3f ms\n", timeit(20, [&]{
        g_src.upload(img); mf->apply(g_src, g_dst); g_dst.download(cpu); }));
    return 0;
}
