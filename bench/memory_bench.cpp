// Аудит работы с памятью OpenCV: churn Mat, first-touch, параллельные аллокации
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace cv;
using Clock = std::chrono::steady_clock;

static double us(const Clock::time_point& a, const Clock::time_point& b, int n) {
    return std::chrono::duration<double, std::micro>(b - a).count() / n;
}

int main() {
    printf("=== churn create/release (мкс на операцию) ===\n");
    for (size_t sz : {(size_t)64 * 1024, (size_t)512 * 1024, (size_t)6 * 1024 * 1024, (size_t)24 * 1024 * 1024}) {
        const int N = 200;
        // прогрев
        { Mat m(1080, 1920, CV_8UC1); }
        auto t0 = Clock::now();
        for (int i = 0; i < N; i++) {
            Mat m(1, (int)sz, CV_8UC1); // create+destroy
        }
        auto t1 = Clock::now();
        // create + first touch (запись) + destroy
        auto t2 = Clock::now();
        for (int i = 0; i < N; i++) {
            Mat m(1, (int)sz, CV_8UC1);
            memset(m.data, i & 0xFF, sz);
        }
        auto t3 = Clock::now();
        // malloc/free напрямую
        auto t4 = Clock::now();
        for (int i = 0; i < N; i++) {
            void* p = malloc(sz);
            free(p);
        }
        auto t5 = Clock::now();
        printf("size %6zu KB: Mat create %.2f us | +touch %.2f us | raw malloc %.2f us\n",
               sz / 1024, us(t0, t1, N), us(t2, t3, N), us(t4, t5, N));
    }

    printf("=== параллельный churn (parallel_for, внутри - Mat 1MB) ===\n");
    auto parTest = [&](int nthreads_mode, const char* name) {
        const int N = 200;
        auto t0 = Clock::now();
        parallel_for_(Range(0, N), [&](const Range& rg) {
            for (int i = rg.start; i < rg.end; i++) {
                Mat m(512, 2048, CV_8UC1);
                memset(m.data, 1, 1024); // лёгкое касание
            }
        });
        auto t1 = Clock::now();
        printf("%-28s %.2f us/оп\n", name, us(t0, t1, N));
    };
    parTest(0, "default threads");

    printf("=== RT-паттерн: blur в свежий Mat vs переиспользование ===\n");
    Mat img(1080, 1920, CV_8UC1, Scalar(77));
    const int N = 100;
    auto t0 = Clock::now();
    for (int i = 0; i < N; i++) {
        Mat tmp, out;
        GaussianBlur(img, tmp, {9, 9}, 2);   // tmp создаётся внутри
        tmp.convertTo(out, CV_8U);           // out создаётся
    }
    auto t1 = Clock::now();
    Mat tmp(1080, 1920, CV_8UC1), out(1080, 1920, CV_8UC1);
    auto t2 = Clock::now();
    for (int i = 0; i < N; i++) {
        GaussianBlur(img, tmp, {9, 9}, 2);
        tmp.convertTo(out, CV_8U);
    }
    auto t3 = Clock::now();
    printf("fresh buffers: %.3f ms/iter | reused: %.3f ms/iter (разница %.3f ms)\n",
           us(t0, t1, N) / 1000.0, us(t2, t3, N) / 1000.0,
           (us(t0, t1, N) - us(t2, t3, N)) / 1000.0);

    printf("=== autoBuffer/Mat header overhead: create без данных ===\n");
    {
        const int M = 100000;
        auto t4 = Clock::now();
        for (int i = 0; i < M; i++) { Mat m; }                       // пустой Mat
        auto t5 = Clock::now();
        printf("empty Mat: %.3f ns/оп\n", us(t4, t5, M) * 1000.0);
        auto t6 = Clock::now();
        for (int i = 0; i < M; i++) { Mat m(64, 64, CV_8UC1); }      // 4KB
        auto t7 = Clock::now();
        printf("Mat 4KB:   %.3f us/оп\n", us(t6, t7, M));
    }
    return 0;
}
