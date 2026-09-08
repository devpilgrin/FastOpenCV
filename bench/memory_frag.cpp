// Фрагментация и возврат памяти ОС при долгом churn смешанных размеров
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace cv;
static double rssMB() {
    FILE* f = fopen("/proc/self/status", "r");
    char line[256]; double v = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "VmRSS: %lf", &v) == 1) break;
    fclose(f);
    return v / 1024.0;
}
int main() {
    printf("RSS старт: %.1f MB\n", rssMB());
    {
        std::vector<Mat> live;
        for (int i = 0; i < 500; i++) {
            live.emplace_back(1080, 1920, CV_8UC3);
            memset(live.back().data, 1, 64);
        }
        printf("RSS после 500x6MB: %.1f MB\n", rssMB());
    }
    printf("RSS после освобождения: %.1f MB\n", rssMB());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20000; i++) {
        int kind = i % 5;
        Mat m(kind == 0 ? 1080 : 64 + (i % 512), kind == 0 ? 1920 : 64 + (i % 256),
              kind == 0 ? CV_8UC3 : CV_8UC1);
        if ((i % 8) == 0) memset(m.data, 1, std::min<size_t>(m.total(), 4096));
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("churn 20000 mixed: %.1f ms, RSS: %.1f MB\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count(), rssMB());
    {
        std::vector<Mat> live;
        for (int i = 0; i < 500; i++) live.emplace_back(1080, 1920, CV_8UC3);
        printf("RSS повторный пик 500x6MB: %.1f MB\n", rssMB());
    }
    printf("RSS финал: %.1f MB\n", rssMB());
    return 0;
}
