// DNN: OpenCV dnn CPU/CUDA-FP16 vs onnxruntime-cpu (MobileNetV2-12, 224x224)
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <chrono>
#include <cstdio>
using namespace cv;
using Clock = std::chrono::steady_clock;
template <typename F> double timeit(int iters, F&& f) {
    for (int i = 0; i < 3; i++) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}
int main() {
    const char* model = "../models/mobilenetv2-12.onnx";
    Mat blob = Mat(Size(224, 224), CV_32FC3, Scalar(0.5));
    blob = dnn::blobFromImage(blob, 1.0/255, Size(224,224), Scalar(), true);

    // OpenCV CPU
    {
        dnn::Net net = dnn::readNetFromONNX(model);
        net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        Mat out;
        printf("OpenCV dnn CPU:     %7.3f ms\n", timeit(30, [&]{ net.setInput(blob); out = net.forward(); }));
    }
    // OpenCV CUDA FP16
    {
        dnn::Net net = dnn::readNetFromONNX(model);
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
        Mat out;
        printf("OpenCV dnn CUDA16:  %7.3f ms\n", timeit(30, [&]{ net.setInput(blob); out = net.forward(); }));
    }
    // onnxruntime CPU
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
        Ort::SessionOptions so;
        Ort::Session session(env, model, so);
        std::vector<float> input(1 * 3 * 224 * 224);
        memcpy(input.data(), blob.ptr<float>(), input.size() * 4);
        std::array<int64_t, 4> shape{1, 3, 224, 224};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tin = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), 4);
        const char* inames[] = {"input"};  // MobilenetV2-12 input name: "input"
        Ort::AllocatorWithDefaultOptions alloc;
        auto iname = session.GetInputNameAllocated(0, alloc);
        const char* in_[] = {iname.get()};
        auto oname = session.GetOutputNameAllocated(0, alloc);
        const char* out_[] = {oname.get()};
        printf("ORT input: %s output: %s\n", iname.get(), oname.get());
        printf("onnxruntime CPU:    %7.3f ms\n", timeit(30, [&]{
            auto outs = session.Run(Ort::RunOptions{nullptr}, in_, &tin, 1, out_, 1);
        }));
    }
    return 0;
}
