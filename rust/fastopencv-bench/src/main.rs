use anyhow::Result;
use opencv::{core, imgproc, prelude::*};
use std::time::Instant;

fn bench<F: FnMut() -> opencv::Result<()>>(name: &str, iters: u32, mut f: F) -> Result<f64> {
    for _ in 0..3 {
        f()?; // прогрев
    }
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("{:<40} {:>8.3} ms/op", name, ms);
    Ok(ms)
}

fn main() -> Result<()> {
    println!("OpenCV: {}", core::CV_VERSION);
    println!("CPU: {}", core::get_cpu_features_line().unwrap_or_default());

    // Memory-bound: GaussianBlur 1080p 8UC3
    let img = core::Mat::new_rows_cols_with_default(
        1080, 1920, core::CV_8UC3, core::Scalar::from((128.0, 64.0, 32.0)),
    )?;
    let mut dst = core::Mat::default();
    bench("GaussianBlur 9x9 1080p (mem-bound)", 100, || {
        imgproc::gaussian_blur(&img, &mut dst, core::Size::new(9, 9), 2.0, 2.0,
                               core::BORDER_DEFAULT, core::AlgorithmHint::ALGO_HINT_DEFAULT)
    })?;

    // Compute-bound: DFT 2048x2048 float
    let fimg = core::Mat::new_rows_cols_with_default(
        2048, 2048, core::CV_32FC1, core::Scalar::from(1.5),
    )?;
    let mut fdst = core::Mat::default();
    bench("DFT 2048x2048 32F (compute-bound)", 20, || {
        core::dft_def(&fimg, &mut fdst)
    })?;

    // Compute-bound: GEMM 1024x1024
    let a = core::Mat::new_rows_cols_with_default(1024, 1024, core::CV_64FC1, core::Scalar::from(0.5))?;
    let b = core::Mat::new_rows_cols_with_default(1024, 1024, core::CV_64FC1, core::Scalar::from(0.25))?;
    let mut c = core::Mat::default();
    bench("GEMM 1024x1024 64F (compute-bound)", 20, || {
        core::gemm(&a, &b, 1.0, &core::Mat::default(), 0.0, &mut c, 0)
    })?;

    // Resize INTER_AREA 4K -> 720p
    let big = core::Mat::new_rows_cols_with_default(
        2160, 3840, core::CV_8UC3, core::Scalar::from((10.0, 20.0, 30.0)),
    )?;
    let mut small = core::Mat::default();
    bench("Resize 4K->720p INTER_AREA", 50, || {
        imgproc::resize(&big, &mut small, core::Size::new(1280, 720), 0.0, 0.0,
                        imgproc::INTER_AREA)
    })?;

    // cvtColor 4K BGR->GRAY
    let mut gray = core::Mat::default();
    bench("cvtColor 4K BGR2GRAY", 50, || {
        imgproc::cvt_color(&big, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)
    })?;

    // Sobel 1080p
    let mut sob = core::Mat::default();
    bench("Sobel 3x3 1080p 8U->16S", 100, || {
        imgproc::sobel(&img, &mut sob, core::CV_16S, 1, 0, 3, 1.0, 0.0, core::BORDER_DEFAULT)
    })?;

    // GPU: тот же GaussianBlur через CUDA (cudaarithm/cudafilters)
    opencv::opencv_has_module_cudafilters! {{
        use opencv::core::GpuMat;
        let n_cuda = opencv::core::get_cuda_enabled_device_count().unwrap_or(0);
        println!("CUDA devices: {}", n_cuda);
        if n_cuda > 0 {
            opencv::core::set_device(0)?;
            let mut g_src = GpuMat::new_def()?;
            g_src.upload(&img)?;
            let mut g_dst = GpuMat::new_def()?;
            let mut stream = core::Stream::default()?;
            let mut filter = opencv::cudafilters::create_gaussian_filter(
                core::CV_8UC3, core::CV_8UC3, core::Size::new(9, 9), 2.0, 2.0,
                core::BORDER_DEFAULT, core::BORDER_DEFAULT,
            )?;
            // прогрев
            for _ in 0..5 { filter.apply(&g_src, &mut g_dst, &mut stream)?; }
            stream.wait_for_completion()?;
            let t = Instant::now();
            let iters = 200;
            for _ in 0..iters {
                filter.apply(&g_src, &mut g_dst, &mut stream)?;
            }
            stream.wait_for_completion()?;
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("{:<40} {:>8.3} ms/op", "GaussianBlur 9x9 1080p (CUDA)", ms);
        }
    }}

    // DNN: MobileNetV2 инференс CPU vs CUDA
    let model_path = "/home/roman/workspace/FastOpemCV/models/mobilenetv2-12.onnx";
    if std::path::Path::new(model_path).exists() {
        let input_img = core::Mat::new_rows_cols_with_default(
            224, 224, core::CV_8UC3, core::Scalar::from((100.0, 120.0, 140.0)),
        )?;
        for (name, backend, target) in [
            ("CPU", opencv::dnn::DNN_BACKEND_OPENCV, opencv::dnn::DNN_TARGET_CPU),
            ("CUDA FP32", opencv::dnn::DNN_BACKEND_CUDA, opencv::dnn::DNN_TARGET_CUDA),
            ("CUDA FP16", opencv::dnn::DNN_BACKEND_CUDA, opencv::dnn::DNN_TARGET_CUDA_FP16),
        ] {
            let mut net = opencv::dnn::read_net_from_onnx(model_path)?;
            net.set_preferable_backend(backend)?;
            net.set_preferable_target(target)?;
            let blob = opencv::dnn::blob_from_image(
                &input_img, 1.0 / 255.0, core::Size::new(224, 224),
                core::Scalar::default(), true, false, core::CV_32F,
            )?;
            net.set_input_def(&blob)?;
            // прогрев (CUDA: компиляция cudnn-графа при первом прогоне)
            for _ in 0..3 { let _ = net.forward_single_def()?; }
            let iters = 50;
            let t = Instant::now();
            for _ in 0..iters { let _ = net.forward_single_def()?; }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("{:<40} {:>8.3} ms/op", format!("DNN MobileNetV2 224x224 ({})", name), ms);
        }
    }

    Ok(())
}
