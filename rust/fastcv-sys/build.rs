fn main() {
    let root = std::env::var("FASTOPENCV_ROOT")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../..").to_string());
    let ocv = format!("{}/install/linux-dispatch", root);
    println!("cargo:rustc-link-search=native={}/build/fastcv", root);
    println!("cargo:rustc-link-search=native={}/lib", ocv);
    println!("cargo:rustc-link-lib=static=fastcv");
    for lib in ["opencv_calib3d", "opencv_imgproc", "opencv_flann", "opencv_features2d", "opencv_core"] {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }
    for lib in ["openblas", "fftw3f", "fftw3f_threads", "tbb", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", ocv);
}
