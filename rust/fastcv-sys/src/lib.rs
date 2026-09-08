//! FFI-биндинги к fastcv (FastOpenCV). Плоские буферы, без cv::Mat.
//! dst-буферы выделяются вызывающим.

extern "C" {
    fn fastcv_median_blur_8u(src: *const u8, dst: *mut u8, rows: i32, cols: i32, ksize: i32);
    fn fastcv_median_blur_8uc3(src: *const u8, dst: *mut u8, rows: i32, cols: i32, ksize: i32);
    fn fastcv_sobel_8u16s(src: *const u8, dst: *mut i16, rows: i32, cols: i32, channels: i32,
                          dx: i32, dy: i32, ksize: i32);
    fn fastcv_sep_filter_8u(src: *const u8, dst: *mut u8, rows: i32, cols: i32, channels: i32,
                            kx: *const f32, kw: i32, ky: *const f32, kh: i32);
    fn fastcv_calc_hist_8u(src: *const u8, rows: i32, cols: i32, hist256: *mut f32);
    fn fastcv_resize_area_int_8u(src: *const u8, dst: *mut u8, rows: i32, cols: i32,
                                 channels: i32, factor: i32);
    fn fastcv_sum_8u_c1(src: *const u8, rows: i32, cols: i32) -> f64;
    fn fastcv_dft_r2c_32f(src: *const f32, dst: *mut f32, rows: i32, cols: i32);
}

/// Медианный фильтр 8UC1 (k>5: 14-17x к cv::medianBlur, побитово точен)
pub fn median_blur_8u(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, ksize: i32) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), src.len());
    unsafe { fastcv_median_blur_8u(src.as_ptr(), dst.as_mut_ptr(), rows as i32, cols as i32, ksize) }
}

/// Медианный фильтр 8UC3 (k>5: ~16x, побитово)
pub fn median_blur_8uc3(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, ksize: i32) {
    assert_eq!(src.len(), rows * cols * 3);
    assert_eq!(dst.len(), src.len());
    unsafe { fastcv_median_blur_8uc3(src.as_ptr(), dst.as_mut_ptr(), rows as i32, cols as i32, ksize) }
}

/// Sobel больших ядер (k>=5; k=3 делегируется cv::Sobel внутри). dst: i16
pub fn sobel_8u16s(src: &[u8], dst: &mut [i16], rows: usize, cols: usize, channels: usize,
                   dx: i32, dy: i32, ksize: i32) {
    assert_eq!(src.len(), rows * cols * channels);
    assert_eq!(dst.len(), src.len());
    unsafe {
        fastcv_sobel_8u16s(src.as_ptr(), dst.as_mut_ptr(), rows as i32, cols as i32,
                           channels as i32, dx, dy, ksize)
    }
}

/// Сепарабельный фильтр float-ядрами (2-3x к cv::sepFilter2D)
pub fn sep_filter_8u(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, channels: usize,
                     kx: &[f32], ky: &[f32]) {
    assert_eq!(dst.len(), src.len());
    unsafe {
        fastcv_sep_filter_8u(src.as_ptr(), dst.as_mut_ptr(), rows as i32, cols as i32,
                             channels as i32, kx.as_ptr(), kx.len() as i32,
                             ky.as_ptr(), ky.len() as i32)
    }
}

/// Гистограмма 256 бин (5x к cv::calcHist)
pub fn calc_hist_8u(src: &[u8], rows: usize, cols: usize) -> [f32; 256] {
    assert_eq!(src.len(), rows * cols);
    let mut h = [0.0f32; 256];
    unsafe { fastcv_calc_hist_8u(src.as_ptr(), rows as i32, cols as i32, h.as_mut_ptr()) }
    h
}

/// resize INTER_AREA с целым фактором (x4: до 18.7x). dst размером (rows/f)*(cols/f)*ch
pub fn resize_area_int_8u(src: &[u8], dst: &mut [u8], rows: usize, cols: usize,
                          channels: usize, factor: i32) {
    let dr = rows / factor as usize;
    let dc = cols / factor as usize;
    assert_eq!(src.len(), rows * cols * channels);
    assert_eq!(dst.len(), dr * dc * channels);
    unsafe {
        fastcv_resize_area_int_8u(src.as_ptr(), dst.as_mut_ptr(), rows as i32, cols as i32,
                                  channels as i32, factor)
    }
}

/// Адаптивная сумма 8UC1 (cv::sum на малых, параллельная на >= 2M px)
pub fn sum_8u(src: &[u8], rows: usize, cols: usize) -> f64 {
    assert_eq!(src.len(), rows * cols);
    unsafe { fastcv_sum_8u_c1(src.as_ptr(), rows as i32, cols as i32) }
}

/// DFT r2c через FFTW. dst: rows*(cols/2+1)*2 float (interleaved re/im, layout FFTW)
pub fn dft_r2c_32f(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * (cols / 2 + 1) * 2);
    unsafe { fastcv_dft_r2c_32f(src.as_ptr(), dst.as_mut_ptr(), rows as i32, cols as i32) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_blur_smoke() {
        let rows = 64;
        let cols = 64;
        let mut src = vec![0u8; rows * cols];
        // вертикальная полоса: медиана 3x3 должна её сохранить по центру
        for y in 0..rows {
            for x in 30..34 {
                src[y * cols + x] = 200;
            }
        }
        let mut dst = vec![0u8; rows * cols];
        median_blur_8u(&src, &mut dst, rows, cols, 3);
        assert_eq!(dst[32 * cols + 31], 200);
        assert_eq!(dst[32 * cols + 10], 0);
    }

    #[test]
    fn hist_and_sum() {
        let rows = 100;
        let cols = 100;
        let src = vec![7u8; rows * cols];
        let h = calc_hist_8u(&src, rows, cols);
        assert_eq!(h[7], (rows * cols) as f32);
        assert_eq!(sum_8u(&src, rows, cols), (7 * rows * cols) as f64);
    }

    #[test]
    fn sobel_and_resize() {
        let rows = 64;
        let cols = 64;
        let src = vec![100u8; rows * cols];
        let mut dst = vec![0i16; rows * cols];
        sobel_8u16s(&src, &mut dst, rows, cols, 1, 1, 0, 5);
        assert!(dst.iter().all(|&v| v == 0)); // константное изображение -> нулевой градиент
        let mut small = vec![0u8; 16 * 16];
        resize_area_int_8u(&src, &mut small, rows, cols, 1, 4);
        assert!(small.iter().all(|&v| v == 100));
    }

    #[test]
    fn dft_dc() {
        // константное изображение: вся энергия в DC
        let n = 64;
        let src = vec![1.0f32; n * n];
        let mut dst = vec![0.0f32; n * (n / 2 + 1) * 2];
        dft_r2c_32f(&src, &mut dst, n, n);
        assert!((dst[0] - (n * n) as f32).abs() < 1.0); // DC = сумма
        assert!(dst[2].abs() < 0.5); // F(0,1) ~ 0
    }
}

