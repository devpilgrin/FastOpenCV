// FastOpenCV: undistort - у OpenCV stripe_size0 = 4096/cols (2 строки для 1920!),
// initUndistortRectifyMap + remap вызываются ~540 раз на кадр (оверхед доминирует: 8.5 мс).
// Здесь: карта один раз на весь кадр (или большими полосами) + один remap.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace fastcv {

using namespace cv;

// один раз на калибровку: вернуть карты для повторного использования
void buildUndistortMaps(const Mat& cameraMatrix, const Mat& distCoeffs,
                        const Mat& newCameraMatrix, Size imgSize,
                        Mat& map1, Mat& map2) {
    Mat A, Ar;
    cameraMatrix.convertTo(A, CV_64F);
    if (!newCameraMatrix.empty()) newCameraMatrix.convertTo(Ar, CV_64F);
    else A.copyTo(Ar);
    Mat d = distCoeffs.empty() ? Mat::zeros(5, 1, CV_64F) : Mat_<double>(distCoeffs);
    initUndistortRectifyMap(A, d, Mat(), Ar, imgSize, CV_16SC2, map1, map2);
}

// быстрый путь: карты снаружи (RT-использование)
void undistortPrecomputed(const Mat& src, Mat& dst, const Mat& map1, const Mat& map2) {
    remap(src, dst, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}

// drop-in замена cv::undistort: карты целиком за один вызов
void undistort8u(const Mat& src, Mat& dst, const Mat& cameraMatrix,
                 const Mat& distCoeffs, const Mat& newCameraMatrix = Mat()) {
    dst.create(src.size(), src.type());
    Mat map1, map2;
    buildUndistortMaps(cameraMatrix, distCoeffs, newCameraMatrix, src.size(), map1, map2);
    remap(src, dst, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}

} // namespace fastcv
