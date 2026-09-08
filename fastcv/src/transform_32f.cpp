// FastOpenCV: параллельный SIMD cv::transform для 32F (у OpenCV - однопоточный
// NAryMatIterator). Основной путь: scn=dcn=3 (BGR affine), общий путь: scn/dcn<=4.
// dst[x,c] = sum_j M[c][j]*src[x,j] + M[c][scn]
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

namespace fastcv {

using namespace cv;

void transform32f(const Mat& src, Mat& dst, const Mat& mtx) {
    CV_Assert(src.depth() == CV_32F);
    const int scn = src.channels(), dcn = mtx.rows;
    CV_Assert(mtx.cols == scn || mtx.cols == scn + 1);
    CV_Assert(scn >= 1 && scn <= 4 && dcn >= 1 && dcn <= 4);
    dst.create(src.size(), CV_MAKETYPE(CV_32F, dcn));

    float M[4][5] = {};
    Mat mf;
    mtx.convertTo(mf, CV_32F);
    for (int i = 0; i < dcn; i++)
        for (int j = 0; j < scn + 1; j++)
            M[i][j] = j < mtx.cols ? mf.at<float>(i, j) : 0.f;

    parallel_for_(Range(0, src.rows), [&](const Range& rg) {
        for (int y = rg.start; y < rg.end; y++) {
            const float* sp = src.ptr<float>(y);
            float* dp = dst.ptr<float>(y);
            const int W = src.cols;
            int x = 0;
#if CV_SIMD
            if (scn == 3 && dcn == 3) {
                const v_float32 m00 = vx_setall_f32(M[0][0]), m01 = vx_setall_f32(M[0][1]),
                                m02 = vx_setall_f32(M[0][2]), m03 = vx_setall_f32(M[0][3]);
                const v_float32 m10 = vx_setall_f32(M[1][0]), m11 = vx_setall_f32(M[1][1]),
                                m12 = vx_setall_f32(M[1][2]), m13 = vx_setall_f32(M[1][3]);
                const v_float32 m20 = vx_setall_f32(M[2][0]), m21 = vx_setall_f32(M[2][1]),
                                m22 = vx_setall_f32(M[2][2]), m23 = vx_setall_f32(M[2][3]);
                const int NF = v_float32::nlanes;
                for (; x <= W - NF; x += NF) {
                    v_float32 c0, c1, c2;
                    v_load_deinterleave(sp + x * 3, c0, c1, c2);
                    v_float32 o0 = v_muladd(m00, c0, v_muladd(m01, c1, v_muladd(m02, c2, m03)));
                    v_float32 o1 = v_muladd(m10, c0, v_muladd(m11, c1, v_muladd(m12, c2, m13)));
                    v_float32 o2 = v_muladd(m20, c0, v_muladd(m21, c1, v_muladd(m22, c2, m23)));
                    v_store_interleave(dp + x * 3, o0, o1, o2);
                }
            }
#endif
            for (; x < W; x++) {
                float in[4] = {0, 0, 0, 1.f};
                for (int j = 0; j < scn; j++) in[j] = sp[x * scn + j];
                for (int c = 0; c < dcn; c++) {
                    float s = M[c][scn];
                    for (int j = 0; j < scn; j++) s += M[c][j] * in[j];
                    dp[x * dcn + c] = s;
                }
            }
        }
    });
}

} // namespace fastcv
