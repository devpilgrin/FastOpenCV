// Референс/проверка DFT и DCT: дамп контрольных сумм и точечных значений
#include <opencv2/core.hpp>
#include <cstdio>
using namespace cv;
int main(int argc, char** argv) {
    RNG rng(7);
    for (int n : {256, 1024, 2048}) {
        Mat img(n, n, CV_32FC1);
        rng.fill(img, RNG::UNIFORM, 0, 1);
        Mat d;
        dft(img, d);
        // контрольная сводка: сумма, норма, несколько точек (r2c - CCS packed 32FC1)
        Scalar s = sum(d);
        double nrm = norm(d);
        printf("dft %4d r2c: sum=%.12g norm=%.12g d[5,14]=%.9g d[5,15]=%.9g d[100,100]=%.9g\n",
               n, s[0], nrm,
               d.at<float>(5, 14), d.at<float>(5, 15), d.at<float>(100, 100));
        Mat cplx;
        dft(img, cplx, DFT_COMPLEX_OUTPUT);
        dft(cplx, d, DFT_COMPLEX_INPUT);
        s = sum(d); nrm = norm(d);
        printf("dft %4d c2c: sum=(%.9g, %.9g) norm=%.9g d[5,7]=(%.9g, %.9g)\n",
               n, s[0], s[1], nrm, d.at<Vec2f>(5, 7)[0], d.at<Vec2f>(5, 7)[1]);
        Mat dc;
        dct(img, dc);
        Scalar sd = sum(dc);
        printf("dct %4d:     sum=%.9g norm=%.9g d[5,7]=%.9g d[100,200]=%.9g\n",
               n, sd[0], norm(dc), dc.at<float>(5, 7), dc.at<float>(100, 200));
        // обратное
        Mat inv;
        idct(dc, inv);
        printf("idct %4d err: %.6g\n", n, norm(inv, img, NORM_INF));
        Mat id;
        idft(d, id, DFT_SCALE | DFT_COMPLEX_INPUT | DFT_REAL_OUTPUT);
        printf("idft %4d err: %.6g\n", n, norm(id, img, NORM_INF));
    }
    return 0;
}
