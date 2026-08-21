#include "resize.h"

void resize_nearest_nchw(
    const float *input,
    float *output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out
) {
    for (int batch = 0; batch < n; batch++) {
        for (int ch = 0; ch < c; ch++) {
            for (int oy = 0; oy < h_out; oy++) {
                int iy = (oy * h_in) / h_out;
                if (iy >= h_in) iy = h_in - 1;

                for (int ox = 0; ox < w_out; ox++) {
                    int ix = (ox * w_in) / w_out;
                    if (ix >= w_in) ix = w_in - 1;

                    int in_idx = ((batch * c + ch) * h_in + iy) * w_in + ix;
                    int out_idx = ((batch * c + ch) * h_out + oy) * w_out + ox;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}
