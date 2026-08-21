#include "conv2d.h"

void conv2d_nchw(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int pad_top,
    int pad_left,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int c_in_per_group = c_in / groups;
    int c_out_per_group = c_out / groups;

    for (int batch = 0; batch < n; batch++) {
        for (int oc = 0; oc < c_out; oc++) {
            int group_id = oc / c_out_per_group;
            int ic_start = group_id * c_in_per_group;

            for (int oy = 0; oy < h_out; oy++) {
                for (int ox = 0; ox < w_out; ox++) {
                    float acc = bias ? bias[oc] : 0.0f;

                    for (int icg = 0; icg < c_in_per_group; icg++) {
                        int ic = ic_start + icg;

                        for (int ky = 0; ky < k_h; ky++) {
                            for (int kx = 0; kx < k_w; kx++) {
                                int iy = oy * stride_h + ky * dilation_h - pad_top;
                                int ix = ox * stride_w + kx * dilation_w - pad_left;

                                if (iy >= 0 && iy < h_in && ix >= 0 && ix < w_in) {
                                    int input_idx =
                                        ((batch * c_in + ic) * h_in + iy) * w_in + ix;

                                    int weight_idx =
                                        ((oc * c_in_per_group + icg) * k_h + ky) * k_w + kx;

                                    acc += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }

                    int out_idx =
                        ((batch * c_out + oc) * h_out + oy) * w_out + ox;

                    output[out_idx] = acc;
                }
            }
        }
    }
}
