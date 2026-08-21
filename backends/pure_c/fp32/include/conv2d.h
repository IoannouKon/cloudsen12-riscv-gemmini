#ifndef CONV2D_H
#define CONV2D_H

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
);

#endif
