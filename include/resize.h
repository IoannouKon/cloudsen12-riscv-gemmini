#ifndef RESIZE_H
#define RESIZE_H

void resize_nearest_nchw(
    const float *input,
    float *output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out
);

#endif
