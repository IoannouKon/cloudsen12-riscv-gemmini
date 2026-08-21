#ifndef CONCAT_H
#define CONCAT_H

void concat_channel_nchw(
    const float **inputs,
    const int *channels,
    int num_inputs,
    float *output,
    int n,
    int h,
    int w
);

#endif
