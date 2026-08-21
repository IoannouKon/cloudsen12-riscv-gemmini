#include "concat.h"

void concat_channel_nchw(
    const float **inputs,
    const int *channels,
    int num_inputs,
    float *output,
    int n,
    int h,
    int w
) {
    int total_c = 0;
    for (int i = 0; i < num_inputs; i++) total_c += channels[i];

    int out_c_offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        const float *inp = inputs[i];
        int c = channels[i];
        for (int batch = 0; batch < n; batch++) {
            for (int ch = 0; ch < c; ch++) {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int in_idx = ((batch * c + ch) * h + y) * w + x;
                        int out_idx = ((batch * total_c + (out_c_offset + ch)) * h + y) * w + x;
                        output[out_idx] = inp[in_idx];
                    }
                }
            }
        }
        out_c_offset += c;
    }
}
