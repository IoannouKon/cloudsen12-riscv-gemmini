#include "concat.h"
#include "first_concat_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int read_float_file(const char *path, float *data, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(data, sizeof(float), count, f);
    fclose(f);
    return n == count;
}

int main(void) {
#if FCAT_NUM_INPUTS != 2
    printf("This test currently supports exactly 2 concat inputs. Found %d\n", FCAT_NUM_INPUTS);
    return 1;
#endif
    const int channels[2] = {FCAT_C_IN_0, FCAT_C_IN_1};
    size_t count0 = (size_t)FCAT_N * FCAT_C_IN_0 * FCAT_H * FCAT_W;
    size_t count1 = (size_t)FCAT_N * FCAT_C_IN_1 * FCAT_H * FCAT_W;
    size_t out_count = (size_t)FCAT_N * FCAT_C_OUT * FCAT_H * FCAT_W;

    float *in0 = (float *)malloc(count0 * sizeof(float));
    float *in1 = (float *)malloc(count1 * sizeof(float));
    float *out = (float *)malloc(out_count * sizeof(float));
    float *ref = (float *)malloc(out_count * sizeof(float));
    if (!in0 || !in1 || !out || !ref) { printf("Allocation failed\n"); return 1; }

    if (!read_float_file("experiments/operator_tests/first_concat/concat_in_0.bin", in0, count0)) { printf("Failed input 0\n"); return 1; }
    if (!read_float_file("experiments/operator_tests/first_concat/concat_in_1.bin", in1, count1)) { printf("Failed input 1\n"); return 1; }
    if (!read_float_file("experiments/operator_tests/first_concat/concat_ref.bin", ref, out_count)) { printf("Failed ref\n"); return 1; }

    const float *inputs[2] = {in0, in1};

    printf("Running pure C first Concat...\n");
    concat_channel_nchw(inputs, channels, 2, out, FCAT_N, FCAT_H, FCAT_W);

    double max_abs = 0.0, mean_abs = 0.0;
    for (size_t i=0;i<out_count;i++) {
        double diff = fabs((double)out[i] - (double)ref[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (double)out_count;
    printf("Output count: %zu\n", out_count);
    printf("Mean abs error vs ONNX Runtime: %.10f\n", mean_abs);
    printf("Max abs error vs ONNX Runtime: %.10f\n", max_abs);

    free(in0); free(in1); free(out); free(ref);
    return 0;
}
