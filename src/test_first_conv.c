#include "conv2d.h"
#include "first_conv_meta.h"

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
    size_t input_count = (size_t)FC_N * FC_C_IN * FC_H_IN * FC_W_IN;
    size_t weight_count = (size_t)FC_C_OUT * (FC_C_IN / FC_GROUP) * FC_KH * FC_KW;
    size_t bias_count = (size_t)FC_C_OUT;
    size_t output_count = (size_t)FC_N * FC_C_OUT * FC_H_OUT * FC_W_OUT;

    float *input = (float *)malloc(input_count * sizeof(float));
    float *weight = (float *)malloc(weight_count * sizeof(float));
    float *bias = (float *)malloc(bias_count * sizeof(float));
    float *out = (float *)malloc(output_count * sizeof(float));
    float *ref = (float *)malloc(output_count * sizeof(float));

    if (!input || !weight || !bias || !out || !ref) {
        printf("Allocation failed\n");
        return 1;
    }

    if (!read_float_file("inputs/input_1x13x512x512_float32.bin", input, input_count)) {
        printf("Failed to read input\n");
        return 1;
    }

    if (!read_float_file("experiments/operator_tests/first_conv/conv_w.bin", weight, weight_count)) {
        printf("Failed to read weights\n");
        return 1;
    }

    if (!read_float_file("experiments/operator_tests/first_conv/conv_b.bin", bias, bias_count)) {
        printf("Failed to read bias\n");
        return 1;
    }

    if (!read_float_file("experiments/operator_tests/first_conv/conv_ref.bin", ref, output_count)) {
        printf("Failed to read reference\n");
        return 1;
    }

    printf("Running pure C first Conv...\n");

    conv2d_nchw(
        input,
        weight,
        bias,
        out,
        FC_N,
        FC_C_IN,
        FC_H_IN,
        FC_W_IN,
        FC_C_OUT,
        FC_KH,
        FC_KW,
        FC_H_OUT,
        FC_W_OUT,
        FC_PAD_TOP,
        FC_PAD_LEFT,
        FC_STRIDE_H,
        FC_STRIDE_W,
        FC_DILATION_H,
        FC_DILATION_W,
        FC_GROUP
    );

    double max_abs = 0.0;
    double mean_abs = 0.0;

    for (size_t i = 0; i < output_count; i++) {
        double diff = fabs((double)out[i] - (double)ref[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }

    mean_abs /= (double)output_count;

    printf("Output count: %zu\n", output_count);
    printf("Mean abs error vs ONNX Runtime: %.10f\n", mean_abs);
    printf("Max abs error vs ONNX Runtime: %.10f\n", max_abs);

    free(input);
    free(weight);
    free(bias);
    free(out);
    free(ref);

    return 0;
}
