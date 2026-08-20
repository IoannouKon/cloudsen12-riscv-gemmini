#include "relu.h"
#include "first_relu_meta.h"

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
    size_t count = (size_t)FR_COUNT;

    float *input = (float *)malloc(count * sizeof(float));
    float *out = (float *)malloc(count * sizeof(float));
    float *ref = (float *)malloc(count * sizeof(float));

    if (!input || !out || !ref) {
        printf("Allocation failed\n");
        return 1;
    }

    if (!read_float_file("experiments/operator_tests/first_relu/relu_input.bin", input, count)) {
        printf("Failed to read Relu input\n");
        return 1;
    }

    if (!read_float_file("experiments/operator_tests/first_relu/relu_ref.bin", ref, count)) {
        printf("Failed to read Relu reference\n");
        return 1;
    }

    printf("Running pure C first Relu...\n");

    relu_float(input, out, (int)count);

    double max_abs = 0.0;
    double mean_abs = 0.0;

    for (size_t i = 0; i < count; i++) {
        double diff = fabs((double)out[i] - (double)ref[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        mean_abs += diff;
    }

    mean_abs /= (double)count;

    printf("Output count: %zu\n", count);
    printf("Mean abs error vs ONNX Runtime: %.10f\n", mean_abs);
    printf("Max abs error vs ONNX Runtime: %.10f\n", max_abs);

    free(input);
    free(out);
    free(ref);

    return 0;
}
