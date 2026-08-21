#include "argmax.h"
#include "final_argmax_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int read_float_file(const char *path, float *data, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(data, sizeof(float), count, f);
    fclose(f);
    return n == count;
}

static int read_uint8_file(const char *path, uint8_t *data, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(data, sizeof(uint8_t), count, f);
    fclose(f);
    return n == count;
}

static int write_pgm(const char *path, const uint8_t *img, int width, int height) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        uint8_t v = (uint8_t)(img[i] * 80);
        fwrite(&v, sizeof(uint8_t), 1, f);
    }
    fclose(f);
    return 1;
}

int main(void) {
    size_t logits_count = (size_t)FARG_N * FARG_CLASSES * FARG_H * FARG_W;
    size_t pred_count = (size_t)FARG_N * FARG_H * FARG_W;

    float *logits = (float *)malloc(logits_count * sizeof(float));
    uint8_t *pred = (uint8_t *)malloc(pred_count * sizeof(uint8_t));
    uint8_t *ref = (uint8_t *)malloc(pred_count * sizeof(uint8_t));

    if (!logits || !pred || !ref) {
        printf("Allocation failed\n");
        return 1;
    }

    if (!read_float_file("experiments/operator_tests/final_argmax/final_logits_1x4x512x512_float32.bin", logits, logits_count)) {
        printf("Failed to read final logits\n");
        return 1;
    }

    if (!read_uint8_file("experiments/operator_tests/final_argmax/onnx_argmax_512x512_uint8.bin", ref, pred_count)) {
        printf("Failed to read ONNX argmax reference\n");
        return 1;
    }

    printf("Running pure C final Argmax...\n");
    argmax_nchw_classes(logits, pred, FARG_N, FARG_CLASSES, FARG_H, FARG_W);

    size_t mismatches = 0;
    size_t cloud_pixels = 0;

    for (size_t i = 0; i < pred_count; i++) {
        if (pred[i] != ref[i]) mismatches++;
        if (pred[i] == 1 || pred[i] == 2) cloud_pixels++;
    }

    double mismatch_rate = (double)mismatches / (double)pred_count;
    double cloud_fraction = (double)cloud_pixels / (double)pred_count;

    printf("Output pixels: %zu\n", pred_count);
    printf("Pure C argmax cloud fraction: %.10f\n", cloud_fraction);
    printf("Mismatch vs ONNX argmax: %.10f\n", mismatch_rate);

    write_pgm("outputs/predictions/pure_c_argmax_pred.pgm", pred, FARG_W, FARG_H);
    printf("Saved: outputs/predictions/pure_c_argmax_pred.pgm\n");

    free(logits);
    free(pred);
    free(ref);
    return 0;
}
