#include "relu.h"

void relu_float(const float *input, float *output, int count) {
    for (int i = 0; i < count; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}
