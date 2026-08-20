#include "clip.h"

void clip_float(const float *input, float *output, int count, float min_val, float max_val) {
    for (int i = 0; i < count; i++) {
        float v = input[i];

        if (v < min_val) {
            v = min_val;
        }

        if (v > max_val) {
            v = max_val;
        }

        output[i] = v;
    }
}
