#include "add.h"

void add_float(const float *a, const float *b, float *out, int count) {
    for (int i = 0; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}
