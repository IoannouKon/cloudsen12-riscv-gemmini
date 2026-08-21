#ifndef ARGMAX_H
#define ARGMAX_H

#include <stdint.h>

void argmax_nchw_classes(
    const float *logits,
    uint8_t *pred,
    int n,
    int classes,
    int h,
    int w
);

#endif
