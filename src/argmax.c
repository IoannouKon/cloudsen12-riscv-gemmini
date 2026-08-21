#include "argmax.h"

void argmax_nchw_classes(
    const float *logits,
    uint8_t *pred,
    int n,
    int classes,
    int h,
    int w
) {
    int hw = h * w;

    for (int batch = 0; batch < n; batch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pix = y * w + x;

                int best_class = 0;
                float best_value = logits[(batch * classes + 0) * hw + pix];

                for (int cls = 1; cls < classes; cls++) {
                    float value = logits[(batch * classes + cls) * hw + pix];

                    if (value > best_value) {
                        best_value = value;
                        best_class = cls;
                    }
                }

                pred[batch * hw + pix] = (uint8_t)best_class;
            }
        }
    }
}
