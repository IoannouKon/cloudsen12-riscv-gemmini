/* Auto-generated standalone FP32 single-file build unit */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/* BEGIN INLINE conv2d.h */
#ifndef CONV2D_H
#define CONV2D_H

void conv2d_nchw(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int pad_top,
    int pad_left,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int groups
);

#endif
/* END INLINE conv2d.h */

/* BEGIN INLINE clip.h */
#ifndef CLIP_H
#define CLIP_H

void clip_float(const float *input, float *output, int count, float min_val, float max_val);

#endif
/* END INLINE clip.h */

/* BEGIN INLINE relu.h */
#ifndef RELU_H
#define RELU_H

void relu_float(const float *input, float *output, int count);

#endif
/* END INLINE relu.h */

/* BEGIN INLINE add.h */
#ifndef ADD_H
#define ADD_H

void add_float(const float *a, const float *b, float *out, int count);

#endif
/* END INLINE add.h */

/* BEGIN INLINE concat.h */
#ifndef CONCAT_H
#define CONCAT_H

void concat_channel_nchw(
    const float **inputs,
    const int *channels,
    int num_inputs,
    float *output,
    int n,
    int h,
    int w
);

#endif
/* END INLINE concat.h */

/* BEGIN INLINE resize.h */
#ifndef RESIZE_H
#define RESIZE_H

void resize_nearest_nchw(
    const float *input,
    float *output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out
);

#endif
/* END INLINE resize.h */

/* BEGIN INLINE argmax.h */
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
/* END INLINE argmax.h */

/* BEGIN INLINE partial_forward_v2_meta.h */
#ifndef PARTIAL_FORWARD_V2_META_H
#define PARTIAL_FORWARD_V2_META_H

#define PFV2_LAST_COUNT 262144
#endif
/* END INLINE partial_forward_v2_meta.h */

/* BEGIN INLINE conv2d.c */

void conv2d_nchw(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int pad_top,
    int pad_left,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int c_in_per_group = c_in / groups;
    int c_out_per_group = c_out / groups;

    for (int batch = 0; batch < n; batch++) {
        for (int oc = 0; oc < c_out; oc++) {
            int group_id = oc / c_out_per_group;
            int ic_start = group_id * c_in_per_group;

            for (int oy = 0; oy < h_out; oy++) {
                for (int ox = 0; ox < w_out; ox++) {
                    float acc = bias ? bias[oc] : 0.0f;

                    for (int icg = 0; icg < c_in_per_group; icg++) {
                        int ic = ic_start + icg;

                        for (int ky = 0; ky < k_h; ky++) {
                            for (int kx = 0; kx < k_w; kx++) {
                                int iy = oy * stride_h + ky * dilation_h - pad_top;
                                int ix = ox * stride_w + kx * dilation_w - pad_left;

                                if (iy >= 0 && iy < h_in && ix >= 0 && ix < w_in) {
                                    int input_idx =
                                        ((batch * c_in + ic) * h_in + iy) * w_in + ix;

                                    int weight_idx =
                                        ((oc * c_in_per_group + icg) * k_h + ky) * k_w + kx;

                                    acc += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }

                    int out_idx =
                        ((batch * c_out + oc) * h_out + oy) * w_out + ox;

                    output[out_idx] = acc;
                }
            }
        }
    }
}
/* END INLINE conv2d.c */

/* BEGIN INLINE clip.c */

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
/* END INLINE clip.c */

/* BEGIN INLINE relu.c */

void relu_float(const float *input, float *output, int count) {
    for (int i = 0; i < count; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}
/* END INLINE relu.c */

/* BEGIN INLINE add.c */

void add_float(const float *a, const float *b, float *out, int count) {
    for (int i = 0; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}
/* END INLINE add.c */

/* BEGIN INLINE concat.c */

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
/* END INLINE concat.c */

/* BEGIN INLINE resize.c */

void resize_nearest_nchw(
    const float *input,
    float *output,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out
) {
    for (int batch = 0; batch < n; batch++) {
        for (int ch = 0; ch < c; ch++) {
            for (int oy = 0; oy < h_out; oy++) {
                int iy = (oy * h_in) / h_out;
                if (iy >= h_in) iy = h_in - 1;

                for (int ox = 0; ox < w_out; ox++) {
                    int ix = (ox * w_in) / w_out;
                    if (ix >= w_in) ix = w_in - 1;

                    int in_idx = ((batch * c + ch) * h_in + iy) * w_in + ix;
                    int out_idx = ((batch * c + ch) * h_out + oy) * w_out + ox;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}
/* END INLINE resize.c */

/* BEGIN INLINE argmax.c */

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
/* END INLINE argmax.c */

/* BEGIN INLINE cloudsen12_full_c.c */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static int read_float_file(const char *path, float *data, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(data, sizeof(float), count, f);
    fclose(f);
    return n == count;
}

int main(void) {
    size_t input_count = (size_t)1 * 13 * 512 * 512;
    float *input = (float *)malloc(input_count * sizeof(float));
    float *ref = (float *)malloc((size_t)PFV2_LAST_COUNT * sizeof(float));
    if (!input || !ref) { printf("Allocation failed\n"); return 1; }
    if (!read_float_file("inputs/input_1x13x512x512_float32.bin", input, input_count)) { printf("Failed input\n"); return 1; }
    if (!read_float_file("experiments/partial_forward/partial_forward_v2/partial_ref.bin", ref, (size_t)PFV2_LAST_COUNT)) { printf("Failed partial ref\n"); return 1; }
    float *onnx__Conv_715 = (float *)malloc((size_t)(32 * 13 * 3 * 3) * sizeof(float));
    float *onnx__Conv_716 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_715.bin", onnx__Conv_715, (size_t)(32 * 13 * 3 * 3))) { printf("Failed weight onnx::Conv_715\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_716.bin", onnx__Conv_716, (size_t)(32))) { printf("Failed bias onnx::Conv_716\n"); return 1; }
    float *onnx__Conv_718 = (float *)malloc((size_t)(32 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_719 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_718.bin", onnx__Conv_718, (size_t)(32 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_718\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_719.bin", onnx__Conv_719, (size_t)(32))) { printf("Failed bias onnx::Conv_719\n"); return 1; }
    float *onnx__Conv_721 = (float *)malloc((size_t)(16 * 32 * 1 * 1) * sizeof(float));
    float *onnx__Conv_722 = (float *)malloc((size_t)(16) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_721.bin", onnx__Conv_721, (size_t)(16 * 32 * 1 * 1))) { printf("Failed weight onnx::Conv_721\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_722.bin", onnx__Conv_722, (size_t)(16))) { printf("Failed bias onnx::Conv_722\n"); return 1; }
    float *onnx__Conv_724 = (float *)malloc((size_t)(96 * 16 * 1 * 1) * sizeof(float));
    float *onnx__Conv_725 = (float *)malloc((size_t)(96) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_724.bin", onnx__Conv_724, (size_t)(96 * 16 * 1 * 1))) { printf("Failed weight onnx::Conv_724\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_725.bin", onnx__Conv_725, (size_t)(96))) { printf("Failed bias onnx::Conv_725\n"); return 1; }
    float *onnx__Conv_727 = (float *)malloc((size_t)(96 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_728 = (float *)malloc((size_t)(96) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_727.bin", onnx__Conv_727, (size_t)(96 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_727\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_728.bin", onnx__Conv_728, (size_t)(96))) { printf("Failed bias onnx::Conv_728\n"); return 1; }
    float *onnx__Conv_730 = (float *)malloc((size_t)(24 * 96 * 1 * 1) * sizeof(float));
    float *onnx__Conv_731 = (float *)malloc((size_t)(24) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_730.bin", onnx__Conv_730, (size_t)(24 * 96 * 1 * 1))) { printf("Failed weight onnx::Conv_730\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_731.bin", onnx__Conv_731, (size_t)(24))) { printf("Failed bias onnx::Conv_731\n"); return 1; }
    float *onnx__Conv_733 = (float *)malloc((size_t)(144 * 24 * 1 * 1) * sizeof(float));
    float *onnx__Conv_734 = (float *)malloc((size_t)(144) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_733.bin", onnx__Conv_733, (size_t)(144 * 24 * 1 * 1))) { printf("Failed weight onnx::Conv_733\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_734.bin", onnx__Conv_734, (size_t)(144))) { printf("Failed bias onnx::Conv_734\n"); return 1; }
    float *onnx__Conv_736 = (float *)malloc((size_t)(144 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_737 = (float *)malloc((size_t)(144) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_736.bin", onnx__Conv_736, (size_t)(144 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_736\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_737.bin", onnx__Conv_737, (size_t)(144))) { printf("Failed bias onnx::Conv_737\n"); return 1; }
    float *onnx__Conv_739 = (float *)malloc((size_t)(24 * 144 * 1 * 1) * sizeof(float));
    float *onnx__Conv_740 = (float *)malloc((size_t)(24) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_739.bin", onnx__Conv_739, (size_t)(24 * 144 * 1 * 1))) { printf("Failed weight onnx::Conv_739\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_740.bin", onnx__Conv_740, (size_t)(24))) { printf("Failed bias onnx::Conv_740\n"); return 1; }
    float *onnx__Conv_742 = (float *)malloc((size_t)(144 * 24 * 1 * 1) * sizeof(float));
    float *onnx__Conv_743 = (float *)malloc((size_t)(144) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_742.bin", onnx__Conv_742, (size_t)(144 * 24 * 1 * 1))) { printf("Failed weight onnx::Conv_742\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_743.bin", onnx__Conv_743, (size_t)(144))) { printf("Failed bias onnx::Conv_743\n"); return 1; }
    float *onnx__Conv_745 = (float *)malloc((size_t)(144 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_746 = (float *)malloc((size_t)(144) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_745.bin", onnx__Conv_745, (size_t)(144 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_745\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_746.bin", onnx__Conv_746, (size_t)(144))) { printf("Failed bias onnx::Conv_746\n"); return 1; }
    float *onnx__Conv_748 = (float *)malloc((size_t)(32 * 144 * 1 * 1) * sizeof(float));
    float *onnx__Conv_749 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_748.bin", onnx__Conv_748, (size_t)(32 * 144 * 1 * 1))) { printf("Failed weight onnx::Conv_748\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_749.bin", onnx__Conv_749, (size_t)(32))) { printf("Failed bias onnx::Conv_749\n"); return 1; }
    float *onnx__Conv_751 = (float *)malloc((size_t)(192 * 32 * 1 * 1) * sizeof(float));
    float *onnx__Conv_752 = (float *)malloc((size_t)(192) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_751.bin", onnx__Conv_751, (size_t)(192 * 32 * 1 * 1))) { printf("Failed weight onnx::Conv_751\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_752.bin", onnx__Conv_752, (size_t)(192))) { printf("Failed bias onnx::Conv_752\n"); return 1; }
    float *onnx__Conv_754 = (float *)malloc((size_t)(192 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_755 = (float *)malloc((size_t)(192) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_754.bin", onnx__Conv_754, (size_t)(192 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_754\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_755.bin", onnx__Conv_755, (size_t)(192))) { printf("Failed bias onnx::Conv_755\n"); return 1; }
    float *onnx__Conv_757 = (float *)malloc((size_t)(32 * 192 * 1 * 1) * sizeof(float));
    float *onnx__Conv_758 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_757.bin", onnx__Conv_757, (size_t)(32 * 192 * 1 * 1))) { printf("Failed weight onnx::Conv_757\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_758.bin", onnx__Conv_758, (size_t)(32))) { printf("Failed bias onnx::Conv_758\n"); return 1; }
    float *onnx__Conv_760 = (float *)malloc((size_t)(192 * 32 * 1 * 1) * sizeof(float));
    float *onnx__Conv_761 = (float *)malloc((size_t)(192) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_760.bin", onnx__Conv_760, (size_t)(192 * 32 * 1 * 1))) { printf("Failed weight onnx::Conv_760\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_761.bin", onnx__Conv_761, (size_t)(192))) { printf("Failed bias onnx::Conv_761\n"); return 1; }
    float *onnx__Conv_763 = (float *)malloc((size_t)(192 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_764 = (float *)malloc((size_t)(192) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_763.bin", onnx__Conv_763, (size_t)(192 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_763\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_764.bin", onnx__Conv_764, (size_t)(192))) { printf("Failed bias onnx::Conv_764\n"); return 1; }
    float *onnx__Conv_766 = (float *)malloc((size_t)(32 * 192 * 1 * 1) * sizeof(float));
    float *onnx__Conv_767 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_766.bin", onnx__Conv_766, (size_t)(32 * 192 * 1 * 1))) { printf("Failed weight onnx::Conv_766\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_767.bin", onnx__Conv_767, (size_t)(32))) { printf("Failed bias onnx::Conv_767\n"); return 1; }
    float *onnx__Conv_769 = (float *)malloc((size_t)(192 * 32 * 1 * 1) * sizeof(float));
    float *onnx__Conv_770 = (float *)malloc((size_t)(192) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_769.bin", onnx__Conv_769, (size_t)(192 * 32 * 1 * 1))) { printf("Failed weight onnx::Conv_769\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_770.bin", onnx__Conv_770, (size_t)(192))) { printf("Failed bias onnx::Conv_770\n"); return 1; }
    float *onnx__Conv_772 = (float *)malloc((size_t)(192 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_773 = (float *)malloc((size_t)(192) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_772.bin", onnx__Conv_772, (size_t)(192 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_772\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_773.bin", onnx__Conv_773, (size_t)(192))) { printf("Failed bias onnx::Conv_773\n"); return 1; }
    float *onnx__Conv_775 = (float *)malloc((size_t)(64 * 192 * 1 * 1) * sizeof(float));
    float *onnx__Conv_776 = (float *)malloc((size_t)(64) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_775.bin", onnx__Conv_775, (size_t)(64 * 192 * 1 * 1))) { printf("Failed weight onnx::Conv_775\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_776.bin", onnx__Conv_776, (size_t)(64))) { printf("Failed bias onnx::Conv_776\n"); return 1; }
    float *onnx__Conv_778 = (float *)malloc((size_t)(384 * 64 * 1 * 1) * sizeof(float));
    float *onnx__Conv_779 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_778.bin", onnx__Conv_778, (size_t)(384 * 64 * 1 * 1))) { printf("Failed weight onnx::Conv_778\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_779.bin", onnx__Conv_779, (size_t)(384))) { printf("Failed bias onnx::Conv_779\n"); return 1; }
    float *onnx__Conv_781 = (float *)malloc((size_t)(384 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_782 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_781.bin", onnx__Conv_781, (size_t)(384 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_781\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_782.bin", onnx__Conv_782, (size_t)(384))) { printf("Failed bias onnx::Conv_782\n"); return 1; }
    float *onnx__Conv_784 = (float *)malloc((size_t)(64 * 384 * 1 * 1) * sizeof(float));
    float *onnx__Conv_785 = (float *)malloc((size_t)(64) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_784.bin", onnx__Conv_784, (size_t)(64 * 384 * 1 * 1))) { printf("Failed weight onnx::Conv_784\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_785.bin", onnx__Conv_785, (size_t)(64))) { printf("Failed bias onnx::Conv_785\n"); return 1; }
    float *onnx__Conv_787 = (float *)malloc((size_t)(384 * 64 * 1 * 1) * sizeof(float));
    float *onnx__Conv_788 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_787.bin", onnx__Conv_787, (size_t)(384 * 64 * 1 * 1))) { printf("Failed weight onnx::Conv_787\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_788.bin", onnx__Conv_788, (size_t)(384))) { printf("Failed bias onnx::Conv_788\n"); return 1; }
    float *onnx__Conv_790 = (float *)malloc((size_t)(384 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_791 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_790.bin", onnx__Conv_790, (size_t)(384 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_790\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_791.bin", onnx__Conv_791, (size_t)(384))) { printf("Failed bias onnx::Conv_791\n"); return 1; }
    float *onnx__Conv_793 = (float *)malloc((size_t)(64 * 384 * 1 * 1) * sizeof(float));
    float *onnx__Conv_794 = (float *)malloc((size_t)(64) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_793.bin", onnx__Conv_793, (size_t)(64 * 384 * 1 * 1))) { printf("Failed weight onnx::Conv_793\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_794.bin", onnx__Conv_794, (size_t)(64))) { printf("Failed bias onnx::Conv_794\n"); return 1; }
    float *onnx__Conv_796 = (float *)malloc((size_t)(384 * 64 * 1 * 1) * sizeof(float));
    float *onnx__Conv_797 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_796.bin", onnx__Conv_796, (size_t)(384 * 64 * 1 * 1))) { printf("Failed weight onnx::Conv_796\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_797.bin", onnx__Conv_797, (size_t)(384))) { printf("Failed bias onnx::Conv_797\n"); return 1; }
    float *onnx__Conv_799 = (float *)malloc((size_t)(384 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_800 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_799.bin", onnx__Conv_799, (size_t)(384 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_799\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_800.bin", onnx__Conv_800, (size_t)(384))) { printf("Failed bias onnx::Conv_800\n"); return 1; }
    float *onnx__Conv_802 = (float *)malloc((size_t)(64 * 384 * 1 * 1) * sizeof(float));
    float *onnx__Conv_803 = (float *)malloc((size_t)(64) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_802.bin", onnx__Conv_802, (size_t)(64 * 384 * 1 * 1))) { printf("Failed weight onnx::Conv_802\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_803.bin", onnx__Conv_803, (size_t)(64))) { printf("Failed bias onnx::Conv_803\n"); return 1; }
    float *onnx__Conv_805 = (float *)malloc((size_t)(384 * 64 * 1 * 1) * sizeof(float));
    float *onnx__Conv_806 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_805.bin", onnx__Conv_805, (size_t)(384 * 64 * 1 * 1))) { printf("Failed weight onnx::Conv_805\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_806.bin", onnx__Conv_806, (size_t)(384))) { printf("Failed bias onnx::Conv_806\n"); return 1; }
    float *onnx__Conv_808 = (float *)malloc((size_t)(384 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_809 = (float *)malloc((size_t)(384) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_808.bin", onnx__Conv_808, (size_t)(384 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_808\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_809.bin", onnx__Conv_809, (size_t)(384))) { printf("Failed bias onnx::Conv_809\n"); return 1; }
    float *onnx__Conv_811 = (float *)malloc((size_t)(96 * 384 * 1 * 1) * sizeof(float));
    float *onnx__Conv_812 = (float *)malloc((size_t)(96) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_811.bin", onnx__Conv_811, (size_t)(96 * 384 * 1 * 1))) { printf("Failed weight onnx::Conv_811\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_812.bin", onnx__Conv_812, (size_t)(96))) { printf("Failed bias onnx::Conv_812\n"); return 1; }
    float *onnx__Conv_814 = (float *)malloc((size_t)(576 * 96 * 1 * 1) * sizeof(float));
    float *onnx__Conv_815 = (float *)malloc((size_t)(576) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_814.bin", onnx__Conv_814, (size_t)(576 * 96 * 1 * 1))) { printf("Failed weight onnx::Conv_814\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_815.bin", onnx__Conv_815, (size_t)(576))) { printf("Failed bias onnx::Conv_815\n"); return 1; }
    float *onnx__Conv_817 = (float *)malloc((size_t)(576 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_818 = (float *)malloc((size_t)(576) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_817.bin", onnx__Conv_817, (size_t)(576 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_817\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_818.bin", onnx__Conv_818, (size_t)(576))) { printf("Failed bias onnx::Conv_818\n"); return 1; }
    float *onnx__Conv_820 = (float *)malloc((size_t)(96 * 576 * 1 * 1) * sizeof(float));
    float *onnx__Conv_821 = (float *)malloc((size_t)(96) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_820.bin", onnx__Conv_820, (size_t)(96 * 576 * 1 * 1))) { printf("Failed weight onnx::Conv_820\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_821.bin", onnx__Conv_821, (size_t)(96))) { printf("Failed bias onnx::Conv_821\n"); return 1; }
    float *onnx__Conv_823 = (float *)malloc((size_t)(576 * 96 * 1 * 1) * sizeof(float));
    float *onnx__Conv_824 = (float *)malloc((size_t)(576) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_823.bin", onnx__Conv_823, (size_t)(576 * 96 * 1 * 1))) { printf("Failed weight onnx::Conv_823\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_824.bin", onnx__Conv_824, (size_t)(576))) { printf("Failed bias onnx::Conv_824\n"); return 1; }
    float *onnx__Conv_826 = (float *)malloc((size_t)(576 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_827 = (float *)malloc((size_t)(576) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_826.bin", onnx__Conv_826, (size_t)(576 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_826\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_827.bin", onnx__Conv_827, (size_t)(576))) { printf("Failed bias onnx::Conv_827\n"); return 1; }
    float *onnx__Conv_829 = (float *)malloc((size_t)(96 * 576 * 1 * 1) * sizeof(float));
    float *onnx__Conv_830 = (float *)malloc((size_t)(96) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_829.bin", onnx__Conv_829, (size_t)(96 * 576 * 1 * 1))) { printf("Failed weight onnx::Conv_829\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_830.bin", onnx__Conv_830, (size_t)(96))) { printf("Failed bias onnx::Conv_830\n"); return 1; }
    float *onnx__Conv_832 = (float *)malloc((size_t)(576 * 96 * 1 * 1) * sizeof(float));
    float *onnx__Conv_833 = (float *)malloc((size_t)(576) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_832.bin", onnx__Conv_832, (size_t)(576 * 96 * 1 * 1))) { printf("Failed weight onnx::Conv_832\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_833.bin", onnx__Conv_833, (size_t)(576))) { printf("Failed bias onnx::Conv_833\n"); return 1; }
    float *onnx__Conv_835 = (float *)malloc((size_t)(576 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_836 = (float *)malloc((size_t)(576) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_835.bin", onnx__Conv_835, (size_t)(576 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_835\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_836.bin", onnx__Conv_836, (size_t)(576))) { printf("Failed bias onnx::Conv_836\n"); return 1; }
    float *onnx__Conv_838 = (float *)malloc((size_t)(160 * 576 * 1 * 1) * sizeof(float));
    float *onnx__Conv_839 = (float *)malloc((size_t)(160) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_838.bin", onnx__Conv_838, (size_t)(160 * 576 * 1 * 1))) { printf("Failed weight onnx::Conv_838\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_839.bin", onnx__Conv_839, (size_t)(160))) { printf("Failed bias onnx::Conv_839\n"); return 1; }
    float *onnx__Conv_841 = (float *)malloc((size_t)(960 * 160 * 1 * 1) * sizeof(float));
    float *onnx__Conv_842 = (float *)malloc((size_t)(960) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_841.bin", onnx__Conv_841, (size_t)(960 * 160 * 1 * 1))) { printf("Failed weight onnx::Conv_841\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_842.bin", onnx__Conv_842, (size_t)(960))) { printf("Failed bias onnx::Conv_842\n"); return 1; }
    float *onnx__Conv_844 = (float *)malloc((size_t)(960 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_845 = (float *)malloc((size_t)(960) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_844.bin", onnx__Conv_844, (size_t)(960 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_844\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_845.bin", onnx__Conv_845, (size_t)(960))) { printf("Failed bias onnx::Conv_845\n"); return 1; }
    float *onnx__Conv_847 = (float *)malloc((size_t)(160 * 960 * 1 * 1) * sizeof(float));
    float *onnx__Conv_848 = (float *)malloc((size_t)(160) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_847.bin", onnx__Conv_847, (size_t)(160 * 960 * 1 * 1))) { printf("Failed weight onnx::Conv_847\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_848.bin", onnx__Conv_848, (size_t)(160))) { printf("Failed bias onnx::Conv_848\n"); return 1; }
    float *onnx__Conv_850 = (float *)malloc((size_t)(960 * 160 * 1 * 1) * sizeof(float));
    float *onnx__Conv_851 = (float *)malloc((size_t)(960) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_850.bin", onnx__Conv_850, (size_t)(960 * 160 * 1 * 1))) { printf("Failed weight onnx::Conv_850\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_851.bin", onnx__Conv_851, (size_t)(960))) { printf("Failed bias onnx::Conv_851\n"); return 1; }
    float *onnx__Conv_853 = (float *)malloc((size_t)(960 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_854 = (float *)malloc((size_t)(960) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_853.bin", onnx__Conv_853, (size_t)(960 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_853\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_854.bin", onnx__Conv_854, (size_t)(960))) { printf("Failed bias onnx::Conv_854\n"); return 1; }
    float *onnx__Conv_856 = (float *)malloc((size_t)(160 * 960 * 1 * 1) * sizeof(float));
    float *onnx__Conv_857 = (float *)malloc((size_t)(160) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_856.bin", onnx__Conv_856, (size_t)(160 * 960 * 1 * 1))) { printf("Failed weight onnx::Conv_856\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_857.bin", onnx__Conv_857, (size_t)(160))) { printf("Failed bias onnx::Conv_857\n"); return 1; }
    float *onnx__Conv_859 = (float *)malloc((size_t)(960 * 160 * 1 * 1) * sizeof(float));
    float *onnx__Conv_860 = (float *)malloc((size_t)(960) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_859.bin", onnx__Conv_859, (size_t)(960 * 160 * 1 * 1))) { printf("Failed weight onnx::Conv_859\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_860.bin", onnx__Conv_860, (size_t)(960))) { printf("Failed bias onnx::Conv_860\n"); return 1; }
    float *onnx__Conv_862 = (float *)malloc((size_t)(960 * 1 * 3 * 3) * sizeof(float));
    float *onnx__Conv_863 = (float *)malloc((size_t)(960) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_862.bin", onnx__Conv_862, (size_t)(960 * 1 * 3 * 3))) { printf("Failed weight onnx::Conv_862\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_863.bin", onnx__Conv_863, (size_t)(960))) { printf("Failed bias onnx::Conv_863\n"); return 1; }
    float *onnx__Conv_865 = (float *)malloc((size_t)(320 * 960 * 1 * 1) * sizeof(float));
    float *onnx__Conv_866 = (float *)malloc((size_t)(320) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_865.bin", onnx__Conv_865, (size_t)(320 * 960 * 1 * 1))) { printf("Failed weight onnx::Conv_865\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_866.bin", onnx__Conv_866, (size_t)(320))) { printf("Failed bias onnx::Conv_866\n"); return 1; }
    float *onnx__Conv_868 = (float *)malloc((size_t)(1280 * 320 * 1 * 1) * sizeof(float));
    float *onnx__Conv_869 = (float *)malloc((size_t)(1280) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_868.bin", onnx__Conv_868, (size_t)(1280 * 320 * 1 * 1))) { printf("Failed weight onnx::Conv_868\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_869.bin", onnx__Conv_869, (size_t)(1280))) { printf("Failed bias onnx::Conv_869\n"); return 1; }
    float *onnx__Conv_871 = (float *)malloc((size_t)(256 * 1376 * 3 * 3) * sizeof(float));
    float *onnx__Conv_872 = (float *)malloc((size_t)(256) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_871.bin", onnx__Conv_871, (size_t)(256 * 1376 * 3 * 3))) { printf("Failed weight onnx::Conv_871\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_872.bin", onnx__Conv_872, (size_t)(256))) { printf("Failed bias onnx::Conv_872\n"); return 1; }
    float *onnx__Conv_874 = (float *)malloc((size_t)(256 * 256 * 3 * 3) * sizeof(float));
    float *onnx__Conv_875 = (float *)malloc((size_t)(256) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_874.bin", onnx__Conv_874, (size_t)(256 * 256 * 3 * 3))) { printf("Failed weight onnx::Conv_874\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_875.bin", onnx__Conv_875, (size_t)(256))) { printf("Failed bias onnx::Conv_875\n"); return 1; }
    float *onnx__Conv_877 = (float *)malloc((size_t)(128 * 288 * 3 * 3) * sizeof(float));
    float *onnx__Conv_878 = (float *)malloc((size_t)(128) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_877.bin", onnx__Conv_877, (size_t)(128 * 288 * 3 * 3))) { printf("Failed weight onnx::Conv_877\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_878.bin", onnx__Conv_878, (size_t)(128))) { printf("Failed bias onnx::Conv_878\n"); return 1; }
    float *onnx__Conv_880 = (float *)malloc((size_t)(128 * 128 * 3 * 3) * sizeof(float));
    float *onnx__Conv_881 = (float *)malloc((size_t)(128) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_880.bin", onnx__Conv_880, (size_t)(128 * 128 * 3 * 3))) { printf("Failed weight onnx::Conv_880\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_881.bin", onnx__Conv_881, (size_t)(128))) { printf("Failed bias onnx::Conv_881\n"); return 1; }
    float *onnx__Conv_883 = (float *)malloc((size_t)(64 * 152 * 3 * 3) * sizeof(float));
    float *onnx__Conv_884 = (float *)malloc((size_t)(64) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_883.bin", onnx__Conv_883, (size_t)(64 * 152 * 3 * 3))) { printf("Failed weight onnx::Conv_883\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_884.bin", onnx__Conv_884, (size_t)(64))) { printf("Failed bias onnx::Conv_884\n"); return 1; }
    float *onnx__Conv_886 = (float *)malloc((size_t)(64 * 64 * 3 * 3) * sizeof(float));
    float *onnx__Conv_887 = (float *)malloc((size_t)(64) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_886.bin", onnx__Conv_886, (size_t)(64 * 64 * 3 * 3))) { printf("Failed weight onnx::Conv_886\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_887.bin", onnx__Conv_887, (size_t)(64))) { printf("Failed bias onnx::Conv_887\n"); return 1; }
    float *onnx__Conv_889 = (float *)malloc((size_t)(32 * 80 * 3 * 3) * sizeof(float));
    float *onnx__Conv_890 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_889.bin", onnx__Conv_889, (size_t)(32 * 80 * 3 * 3))) { printf("Failed weight onnx::Conv_889\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_890.bin", onnx__Conv_890, (size_t)(32))) { printf("Failed bias onnx::Conv_890\n"); return 1; }
    float *onnx__Conv_892 = (float *)malloc((size_t)(32 * 32 * 3 * 3) * sizeof(float));
    float *onnx__Conv_893 = (float *)malloc((size_t)(32) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_892.bin", onnx__Conv_892, (size_t)(32 * 32 * 3 * 3))) { printf("Failed weight onnx::Conv_892\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_893.bin", onnx__Conv_893, (size_t)(32))) { printf("Failed bias onnx::Conv_893\n"); return 1; }
    float *onnx__Conv_895 = (float *)malloc((size_t)(16 * 32 * 3 * 3) * sizeof(float));
    float *onnx__Conv_896 = (float *)malloc((size_t)(16) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_895.bin", onnx__Conv_895, (size_t)(16 * 32 * 3 * 3))) { printf("Failed weight onnx::Conv_895\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_896.bin", onnx__Conv_896, (size_t)(16))) { printf("Failed bias onnx::Conv_896\n"); return 1; }
    float *onnx__Conv_898 = (float *)malloc((size_t)(16 * 16 * 3 * 3) * sizeof(float));
    float *onnx__Conv_899 = (float *)malloc((size_t)(16) * sizeof(float));
    if (!read_float_file("weights/onnx__Conv_898.bin", onnx__Conv_898, (size_t)(16 * 16 * 3 * 3))) { printf("Failed weight onnx::Conv_898\n"); return 1; }
    if (!read_float_file("weights/onnx__Conv_899.bin", onnx__Conv_899, (size_t)(16))) { printf("Failed bias onnx::Conv_899\n"); return 1; }
    float *segmentation_head_0_weight = (float *)malloc((size_t)(4 * 16 * 3 * 3) * sizeof(float));
    float *segmentation_head_0_bias = (float *)malloc((size_t)(4) * sizeof(float));
    if (!read_float_file("weights/segmentation_head_0_weight.bin", segmentation_head_0_weight, (size_t)(4 * 16 * 3 * 3))) { printf("Failed weight segmentation_head.0.weight\n"); return 1; }
    if (!read_float_file("weights/segmentation_head_0_bias.bin", segmentation_head_0_bias, (size_t)(4))) { printf("Failed bias segmentation_head.0.bias\n"); return 1; }
    float *encoder_features_0_features_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *encoder_features_0_features_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *encoder_features_1_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *encoder_features_1_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *encoder_features_1_conv_conv_1_Conv_output_0 = (float *)malloc((size_t)(1 * 16 * 256 * 256) * sizeof(float));
    float *encoder_features_2_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 96 * 256 * 256) * sizeof(float));
    float *encoder_features_2_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 96 * 256 * 256) * sizeof(float));
    float *encoder_features_2_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 96 * 128 * 128) * sizeof(float));
    float *encoder_features_2_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 96 * 128 * 128) * sizeof(float));
    float *encoder_features_2_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 24 * 128 * 128) * sizeof(float));
    float *encoder_features_3_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 144 * 128 * 128) * sizeof(float));
    float *encoder_features_3_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 144 * 128 * 128) * sizeof(float));
    float *encoder_features_3_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 144 * 128 * 128) * sizeof(float));
    float *encoder_features_3_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 144 * 128 * 128) * sizeof(float));
    float *encoder_features_3_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 24 * 128 * 128) * sizeof(float));
    float *encoder_features_3_Add_output_0 = (float *)malloc((size_t)(1 * 24 * 128 * 128) * sizeof(float));
    float *encoder_features_4_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 144 * 128 * 128) * sizeof(float));
    float *encoder_features_4_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 144 * 128 * 128) * sizeof(float));
    float *encoder_features_4_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 144 * 64 * 64) * sizeof(float));
    float *encoder_features_4_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 144 * 64 * 64) * sizeof(float));
    float *encoder_features_4_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 64 * 64) * sizeof(float));
    float *encoder_features_5_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_5_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_5_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_5_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_5_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 64 * 64) * sizeof(float));
    float *encoder_features_5_Add_output_0 = (float *)malloc((size_t)(1 * 32 * 64 * 64) * sizeof(float));
    float *encoder_features_6_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_6_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_6_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_6_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_6_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 64 * 64) * sizeof(float));
    float *encoder_features_6_Add_output_0 = (float *)malloc((size_t)(1 * 32 * 64 * 64) * sizeof(float));
    float *encoder_features_7_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_7_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 192 * 64 * 64) * sizeof(float));
    float *encoder_features_7_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 192 * 32 * 32) * sizeof(float));
    float *encoder_features_7_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 192 * 32 * 32) * sizeof(float));
    float *encoder_features_7_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_8_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_8_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_8_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_8_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_8_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_8_Add_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_9_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_9_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_9_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_9_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_9_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_9_Add_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_10_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_10_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_10_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_10_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_10_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_10_Add_output_0 = (float *)malloc((size_t)(1 * 64 * 32 * 32) * sizeof(float));
    float *encoder_features_11_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_11_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_11_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_11_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 384 * 32 * 32) * sizeof(float));
    float *encoder_features_11_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 96 * 32 * 32) * sizeof(float));
    float *encoder_features_12_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_12_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_12_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_12_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_12_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 96 * 32 * 32) * sizeof(float));
    float *encoder_features_12_Add_output_0 = (float *)malloc((size_t)(1 * 96 * 32 * 32) * sizeof(float));
    float *encoder_features_13_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_13_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_13_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_13_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_13_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 96 * 32 * 32) * sizeof(float));
    float *encoder_features_13_Add_output_0 = (float *)malloc((size_t)(1 * 96 * 32 * 32) * sizeof(float));
    float *encoder_features_14_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_14_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 576 * 32 * 32) * sizeof(float));
    float *encoder_features_14_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 576 * 16 * 16) * sizeof(float));
    float *encoder_features_14_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 576 * 16 * 16) * sizeof(float));
    float *encoder_features_14_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 160 * 16 * 16) * sizeof(float));
    float *encoder_features_15_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_15_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_15_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_15_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_15_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 160 * 16 * 16) * sizeof(float));
    float *encoder_features_15_Add_output_0 = (float *)malloc((size_t)(1 * 160 * 16 * 16) * sizeof(float));
    float *encoder_features_16_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_16_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_16_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_16_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_16_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 160 * 16 * 16) * sizeof(float));
    float *encoder_features_16_Add_output_0 = (float *)malloc((size_t)(1 * 160 * 16 * 16) * sizeof(float));
    float *encoder_features_17_conv_conv_0_conv_0_0_Conv_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_17_conv_conv_0_conv_0_2_Clip_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_17_conv_conv_1_conv_1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_17_conv_conv_1_conv_1_2_Clip_output_0 = (float *)malloc((size_t)(1 * 960 * 16 * 16) * sizeof(float));
    float *encoder_features_17_conv_conv_2_Conv_output_0 = (float *)malloc((size_t)(1 * 320 * 16 * 16) * sizeof(float));
    float *encoder_features_18_features_18_0_Conv_output_0 = (float *)malloc((size_t)(1 * 1280 * 16 * 16) * sizeof(float));
    float *encoder_features_18_features_18_2_Clip_output_0 = (float *)malloc((size_t)(1 * 1280 * 16 * 16) * sizeof(float));
    float *decoder_blocks_0_Resize_output_0 = (float *)malloc((size_t)(1 * 1280 * 32 * 32) * sizeof(float));
    float *decoder_blocks_0_Concat_1_output_0 = (float *)malloc((size_t)(1 * 1376 * 32 * 32) * sizeof(float));
    float *decoder_blocks_0_conv1_conv1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 256 * 32 * 32) * sizeof(float));
    float *decoder_blocks_0_conv1_conv1_2_Relu_output_0 = (float *)malloc((size_t)(1 * 256 * 32 * 32) * sizeof(float));
    float *decoder_blocks_0_conv2_conv2_0_Conv_output_0 = (float *)malloc((size_t)(1 * 256 * 32 * 32) * sizeof(float));
    float *decoder_blocks_0_conv2_conv2_2_Relu_output_0 = (float *)malloc((size_t)(1 * 256 * 32 * 32) * sizeof(float));
    float *decoder_blocks_1_Resize_output_0 = (float *)malloc((size_t)(1 * 256 * 64 * 64) * sizeof(float));
    float *decoder_blocks_1_Concat_1_output_0 = (float *)malloc((size_t)(1 * 288 * 64 * 64) * sizeof(float));
    float *decoder_blocks_1_conv1_conv1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 128 * 64 * 64) * sizeof(float));
    float *decoder_blocks_1_conv1_conv1_2_Relu_output_0 = (float *)malloc((size_t)(1 * 128 * 64 * 64) * sizeof(float));
    float *decoder_blocks_1_conv2_conv2_0_Conv_output_0 = (float *)malloc((size_t)(1 * 128 * 64 * 64) * sizeof(float));
    float *decoder_blocks_1_conv2_conv2_2_Relu_output_0 = (float *)malloc((size_t)(1 * 128 * 64 * 64) * sizeof(float));
    float *decoder_blocks_2_Resize_output_0 = (float *)malloc((size_t)(1 * 128 * 128 * 128) * sizeof(float));
    float *decoder_blocks_2_Concat_1_output_0 = (float *)malloc((size_t)(1 * 152 * 128 * 128) * sizeof(float));
    float *decoder_blocks_2_conv1_conv1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 64 * 128 * 128) * sizeof(float));
    float *decoder_blocks_2_conv1_conv1_2_Relu_output_0 = (float *)malloc((size_t)(1 * 64 * 128 * 128) * sizeof(float));
    float *decoder_blocks_2_conv2_conv2_0_Conv_output_0 = (float *)malloc((size_t)(1 * 64 * 128 * 128) * sizeof(float));
    float *decoder_blocks_2_conv2_conv2_2_Relu_output_0 = (float *)malloc((size_t)(1 * 64 * 128 * 128) * sizeof(float));
    float *decoder_blocks_3_Resize_output_0 = (float *)malloc((size_t)(1 * 64 * 256 * 256) * sizeof(float));
    float *decoder_blocks_3_Concat_1_output_0 = (float *)malloc((size_t)(1 * 80 * 256 * 256) * sizeof(float));
    float *decoder_blocks_3_conv1_conv1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *decoder_blocks_3_conv1_conv1_2_Relu_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *decoder_blocks_3_conv2_conv2_0_Conv_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *decoder_blocks_3_conv2_conv2_2_Relu_output_0 = (float *)malloc((size_t)(1 * 32 * 256 * 256) * sizeof(float));
    float *decoder_blocks_4_Resize_output_0 = (float *)malloc((size_t)(1 * 32 * 512 * 512) * sizeof(float));
    float *decoder_blocks_4_conv1_conv1_0_Conv_output_0 = (float *)malloc((size_t)(1 * 16 * 512 * 512) * sizeof(float));
    float *decoder_blocks_4_conv1_conv1_2_Relu_output_0 = (float *)malloc((size_t)(1 * 16 * 512 * 512) * sizeof(float));
    float *decoder_blocks_4_conv2_conv2_0_Conv_output_0 = (float *)malloc((size_t)(1 * 16 * 512 * 512) * sizeof(float));
    float *decoder_blocks_4_conv2_conv2_2_Relu_output_0 = (float *)malloc((size_t)(1 * 16 * 512 * 512) * sizeof(float));
    float *logits = (float *)malloc((size_t)(1 * 4 * 512 * 512) * sizeof(float));
    printf("Running generated partial pure C forward v2...\n");
    // Node 0: Conv
    conv2d_nchw(input, onnx__Conv_715, onnx__Conv_716, encoder_features_0_features_0_0_Conv_output_0, 1, 13, 512, 512, 32, 3, 3, 256, 256, 1, 1, 2, 2, 1, 1, 1);
    // Node 3: Clip/ReLU6
    clip_float(encoder_features_0_features_0_0_Conv_output_0, encoder_features_0_features_0_2_Clip_output_0, (int)(1 * 32 * 256 * 256), 0.0f, 6.0f);
    // Node 4: Conv
    conv2d_nchw(encoder_features_0_features_0_2_Clip_output_0, onnx__Conv_718, onnx__Conv_719, encoder_features_1_conv_conv_0_conv_0_0_Conv_output_0, 1, 32, 256, 256, 32, 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 32);
    // Node 7: Clip/ReLU6
    clip_float(encoder_features_1_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_1_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 32 * 256 * 256), 0.0f, 6.0f);
    // Node 8: Conv
    conv2d_nchw(encoder_features_1_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_721, onnx__Conv_722, encoder_features_1_conv_conv_1_Conv_output_0, 1, 32, 256, 256, 16, 1, 1, 256, 256, 0, 0, 1, 1, 1, 1, 1);
    // Node 9: Conv
    conv2d_nchw(encoder_features_1_conv_conv_1_Conv_output_0, onnx__Conv_724, onnx__Conv_725, encoder_features_2_conv_conv_0_conv_0_0_Conv_output_0, 1, 16, 256, 256, 96, 1, 1, 256, 256, 0, 0, 1, 1, 1, 1, 1);
    // Node 12: Clip/ReLU6
    clip_float(encoder_features_2_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_2_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 96 * 256 * 256), 0.0f, 6.0f);
    // Node 13: Conv
    conv2d_nchw(encoder_features_2_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_727, onnx__Conv_728, encoder_features_2_conv_conv_1_conv_1_0_Conv_output_0, 1, 96, 256, 256, 96, 3, 3, 128, 128, 1, 1, 2, 2, 1, 1, 96);
    // Node 16: Clip/ReLU6
    clip_float(encoder_features_2_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_2_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 96 * 128 * 128), 0.0f, 6.0f);
    // Node 17: Conv
    conv2d_nchw(encoder_features_2_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_730, onnx__Conv_731, encoder_features_2_conv_conv_2_Conv_output_0, 1, 96, 128, 128, 24, 1, 1, 128, 128, 0, 0, 1, 1, 1, 1, 1);
    // Node 18: Conv
    conv2d_nchw(encoder_features_2_conv_conv_2_Conv_output_0, onnx__Conv_733, onnx__Conv_734, encoder_features_3_conv_conv_0_conv_0_0_Conv_output_0, 1, 24, 128, 128, 144, 1, 1, 128, 128, 0, 0, 1, 1, 1, 1, 1);
    // Node 21: Clip/ReLU6
    clip_float(encoder_features_3_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_3_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 144 * 128 * 128), 0.0f, 6.0f);
    // Node 22: Conv
    conv2d_nchw(encoder_features_3_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_736, onnx__Conv_737, encoder_features_3_conv_conv_1_conv_1_0_Conv_output_0, 1, 144, 128, 128, 144, 3, 3, 128, 128, 1, 1, 1, 1, 1, 1, 144);
    // Node 25: Clip/ReLU6
    clip_float(encoder_features_3_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_3_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 144 * 128 * 128), 0.0f, 6.0f);
    // Node 26: Conv
    conv2d_nchw(encoder_features_3_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_739, onnx__Conv_740, encoder_features_3_conv_conv_2_Conv_output_0, 1, 144, 128, 128, 24, 1, 1, 128, 128, 0, 0, 1, 1, 1, 1, 1);
    // Node 27: Add
    add_float(encoder_features_2_conv_conv_2_Conv_output_0, encoder_features_3_conv_conv_2_Conv_output_0, encoder_features_3_Add_output_0, (int)(1 * 24 * 128 * 128));
    // Node 28: Conv
    conv2d_nchw(encoder_features_3_Add_output_0, onnx__Conv_742, onnx__Conv_743, encoder_features_4_conv_conv_0_conv_0_0_Conv_output_0, 1, 24, 128, 128, 144, 1, 1, 128, 128, 0, 0, 1, 1, 1, 1, 1);
    // Node 31: Clip/ReLU6
    clip_float(encoder_features_4_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_4_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 144 * 128 * 128), 0.0f, 6.0f);
    // Node 32: Conv
    conv2d_nchw(encoder_features_4_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_745, onnx__Conv_746, encoder_features_4_conv_conv_1_conv_1_0_Conv_output_0, 1, 144, 128, 128, 144, 3, 3, 64, 64, 1, 1, 2, 2, 1, 1, 144);
    // Node 35: Clip/ReLU6
    clip_float(encoder_features_4_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_4_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 144 * 64 * 64), 0.0f, 6.0f);
    // Node 36: Conv
    conv2d_nchw(encoder_features_4_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_748, onnx__Conv_749, encoder_features_4_conv_conv_2_Conv_output_0, 1, 144, 64, 64, 32, 1, 1, 64, 64, 0, 0, 1, 1, 1, 1, 1);
    // Node 37: Conv
    conv2d_nchw(encoder_features_4_conv_conv_2_Conv_output_0, onnx__Conv_751, onnx__Conv_752, encoder_features_5_conv_conv_0_conv_0_0_Conv_output_0, 1, 32, 64, 64, 192, 1, 1, 64, 64, 0, 0, 1, 1, 1, 1, 1);
    // Node 40: Clip/ReLU6
    clip_float(encoder_features_5_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_5_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 192 * 64 * 64), 0.0f, 6.0f);
    // Node 41: Conv
    conv2d_nchw(encoder_features_5_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_754, onnx__Conv_755, encoder_features_5_conv_conv_1_conv_1_0_Conv_output_0, 1, 192, 64, 64, 192, 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 192);
    // Node 44: Clip/ReLU6
    clip_float(encoder_features_5_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_5_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 192 * 64 * 64), 0.0f, 6.0f);
    // Node 45: Conv
    conv2d_nchw(encoder_features_5_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_757, onnx__Conv_758, encoder_features_5_conv_conv_2_Conv_output_0, 1, 192, 64, 64, 32, 1, 1, 64, 64, 0, 0, 1, 1, 1, 1, 1);
    // Node 46: Add
    add_float(encoder_features_4_conv_conv_2_Conv_output_0, encoder_features_5_conv_conv_2_Conv_output_0, encoder_features_5_Add_output_0, (int)(1 * 32 * 64 * 64));
    // Node 47: Conv
    conv2d_nchw(encoder_features_5_Add_output_0, onnx__Conv_760, onnx__Conv_761, encoder_features_6_conv_conv_0_conv_0_0_Conv_output_0, 1, 32, 64, 64, 192, 1, 1, 64, 64, 0, 0, 1, 1, 1, 1, 1);
    // Node 50: Clip/ReLU6
    clip_float(encoder_features_6_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_6_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 192 * 64 * 64), 0.0f, 6.0f);
    // Node 51: Conv
    conv2d_nchw(encoder_features_6_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_763, onnx__Conv_764, encoder_features_6_conv_conv_1_conv_1_0_Conv_output_0, 1, 192, 64, 64, 192, 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 192);
    // Node 54: Clip/ReLU6
    clip_float(encoder_features_6_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_6_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 192 * 64 * 64), 0.0f, 6.0f);
    // Node 55: Conv
    conv2d_nchw(encoder_features_6_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_766, onnx__Conv_767, encoder_features_6_conv_conv_2_Conv_output_0, 1, 192, 64, 64, 32, 1, 1, 64, 64, 0, 0, 1, 1, 1, 1, 1);
    // Node 56: Add
    add_float(encoder_features_5_Add_output_0, encoder_features_6_conv_conv_2_Conv_output_0, encoder_features_6_Add_output_0, (int)(1 * 32 * 64 * 64));
    // Node 57: Conv
    conv2d_nchw(encoder_features_6_Add_output_0, onnx__Conv_769, onnx__Conv_770, encoder_features_7_conv_conv_0_conv_0_0_Conv_output_0, 1, 32, 64, 64, 192, 1, 1, 64, 64, 0, 0, 1, 1, 1, 1, 1);
    // Node 60: Clip/ReLU6
    clip_float(encoder_features_7_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_7_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 192 * 64 * 64), 0.0f, 6.0f);
    // Node 61: Conv
    conv2d_nchw(encoder_features_7_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_772, onnx__Conv_773, encoder_features_7_conv_conv_1_conv_1_0_Conv_output_0, 1, 192, 64, 64, 192, 3, 3, 32, 32, 1, 1, 2, 2, 1, 1, 192);
    // Node 64: Clip/ReLU6
    clip_float(encoder_features_7_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_7_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 192 * 32 * 32), 0.0f, 6.0f);
    // Node 65: Conv
    conv2d_nchw(encoder_features_7_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_775, onnx__Conv_776, encoder_features_7_conv_conv_2_Conv_output_0, 1, 192, 32, 32, 64, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 66: Conv
    conv2d_nchw(encoder_features_7_conv_conv_2_Conv_output_0, onnx__Conv_778, onnx__Conv_779, encoder_features_8_conv_conv_0_conv_0_0_Conv_output_0, 1, 64, 32, 32, 384, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 69: Clip/ReLU6
    clip_float(encoder_features_8_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_8_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 70: Conv
    conv2d_nchw(encoder_features_8_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_781, onnx__Conv_782, encoder_features_8_conv_conv_1_conv_1_0_Conv_output_0, 1, 384, 32, 32, 384, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 384);
    // Node 73: Clip/ReLU6
    clip_float(encoder_features_8_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_8_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 74: Conv
    conv2d_nchw(encoder_features_8_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_784, onnx__Conv_785, encoder_features_8_conv_conv_2_Conv_output_0, 1, 384, 32, 32, 64, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 75: Add
    add_float(encoder_features_7_conv_conv_2_Conv_output_0, encoder_features_8_conv_conv_2_Conv_output_0, encoder_features_8_Add_output_0, (int)(1 * 64 * 32 * 32));
    // Node 76: Conv
    conv2d_nchw(encoder_features_8_Add_output_0, onnx__Conv_787, onnx__Conv_788, encoder_features_9_conv_conv_0_conv_0_0_Conv_output_0, 1, 64, 32, 32, 384, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 79: Clip/ReLU6
    clip_float(encoder_features_9_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_9_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 80: Conv
    conv2d_nchw(encoder_features_9_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_790, onnx__Conv_791, encoder_features_9_conv_conv_1_conv_1_0_Conv_output_0, 1, 384, 32, 32, 384, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 384);
    // Node 83: Clip/ReLU6
    clip_float(encoder_features_9_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_9_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 84: Conv
    conv2d_nchw(encoder_features_9_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_793, onnx__Conv_794, encoder_features_9_conv_conv_2_Conv_output_0, 1, 384, 32, 32, 64, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 85: Add
    add_float(encoder_features_8_Add_output_0, encoder_features_9_conv_conv_2_Conv_output_0, encoder_features_9_Add_output_0, (int)(1 * 64 * 32 * 32));
    // Node 86: Conv
    conv2d_nchw(encoder_features_9_Add_output_0, onnx__Conv_796, onnx__Conv_797, encoder_features_10_conv_conv_0_conv_0_0_Conv_output_0, 1, 64, 32, 32, 384, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 89: Clip/ReLU6
    clip_float(encoder_features_10_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_10_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 90: Conv
    conv2d_nchw(encoder_features_10_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_799, onnx__Conv_800, encoder_features_10_conv_conv_1_conv_1_0_Conv_output_0, 1, 384, 32, 32, 384, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 384);
    // Node 93: Clip/ReLU6
    clip_float(encoder_features_10_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_10_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 94: Conv
    conv2d_nchw(encoder_features_10_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_802, onnx__Conv_803, encoder_features_10_conv_conv_2_Conv_output_0, 1, 384, 32, 32, 64, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 95: Add
    add_float(encoder_features_9_Add_output_0, encoder_features_10_conv_conv_2_Conv_output_0, encoder_features_10_Add_output_0, (int)(1 * 64 * 32 * 32));
    // Node 96: Conv
    conv2d_nchw(encoder_features_10_Add_output_0, onnx__Conv_805, onnx__Conv_806, encoder_features_11_conv_conv_0_conv_0_0_Conv_output_0, 1, 64, 32, 32, 384, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 99: Clip/ReLU6
    clip_float(encoder_features_11_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_11_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 100: Conv
    conv2d_nchw(encoder_features_11_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_808, onnx__Conv_809, encoder_features_11_conv_conv_1_conv_1_0_Conv_output_0, 1, 384, 32, 32, 384, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 384);
    // Node 103: Clip/ReLU6
    clip_float(encoder_features_11_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_11_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 384 * 32 * 32), 0.0f, 6.0f);
    // Node 104: Conv
    conv2d_nchw(encoder_features_11_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_811, onnx__Conv_812, encoder_features_11_conv_conv_2_Conv_output_0, 1, 384, 32, 32, 96, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 105: Conv
    conv2d_nchw(encoder_features_11_conv_conv_2_Conv_output_0, onnx__Conv_814, onnx__Conv_815, encoder_features_12_conv_conv_0_conv_0_0_Conv_output_0, 1, 96, 32, 32, 576, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 108: Clip/ReLU6
    clip_float(encoder_features_12_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_12_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 576 * 32 * 32), 0.0f, 6.0f);
    // Node 109: Conv
    conv2d_nchw(encoder_features_12_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_817, onnx__Conv_818, encoder_features_12_conv_conv_1_conv_1_0_Conv_output_0, 1, 576, 32, 32, 576, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 576);
    // Node 112: Clip/ReLU6
    clip_float(encoder_features_12_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_12_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 576 * 32 * 32), 0.0f, 6.0f);
    // Node 113: Conv
    conv2d_nchw(encoder_features_12_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_820, onnx__Conv_821, encoder_features_12_conv_conv_2_Conv_output_0, 1, 576, 32, 32, 96, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 114: Add
    add_float(encoder_features_11_conv_conv_2_Conv_output_0, encoder_features_12_conv_conv_2_Conv_output_0, encoder_features_12_Add_output_0, (int)(1 * 96 * 32 * 32));
    // Node 115: Conv
    conv2d_nchw(encoder_features_12_Add_output_0, onnx__Conv_823, onnx__Conv_824, encoder_features_13_conv_conv_0_conv_0_0_Conv_output_0, 1, 96, 32, 32, 576, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 118: Clip/ReLU6
    clip_float(encoder_features_13_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_13_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 576 * 32 * 32), 0.0f, 6.0f);
    // Node 119: Conv
    conv2d_nchw(encoder_features_13_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_826, onnx__Conv_827, encoder_features_13_conv_conv_1_conv_1_0_Conv_output_0, 1, 576, 32, 32, 576, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 576);
    // Node 122: Clip/ReLU6
    clip_float(encoder_features_13_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_13_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 576 * 32 * 32), 0.0f, 6.0f);
    // Node 123: Conv
    conv2d_nchw(encoder_features_13_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_829, onnx__Conv_830, encoder_features_13_conv_conv_2_Conv_output_0, 1, 576, 32, 32, 96, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 124: Add
    add_float(encoder_features_12_Add_output_0, encoder_features_13_conv_conv_2_Conv_output_0, encoder_features_13_Add_output_0, (int)(1 * 96 * 32 * 32));
    // Node 125: Conv
    conv2d_nchw(encoder_features_13_Add_output_0, onnx__Conv_832, onnx__Conv_833, encoder_features_14_conv_conv_0_conv_0_0_Conv_output_0, 1, 96, 32, 32, 576, 1, 1, 32, 32, 0, 0, 1, 1, 1, 1, 1);
    // Node 128: Clip/ReLU6
    clip_float(encoder_features_14_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_14_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 576 * 32 * 32), 0.0f, 6.0f);
    // Node 129: Conv
    conv2d_nchw(encoder_features_14_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_835, onnx__Conv_836, encoder_features_14_conv_conv_1_conv_1_0_Conv_output_0, 1, 576, 32, 32, 576, 3, 3, 16, 16, 1, 1, 2, 2, 1, 1, 576);
    // Node 132: Clip/ReLU6
    clip_float(encoder_features_14_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_14_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 576 * 16 * 16), 0.0f, 6.0f);
    // Node 133: Conv
    conv2d_nchw(encoder_features_14_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_838, onnx__Conv_839, encoder_features_14_conv_conv_2_Conv_output_0, 1, 576, 16, 16, 160, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 134: Conv
    conv2d_nchw(encoder_features_14_conv_conv_2_Conv_output_0, onnx__Conv_841, onnx__Conv_842, encoder_features_15_conv_conv_0_conv_0_0_Conv_output_0, 1, 160, 16, 16, 960, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 137: Clip/ReLU6
    clip_float(encoder_features_15_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_15_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 960 * 16 * 16), 0.0f, 6.0f);
    // Node 138: Conv
    conv2d_nchw(encoder_features_15_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_844, onnx__Conv_845, encoder_features_15_conv_conv_1_conv_1_0_Conv_output_0, 1, 960, 16, 16, 960, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 960);
    // Node 141: Clip/ReLU6
    clip_float(encoder_features_15_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_15_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 960 * 16 * 16), 0.0f, 6.0f);
    // Node 142: Conv
    conv2d_nchw(encoder_features_15_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_847, onnx__Conv_848, encoder_features_15_conv_conv_2_Conv_output_0, 1, 960, 16, 16, 160, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 143: Add
    add_float(encoder_features_14_conv_conv_2_Conv_output_0, encoder_features_15_conv_conv_2_Conv_output_0, encoder_features_15_Add_output_0, (int)(1 * 160 * 16 * 16));
    // Node 144: Conv
    conv2d_nchw(encoder_features_15_Add_output_0, onnx__Conv_850, onnx__Conv_851, encoder_features_16_conv_conv_0_conv_0_0_Conv_output_0, 1, 160, 16, 16, 960, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 147: Clip/ReLU6
    clip_float(encoder_features_16_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_16_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 960 * 16 * 16), 0.0f, 6.0f);
    // Node 148: Conv
    conv2d_nchw(encoder_features_16_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_853, onnx__Conv_854, encoder_features_16_conv_conv_1_conv_1_0_Conv_output_0, 1, 960, 16, 16, 960, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 960);
    // Node 151: Clip/ReLU6
    clip_float(encoder_features_16_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_16_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 960 * 16 * 16), 0.0f, 6.0f);
    // Node 152: Conv
    conv2d_nchw(encoder_features_16_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_856, onnx__Conv_857, encoder_features_16_conv_conv_2_Conv_output_0, 1, 960, 16, 16, 160, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 153: Add
    add_float(encoder_features_15_Add_output_0, encoder_features_16_conv_conv_2_Conv_output_0, encoder_features_16_Add_output_0, (int)(1 * 160 * 16 * 16));
    // Node 154: Conv
    conv2d_nchw(encoder_features_16_Add_output_0, onnx__Conv_859, onnx__Conv_860, encoder_features_17_conv_conv_0_conv_0_0_Conv_output_0, 1, 160, 16, 16, 960, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 157: Clip/ReLU6
    clip_float(encoder_features_17_conv_conv_0_conv_0_0_Conv_output_0, encoder_features_17_conv_conv_0_conv_0_2_Clip_output_0, (int)(1 * 960 * 16 * 16), 0.0f, 6.0f);
    // Node 158: Conv
    conv2d_nchw(encoder_features_17_conv_conv_0_conv_0_2_Clip_output_0, onnx__Conv_862, onnx__Conv_863, encoder_features_17_conv_conv_1_conv_1_0_Conv_output_0, 1, 960, 16, 16, 960, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 960);
    // Node 161: Clip/ReLU6
    clip_float(encoder_features_17_conv_conv_1_conv_1_0_Conv_output_0, encoder_features_17_conv_conv_1_conv_1_2_Clip_output_0, (int)(1 * 960 * 16 * 16), 0.0f, 6.0f);
    // Node 162: Conv
    conv2d_nchw(encoder_features_17_conv_conv_1_conv_1_2_Clip_output_0, onnx__Conv_865, onnx__Conv_866, encoder_features_17_conv_conv_2_Conv_output_0, 1, 960, 16, 16, 320, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 163: Conv
    conv2d_nchw(encoder_features_17_conv_conv_2_Conv_output_0, onnx__Conv_868, onnx__Conv_869, encoder_features_18_features_18_0_Conv_output_0, 1, 320, 16, 16, 1280, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1, 1);
    // Node 166: Clip/ReLU6
    clip_float(encoder_features_18_features_18_0_Conv_output_0, encoder_features_18_features_18_2_Clip_output_0, (int)(1 * 1280 * 16 * 16), 0.0f, 6.0f);
    // Node 174: Resize nearest
    resize_nearest_nchw(encoder_features_18_features_18_2_Clip_output_0, decoder_blocks_0_Resize_output_0, 1, 1280, 16, 16, 32, 32);
    // Node 175: Concat axis=1
    const float *concat_inputs_175[] = {decoder_blocks_0_Resize_output_0, encoder_features_13_Add_output_0};
    const int concat_channels_175[] = {1280, 96};
    concat_channel_nchw(concat_inputs_175, concat_channels_175, 2, decoder_blocks_0_Concat_1_output_0, 1, 32, 32);
    // Node 176: Conv
    conv2d_nchw(decoder_blocks_0_Concat_1_output_0, onnx__Conv_871, onnx__Conv_872, decoder_blocks_0_conv1_conv1_0_Conv_output_0, 1, 1376, 32, 32, 256, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 1);
    // Node 177: Relu
    relu_float(decoder_blocks_0_conv1_conv1_0_Conv_output_0, decoder_blocks_0_conv1_conv1_2_Relu_output_0, (int)(1 * 256 * 32 * 32));
    // Node 178: Conv
    conv2d_nchw(decoder_blocks_0_conv1_conv1_2_Relu_output_0, onnx__Conv_874, onnx__Conv_875, decoder_blocks_0_conv2_conv2_0_Conv_output_0, 1, 256, 32, 32, 256, 3, 3, 32, 32, 1, 1, 1, 1, 1, 1, 1);
    // Node 179: Relu
    relu_float(decoder_blocks_0_conv2_conv2_0_Conv_output_0, decoder_blocks_0_conv2_conv2_2_Relu_output_0, (int)(1 * 256 * 32 * 32));
    // Node 187: Resize nearest
    resize_nearest_nchw(decoder_blocks_0_conv2_conv2_2_Relu_output_0, decoder_blocks_1_Resize_output_0, 1, 256, 32, 32, 64, 64);
    // Node 188: Concat axis=1
    const float *concat_inputs_188[] = {decoder_blocks_1_Resize_output_0, encoder_features_6_Add_output_0};
    const int concat_channels_188[] = {256, 32};
    concat_channel_nchw(concat_inputs_188, concat_channels_188, 2, decoder_blocks_1_Concat_1_output_0, 1, 64, 64);
    // Node 189: Conv
    conv2d_nchw(decoder_blocks_1_Concat_1_output_0, onnx__Conv_877, onnx__Conv_878, decoder_blocks_1_conv1_conv1_0_Conv_output_0, 1, 288, 64, 64, 128, 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 1);
    // Node 190: Relu
    relu_float(decoder_blocks_1_conv1_conv1_0_Conv_output_0, decoder_blocks_1_conv1_conv1_2_Relu_output_0, (int)(1 * 128 * 64 * 64));
    // Node 191: Conv
    conv2d_nchw(decoder_blocks_1_conv1_conv1_2_Relu_output_0, onnx__Conv_880, onnx__Conv_881, decoder_blocks_1_conv2_conv2_0_Conv_output_0, 1, 128, 64, 64, 128, 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 1);
    // Node 192: Relu
    relu_float(decoder_blocks_1_conv2_conv2_0_Conv_output_0, decoder_blocks_1_conv2_conv2_2_Relu_output_0, (int)(1 * 128 * 64 * 64));
    // Node 200: Resize nearest
    resize_nearest_nchw(decoder_blocks_1_conv2_conv2_2_Relu_output_0, decoder_blocks_2_Resize_output_0, 1, 128, 64, 64, 128, 128);
    // Node 201: Concat axis=1
    const float *concat_inputs_201[] = {decoder_blocks_2_Resize_output_0, encoder_features_3_Add_output_0};
    const int concat_channels_201[] = {128, 24};
    concat_channel_nchw(concat_inputs_201, concat_channels_201, 2, decoder_blocks_2_Concat_1_output_0, 1, 128, 128);
    // Node 202: Conv
    conv2d_nchw(decoder_blocks_2_Concat_1_output_0, onnx__Conv_883, onnx__Conv_884, decoder_blocks_2_conv1_conv1_0_Conv_output_0, 1, 152, 128, 128, 64, 3, 3, 128, 128, 1, 1, 1, 1, 1, 1, 1);
    // Node 203: Relu
    relu_float(decoder_blocks_2_conv1_conv1_0_Conv_output_0, decoder_blocks_2_conv1_conv1_2_Relu_output_0, (int)(1 * 64 * 128 * 128));
    // Node 204: Conv
    conv2d_nchw(decoder_blocks_2_conv1_conv1_2_Relu_output_0, onnx__Conv_886, onnx__Conv_887, decoder_blocks_2_conv2_conv2_0_Conv_output_0, 1, 64, 128, 128, 64, 3, 3, 128, 128, 1, 1, 1, 1, 1, 1, 1);
    // Node 205: Relu
    relu_float(decoder_blocks_2_conv2_conv2_0_Conv_output_0, decoder_blocks_2_conv2_conv2_2_Relu_output_0, (int)(1 * 64 * 128 * 128));
    // Node 213: Resize nearest
    resize_nearest_nchw(decoder_blocks_2_conv2_conv2_2_Relu_output_0, decoder_blocks_3_Resize_output_0, 1, 64, 128, 128, 256, 256);
    // Node 214: Concat axis=1
    const float *concat_inputs_214[] = {decoder_blocks_3_Resize_output_0, encoder_features_1_conv_conv_1_Conv_output_0};
    const int concat_channels_214[] = {64, 16};
    concat_channel_nchw(concat_inputs_214, concat_channels_214, 2, decoder_blocks_3_Concat_1_output_0, 1, 256, 256);
    // Node 215: Conv
    conv2d_nchw(decoder_blocks_3_Concat_1_output_0, onnx__Conv_889, onnx__Conv_890, decoder_blocks_3_conv1_conv1_0_Conv_output_0, 1, 80, 256, 256, 32, 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 1);
    // Node 216: Relu
    relu_float(decoder_blocks_3_conv1_conv1_0_Conv_output_0, decoder_blocks_3_conv1_conv1_2_Relu_output_0, (int)(1 * 32 * 256 * 256));
    // Node 217: Conv
    conv2d_nchw(decoder_blocks_3_conv1_conv1_2_Relu_output_0, onnx__Conv_892, onnx__Conv_893, decoder_blocks_3_conv2_conv2_0_Conv_output_0, 1, 32, 256, 256, 32, 3, 3, 256, 256, 1, 1, 1, 1, 1, 1, 1);
    // Node 218: Relu
    relu_float(decoder_blocks_3_conv2_conv2_0_Conv_output_0, decoder_blocks_3_conv2_conv2_2_Relu_output_0, (int)(1 * 32 * 256 * 256));
    // Node 226: Resize nearest
    resize_nearest_nchw(decoder_blocks_3_conv2_conv2_2_Relu_output_0, decoder_blocks_4_Resize_output_0, 1, 32, 256, 256, 512, 512);
    // Node 227: Conv
    conv2d_nchw(decoder_blocks_4_Resize_output_0, onnx__Conv_895, onnx__Conv_896, decoder_blocks_4_conv1_conv1_0_Conv_output_0, 1, 32, 512, 512, 16, 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1);
    // Node 228: Relu
    relu_float(decoder_blocks_4_conv1_conv1_0_Conv_output_0, decoder_blocks_4_conv1_conv1_2_Relu_output_0, (int)(1 * 16 * 512 * 512));
    // Node 229: Conv
    conv2d_nchw(decoder_blocks_4_conv1_conv1_2_Relu_output_0, onnx__Conv_898, onnx__Conv_899, decoder_blocks_4_conv2_conv2_0_Conv_output_0, 1, 16, 512, 512, 16, 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1);
    // Node 230: Relu
    relu_float(decoder_blocks_4_conv2_conv2_0_Conv_output_0, decoder_blocks_4_conv2_conv2_2_Relu_output_0, (int)(1 * 16 * 512 * 512));
    // Node 231: Conv
    conv2d_nchw(decoder_blocks_4_conv2_conv2_2_Relu_output_0, segmentation_head_0_weight, segmentation_head_0_bias, logits, 1, 16, 512, 512, 4, 3, 3, 512, 512, 1, 1, 1, 1, 1, 1, 1);
    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (size_t i = 0; i < (size_t)PFV2_LAST_COUNT; i++) {
        double diff = fabs((double)logits[i] - (double)ref[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (double)PFV2_LAST_COUNT;
    printf("Partial output count: %d\n", PFV2_LAST_COUNT);
    printf("Mean abs error vs ONNX Runtime: %.10f\n", mean_abs);
    printf("Max abs error vs ONNX Runtime: %.10f\n", max_abs);
        // Final argmax and comparison against PyTorch prediction mask.
    size_t pred_count = (size_t)1 * 512 * 512;

    uint8_t *pred = (uint8_t *)malloc(pred_count * sizeof(uint8_t));
    uint8_t *py_ref = (uint8_t *)malloc(pred_count * sizeof(uint8_t));

    if (!pred || !py_ref) {
        printf("Argmax allocation failed\n");
        return 1;
    }

    FILE *fmask = fopen("inputs/pytorch_pred_512x512_uint8.bin", "rb");
    if (!fmask) {
        printf("Failed to open inputs/pytorch_pred_512x512_uint8.bin\n");
        return 1;
    }

    fread(py_ref, sizeof(uint8_t), pred_count, fmask);
    fclose(fmask);

    argmax_nchw_classes(logits, pred, 1, 4, 512, 512);

    size_t mismatches = 0;
    size_t cloud_pixels = 0;

    for (size_t i = 0; i < pred_count; i++) {
        if (pred[i] != py_ref[i]) {
            mismatches++;
        }

        if (pred[i] == 1 || pred[i] == 2) {
            cloud_pixels++;
        }
    }

    double mask_mismatch = (double)mismatches / (double)pred_count;
    double cloud_fraction = (double)cloud_pixels / (double)pred_count;

    printf("Pure C mask cloud fraction: %.10f\n", cloud_fraction);
    printf("Mismatch vs PyTorch mask: %.10f\n", mask_mismatch);

    FILE *pgm = fopen("outputs/predictions/pure_c_full_pred.pgm", "wb");
    if (pgm) {
        fprintf(pgm, "P5\n512 512\n255\n");

        for (size_t i = 0; i < pred_count; i++) {
            uint8_t v = (uint8_t)(pred[i] * 80);
            fwrite(&v, sizeof(uint8_t), 1, pgm);
        }

        fclose(pgm);
        printf("Saved: outputs/predictions/pure_c_full_pred.pgm\n");
    }

    free(pred);
    free(py_ref);

    free(encoder_features_0_features_0_0_Conv_output_0);
    free(onnx__Conv_715);
    free(onnx__Conv_716);
    free(encoder_features_0_features_0_2_Clip_output_0);
    free(encoder_features_1_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_718);
    free(onnx__Conv_719);
    free(encoder_features_1_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_1_conv_conv_1_Conv_output_0);
    free(onnx__Conv_721);
    free(onnx__Conv_722);
    free(encoder_features_2_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_724);
    free(onnx__Conv_725);
    free(encoder_features_2_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_2_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_727);
    free(onnx__Conv_728);
    free(encoder_features_2_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_2_conv_conv_2_Conv_output_0);
    free(onnx__Conv_730);
    free(onnx__Conv_731);
    free(encoder_features_3_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_733);
    free(onnx__Conv_734);
    free(encoder_features_3_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_3_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_736);
    free(onnx__Conv_737);
    free(encoder_features_3_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_3_conv_conv_2_Conv_output_0);
    free(onnx__Conv_739);
    free(onnx__Conv_740);
    free(encoder_features_3_Add_output_0);
    free(encoder_features_4_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_742);
    free(onnx__Conv_743);
    free(encoder_features_4_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_4_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_745);
    free(onnx__Conv_746);
    free(encoder_features_4_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_4_conv_conv_2_Conv_output_0);
    free(onnx__Conv_748);
    free(onnx__Conv_749);
    free(encoder_features_5_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_751);
    free(onnx__Conv_752);
    free(encoder_features_5_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_5_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_754);
    free(onnx__Conv_755);
    free(encoder_features_5_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_5_conv_conv_2_Conv_output_0);
    free(onnx__Conv_757);
    free(onnx__Conv_758);
    free(encoder_features_5_Add_output_0);
    free(encoder_features_6_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_760);
    free(onnx__Conv_761);
    free(encoder_features_6_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_6_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_763);
    free(onnx__Conv_764);
    free(encoder_features_6_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_6_conv_conv_2_Conv_output_0);
    free(onnx__Conv_766);
    free(onnx__Conv_767);
    free(encoder_features_6_Add_output_0);
    free(encoder_features_7_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_769);
    free(onnx__Conv_770);
    free(encoder_features_7_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_7_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_772);
    free(onnx__Conv_773);
    free(encoder_features_7_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_7_conv_conv_2_Conv_output_0);
    free(onnx__Conv_775);
    free(onnx__Conv_776);
    free(encoder_features_8_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_778);
    free(onnx__Conv_779);
    free(encoder_features_8_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_8_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_781);
    free(onnx__Conv_782);
    free(encoder_features_8_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_8_conv_conv_2_Conv_output_0);
    free(onnx__Conv_784);
    free(onnx__Conv_785);
    free(encoder_features_8_Add_output_0);
    free(encoder_features_9_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_787);
    free(onnx__Conv_788);
    free(encoder_features_9_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_9_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_790);
    free(onnx__Conv_791);
    free(encoder_features_9_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_9_conv_conv_2_Conv_output_0);
    free(onnx__Conv_793);
    free(onnx__Conv_794);
    free(encoder_features_9_Add_output_0);
    free(encoder_features_10_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_796);
    free(onnx__Conv_797);
    free(encoder_features_10_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_10_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_799);
    free(onnx__Conv_800);
    free(encoder_features_10_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_10_conv_conv_2_Conv_output_0);
    free(onnx__Conv_802);
    free(onnx__Conv_803);
    free(encoder_features_10_Add_output_0);
    free(encoder_features_11_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_805);
    free(onnx__Conv_806);
    free(encoder_features_11_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_11_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_808);
    free(onnx__Conv_809);
    free(encoder_features_11_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_11_conv_conv_2_Conv_output_0);
    free(onnx__Conv_811);
    free(onnx__Conv_812);
    free(encoder_features_12_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_814);
    free(onnx__Conv_815);
    free(encoder_features_12_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_12_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_817);
    free(onnx__Conv_818);
    free(encoder_features_12_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_12_conv_conv_2_Conv_output_0);
    free(onnx__Conv_820);
    free(onnx__Conv_821);
    free(encoder_features_12_Add_output_0);
    free(encoder_features_13_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_823);
    free(onnx__Conv_824);
    free(encoder_features_13_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_13_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_826);
    free(onnx__Conv_827);
    free(encoder_features_13_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_13_conv_conv_2_Conv_output_0);
    free(onnx__Conv_829);
    free(onnx__Conv_830);
    free(encoder_features_13_Add_output_0);
    free(encoder_features_14_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_832);
    free(onnx__Conv_833);
    free(encoder_features_14_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_14_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_835);
    free(onnx__Conv_836);
    free(encoder_features_14_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_14_conv_conv_2_Conv_output_0);
    free(onnx__Conv_838);
    free(onnx__Conv_839);
    free(encoder_features_15_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_841);
    free(onnx__Conv_842);
    free(encoder_features_15_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_15_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_844);
    free(onnx__Conv_845);
    free(encoder_features_15_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_15_conv_conv_2_Conv_output_0);
    free(onnx__Conv_847);
    free(onnx__Conv_848);
    free(encoder_features_15_Add_output_0);
    free(encoder_features_16_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_850);
    free(onnx__Conv_851);
    free(encoder_features_16_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_16_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_853);
    free(onnx__Conv_854);
    free(encoder_features_16_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_16_conv_conv_2_Conv_output_0);
    free(onnx__Conv_856);
    free(onnx__Conv_857);
    free(encoder_features_16_Add_output_0);
    free(encoder_features_17_conv_conv_0_conv_0_0_Conv_output_0);
    free(onnx__Conv_859);
    free(onnx__Conv_860);
    free(encoder_features_17_conv_conv_0_conv_0_2_Clip_output_0);
    free(encoder_features_17_conv_conv_1_conv_1_0_Conv_output_0);
    free(onnx__Conv_862);
    free(onnx__Conv_863);
    free(encoder_features_17_conv_conv_1_conv_1_2_Clip_output_0);
    free(encoder_features_17_conv_conv_2_Conv_output_0);
    free(onnx__Conv_865);
    free(onnx__Conv_866);
    free(encoder_features_18_features_18_0_Conv_output_0);
    free(onnx__Conv_868);
    free(onnx__Conv_869);
    free(encoder_features_18_features_18_2_Clip_output_0);
    free(decoder_blocks_0_Resize_output_0);
    free(decoder_blocks_0_Concat_1_output_0);
    free(decoder_blocks_0_conv1_conv1_0_Conv_output_0);
    free(onnx__Conv_871);
    free(onnx__Conv_872);
    free(decoder_blocks_0_conv1_conv1_2_Relu_output_0);
    free(decoder_blocks_0_conv2_conv2_0_Conv_output_0);
    free(onnx__Conv_874);
    free(onnx__Conv_875);
    free(decoder_blocks_0_conv2_conv2_2_Relu_output_0);
    free(decoder_blocks_1_Resize_output_0);
    free(decoder_blocks_1_Concat_1_output_0);
    free(decoder_blocks_1_conv1_conv1_0_Conv_output_0);
    free(onnx__Conv_877);
    free(onnx__Conv_878);
    free(decoder_blocks_1_conv1_conv1_2_Relu_output_0);
    free(decoder_blocks_1_conv2_conv2_0_Conv_output_0);
    free(onnx__Conv_880);
    free(onnx__Conv_881);
    free(decoder_blocks_1_conv2_conv2_2_Relu_output_0);
    free(decoder_blocks_2_Resize_output_0);
    free(decoder_blocks_2_Concat_1_output_0);
    free(decoder_blocks_2_conv1_conv1_0_Conv_output_0);
    free(onnx__Conv_883);
    free(onnx__Conv_884);
    free(decoder_blocks_2_conv1_conv1_2_Relu_output_0);
    free(decoder_blocks_2_conv2_conv2_0_Conv_output_0);
    free(onnx__Conv_886);
    free(onnx__Conv_887);
    free(decoder_blocks_2_conv2_conv2_2_Relu_output_0);
    free(decoder_blocks_3_Resize_output_0);
    free(decoder_blocks_3_Concat_1_output_0);
    free(decoder_blocks_3_conv1_conv1_0_Conv_output_0);
    free(onnx__Conv_889);
    free(onnx__Conv_890);
    free(decoder_blocks_3_conv1_conv1_2_Relu_output_0);
    free(decoder_blocks_3_conv2_conv2_0_Conv_output_0);
    free(onnx__Conv_892);
    free(onnx__Conv_893);
    free(decoder_blocks_3_conv2_conv2_2_Relu_output_0);
    free(decoder_blocks_4_Resize_output_0);
    free(decoder_blocks_4_conv1_conv1_0_Conv_output_0);
    free(onnx__Conv_895);
    free(onnx__Conv_896);
    free(decoder_blocks_4_conv1_conv1_2_Relu_output_0);
    free(decoder_blocks_4_conv2_conv2_0_Conv_output_0);
    free(onnx__Conv_898);
    free(onnx__Conv_899);
    free(decoder_blocks_4_conv2_conv2_2_Relu_output_0);
    free(logits);
    free(segmentation_head_0_weight);
    free(segmentation_head_0_bias);
    free(input);
    free(ref);
    return 0;
}
/* END INLINE cloudsen12_full_c.c */

