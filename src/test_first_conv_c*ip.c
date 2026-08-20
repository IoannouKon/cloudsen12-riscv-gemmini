#include "conv2d.h"
#include "clip.h"
#include "first_conv_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int read_float_file(const char *path, float *data, size_t count) {*    FILE *f = fopen(path, "rb");
 *  if (!f) return 0;
    size_t n =*fread(data, sizeof(float), count, *);
    fclose(f);
    return n == *ount;
}

int main(void) {
    size*t input_count = (size_t)FC_N * FC_*_IN * FC_H_IN * FC_W_IN;
    size_* weight_count = (size_t)FC_C_OUT **(FC_C_IN / FC_GROUP) * FC_KH * FC_*W;
    size_t bias_count = (size_t*FC_C_OUT;
    size_t output_count * (size_t)FC_N * FC_C_OUT * FC_H_OU* * FC_W_OUT;

    float *input = (*loat *)malloc(input_count * sizeof*float));
    float *weight = (floa* *)malloc(weight_count * sizeof(fl*at));
    float *bias = (float *)m*lloc(bias_count * sizeof(float));
*   float *conv_out = (float *)mall*c(output_count * sizeof(float));
 *  float *clip_out = (float *)mallo*(output_count * sizeof(float));
  * float *ref = (float *)malloc(outp*t_count * sizeof(float));

    if *!input || !weight || !bias || !con*_out || !clip_out || !ref) {
     *  printf("Allocation failed\n");
 *      return 1;
    }

    if (!re*d_float_file("input_1x13x512x512_f*oat32.bin", input, input_count)) {*        printf("Failed to read inp*t\n");
        return 1;
    }

  * if (!read_float_file("experiments/operator_tests/first_conv/*onv_w.bin", weight, weight_count))*{
        printf("Failed to read w*ights\n");
        return 1;
    }*
    if (!read_float_file("first_c*nv/conv_b.bin", bias, bias_count))*{
        printf("Failed to read b*as\n");
        return 1;
    }

 *  if (!read_float_file("first_clip*clip_ref.bin", ref, output_count))*{
        printf("Failed to read C*ip reference\n");
        return 1*
    }

    printf("Running pure C*first Conv + Clip...\n");

    con*2d_nchw(
        input,
        we*ght,
        bias,
        conv_ou*,
        FC_N,
        FC_C_IN,
 *      FC_H_IN,
        FC_W_IN,
  *     FC_C_OUT,
        FC_KH,
    *   FC_KW,
        FC_H_OUT,
      * FC_W_OUT,
        FC_PAD_TOP,
   *    FC_PAD_LEFT,
        FC_STRIDE*H,
        FC_STRIDE_W,
        FC*DILATION_H,
        FC_DILATION_W,*        FC_GROUP
    );

    clip_*loat(conv_out, clip_out, (int)outp*t_count, 0.0f, 6.0f);

    double *ax_abs = 0.0;
    double mean_abs * 0.0;

    for (size_t i = 0; i < *utput_count; i++) {
        double*diff = fabs((double)clip_out[i] - *double)ref[i]);
        if (diff >*max_abs) {
            max_abs = d*ff;
        }
        mean_abs += *iff;
    }

    mean_abs /= (doubl*)output_count;

    printf("Output*count: %zu\n", output_count);
    *rintf("Mean abs error vs ONNX Runt*me: %.10f\n", mean_abs);
    print*("Max abs error vs ONNX Runtime: %*10f\n", max_abs);

    free(input)*
    free(weight);
    free(bias);*    free(conv_out);
    free(clip_*ut);
    free(ref);

    return 0;*}
