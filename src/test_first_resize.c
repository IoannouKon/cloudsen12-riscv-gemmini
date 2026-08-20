#include "resize.h"
#include "first_resize_meta.h"

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
    size_t in_count = (size_t)FRS_N * FRS_C * FRS_H_IN * FRS_W_IN;
    size_t out_count = (size_t)FRS_N * FRS_C * FRS_H_OUT * FRS_W_OUT;
    float *input=(float*)malloc(in_count*sizeof(float));
    float *out=(float*)malloc(out_count*sizeof(float));
    float *ref=(float*)malloc(out_count*sizeof(float));
    if(!input||!out||!ref){printf("Allocation failed\n"); return 1;}
    if(!read_float_file("experiments/operator_tests/first_resize/resize_input.bin",input,in_count)){printf("Failed resize input\n"); return 1;}
    if(!read_float_file("experiments/operator_tests/first_resize/resize_ref.bin",ref,out_count)){printf("Failed resize ref\n"); return 1;}

    printf("Running pure C first Resize nearest...\n");
    resize_nearest_nchw(input,out,FRS_N,FRS_C,FRS_H_IN,FRS_W_IN,FRS_H_OUT,FRS_W_OUT);

    double max_abs=0.0, mean_abs=0.0;
    for(size_t i=0;i<out_count;i++){
        double diff=fabs((double)out[i]-(double)ref[i]);
        if(diff>max_abs) max_abs=diff;
        mean_abs+=diff;
    }
    mean_abs/=(double)out_count;
    printf("Output count: %zu\n",out_count);
    printf("Mean abs error vs ONNX Runtime: %.10f\n",mean_abs);
    printf("Max abs error vs ONNX Runtime: %.10f\n",max_abs);
    free(input); free(out); free(ref);
    return 0;
}
