#include "argmax.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int read_float_file(const char *path, float *data, size_t count) {
    FILE *f=fopen(path,"rb"); if(!f) return 0; size_t n=fread(data,sizeof(float),count,f); fclose(f); return n==count;
}
static int read_uint8_file(const char *path, uint8_t *data, size_t count) {
    FILE *f=fopen(path,"rb"); if(!f) return 0; size_t n=fread(data,sizeof(uint8_t),count,f); fclose(f); return n==count;
}

int main(void){
    printf("Argmax test needs final logits exported later. For now argmax kernel is compiled only.\n");
    return 0;
}
