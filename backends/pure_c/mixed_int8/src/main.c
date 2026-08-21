#include "v9_runtime.h"
#include <stdio.h>
int main(int argc,char**argv){if(argc!=4){fprintf(stderr,"Usage: %s INPUT_INT8.bin PACKAGE_ROOT OUTPUT_U8.bin\n",argv[0]);return 2;}return v9_run(argv[1],argv[2],argv[3]);}
