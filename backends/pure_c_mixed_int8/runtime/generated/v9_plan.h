#pragma once
#include <stdint.h>
typedef enum {OP_CONV,OP_RELU6,OP_RELU,OP_ADD,OP_RESIZE,OP_CONCAT} OpKind;
typedef struct {int kind,out,a,b,layer; int32_t m1,s1,m2,s2; double r1,r2; int n,c,h,w;} Op;
typedef struct {int oc,icg,kh,kw,sh,sw,ph,pw,dh,dw,groups; const char *wf,*bf,*mf,*sf;} Layer;
extern const Op OPS[]; extern const int NOPS,NTENSORS; extern const Layer LAYERS[]; extern const int NLAYERS,FINAL_TENSOR;
