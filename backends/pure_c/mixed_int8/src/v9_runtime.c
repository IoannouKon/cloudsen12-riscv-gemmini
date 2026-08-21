#include "v9_runtime.h"
#include "v9_plan.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
typedef struct {void *p; int type,n,c,h,w;} Tensor;
static void *read_file(const char *root,const char *rel,size_t bytes){char p[4096];snprintf(p,sizeof(p),"%s/%s",root,rel);FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open failed %s\n",p);return 0;}void*x=malloc(bytes);if(!x){fclose(f);return 0;}size_t n=fread(x,1,bytes,f);fclose(f);if(n!=bytes){fprintf(stderr,"short read %s\n",p);free(x);return 0;}return x;}
static int32_t rq64(int64_t x,int32_t m,int s){int64_t p=x*(int64_t)m;int64_t q=(llabs(p)+(1LL<<30))>>31;if(p<0)q=-q;if(s>0)q<<=s;else if(s<0){int r=-s;int64_t a=llabs(q);a=(a+(1LL<<(r-1)))>>r;q=q<0?-a:a;}if(q>INT32_MAX)q=INT32_MAX;if(q<INT32_MIN)q=INT32_MIN;return(int32_t)q;}
static int8_t rq8(int64_t x,int32_t m,int s){int32_t q=rq64(x,m,s);return q>127?127:q<-127?-127:(int8_t)q;}
static int8_t fr8(int8_t x,double r){long q=lrint((double)x*r);return q>127?127:q<-127?-127:(int8_t)q;}
static int alloc8(Tensor*t,const Op*o){t->type=8;t->n=o->n;t->c=o->c;t->h=o->h;t->w=o->w;t->p=malloc((size_t)o->n*o->c*o->h*o->w);return t->p!=0;}
static int alloc32(Tensor*t,const Op*o){t->type=32;t->n=o->n;t->c=o->c;t->h=o->h;t->w=o->w;t->p=malloc((size_t)o->n*o->c*o->h*o->w*4);return t->p!=0;}
int v9_run(const char*ip,const char*root,const char*op){Tensor*T=calloc(NTENSORS,sizeof(*T));if(!T)return 2;FILE*f=fopen(ip,"rb");if(!f)return 3;T[0]=(Tensor){malloc(13*256*256),8,1,13,256,256};if(!T[0].p||fread(T[0].p,1,13*256*256,f)!=13*256*256){fclose(f);return 4;}fclose(f);
for(int oi=0;oi<NOPS;oi++){const Op*o=&OPS[oi];Tensor*a=&T[o->a],*z=&T[o->out];if(o->kind==OP_CONV){const Layer*l=&LAYERS[o->layer];size_t wc=(size_t)l->oc*l->icg*l->kh*l->kw;int8_t*w=read_file(root,l->wf,wc);int32_t*b=read_file(root,l->bf,l->oc*4),*mu=read_file(root,l->mf,l->oc*4),*sh=read_file(root,l->sf,l->oc*4);if(!w||!b||!mu||!sh)return 5;int last=o->layer==NLAYERS-1;if(!(last?alloc32(z,o):alloc8(z,o)))return 6;int8_t*x=a->p;int in_c=a->c,in_h=a->h,in_w=a->w;
#pragma omp parallel for collapse(2) schedule(static)
for(int oc=0;oc<l->oc;oc++)for(int oh=0;oh<o->h;oh++)for(int ow=0;ow<o->w;ow++){int g=oc/(l->oc/l->groups);int ic0=g*l->icg;int64_t acc=b[oc];for(int ic=0;ic<l->icg;ic++)for(int kh=0;kh<l->kh;kh++)for(int kw=0;kw<l->kw;kw++){int ih=oh*l->sh-l->ph+kh*l->dh,iw=ow*l->sw-l->pw+kw*l->dw;if((unsigned)ih<(unsigned)in_h&&(unsigned)iw<(unsigned)in_w){int8_t xv=x[((ic0+ic)*in_h+ih)*in_w+iw];xv=fr8(xv,o->r1);int8_t wv=w[(((oc*l->icg+ic)*l->kh+kh)*l->kw+kw)];acc+=(int32_t)xv*(int32_t)wv;}}size_t idx=((size_t)oc*o->h+oh)*o->w+ow;if(last)((int32_t*)z->p)[idx]=rq64(acc,mu[oc],sh[oc]);else((int8_t*)z->p)[idx]=rq8(acc,mu[oc],sh[oc]);}free(w);free(b);free(mu);free(sh);
}else if(o->kind==OP_RELU||o->kind==OP_RELU6){if(!alloc8(z,o))return 7;size_t n=(size_t)o->n*o->c*o->h*o->w;int8_t*x=a->p,*y=z->p;int hi=o->kind==OP_RELU6?o->m2:127;for(size_t i=0;i<n;i++){int v=x[i];y[i]=v<0?0:v>hi?hi:v;}
}else if(o->kind==OP_RESIZE){if(!alloc8(z,o))return 8;int8_t*x=a->p,*y=z->p;for(int c=0;c<o->c;c++)for(int h=0;h<o->h;h++)for(int w0=0;w0<o->w;w0++){int ih=h*a->h/o->h,iw=w0*a->w/o->w;y[(c*o->h+h)*o->w+w0]=x[(c*a->h+ih)*a->w+iw];}
}else if(o->kind==OP_ADD){Tensor*bb=&T[o->b];if(!alloc8(z,o))return 9;size_t n=(size_t)o->n*o->c*o->h*o->w;int8_t*x=a->p,*y=bb->p,*q=z->p;for(size_t i=0;i<n;i++){int v=fr8(x[i],o->r1)+fr8(y[i],o->r2);q[i]=v>127?127:v<-127?-127:v;}
}else if(o->kind==OP_CONCAT){Tensor*bb=&T[o->b];if(!alloc8(z,o))return 10;int8_t*q=z->p,*x=a->p,*y=bb->p;size_t plane=(size_t)o->h*o->w;for(int c=0;c<a->c;c++)for(size_t i=0;i<plane;i++)q[(size_t)c*plane+i]=fr8(x[(size_t)c*plane+i],o->r1);for(int c=0;c<bb->c;c++)for(size_t i=0;i<plane;i++)q[(size_t)(a->c+c)*plane+i]=fr8(y[(size_t)c*plane+i],o->r2);}}
Tensor*z=&T[FINAL_TENSOR];if(z->type!=32||z->c!=4)return 11;uint8_t*mask=malloc((size_t)z->h*z->w);int32_t*v=z->p;for(int p=0;p<z->h*z->w;p++){int best=0;for(int c=1;c<4;c++)if(v[c*z->h*z->w+p]>v[best*z->h*z->w+p])best=c;mask[p]=best;}f=fopen(op,"wb");if(!f)return 12;fwrite(mask,1,(size_t)z->h*z->w,f);fclose(f);free(mask);for(int i=0;i<NTENSORS;i++)free(T[i].p);free(T);return 0;}
