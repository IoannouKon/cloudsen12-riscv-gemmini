#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/User/cloudsen12_pure_c
cp -f pure_c_int8/tools/generate_plan.py pure_c_int8/tools/generate_plan.py.before_float_align
cp -f pure_c_int8/src/v9_runtime.c pure_c_int8/src/v9_runtime.c.before_float_align
python3 - <<'PY'
from pathlib import Path
p=Path('pure_c_int8/tools/generate_plan.py');s=p.read_text()
s=s.replace('int32_t m1,s1,m2,s2; int n,c,h,w;', 'int32_t m1,s1,m2,s2; double r1,r2; int n,c,h,w;')
s=s.replace("c.append('{%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d},'%(kind[k],oid,a,b,li,m1,s1,m2,s2,*sh))", "c.append('{%s,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d},'%(kind[k],oid,a,b,li,m1,s1,m2,s2,(scales[a]/(nextscale(name) or scales[a])) if k in ('ADD','CONCAT') else ((scales[a]/float(layers[li]['input_scale'])) if k=='CONV' else 1.0),(scales[b]/(nextscale(name) or scales[b])) if k in ('ADD','CONCAT') else 1.0,*sh))")
p.write_text(s)
PY
python3 - <<'PY'
from pathlib import Path
p=Path('pure_c_int8/src/v9_runtime.c');s=p.read_text()
s=s.replace('#include <limits.h>', '#include <limits.h>\n#include <math.h>')
s=s.replace('static int8_t rq8(int64_t x,int32_t m,int s){int32_t q=rq64(x,m,s);return q>127?127:q<-127?-127:(int8_t)q;}', 'static int8_t rq8(int64_t x,int32_t m,int s){int32_t q=rq64(x,m,s);return q>127?127:q<-127?-127:(int8_t)q;}\nstatic int8_t fr8(int8_t x,double r){long q=lrint((double)x*r);return q>127?127:q<-127?-127:(int8_t)q;}')
s=s.replace('xv=rq8(xv,o->m1,o->s1);', 'xv=fr8(xv,o->r1);')
s=s.replace('int v=rq8(x[i],o->m1,o->s1)+rq8(y[i],o->m2,o->s2);', 'int v=fr8(x[i],o->r1)+fr8(y[i],o->r2);')
s=s.replace('rq8(x[(size_t)c*plane+i],o->m1,o->s1)', 'fr8(x[(size_t)c*plane+i],o->r1)')
s=s.replace('rq8(y[(size_t)c*plane+i],o->m2,o->s2)', 'fr8(y[(size_t)c*plane+i],o->r2)')
p.write_text(s)
PY
python3 pure_c_int8/tools/generate_plan.py .
mkdir -p build pure_c_int8/results/float_align_final
gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra -Ipure_c_int8/include -Ipure_c_int8/generated pure_c_int8/src/main.c pure_c_int8/src/v9_runtime.c pure_c_int8/generated/v9_plan.c -lm -o build/cloudsen12_v9_int8_float_align
python3 pure_c_int8/tools/prepare_input.py . qat/data/test/patch_00000.npz pure_c_int8/results/float_align_final/input.bin >/dev/null
/usr/bin/time -f '%e,%M' -o pure_c_int8/results/float_align_final/time.csv ./build/cloudsen12_v9_int8_float_align pure_c_int8/results/float_align_final/input.bin qat/deployment/v9/approved_pure_c pure_c_int8/results/float_align_final/mask.bin
python3 - <<'PY'
import json,numpy as np
from pathlib import Path
m=np.fromfile('pure_c_int8/results/float_align_final/mask.bin',np.uint8)
r=np.fromfile('qat/deployment/v9/corrected_int8/validation/corrected_real_int8_mask.bin',np.uint8)
with np.load('qat/data/test/patch_00000.npz') as d:y=d['target'].astype(np.uint8).ravel()
z={'agreement_vs_approved_python_integer':float(np.mean(m==r)),'pixel_accuracy':float(np.mean(m==y)),'mismatches_vs_python':int(np.count_nonzero(m!=r)),'continue_25patch':bool(np.mean(m==r)>=.999)}
Path('pure_c_int8/results/float_align_final/patch0.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
PY
cat pure_c_int8/results/float_align_final/time.csv
