#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/User/cloudsen12_pure_c
# Restore the generator from the automatic backup made before the failed edit.
cp -f pure_c_int8/tools/generate_plan.py.before_float_align pure_c_int8/tools/generate_plan.py
python3 - <<'PY'
from pathlib import Path
p=Path('pure_c_int8/tools/generate_plan.py')
s=p.read_text()
s=s.replace('int32_t m1,s1,m2,s2; int n,c,h,w;', 'int32_t m1,s1,m2,s2; double r1,r2; int n,c,h,w;')
s=s.replace("ops.append(('CONV',oid,ids[src],-1,i,m,s,0,0,shape))", "ops.append(('CONV',oid,ids[src],-1,i,m,s,0,0,ss/ts,1.0,shape))")
s=s.replace("ops.append(('RELU6',oid,ids[src],-1,-1,0,0,round(6/scales[src]),0,shape))", "ops.append(('RELU6',oid,ids[src],-1,-1,0,0,round(6/scales[src]),0,1.0,1.0,shape))")
s=s.replace("ops.append(('RELU',oid,ids[src],-1,-1,0,0,0,0,shape))", "ops.append(('RELU',oid,ids[src],-1,-1,0,0,0,0,1.0,1.0,shape))")
s=s.replace("ops.append(('ADD',oid,ids[a],ids[b],-1,m1,s1,m2,s2,shape))", "ops.append(('ADD',oid,ids[a],ids[b],-1,m1,s1,m2,s2,scales[a]/ts,scales[b]/ts,shape))")
s=s.replace("ops.append(('RESIZE',oid,ids[src],-1,-1,0,0,0,0,shape))", "ops.append(('RESIZE',oid,ids[src],-1,-1,0,0,0,0,1.0,1.0,shape))")
s=s.replace("ops.append(('CONCAT',oid,ids[a],ids[b],-1,m1,s1,m2,s2,shape))", "ops.append(('CONCAT',oid,ids[a],ids[b],-1,m1,s1,m2,s2,scales[a]/ts,scales[b]/ts,shape))")
s=s.replace("k,oid,a,b,li,m1,s1,m2,s2,sh=o;c.append('{%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d},'%(kind[k],oid,a,b,li,m1,s1,m2,s2,*sh))", "k,oid,a,b,li,m1,s1,m2,s2,r1,r2,sh=o;c.append('{%s,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d},'%(kind[k],oid,a,b,li,m1,s1,m2,s2,r1,r2,*sh))")
required=['double r1,r2','ss/ts,1.0','scales[a]/ts,scales[b]/ts','r1,r2,sh=o']
for x in required: assert x in s,x
p.write_text(s)
print('Generator float-ratio patch applied.')
PY
# Runtime was already patched by the previous package. Validate required changes.
python3 - <<'PY'
from pathlib import Path
s=Path('pure_c_int8/src/v9_runtime.c').read_text()
for x in ['static int8_t fr8','xv=fr8(xv,o->r1)','fr8(x[i],o->r1)','fr8(y[i],o->r2)']: assert x in s,x
print('Runtime float-alignment patch present.')
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
