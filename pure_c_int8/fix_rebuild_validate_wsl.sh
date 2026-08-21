#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/User/cloudsen12_pure_c
src=pure_c_int8/src/v9_runtime.c
cp -f "$src" "$src.before_int64_fix.bak"
python3 - <<'PY'
from pathlib import Path
p=Path('pure_c_int8/src/v9_runtime.c')
s=p.read_text()
s=s.replace('static int8_t rq8(int32_t x,int32_t m,int s)', 'static int8_t rq8(int64_t x,int32_t m,int s)')
s=s.replace('rq8((int32_t)acc,mu[oc],sh[oc])', 'rq8(acc,mu[oc],sh[oc])')
p.write_text(s)
assert 'static int8_t rq8(int64_t x' in s
assert 'rq8(acc,mu[oc],sh[oc])' in s
print('Applied INT64 accumulator requantization fix.')
PY
python3 pure_c_int8/tools/generate_plan.py .
mkdir -p build pure_c_int8/results/q31_fix
gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra \
  -Ipure_c_int8/include -Ipure_c_int8/generated \
  pure_c_int8/src/main.c pure_c_int8/src/v9_runtime.c \
  pure_c_int8/generated/v9_plan.c -lm -o build/cloudsen12_v9_int8
python3 pure_c_int8/tools/prepare_input.py . qat/data/test/patch_00000.npz pure_c_int8/results/q31_fix/patch_00000_int8.bin
/usr/bin/time -f '%e,%M' -o pure_c_int8/results/q31_fix/time.csv \
  ./build/cloudsen12_v9_int8 \
  pure_c_int8/results/q31_fix/patch_00000_int8.bin \
  qat/deployment/v9/approved_pure_c \
  pure_c_int8/results/q31_fix/patch_00000_c_mask.bin
python3 - <<'PY'
from pathlib import Path
import json, numpy as np
c=np.fromfile('pure_c_int8/results/q31_fix/patch_00000_c_mask.bin',np.uint8)
ref=Path('qat/deployment/v9/corrected_int8/validation/corrected_real_int8_mask.bin')
r=np.fromfile(ref,np.uint8) if ref.exists() else None
result={'mask_pixels':int(c.size),'mask_bytes':int(c.nbytes),'int64_accumulator_fix_applied':True}
if r is not None:
 result['agreement_vs_old_float_alignment_reference']=float(np.mean(c==r))
 result['mismatched_pixels_vs_old_reference']=int(np.count_nonzero(c!=r))
result['note']='Old Python mask used float inter-tensor alignment, so less than 100% here is expected. This run tests the INT64 accumulator correction.'
Path('pure_c_int8/results/q31_fix/result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
echo 'Timing seconds,max_rss_kb:'
cat pure_c_int8/results/q31_fix/time.csv
echo 'Saved: pure_c_int8/results/q31_fix/result.json'
