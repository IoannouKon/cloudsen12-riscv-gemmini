#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 pure_c_int8/tools/generate_plan.py .
mkdir -p build pure_c_int8/results
gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra -Ipure_c_int8/include -Ipure_c_int8/generated pure_c_int8/src/main.c pure_c_int8/src/v9_runtime.c pure_c_int8/generated/v9_plan.c -lm -o build/cloudsen12_v9_int8
python3 pure_c_int8/tools/prepare_input.py . qat/data/test/patch_00000.npz pure_c_int8/test_input_int8.bin
/usr/bin/time -v ./build/cloudsen12_v9_int8 pure_c_int8/test_input_int8.bin qat/deployment/v9/approved_pure_c pure_c_int8/results/c_mask.bin 2> pure_c_int8/results/time.txt
python3 - <<'PY'
from pathlib import Path
import numpy as np
c=np.fromfile('pure_c_int8/results/c_mask.bin',np.uint8)
refs=[Path('qat/deployment/v9/corrected_int8/validation/corrected_real_int8_mask.bin'),Path('qat/deployment/v9/integer_reference/full_run/real_int8_mask_patch_00000.bin')]
r=next((p for p in refs if p.exists()),None)
print('C mask pixels:',c.size)
if r:
 x=np.fromfile(r,np.uint8);m=np.mean(c!=x);print('Reference:',r);print('Agreement:',1-m,'mismatched pixels:',np.count_nonzero(c!=x));
 if m: raise SystemExit('C mask is not bit-exact')
else: print('Reference mask not found; C execution completed but bit-exact comparison skipped.')
PY
echo 'Built: build/cloudsen12_v9_int8'
cat pure_c_int8/results/time.txt
