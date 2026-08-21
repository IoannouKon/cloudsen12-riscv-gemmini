import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1] if len(sys.argv)>1 else '.')
patch=Path(sys.argv[2] if len(sys.argv)>2 else root/'qat/data/test/patch_00000.npz')
out=Path(sys.argv[3] if len(sys.argv)>3 else root/'pure_c_int8/test_input_int8.bin')
man=json.loads((root/'qat/deployment/v9/approved_pure_c/corrected_int8_manifest.json').read_text())
with np.load(patch) as d:x=d['image'].astype(np.float32)/10000.0
s=float(man['layers'][0]['input_scale']);q=np.clip(np.rint(x/s),-127,127).astype(np.int8);out.parent.mkdir(parents=True,exist_ok=True);q.tofile(out);print(out,q.shape,s)
