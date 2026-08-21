#!/usr/bin/env python3
import csv,json,time
from pathlib import Path
import numpy as np,torch
from cloudsen12_models import cloudsen12
R=Path.cwd(); O=R/'benchmark/results/final_four_backend'; O.mkdir(parents=True,exist_ok=True)
model=cloudsen12.load_model_by_name(name='UNetMobV2_V2',weights_folder=str(R/'weights'),device=torch.device('cpu'))
rows=[]
for i,p in enumerate(sorted((R/'benchmark/data').glob('*.bin'))[:25],1):
 x=np.fromfile(p,np.float32).reshape(13,512,512); t=time.perf_counter(); y=model.predict(x).astype(np.uint8); dt=time.perf_counter()-t; y.tofile(O/f'{p.stem}_pytorch.bin'); rows.append((p.name,dt)); print(f'[PyTorch FP32 512] {i}/25 {dt:.6f}s',flush=True)
with (O/'pytorch_fp32.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['image','seconds']);w.writerows(rows)
# Same 25 256 patches for the INT8 agreement/target comparison.
rows=[]
for i,p in enumerate(sorted((R/'qat/data/test').glob('patch_*.npz'))[:25],1):
 with np.load(p) as d:x=d['image'].astype(np.float32)/10000.;target=d['target'].astype(np.uint8)
 t=time.perf_counter();y=model.predict(x).astype(np.uint8);dt=time.perf_counter()-t;y.tofile(O/f'{p.stem}_pytorch_256.bin');target.tofile(O/f'{p.stem}_target_256.bin');rows.append((p.name,dt));print(f'[PyTorch FP32 256] {i}/25 {dt:.6f}s',flush=True)
with (O/'pytorch_fp32_256.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['image','seconds']);w.writerows(rows)
