#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/User/cloudsen12_pure_c
O=benchmark/results/final_four_backend; mkdir -p "$O"
# ONNX C++ FP32 and Pure-C FP32 on identical 512x512 benchmark images.
echo 'image,seconds,max_rss_kb,mismatched_pixels,total_pixels' > "$O/cpp_onnx_fp32.csv"
echo 'image,seconds,max_rss_kb,mismatched_pixels,total_pixels' > "$O/pure_c_fp32.csv"
backup_i=$(mktemp); backup_r=$(mktemp); cp inputs/input_1x13x512x512_float32.bin "$backup_i"; cp inputs/pytorch_pred_512x512_uint8.bin "$backup_r"
trap 'cp "$backup_i" inputs/input_1x13x512x512_float32.bin; cp "$backup_r" inputs/pytorch_pred_512x512_uint8.bin; rm -f "$backup_i" "$backup_r"' EXIT
n=0
for input in $(find benchmark/data -maxdepth 1 -name '*.bin' | sort | head -25); do
 n=$((n+1)); stem=$(basename "$input" .bin); py="$O/${stem}_pytorch.bin"; cpp="$O/${stem}_cpp.bin"
 /usr/bin/time -f '%e,%M' -o "$O/t.time" backends/cpp_onnx/run_onnx_opencv_cli models/cloudsen12_unetmobv2_v2.onnx "$input" "$cpp"
 v=$(cat "$O/t.time"); sec=${v%,*}; mem=${v#*,}; mm=$(python3 - "$py" "$cpp" <<'PY'
import sys,numpy as np
a=np.fromfile(sys.argv[1],np.uint8);b=np.fromfile(sys.argv[2],np.uint8);print(np.count_nonzero(a!=b))
PY
); echo "$stem,$sec,$mem,$mm,262144" >> "$O/cpp_onnx_fp32.csv"
 cp "$input" inputs/input_1x13x512x512_float32.bin; cp "$py" inputs/pytorch_pred_512x512_uint8.bin
 /usr/bin/time -f '%e,%M' -o "$O/t.time" build/cloudsen12_full_c >/dev/null
 v=$(cat "$O/t.time"); sec=${v%,*}; mem=${v#*,}; python3 - "$O/${stem}_pure_c.bin" <<'PY'
import sys,numpy as np
p='outputs/predictions/pure_c_full_pred.pgm'
with open(p,'rb') as f:
 assert f.readline().strip()==b'P5'; line=f.readline()
 while line.startswith(b'#'):line=f.readline()
 w,h=map(int,line.split());assert int(f.readline())==255;a=np.frombuffer(f.read(),np.uint8).reshape(h,w)//80
a.astype(np.uint8).tofile(sys.argv[1])
PY
 mm=$(python3 - "$py" "$O/${stem}_pure_c.bin" <<'PY'
import sys,numpy as np
a=np.fromfile(sys.argv[1],np.uint8);b=np.fromfile(sys.argv[2],np.uint8);print(np.count_nonzero(a!=b))
PY
); echo "$stem,$sec,$mem,$mm,262144" >> "$O/pure_c_fp32.csv"; echo "[WSL FP32] $n/25"
done
# Approved mixed INT8 C on 256x256 patches, compared with PyTorch FP32 and target.
echo 'image,seconds,max_rss_kb,mismatched_vs_pytorch,correct_vs_pytorch,target_correct,total_pixels' > "$O/pure_c_mixed_int8.csv"
for n in $(seq 0 24); do i=$(printf '%05d' "$n"); inp="pure_c_int8/results/q31_25patch/inputs/patch_${i}_int8.bin"; out="$O/patch_${i}_int8.bin"; py="$O/patch_${i}_pytorch_256.bin"; tg="$O/patch_${i}_target_256.bin"
 /usr/bin/time -f '%e,%M' -o "$O/t.time" build/cloudsen12_v9_int8_approved "$inp" qat/deployment/v9/approved_pure_c "$out"; v=$(cat "$O/t.time"); sec=${v%,*}; mem=${v#*,}
 vals=$(python3 - "$out" "$py" "$tg" <<'PY'
import sys,numpy as np
a=np.fromfile(sys.argv[1],np.uint8);p=np.fromfile(sys.argv[2],np.uint8);t=np.fromfile(sys.argv[3],np.uint8);m=np.count_nonzero(a!=p);print(f'{m},{a.size-m},{np.count_nonzero(a==t)}')
PY
); echo "patch_${i},$sec,$mem,$vals,65536" >> "$O/pure_c_mixed_int8.csv"; echo "[WSL INT8] $((n+1))/25"
done
python3 - <<'PY'
import csv,json,statistics
from pathlib import Path
O=Path('benchmark/results/final_four_backend')
def read(n):
 with (O/n).open(newline='') as f:return list(csv.DictReader(f))
def times(n):return [float(x['seconds']) for x in read(n)]
def base(n,totalkey='total_pixels',badkey='mismatched_pixels'):
 r=read(n); bad=sum(int(x[badkey]) for x in r); total=sum(int(x[totalkey]) for x in r); ts=[float(x['seconds']) for x in r]; mem=[int(x['max_rss_kb']) for x in r]
 return {'images':len(r),'correct_vs_pytorch':total-bad,'wrong_vs_pytorch':bad,'agreement_vs_pytorch':(total-bad)/total,'mean_seconds':statistics.mean(ts),'median_seconds':statistics.median(ts),'maximum_peak_rss_mib':max(mem)/1024}
py=times('pytorch_fp32.csv'); py256=times('pytorch_fp32_256.csv'); ir=read('pure_c_mixed_int8.csv'); bad=sum(int(x['mismatched_vs_pytorch']) for x in ir); total=sum(int(x['total_pixels']) for x in ir); target=sum(int(x['target_correct']) for x in ir); its=[float(x['seconds']) for x in ir]; im=[int(x['max_rss_kb']) for x in ir]
res={'pytorch_fp32_512x512':{'images':25,'reference':True,'correct_vs_pytorch':25*262144,'wrong_vs_pytorch':0,'agreement_vs_pytorch':1.0,'mean_seconds':statistics.mean(py),'median_seconds':statistics.median(py)},'opencv_onnx_cpp_fp32_512x512':base('cpp_onnx_fp32.csv'),'pure_c_fp32_512x512':base('pure_c_fp32.csv'),'pure_c_mixed_int8_256x256':{'images':25,'correct_vs_pytorch':total-bad,'wrong_vs_pytorch':bad,'agreement_vs_pytorch':(total-bad)/total,'target_correct_pixels':target,'target_wrong_pixels':total-target,'target_pixel_accuracy':target/total,'mean_seconds':statistics.mean(its),'median_seconds':statistics.median(its),'maximum_peak_rss_mib':max(im)/1024},'pytorch_fp32_256x256_timing':{'images':25,'mean_seconds':statistics.mean(py256),'median_seconds':statistics.median(py256)},'notes':['512x512 FP32 backends are directly comparable.','256x256 mixed INT8 is reported separately because dimensions differ.','PyTorch memory is not included because it runs in Windows Python/CUDA while Linux RSS measures WSL processes.']}
(O/'final_summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
PY
