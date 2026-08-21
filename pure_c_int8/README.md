# CloudSEN12 V9 pure-C INT8

This package generates a fixed 256x256 execution plan from the approved 63-layer manifest and the 566-node graph, builds a Linux pure-C executable, prepares an INT8 input from `patch_00000.npz`, runs all Conv/depthwise Conv, ReLU/ReLU6, Add, Concat, nearest resize, and INT32 Argmax operations, and checks the C mask against the saved Python integer mask.

## Install from PowerShell

```powershell
Expand-Archive -Path "Download CloudSEN12 V9 pure C INT8.zip" -DestinationPath "." -Force
& ".\cloudsen12_v9_pure_c_int8\install.ps1"
```

## Build and validate in WSL

```bash
cd /mnt/c/Users/User/cloudsen12_pure_c
sudo apt install -y python3-numpy
bash pure_c_int8/build_and_validate_wsl.sh
```

## Benchmark pure-C INT8

```bash
bash pure_c_int8/benchmark_int8_wsl.sh 10
```

The existing FP32 benchmark remains `python benchmark/benchmark_50.py --root . --count 50`. It uses 512x512 images, while the approved V9 INT8 graph is fixed at 256x256. Do not compare raw latency as a fair precision-only speedup until all four backends use the same input dimensions and timing boundary.
