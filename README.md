# CloudSEN12 Edge AI on RISC-V and Gemmini

A reproducible hardware/software pipeline for onboard satellite cloud detection, progressing from a pretrained CloudSEN12 model to Python, C++/ONNX, full pure-C inference, and future RISC-V and Gemmini acceleration.

## Project Goal

The long-term goal is to run cloud detection onboard a CubeSat-class system using an open-source RISC-V SoC and the Gemmini neural-network accelerator in Chipyard.

The project is inspired by onboard Earth-observation filtering concepts such as ESA Phi-Sat-1 and CloudScout, where cloud-covered images can be detected before downlink. Because an official public CloudScout model is not used here, this repository uses the open CloudSEN12 `UNetMobV2_V2` model with real 13-band Sentinel-2 L1C samples.

The planned hardware path is:

```text
CloudSEN12 model
        |
Python/PyTorch reference
        |
ONNX export
        |
C++ ONNX inference
        |
Full pure-C inference
        |
RISC-V CPU implementation
        |
Chipyard/Verilator simulation
        |
Gemmini acceleration
        |
Optional FPGA prototyping
```

## Current Status

## Quick Start

### 1. Clone and enter the repository

```bash
git clone https://github.com/IoannouKon/cloudsen12-riscv-gemmini.git

cd cloudsen12-riscv-gemmini


### Completed

- Real 13-band Sentinel-2 CloudSEN12 samples loaded locally
- Original Python/PyTorch inference validated
- PyTorch model exported to ONNX
- C++ inference implemented with OpenCV DNN and ONNX
- Pure-C kernels implemented for:
  - Conv2D and grouped/depthwise Conv2D
  - Clip/ReLU6
  - ReLU
  - Add
  - Channel concatenation
  - Nearest-neighbor resize
  - Argmax
- Full `UNetMobV2_V2` forward pass generated and executed in pure C
- Pure-C output compared against Python/PyTorch
- Three-backend benchmark created for Python, C++ ONNX, and pure C

### Verified Full Pure-C Result

```text
Input shape:  1 x 13 x 512 x 512
Logits shape: 1 x 4 x 512 x 512

Mean absolute logits error vs ONNX Runtime: 0.0000151485
Maximum absolute logits error vs ONNX Runtime: 0.0001773834
Pure-C cloud fraction: 0.8547668457
Mismatch vs PyTorch mask: 0.0000000000
```

For the validated real Sentinel-2 sample, the final pure-C segmentation mask matched the PyTorch mask exactly.

## Task and Classes

This is pixel-level semantic segmentation, not only whole-image binary classification.

Each pixel is assigned one of four classes:

```text
0 = clear
1 = thick cloud
2 = thin cloud
3 = cloud shadow
```

An image-level decision can be derived from the fraction of pixels classified as thick or thin cloud.

## Backends

```text
backends/python_original/   Original Python/PyTorch workflow
backends/cpp_onnx/          C++ OpenCV DNN backend using ONNX
backends/pure_c/            Links to the full pure-C backend
src/                        Pure-C implementation and generated forward pass
include/                    Pure-C headers
```

## Repository Structure

```text
.
├── backends/
│   ├── python_original/
│   ├── cpp_onnx/
│   └── pure_c/
├── benchmark/
│   ├── data/
│   ├── results/
│   └── logs/
├── build/
├── config/
├── docs/
├── experiments/
├── gemmini/
├── include/
├── inputs/
├── models/
├── outputs/
├── riscv/
├── scripts/
├── src/
└── weights/
```

## Run the Full Pure-C Model

Run from the repository root.

### Build

```bash
gcc -O2 -Wall -Wextra -std=c11 -Iinclude \
  src/cloudsen12_full_c.c \
  src/conv2d.c \
  src/clip.c \
  src/relu.c \
  src/add.c \
  src/concat.c \
  src/resize.c \
  src/argmax.c \
  -lm \
  -o build/cloudsen12_full_c
```

### Execute

```bash
./build/cloudsen12_full_c
```

The generated prediction is written to:

```text
outputs/predictions/pure_c_full_pred.pgm
```

## Three-Backend Benchmark

Prepare 50 real CloudSEN12 samples:

```bash
python backends/python_original/prepare_50_images.py \
  --count 50 \
  --out benchmark/data
```

Compile the C++ ONNX backend:

```bash
g++ -O2 -std=c++17 \
  backends/cpp_onnx/run_onnx_opencv_cli.cpp \
  -o backends/cpp_onnx/run_onnx_opencv_cli \
  $(pkg-config --cflags --libs opencv4)
```

Run the comparison:

```bash
python benchmark/benchmark_50.py --root . --count 50
```

The benchmark reports:

- Inference time for each backend
- Total and average time over all images
- Mismatch rate against Python/PyTorch
- Number of mismatched pixels

The current pure-C implementation is a correctness-oriented reference and uses naive nested-loop convolution. It is expected to be substantially slower than optimized PyTorch and OpenCV backends.

## Planned RISC-V Work

The next implementation phase will add:

```text
riscv/
├── linux/
├── baremetal/
├── linker/
└── tests/
```

Planned tasks:

1. Cross-compile the pure-C kernels for RISC-V.
2. Run CPU-only inference on RISC-V Linux.
3. Adapt file I/O and memory allocation for bare-metal execution.
4. Run in Chipyard/Verilator.
5. Measure latency, instruction count, cycles, and memory use.

## Planned Gemmini Work

The Gemmini phase will focus on convolution acceleration:

1. Convert convolution into im2col plus GEMM or another Gemmini-compatible mapping.
2. Quantize weights and activations to INT8.
3. Compare FP32 pure C, INT8 RISC-V CPU, and INT8 Gemmini results.
4. Measure latency, throughput, model size, and accuracy loss.
5. Optionally explore FPGA prototyping on a compatible board.

## Research Questions

- How much faster is Gemmini than RISC-V CPU-only inference?
- What accuracy loss is introduced by INT8 quantization?
- What memory capacity is required for onboard inference?
- Can early cloud filtering reduce unnecessary downlink volume?
- Which CloudSEN12 layers dominate runtime and benefit most from acceleration?

## Important Notes

- The ONNX model and raw weight files may be managed with Git LFS or release assets.
- Benchmark datasets and generated predictions are intentionally excluded from normal Git history.
- Commands should be executed from the repository root because current runtime paths are root-relative.
- The current x86 pure-C version prioritizes correctness and portability, not speed.

## Author

Kostis Ioannou  
Electrical and Computer Engineering, National Technical University of Athens  
GitHub: https://github.com/IoannouKon
