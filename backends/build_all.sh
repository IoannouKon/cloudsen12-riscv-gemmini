#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo
echo "[1/3] Building ONNX/OpenCV C++ FP32..."

if [ -f backends/onnx_cpp_fp32/build.sh ]; then
    bash backends/onnx_cpp_fp32/build.sh
elif [ -f backends/cpp_onnx/run_onnx_opencv_cli.cpp ]; then
    mkdir -p build

    g++ \
      -O3 \
      -march=native \
      -std=c++17 \
      backends/cpp_onnx/run_onnx_opencv_cli.cpp \
      -o build/cloudsen12_onnx_cpp_fp32 \
      $(pkg-config --cflags --libs opencv4)
else
    echo "ERROR: ONNX C++ source was not found."
    exit 1
fi

echo
echo "[2/3] Building pure-C FP32..."

bash backends/pure_c_fp32/build.sh

echo
echo "[3/3] Building pure-C mixed INT8..."

bash backends/pure_c_mixed_int8/build.sh

echo
echo "========================================"
echo "ALL COMPILED BACKENDS BUILT"
echo "========================================"

ls -lh \
  build/cloudsen12_onnx_cpp_fp32 \
  build/cloudsen12_pure_c_fp32 \
  build/cloudsen12_full_c \
  build/cloudsen12_pure_c_mixed_int8 \
  build/cloudsen12_v9_int8_approved
