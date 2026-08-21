#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BACKEND="$ROOT/backends/pure_c"

cd "$ROOT"

mkdir -p build

echo "Building pure-C FP32..."

gcc \
  -O3 \
  -march=native \
  -std=c11 \
  -I"$BACKEND/fp32/include" \
  "$BACKEND/fp32/src/cloudsen12_full_c.c" \
  "$BACKEND/fp32/src/conv2d.c" \
  "$BACKEND/fp32/src/clip.c" \
  "$BACKEND/fp32/src/relu.c" \
  "$BACKEND/fp32/src/add.c" \
  "$BACKEND/fp32/src/concat.c" \
  "$BACKEND/fp32/src/resize.c" \
  "$BACKEND/fp32/src/argmax.c" \
  -lm \
  -o build/cloudsen12_full_c

echo "Building pure-C mixed INT8..."

gcc \
  -O3 \
  -march=native \
  -fopenmp \
  -std=c11 \
  -I"$BACKEND/mixed_int8/include" \
  -I"$BACKEND/mixed_int8/generated" \
  "$BACKEND/mixed_int8/src/main.c" \
  "$BACKEND/mixed_int8/src/v9_runtime.c" \
  "$BACKEND/mixed_int8/generated/v9_plan.c" \
  -lm \
  -o build/cloudsen12_v9_int8_approved

echo
echo "Build complete:"
ls -lh \
  build/cloudsen12_full_c \
  build/cloudsen12_v9_int8_approved
