#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BACKEND="$ROOT/backends/pure_c_fp32"

cd "$ROOT"

mkdir -p build

gcc \
  -O3 \
  -march=native \
  -std=c11 \
  -I"$BACKEND/include" \
  "$BACKEND/src/cloudsen12_full_c.c" \
  "$BACKEND/src/conv2d.c" \
  "$BACKEND/src/clip.c" \
  "$BACKEND/src/relu.c" \
  "$BACKEND/src/add.c" \
  "$BACKEND/src/concat.c" \
  "$BACKEND/src/resize.c" \
  "$BACKEND/src/argmax.c" \
  -lm \
  -o build/cloudsen12_pure_c_fp32

cp -f \
  build/cloudsen12_pure_c_fp32 \
  build/cloudsen12_full_c

echo "Built:"
ls -lh \
  build/cloudsen12_pure_c_fp32 \
  build/cloudsen12_full_c
