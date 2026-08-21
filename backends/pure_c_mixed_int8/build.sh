#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BACKEND="$ROOT/backends/pure_c_mixed_int8/runtime"

cd "$ROOT"

mkdir -p build

gcc \
  -O3 \
  -march=native \
  -fopenmp \
  -std=c11 \
  -I"$BACKEND/include" \
  -I"$BACKEND/generated" \
  "$BACKEND/src/main.c" \
  "$BACKEND/src/v9_runtime.c" \
  "$BACKEND/generated/v9_plan.c" \
  -lm \
  -o build/cloudsen12_pure_c_mixed_int8

cp -f \
  build/cloudsen12_pure_c_mixed_int8 \
  build/cloudsen12_v9_int8_approved

echo "Built:"
ls -lh \
  build/cloudsen12_pure_c_mixed_int8 \
  build/cloudsen12_v9_int8_approved
