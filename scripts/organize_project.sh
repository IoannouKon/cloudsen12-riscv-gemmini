#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/cloudsen12_pure_c}"
cd "$ROOT"

mkdir -p backends/python_original backends/cpp_onnx backends/pure_c benchmark/data benchmark/results benchmark/logs

# Preserve the working pure-C project and expose it through a clear backend folder.
ln -sfn ../../src backends/pure_c/src
ln -sfn ../../include backends/pure_c/include
ln -sfn ../../weights backends/pure_c/weights
ln -sfn ../../build/cloudsen12_full_c backends/pure_c/cloudsen12_full_c

# Link shared model and input assets rather than duplicating large files.
ln -sfn ../../models/cloudsen12_unetmobv2_v2.onnx backends/cpp_onnx/cloudsen12_unetmobv2_v2.onnx
ln -sfn ../../models/cloudsen12_unetmobv2_v2.onnx backends/python_original/cloudsen12_unetmobv2_v2.onnx

cat > backends/pure_c/README.md <<'EOF'
# Pure C backend

The implementation remains in the project-level `src/`, `include/`, and `weights/` folders. This directory provides stable links to those files and the final executable.

Run from the repository root:

```bash
./build/cloudsen12_full_c
```
EOF

printf 'Organized benchmark/backend folders under: %s\n' "$ROOT"
printf 'No working source, weights, or model files were moved.\n'
