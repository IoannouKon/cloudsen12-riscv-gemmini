#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FP32="backends/pure_c_fp32"
INT8="backends/pure_c_mixed_int8/runtime"

echo "========================================"
echo "Removing legacy root compatibility links"
echo "Repository: $ROOT"
echo "========================================"

echo
echo "[1/7] Verifying authoritative backend sources..."

required=(
  "$FP32/src/cloudsen12_full_c.c"
  "$FP32/src/conv2d.c"
  "$FP32/src/clip.c"
  "$FP32/src/relu.c"
  "$FP32/src/add.c"
  "$FP32/src/concat.c"
  "$FP32/src/resize.c"
  "$FP32/src/argmax.c"
  "$FP32/include/conv2d.h"
  "$INT8/src/main.c"
  "$INT8/src/v9_runtime.c"
  "$INT8/include/v9_runtime.h"
  "$INT8/generated/v9_plan.c"
  "$INT8/generated/v9_plan.h"
  "$INT8/tools/generate_plan.py"
  "$INT8/tools/prepare_input.py"
)

for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: Required backend file is missing:"
        echo "$path"
        exit 1
    fi
done

echo "Authoritative backend sources verified."

echo
echo "[2/7] Creating safety backup of scripts..."

BACKUP="/tmp/cloudsen12_before_link_removal_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$BACKUP" \
  scripts \
  backends \
  2>/dev/null

echo "Backup: $BACKUP"

echo
echo "[3/7] Updating unified benchmark runner..."

python - <<'PY'
from pathlib import Path

path = Path("scripts/run_all_benchmarks.py")
text = path.read_text(encoding="utf-8")

replacements = {
    '"src/cloudsen12_full_c.c"':
        '"backends/pure_c_fp32/src/cloudsen12_full_c.c"',
    '"src/conv2d.c"':
        '"backends/pure_c_fp32/src/conv2d.c"',
    '"src/clip.c"':
        '"backends/pure_c_fp32/src/clip.c"',
    '"src/relu.c"':
        '"backends/pure_c_fp32/src/relu.c"',
    '"src/add.c"':
        '"backends/pure_c_fp32/src/add.c"',
    '"src/concat.c"':
        '"backends/pure_c_fp32/src/concat.c"',
    '"src/resize.c"':
        '"backends/pure_c_fp32/src/resize.c"',
    '"src/argmax.c"':
        '"backends/pure_c_fp32/src/argmax.c"',
    '"-Iinclude"':
        '"-Ibackends/pure_c_fp32/include"',
    'root / "pure_c_int8/src/main.c"':
        'root / "backends/pure_c_mixed_int8/runtime/src/main.c"',
    'root / "pure_c_int8/src/v9_runtime.c"':
        'root / "backends/pure_c_mixed_int8/runtime/src/v9_runtime.c"',
    'root / "pure_c_int8/generated/v9_plan.c"':
        'root / "backends/pure_c_mixed_int8/runtime/generated/v9_plan.c"',
    '"-Ipure_c_int8/include"':
        '"-Ibackends/pure_c_mixed_int8/runtime/include"',
    '"-Ipure_c_int8/generated"':
        '"-Ibackends/pure_c_mixed_int8/runtime/generated"',
    '"pure_c_int8/src/main.c"':
        '"backends/pure_c_mixed_int8/runtime/src/main.c"',
    '"pure_c_int8/src/v9_runtime.c"':
        '"backends/pure_c_mixed_int8/runtime/src/v9_runtime.c"',
    '"pure_c_int8/generated/v9_plan.c"':
        '"backends/pure_c_mixed_int8/runtime/generated/v9_plan.c"',
}

for old, new in replacements.items():
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("Updated:", path)
PY

echo
echo "[4/7] Updating shell benchmark paths..."

python - <<'PY'
from pathlib import Path

path = Path("scripts/benchmark/run_wsl_backends_final.sh")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "pure_c_int8/results/q31_25patch/inputs/",
    "backends/pure_c_mixed_int8/runtime/"
    "results/q31_25patch/inputs/",
)

path.write_text(text, encoding="utf-8")
path.chmod(path.stat().st_mode | 0o111)

print("Updated:", path)
PY

echo
echo "[5/7] Updating asset generators..."

python - <<'PY'
from pathlib import Path

generator_files = [
    Path("scripts/generate_partial_forward.py"),
    Path("scripts/generate_partial_forward_v2.py"),
    Path("scripts/export_first_conv_test.py"),
]

replacements = {
    '"src/': '"backends/pure_c_fp32/src/',
    "'src/": "'backends/pure_c_fp32/src/",
    '"include/': '"backends/pure_c_fp32/include/',
    "'include/": "'backends/pure_c_fp32/include/",
}

for path in generator_files:
    if not path.exists():
        continue

    text = path.read_text(encoding="utf-8")

    for old, new in replacements.items():
        text = text.replace(old, new)

    path.write_text(text, encoding="utf-8")
    print("Updated:", path)
PY

echo
echo "[6/7] Checking remaining source-path references..."

grep -RInE \
  '(^|[^A-Za-z0-9_])(src/|include/|pure_c_int8/)' \
  scripts \
  backends \
  --include='*.py' \
  --include='*.sh' \
  --include='*.md' \
  2>/dev/null \
  > /tmp/cloudsen12_remaining_legacy_paths.txt || true

echo "Remaining references before link deletion:"

cat /tmp/cloudsen12_remaining_legacy_paths.txt || true

echo
echo "[7/7] Removing root compatibility links..."

for link in src include pure_c_int8; do
    if [ -L "$link" ]; then
        rm "$link"
        echo "Removed symbolic link: $link"
    elif [ -e "$link" ]; then
        echo "ERROR: $link exists but is not a symbolic link."
        echo "It was not removed."
        exit 1
    else
        echo "Already absent: $link"
    fi
done

echo
echo "Compiling Python scripts..."

python -m py_compile \
  scripts/run_all_benchmarks.py \
  scripts/generate_partial_forward.py \
  scripts/generate_partial_forward_v2.py

echo
echo "========================================"
echo "LEGACY LINKS REMOVED"
echo "========================================"

echo
echo "Current top level:"

ls

echo
echo "Authoritative source locations:"

ls -ld \
  backends/pure_c_fp32/src \
  backends/pure_c_fp32/include \
  backends/pure_c_mixed_int8/runtime

echo
echo "Safety backup:"
echo "$BACKUP"
