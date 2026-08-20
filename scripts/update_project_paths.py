#!/usr/bin/env python3

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Exact old root-relative paths and their new locations.
REPLACEMENTS = {
    # Main models
    "models/cloudsen12_unetmobv2_v2_shapes.onnx":
        "models/cloudsen12_unetmobv2_v2_shapes.onnx",

    "models/runtime_shapes_all_outputs.onnx":
        "models/runtime_shapes_all_outputs.onnx",

    "models/runtime_shapes_float_outputs.onnx":
        "models/runtime_shapes_float_outputs.onnx",

    "models/cloudsen12_unetmobv2_v2.onnx":
        "models/cloudsen12_unetmobv2_v2.onnx",

    # Inputs and references
    "inputs/input_1x13x512x512_float32.bin":
        "inputs/input_1x13x512x512_float32.bin",

    "inputs/pytorch_pred_512x512_uint8.bin":
        "inputs/pytorch_pred_512x512_uint8.bin",

    # Configuration
    "config/c_execution_plan.json":
        "config/c_execution_plan.json",

    "config/model_manifest.json":
        "config/model_manifest.json",

    "config/runtime_shapes.json":
        "config/runtime_shapes.json",

    # Operator experiments
    "experiments/operator_tests/first_add/":
        "experiments/operator_tests/first_add/",

    "experiments/operator_tests/first_clip/":
        "experiments/operator_tests/first_clip/",

    "experiments/operator_tests/first_concat/":
        "experiments/operator_tests/first_concat/",

    "experiments/operator_tests/first_conv/":
        "experiments/operator_tests/first_conv/",

    "experiments/operator_tests/first_relu/":
        "experiments/operator_tests/first_relu/",

    "experiments/operator_tests/first_resize/":
        "experiments/operator_tests/first_resize/",

    "experiments/operator_tests/final_argmax/":
        "experiments/operator_tests/final_argmax/",

    # Partial-forward experiments
    "experiments/partial_forward/partial_forward_v2/":
        "experiments/partial_forward/partial_forward_v2/",

    "experiments/partial_forward/partial_forward_logs/":
        "experiments/partial_forward/partial_forward_logs/",

    "experiments/partial_forward/partial_forward/":
        "experiments/partial_forward/partial_forward/",

    "experiments/partial_forward/partial_forward_exploration.csv":
        "experiments/partial_forward/partial_forward_exploration.csv",

    # Predictions
    "outputs/predictions/pure_c_argmax_pred.pgm":
        "outputs/predictions/pure_c_argmax_pred.pgm",

    "outputs/predictions/pure_c_full_pred.pgm":
        "outputs/predictions/pure_c_full_pred.pgm",
}

TEXT_EXTENSIONS = {
    ".c", ".h", ".py", ".md", ".json",
    ".txt", ".cmake", ""
}

# Do not modify historical reports or backups.
SKIP_DIRECTORIES = {
    ".git",
    "build",
    "outputs",
    "weights",
    "src.backup_before_paths",
    "scripts.backup_before_paths",
    "config.backup_before_paths",
    "__pycache__",
}

changed_files = []


def replace_root_relative(text: str, old: str, new: str) -> str:
    """
    Replace old root-relative paths without changing an already-updated path.

    For example, this prevents:
        models/cloudsen12... -> models/models/cloudsen12...
    """

    pattern = rf"(?<![A-Za-z0-9_./]){re.escape(old)}"
    return re.sub(pattern, new, text)


for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    relative = path.relative_to(ROOT)

    if any(part in SKIP_DIRECTORIES for part in relative.parts):
        continue

    if path.suffix not in TEXT_EXTENSIONS:
        continue

    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    updated = original

    # Longer paths first to avoid partial replacements.
    for old, new in sorted(
        REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        updated = replace_root_relative(updated, old, new)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        changed_files.append(str(relative))


# Explicitly ensure the model field in the manifest is correct.
manifest_path = ROOT / "config" / "config/model_manifest.json"

if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "models/cloudsen12_unetmobv2_v2.onnx"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

print("Path migration complete.")
print("Changed files:", len(changed_files))

for filename in changed_files:
    print("  ", filename)

print()
print("Manifest model path:")
print("  models/cloudsen12_unetmobv2_v2.onnx")