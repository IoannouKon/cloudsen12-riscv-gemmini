#!/usr/bin/env python3
"""Build assets/backends and run the complete CloudSEN12 four-backend benchmark.

Run from the repository root with the cloudsen12 Conda environment active:
    python scripts/run_all_benchmarks.py

Optional quick smoke test:
    python scripts/run_all_benchmarks.py --count 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], root: Path, *, env=None) -> None:
    print("\n+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)


def require(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(f"ERROR: {message}\nMissing: {path}")


def replace_benchmark_settings(script: Path, count: int) -> None:
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3:
        raise SystemExit(f"ERROR: Invalid benchmark script: {script}")
    lines[2] = 'cd "$(cd "$(dirname "$0")/../.." && pwd)"'
    text = "\n".join(lines) + "\n"
    text = text.replace("head -25", f"head -{count}")
    text = text.replace("head -2", f"head -{count}")
    text = text.replace("seq 0 24", f"seq 0 {count - 1}")
    text = text.replace("seq 0 1", f"seq 0 {count - 1}")
    # Ensure summary uses actual row counts rather than a hard-coded 25.
    text = text.replace("'images':25,'reference':True", "'images':len(py),'reference':True")
    text = text.replace("'correct_vs_pytorch':25*262144", "'correct_vs_pytorch':len(py)*262144")
    text = text.replace("'images':25,'correct_vs_pytorch':total-bad", "'images':len(ir),'correct_vs_pytorch':total-bad")
    text = text.replace("'images':25,'mean_seconds':statistics.mean(py256)", "'images':len(py256),'mean_seconds':statistics.mean(py256)")
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | 0o111)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root")
    ap.add_argument("--count", type=int, default=25, help="Number of images/patches")
    ap.add_argument("--skip-pytorch", action="store_true", help="Reuse saved PyTorch masks/timings")
    args = ap.parse_args()
    if args.count < 1 or args.count > 25:
        raise SystemExit("ERROR: --count must be between 1 and 25")

    root = Path(args.root).resolve()
    os.chdir(root)
    print(f"Repository: {root}")
    print(f"Python: {sys.executable}")
    print(f"Images: {args.count}")

    require(root / "models/cloudsen12_unetmobv2_v2.onnx", "ONNX model is required")
    require(root / "weights/UNetMobV2_V2.pt", "PyTorch model checkpoint is required")
    require(root / "scripts/export_weights_and_manifest.py", "FP32 weight exporter is required")
    require(root / "scripts/generate_partial_forward_v2.py", "partial-reference generator is required")
    require(root / "scripts/benchmark/run_windows_pytorch_final.py", "PyTorch benchmark script is required")
    require(root / "scripts/benchmark/run_wsl_backends_final.sh", "C/C++ benchmark script is required")

    for directory in [
        "build", "outputs/predictions", "benchmark/results/final_four_backend",
        "experiments/partial_forward/partial_forward_v2",
    ]:
        (root / directory).mkdir(parents=True, exist_ok=True)

    # Generate FP32 C assets when absent.
    weight_bins = list((root / "weights").glob("*.bin"))
    partial_ref = root / "experiments/partial_forward/partial_forward_v2/partial_ref.bin"
    if len(weight_bins) < 126:
        run([sys.executable, "scripts/export_weights_and_manifest.py"], root)
    if not partial_ref.exists():
        run([sys.executable, "scripts/generate_partial_forward_v2.py"], root)

    # Build OpenCV/ONNX C++ backend.
    opencv_flags = subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "opencv4"], text=True
    ).split()
    run([
        "g++", "-O3", "-march=native", "-std=c++17",
        "backends/cpp_onnx/run_onnx_opencv_cli.cpp",
        "-o", "backends/cpp_onnx/run_onnx_opencv_cli",
        *opencv_flags,
    ], root)

    # Build pure-C FP32 backend.
    run([
        "gcc", "-O3", "-march=native", "-std=c11", "-Ibackends/pure_c_fp32/include",
        "backends/pure_c_fp32/src/cloudsen12_full_c.c", "backends/pure_c_fp32/src/conv2d.c", "backends/pure_c_fp32/src/clip.c",
        "backends/pure_c_fp32/src/relu.c", "backends/pure_c_fp32/src/add.c", "backends/pure_c_fp32/src/concat.c", "backends/pure_c_fp32/src/resize.c",
        "backends/pure_c_fp32/src/argmax.c", "-lm", "-o", "build/cloudsen12_full_c",
    ], root)

    # Build approved mixed-INT8 backend if sources are available.
    int8_bin = root / "build/cloudsen12_v9_int8_approved"
    int8_sources = [
        root / "backends/pure_c_mixed_int8/runtime/src/main.c",
        root / "backends/pure_c_mixed_int8/runtime/src/v9_runtime.c",
        root / "backends/pure_c_mixed_int8/runtime/generated/v9_plan.c",
    ]
    if all(p.exists() for p in int8_sources):
        run([
            "gcc", "-O3", "-march=native", "-fopenmp", "-std=c11",
            "-Ibackends/pure_c_mixed_int8/runtime/include", "-Ibackends/pure_c_mixed_int8/runtime/generated",
            "backends/pure_c_mixed_int8/runtime/src/main.c", "backends/pure_c_mixed_int8/runtime/src/v9_runtime.c",
            "backends/pure_c_mixed_int8/runtime/generated/v9_plan.c", "-lm",
            "-o", str(int8_bin.relative_to(root)),
        ], root)
    require(int8_bin, "Approved mixed-INT8 executable could not be built or found")

    # Test pure-C FP32 before the long benchmark.
    first_input = next(iter(sorted((root / "benchmark/data").glob("*.bin"))), None)
    require(first_input or root / "missing", "No benchmark/data/*.bin inputs found")

    # Configure portable benchmark script and requested count.
    backend_script = root / "scripts/benchmark/run_wsl_backends_final.sh"
    replace_benchmark_settings(backend_script, args.count)

    if not args.skip_pytorch:
        env = os.environ.copy()
        env["CLOUDSEN12_BENCHMARK_COUNT"] = str(args.count)
        # Existing PyTorch script is fixed at 25, so make a temporary count-adjusted copy.
        src = root / "scripts/benchmark/run_windows_pytorch_final.py"
        tmp = root / "benchmark/run_pytorch_current.py"
        text = src.read_text(encoding="utf-8")
        text = text.replace("[:25]", f"[:{args.count}]")
        text = text.replace("/25", f"/{args.count}")
        tmp.write_text(text, encoding="utf-8")
        try:
            run([sys.executable, "-u", str(tmp.relative_to(root))], root, env=env)
        finally:
            tmp.unlink(missing_ok=True)

    run(["bash", str(backend_script.relative_to(root))], root)

    summary = root / "benchmark/results/final_four_backend/final_summary.json"
    require(summary, "Final benchmark summary was not created")
    data = json.loads(summary.read_text(encoding="utf-8"))
    print("\n================ FINAL RESULTS ================")
    print(json.dumps(data, indent=2))
    print(f"\nSaved: {summary}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"\nERROR: command failed with exit code {exc.returncode}") from exc
