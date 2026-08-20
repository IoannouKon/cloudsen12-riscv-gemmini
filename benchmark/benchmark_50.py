#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from cloudsen12_models import cloudsen12


def mismatch(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def read_pure_c_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        magic = file.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unexpected PGM format: {magic!r}")

        dimensions = file.readline().strip()
        while dimensions.startswith(b"#"):
            dimensions = file.readline().strip()

        width, height = map(int, dimensions.split())
        max_value = int(file.readline().strip())

        if width != 512 or height != 512 or max_value != 255:
            raise ValueError(
                f"Unexpected PGM metadata: {width}x{height}, max={max_value}"
            )

        image = np.frombuffer(file.read(), dtype=np.uint8)

    if image.size != width * height:
        raise ValueError(
            f"Unexpected PGM size: {image.size}, expected {width * height}"
        )

    return (image.reshape(height, width) // 80).astype(np.uint8)


def run_command(command, cwd: Path):
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, command))}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return elapsed, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--cpp-bin", default="backends/cpp_onnx/run_onnx_opencv_cli")
    ap.add_argument("--pure-c-bin", default="build/cloudsen12_full_c")
    ap.add_argument("--model", default="models/cloudsen12_unetmobv2_v2.onnx")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    data_dir = root / "benchmark/data"
    result_dir = root / "benchmark/results"
    result_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(data_dir.glob("*.bin"))[: args.count]
    if len(images) < args.count:
        raise SystemExit(f"Need {args.count} images in {data_dir}, found {len(images)}")

    # Python baseline: one model load, then time only model.predict for each image.
    print("Loading Python/PyTorch CloudSEN12 model once...")
    py_model = cloudsen12.load_model_by_name(
        name="UNetMobV2_V2",
        weights_folder=str(root / "weights"),
        device=torch.device("cpu"),
    )

    rows = []
    py_outputs = {}
    py_total = 0.0
    cpp_total = 0.0
    c_total = 0.0

    # Save current canonical files because the pure-C binary uses stable root-relative paths.
    canonical_input = root / "inputs/input_1x13x512x512_float32.bin"
    canonical_ref = root / "inputs/pytorch_pred_512x512_uint8.bin"
    backup_input = canonical_input.read_bytes() if canonical_input.exists() else None
    backup_ref = canonical_ref.read_bytes() if canonical_ref.exists() else None

    try:
        for i, image_path in enumerate(images, start=1):
            x = np.fromfile(image_path, dtype=np.float32).reshape(1, 13, 512, 512)

            t0 = time.perf_counter()
            py_pred = py_model.predict(x[0]).astype(np.uint8)
            py_time = time.perf_counter() - t0
            py_total += py_time
            py_outputs[image_path.name] = py_pred
            py_ref_path = result_dir / f"{image_path.stem}_python.bin"
            py_pred.tofile(py_ref_path)

            cpp_out = result_dir / f"{image_path.stem}_cpp.bin"
            cpp_time, _ = run_command(
                [str(root / args.cpp_bin), str(root / args.model), str(image_path), str(cpp_out)],
                root,
            )
            cpp_total += cpp_time
            cpp_pred = np.fromfile(cpp_out, dtype=np.uint8).reshape(512, 512)

            # The generated pure-C program currently uses canonical file paths.
            shutil.copyfile(image_path, canonical_input)
            shutil.copyfile(py_ref_path, canonical_ref)
            c_time, c_stdout = run_command([str(root / args.pure_c_bin)], root)
            c_total += c_time
            c_pred_path = root / "outputs/predictions/pure_c_full_pred.pgm"
            c_pred = read_pure_c_pgm(c_pred_path)

            c_mismatch_pixels = int(np.count_nonzero(c_pred != py_pred))
            c_mismatch_rate = float(c_mismatch_pixels / py_pred.size)
            c_exact = int(c_mismatch_pixels == 0)

            cpp_mismatch_pixels = int(np.count_nonzero(cpp_pred != py_pred))
            cpp_mismatch_rate = float(cpp_mismatch_pixels / py_pred.size)

            row = {
                "image": image_path.name,
                "python_seconds": py_time,
                "cpp_onnx_seconds": cpp_time,
                "pure_c_seconds": c_time,
                "cpp_mismatched_pixels": cpp_mismatch_pixels,
                "cpp_vs_python_mismatch": cpp_mismatch_rate,
                "pure_c_mismatched_pixels": c_mismatch_pixels,
                "pure_c_vs_python_mismatch": c_mismatch_rate,
                "pure_c_vs_python_exact": c_exact,
            }
            rows.append(row)
            print(
                f"[{i:02d}/{len(images)}] {image_path.name} | "
                f"py={py_time:.4f}s cpp={cpp_time:.4f}s c={c_time:.4f}s | "
                f"cpp mismatch={cpp_mismatch_rate:.8f} "
                f"({cpp_mismatch_pixels} px) | "
                f"c mismatch={c_mismatch_rate:.8f} "
                f"({c_mismatch_pixels} px)"
            )
    finally:
        if backup_input is not None:
            canonical_input.write_bytes(backup_input)
        if backup_ref is not None:
            canonical_ref.write_bytes(backup_ref)

    csv_path = result_dir / "benchmark_50.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = result_dir / "benchmark_50_summary.txt"
    summary = (
        f"images={len(images)}\n"
        f"python_total_seconds={py_total:.6f}\n"
        f"cpp_onnx_total_seconds={cpp_total:.6f}\n"
        f"pure_c_total_seconds={c_total:.6f}\n"
        f"python_avg_seconds={py_total/len(images):.6f}\n"
        f"cpp_onnx_avg_seconds={cpp_total/len(images):.6f}\n"
        f"pure_c_avg_seconds={c_total/len(images):.6f}\n"
        f"max_cpp_vs_python_mismatch={max(r['cpp_vs_python_mismatch'] for r in rows):.10f}\n"
        f"total_cpp_mismatched_pixels={sum(r['cpp_mismatched_pixels'] for r in rows)}\n"
        f"max_pure_c_vs_python_mismatch={max(r['pure_c_vs_python_mismatch'] for r in rows):.10f}\n"
        f"total_pure_c_mismatched_pixels={sum(r['pure_c_mismatched_pixels'] for r in rows)}\n"
        f"pure_c_exact_count={sum(r['pure_c_vs_python_exact'] for r in rows)}/{len(images)}\n"
        "timing_note=Python model is loaded once; C++ and pure C timings include process startup and file IO.\n"
    )
    summary_path.write_text(summary)
    print("\n" + summary)
    print("CSV:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
