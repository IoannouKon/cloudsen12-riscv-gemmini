#!/usr/bin/env python3
"""Calibrate a static INT8 QDQ CloudSEN12 ONNX model and benchmark it.

Uses the first N images for calibration and the next M images for validation.
The FP32 Python/PyTorch model is the correctness reference.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from cloudsen12_models import cloudsen12
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class BinCalibrationReader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.files = list(files)
        self.input_name = input_name
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.files):
            return None
        path = self.files[self.pos]
        self.pos += 1
        print(f"  calibration {self.pos:02d}/{len(self.files):02d}: {path.name}", flush=True)
        x = np.fromfile(path, dtype=np.float32).reshape(1, 13, 512, 512)
        return {self.input_name: x}

    def rewind(self):
        self.pos = 0


def calibration_method(name):
    return {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[name]


def mismatch(pred, ref):
    count = int(np.count_nonzero(pred != ref))
    return count, float(count / ref.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--calibration-count", type=int, default=25)
    parser.add_argument("--validation-count", type=int, default=25)
    parser.add_argument(
        "--method",
        choices=["minmax", "entropy", "percentile"],
        default="percentile",
    )
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fp32_model = root / "models/cloudsen12_unetmobv2_v2.onnx"
    data_dir = root / "benchmark/data"
    model_dir = root / "models/quantized"
    result_dir = root / "benchmark/results"
    config_dir = root / "config/quantization"
    for directory in (model_dir, result_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_images = sorted(data_dir.glob("*.bin"))
    required = args.calibration_count + args.validation_count
    if len(all_images) < required:
        raise SystemExit(
            f"Need at least {required} images in {data_dir}; found {len(all_images)}."
        )
    calibration_files = all_images[: args.calibration_count]
    validation_files = all_images[
        args.calibration_count : args.calibration_count + args.validation_count
    ]

    suffix = args.method
    if args.method == "percentile":
        suffix += "_" + str(args.percentile).replace(".", "p")
    int8_model = model_dir / f"cloudsen12_unetmobv2_v2_int8_{suffix}_qdq.onnx"

    if not args.skip_quantize:
        if not fp32_model.exists():
            raise SystemExit(f"Missing FP32 model: {fp32_model}")
        model = onnx.load(str(fp32_model))
        input_name = model.graph.input[0].name
        reader = BinCalibrationReader(calibration_files, input_name)

        extra_options = {
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "DedicatedQDQPair": True,
        }
        if args.method == "percentile":
            extra_options["CalibPercentile"] = args.percentile

        print("Creating static INT8 QDQ model", flush=True)
        print("  source:", fp32_model, flush=True)
        print("  output:", int8_model, flush=True)
        print("  method:", args.method, flush=True)
        print("  calibration images:", len(calibration_files), flush=True)

        quantize_static(
            model_input=str(fp32_model),
            model_output=str(int8_model),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=calibration_method(args.method),
            op_types_to_quantize=["Conv"],
            per_channel=True,
            reduce_range=False,
            extra_options=extra_options,
        )
        print("Static INT8 model created.", flush=True)
    elif not int8_model.exists():
        raise SystemExit(f"--skip-quantize was used but model is missing: {int8_model}")

    print("Loading Python/PyTorch FP32 reference once...", flush=True)
    py_model = cloudsen12.load_model_by_name(
        name="UNetMobV2_V2",
        weights_folder=str(root / "weights"),
        device=torch.device("cpu"),
    )

    print("Loading static INT8 ONNX model once...", flush=True)
    int8_session = ort.InferenceSession(
        str(int8_model), providers=["CPUExecutionProvider"]
    )
    input_name = int8_session.get_inputs()[0].name
    output_name = int8_session.get_outputs()[0].name

    rows = []
    python_total = 0.0
    int8_total = 0.0
    class_intersections = np.zeros(4, dtype=np.int64)
    class_unions = np.zeros(4, dtype=np.int64)

    print("Validating on images not used for calibration...", flush=True)
    for idx, image_path in enumerate(validation_files, start=1):
        x = np.fromfile(image_path, dtype=np.float32).reshape(1, 13, 512, 512)

        start = time.perf_counter()
        py_pred = py_model.predict(x[0]).astype(np.uint8)
        py_seconds = time.perf_counter() - start
        python_total += py_seconds

        start = time.perf_counter()
        logits = int8_session.run([output_name], {input_name: x})[0]
        int8_seconds = time.perf_counter() - start
        int8_total += int8_seconds
        int8_pred = np.argmax(logits, axis=1).squeeze(0).astype(np.uint8)

        diff_pixels, diff_rate = mismatch(int8_pred, py_pred)
        for cls in range(4):
            ref_cls = py_pred == cls
            pred_cls = int8_pred == cls
            class_intersections[cls] += np.count_nonzero(ref_cls & pred_cls)
            class_unions[cls] += np.count_nonzero(ref_cls | pred_cls)

        rows.append(
            {
                "image": image_path.name,
                "python_seconds": py_seconds,
                "static_int8_seconds": int8_seconds,
                "mismatched_pixels": diff_pixels,
                "mismatch_rate": diff_rate,
            }
        )
        print(
            f"[{idx:02d}/{len(validation_files)}] "
            f"py={py_seconds:.4f}s int8={int8_seconds:.4f}s | "
            f"mismatch={diff_rate:.6%} ({diff_pixels} px)",
            flush=True,
        )

    total_pixels = len(rows) * 512 * 512
    total_mismatch = sum(row["mismatched_pixels"] for row in rows)
    class_iou = [
        float(class_intersections[c] / class_unions[c]) if class_unions[c] else 1.0
        for c in range(4)
    ]

    csv_path = result_dir / f"benchmark_static_int8_{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "quantized_model": str(int8_model.relative_to(root)),
        "format": "QDQ",
        "quantized_ops": ["Conv"],
        "activation": "QInt8 symmetric per tensor, statically calibrated",
        "weights": "QInt8 symmetric per output channel",
        "accumulation": "INT32 in quantized Conv kernels",
        "calibration_method": args.method,
        "percentile": args.percentile if args.method == "percentile" else None,
        "calibration_images": len(calibration_files),
        "validation_images": len(validation_files),
        "python_total_seconds": python_total,
        "static_int8_total_seconds": int8_total,
        "python_average_seconds": python_total / len(rows),
        "static_int8_average_seconds": int8_total / len(rows),
        "total_mismatched_pixels": total_mismatch,
        "overall_mismatch_rate": total_mismatch / total_pixels,
        "max_image_mismatch_rate": max(row["mismatch_rate"] for row in rows),
        "class_iou_against_fp32": {
            "clear": class_iou[0],
            "thick_cloud": class_iou[1],
            "thin_cloud": class_iou[2],
            "cloud_shadow": class_iou[3],
        },
    }

    json_path = result_dir / f"benchmark_static_int8_{suffix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    config_path = config_dir / f"static_int8_{suffix}.json"
    config_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nSummary", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("CSV:", csv_path, flush=True)
    print("JSON:", json_path, flush=True)


if __name__ == "__main__":
    main()
