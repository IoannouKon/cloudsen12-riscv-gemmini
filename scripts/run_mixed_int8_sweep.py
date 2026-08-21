#!/usr/bin/env python3
"""Mixed-precision static INT8 sweep for CloudSEN12.

Profiles quantize only selected Conv nodes while sensitive layers remain FP32.
Calibration and validation images are disjoint.
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


class Reader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.files = list(files)
        self.input_name = input_name
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.files):
            return None
        path = self.files[self.pos]
        self.pos += 1
        print(f"    calibration {self.pos:02d}/{len(self.files):02d}: {path.name}", flush=True)
        x = np.fromfile(path, dtype=np.float32).reshape(1, 13, 512, 512)
        return {self.input_name: x}

    def rewind(self):
        self.pos = 0


def attr_int(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def classify_convs(model):
    init = {x.name: x for x in model.graph.initializer}
    convs = []
    for graph_index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2 or node.input[1] not in init:
            continue
        dims = list(init[node.input[1]].dims)
        if len(dims) != 4:
            continue
        out_c, in_per_group, kh, kw = map(int, dims)
        group = attr_int(node, "group", 1)
        depthwise = group > 1 and in_per_group == 1
        convs.append({
            "node": node,
            "graph_index": graph_index,
            "name": node.name,
            "out_c": out_c,
            "in_per_group": in_per_group,
            "kh": kh,
            "kw": kw,
            "group": group,
            "depthwise": depthwise,
            "pointwise": kh == 1 and kw == 1 and group == 1,
        })
    return convs


def profile_nodes(convs, profile):
    first_name = convs[0]["name"]
    last_name = convs[-1]["name"]
    selected = []
    for item in convs:
        name = item["name"]
        protected = name in {first_name, last_name}
        if profile == "pointwise_only":
            use = item["pointwise"] and not protected
        elif profile == "standard_no_edges":
            use = item["group"] == 1 and not protected
        elif profile == "all_no_depthwise_edges":
            use = not item["depthwise"] and not protected
        elif profile == "all_no_edges":
            use = not protected
        else:
            raise ValueError(profile)
        if use:
            selected.append(name)
    return selected, first_name, last_name


def compare(pred, ref):
    pixels = int(np.count_nonzero(pred != ref))
    return pixels, float(pixels / ref.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--calibration-count", type=int, default=25)
    ap.add_argument("--validation-count", type=int, default=25)
    ap.add_argument("--profiles", nargs="+", default=[
        "pointwise_only",
        "standard_no_edges",
        "all_no_depthwise_edges",
        "all_no_edges",
    ])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    fp32_model = root / "models/cloudsen12_unetmobv2_v2.onnx"
    model_dir = root / "models/quantized"
    result_dir = root / "benchmark/results"
    config_dir = root / "config/quantization"
    for d in (model_dir, result_dir, config_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted((root / "benchmark/data").glob("*.bin"))
    required = args.calibration_count + args.validation_count
    if len(images) < required:
        raise SystemExit(f"Need {required} images; found {len(images)}")
    calibration = images[:args.calibration_count]
    validation = images[args.calibration_count:required]

    model = onnx.load(str(fp32_model))
    input_name = model.graph.input[0].name
    convs = classify_convs(model)
    print(f"Found {len(convs)} Conv nodes.", flush=True)

    print("Loading Python/PyTorch FP32 reference once...", flush=True)
    py_model = cloudsen12.load_model_by_name(
        name="UNetMobV2_V2",
        weights_folder=str(root / "weights"),
        device=torch.device("cpu"),
    )

    print("Computing validation reference masks once...", flush=True)
    refs = []
    python_times = []
    validation_inputs = []
    for i, path in enumerate(validation, 1):
        x = np.fromfile(path, dtype=np.float32).reshape(1, 13, 512, 512)
        t0 = time.perf_counter()
        pred = py_model.predict(x[0]).astype(np.uint8)
        seconds = time.perf_counter() - t0
        validation_inputs.append(x)
        refs.append(pred)
        python_times.append(seconds)
        print(f"  FP32 reference {i:02d}/{len(validation)}: {seconds:.4f}s", flush=True)

    all_profile_summaries = []
    all_rows = []

    for profile in args.profiles:
        selected, first_name, last_name = profile_nodes(convs, profile)
        output_model = model_dir / f"cloudsen12_mixed_int8_{profile}_qdq.onnx"
        print("\nProfile:", profile, flush=True)
        print("  selected Conv nodes:", len(selected), flush=True)
        print("  protected first Conv:", first_name, flush=True)
        print("  protected final Conv:", last_name, flush=True)

        reader = Reader(calibration, input_name)
        quantize_static(
            model_input=str(fp32_model),
            model_output=str(output_model),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_quantize=selected,
            per_channel=True,
            reduce_range=False,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "DedicatedQDQPair": True,
            },
        )

        session = ort.InferenceSession(str(output_model), providers=["CPUExecutionProvider"])
        q_input = session.get_inputs()[0].name
        q_output = session.get_outputs()[0].name

        total_mismatch = 0
        total_seconds = 0.0
        intersections = np.zeros(4, dtype=np.int64)
        unions = np.zeros(4, dtype=np.int64)
        profile_rows = []

        for i, (path, x, ref) in enumerate(zip(validation, validation_inputs, refs), 1):
            t0 = time.perf_counter()
            logits = session.run([q_output], {q_input: x})[0]
            seconds = time.perf_counter() - t0
            pred = np.argmax(logits, axis=1).squeeze(0).astype(np.uint8)
            pixels, rate = compare(pred, ref)
            total_mismatch += pixels
            total_seconds += seconds
            for cls in range(4):
                a = ref == cls
                b = pred == cls
                intersections[cls] += np.count_nonzero(a & b)
                unions[cls] += np.count_nonzero(a | b)
            row = {
                "profile": profile,
                "image": path.name,
                "quantized_conv_nodes": len(selected),
                "int8_seconds": seconds,
                "mismatched_pixels": pixels,
                "mismatch_rate": rate,
            }
            profile_rows.append(row)
            all_rows.append(row)
            print(
                f"  [{i:02d}/{len(validation)}] int8={seconds:.4f}s | "
                f"mismatch={rate:.4%} ({pixels} px)",
                flush=True,
            )

        total_pixels = len(validation) * 512 * 512
        iou = [float(intersections[c] / unions[c]) if unions[c] else 1.0 for c in range(4)]
        summary = {
            "profile": profile,
            "model": str(output_model.relative_to(root)),
            "activation": "QUInt8 asymmetric per tensor, static MinMax",
            "weights": "QInt8 symmetric per output channel",
            "quantized_conv_nodes": len(selected),
            "total_conv_nodes": len(convs),
            "calibration_images": len(calibration),
            "validation_images": len(validation),
            "python_average_seconds": float(np.mean(python_times)),
            "int8_average_seconds": total_seconds / len(validation),
            "speedup_vs_python": float(np.mean(python_times)) / (total_seconds / len(validation)),
            "total_mismatched_pixels": total_mismatch,
            "overall_mismatch_rate": total_mismatch / total_pixels,
            "max_image_mismatch_rate": max(r["mismatch_rate"] for r in profile_rows),
            "class_iou_against_fp32": {
                "clear": iou[0],
                "thick_cloud": iou[1],
                "thin_cloud": iou[2],
                "cloud_shadow": iou[3],
            },
        }
        all_profile_summaries.append(summary)
        (config_dir / f"mixed_int8_{profile}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print("  profile summary:", json.dumps(summary, indent=2), flush=True)

    all_profile_summaries.sort(key=lambda x: (x["overall_mismatch_rate"], -x["speedup_vs_python"]))
    final = {
        "recommended_profile": all_profile_summaries[0]["profile"],
        "selection_rule": "lowest overall mismatch, then highest speedup",
        "profiles": all_profile_summaries,
    }

    csv_path = result_dir / "benchmark_mixed_int8_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    json_path = result_dir / "benchmark_mixed_int8_sweep_summary.json"
    json_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    print("\nFINAL MIXED-PRECISION SWEEP", flush=True)
    print(json.dumps(final, indent=2), flush=True)
    print("CSV:", csv_path, flush=True)
    print("JSON:", json_path, flush=True)


if __name__ == "__main__":
    main()
