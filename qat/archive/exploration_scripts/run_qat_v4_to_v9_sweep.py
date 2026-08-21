#!/usr/bin/env python3

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd()
SOURCE = ROOT / "train_qat_int8_v4.py"
V4_CHECKPOINT = ROOT / "qat/checkpoints/qat_int8_v4_best.pth"
V4_RESULTS = ROOT / "qat/results/qat_int8_v4_training.json"

LOG_DIRECTORY = ROOT / "qat/logs"
RESULT_DIRECTORY = ROOT / "qat/results"
CHECKPOINT_DIRECTORY = ROOT / "qat/checkpoints"

LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "version": "v5",
        "soft_teacher": 0.70,
        "hard_teacher": 0.25,
        "ground_truth": 0.05,
        "learning_rate": "3e-8",
        "epochs": 6,
    },
    {
        "version": "v6",
        "soft_teacher": 0.50,
        "hard_teacher": 0.45,
        "ground_truth": 0.05,
        "learning_rate": "5e-8",
        "epochs": 6,
    },
    {
        "version": "v7",
        "soft_teacher": 0.75,
        "hard_teacher": 0.20,
        "ground_truth": 0.05,
        "learning_rate": "1e-7",
        "epochs": 6,
    },
    {
        "version": "v8",
        "soft_teacher": 0.55,
        "hard_teacher": 0.40,
        "ground_truth": 0.05,
        "learning_rate": "2e-8",
        "epochs": 6,
    },
    {
        "version": "v9",
        "soft_teacher": 0.65,
        "hard_teacher": 0.30,
        "ground_truth": 0.05,
        "learning_rate": "1e-8",
        "epochs": 8,
    },
]


def replace_once(text, old, new, description):
    if old not in text:
        raise RuntimeError(
            f"Could not find {description} in Version 4 source."
        )

    return text.replace(old, new, 1)


def build_experiment(experiment):
    version = experiment["version"]
    source_text = SOURCE.read_text(encoding="utf-8-sig")

    script_path = ROOT / f"train_qat_int8_{version}.py"

    source_text = re.sub(
        r"lr=[0-9.eE+-]+,",
        f'lr={experiment["learning_rate"]},',
        source_text,
        count=1,
    )

    source_text = re.sub(
        r"epochs\s*=\s*\d+",
        f'epochs = {experiment["epochs"]}',
        source_text,
        count=1,
    )

    old_loss = """            loss = (
                0.60 * distillation_loss
                + 0.35 * teacher_class_loss
                + 0.05 * ground_truth_loss
            )"""

    new_loss = f"""            loss = (
                {experiment["soft_teacher"]:.2f} * distillation_loss
                + {experiment["hard_teacher"]:.2f} * teacher_class_loss
                + {experiment["ground_truth"]:.2f} * ground_truth_loss
            )"""

    source_text = replace_once(
        source_text,
        old_loss,
        new_loss,
        "Version 4 loss block",
    )

    source_text = replace_once(
        source_text,
        '"qat/checkpoints/qat_int8_v2_best.pth"',
        '"qat/checkpoints/qat_int8_v4_best.pth"',
        "Version 4 input checkpoint path",
    )

    source_text = source_text.replace(
        "qat/checkpoints/qat_int8_v4_best.pth",
        f"qat/checkpoints/qat_int8_{version}_best.pth",
    )

    # Restore the input checkpoint after replacing output paths.
    marker = """version_2_checkpoint = Path(
        f"qat/checkpoints/qat_int8_{version}_best.pth"
    )"""

    restored_marker = """version_2_checkpoint = Path(
        "qat/checkpoints/qat_int8_v4_best.pth"
    )"""

    source_text = source_text.replace(
        marker,
        restored_marker,
        1,
    )

    # Handle the variable if it was renamed in the existing V4 code.
    marker_alt = f"""version_2_checkpoint = Path(
        "qat/checkpoints/qat_int8_{version}_best.pth"
    )"""

    source_text = source_text.replace(
        marker_alt,
        restored_marker,
        1,
    )

    source_text = source_text.replace(
        "qat/results/qat_int8_v4_training.json",
        f"qat/results/qat_int8_{version}_training.json",
    )

    script_path.write_text(
        source_text,
        encoding="utf-8",
    )

    compile(
        source_text,
        str(script_path),
        "exec",
    )

    return script_path


def run_experiment(experiment, script_path):
    version = experiment["version"]
    log_path = LOG_DIRECTORY / f"qat_int8_{version}.log"

    print()
    print("=" * 72)
    print(f"STARTING {version.upper()}")
    print(json.dumps(experiment, indent=2))
    print("Script:", script_path)
    print("Log:", log_path)
    print("=" * 72)

    command = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    with log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{version} failed with exit code {return_code}. "
            f"See {log_path}"
        )

    print(f"{version.upper()} completed successfully.")


def extract_best(version, result_path):
    data = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    epochs = data.get("epochs", [])

    if not epochs:
        raise RuntimeError(
            f"No epoch records found in {result_path}"
        )

    best_mismatch = min(
        epochs,
        key=lambda entry: entry[
            "mismatch_vs_fp32_teacher"
        ],
    )

    best_iou = max(
        epochs,
        key=lambda entry: entry["mean_iou"],
    )

    return {
        "version": version,
        "best_mismatch_epoch": best_mismatch["epoch"],
        "mismatch_vs_fp32": best_mismatch[
            "mismatch_vs_fp32_teacher"
        ],
        "agreement_vs_fp32": (
            1.0
            - best_mismatch[
                "mismatch_vs_fp32_teacher"
            ]
        ),
        "pixel_accuracy_at_best_mismatch": best_mismatch[
            "pixel_accuracy"
        ],
        "mean_iou_at_best_mismatch": best_mismatch[
            "mean_iou"
        ],
        "clear_iou_at_best_mismatch": best_mismatch[
            "clear_iou"
        ],
        "thick_cloud_iou_at_best_mismatch": best_mismatch[
            "thick_cloud_iou"
        ],
        "thin_cloud_iou_at_best_mismatch": best_mismatch[
            "thin_cloud_iou"
        ],
        "cloud_shadow_iou_at_best_mismatch": best_mismatch[
            "cloud_shadow_iou"
        ],
        "best_iou_epoch": best_iou["epoch"],
        "best_mean_iou": best_iou["mean_iou"],
        "mismatch_at_best_iou": best_iou[
            "mismatch_vs_fp32_teacher"
        ],
        "result_file": str(result_path),
    }


def main():
    if not SOURCE.exists():
        raise SystemExit(
            f"Missing source trainer: {SOURCE}"
        )

    if not V4_CHECKPOINT.exists():
        raise SystemExit(
            f"Missing Version 4 checkpoint: {V4_CHECKPOINT}"
        )

    if not V4_RESULTS.exists():
        raise SystemExit(
            f"Missing Version 4 results: {V4_RESULTS}"
        )

    print("Version 4 source:", SOURCE)
    print("Version 4 checkpoint:", V4_CHECKPOINT)
    print("Version 4 results:", V4_RESULTS)

    summaries = [
        extract_best("v4", V4_RESULTS)
    ]

    for experiment in EXPERIMENTS:
        version = experiment["version"]
        result_path = (
            RESULT_DIRECTORY
            / f"qat_int8_{version}_training.json"
        )

        script_path = build_experiment(experiment)

        print(
            f"Created {script_path.name} "
            f"for {version.upper()}."
        )

        run_experiment(
            experiment,
            script_path,
        )

        if not result_path.exists():
            raise RuntimeError(
                f"{version} completed but result file "
                f"is missing: {result_path}"
            )

        summary = extract_best(
            version,
            result_path,
        )

        summary.update(experiment)
        summaries.append(summary)

    by_mismatch = sorted(
        summaries,
        key=lambda item: (
            item["mismatch_vs_fp32"],
            -item["best_mean_iou"],
        ),
    )

    by_iou = sorted(
        summaries,
        key=lambda item: (
            -item["best_mean_iou"],
            item["mismatch_vs_fp32"],
        ),
    )

    comparison = {
        "selection_rule": (
            "Primary: lowest mismatch versus FP32. "
            "Secondary: highest mean IoU."
        ),
        "recommended_version": by_mismatch[0]["version"],
        "best_by_fp32_agreement": by_mismatch[0],
        "best_by_mean_iou": by_iou[0],
        "ranking_by_fp32_agreement": by_mismatch,
        "ranking_by_mean_iou": by_iou,
    }

    json_path = (
        RESULT_DIRECTORY
        / "qat_versions_v4_to_v9_comparison.json"
    )

    csv_path = (
        RESULT_DIRECTORY
        / "qat_versions_v4_to_v9_comparison.csv"
    )

    json_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "version",
        "best_mismatch_epoch",
        "mismatch_vs_fp32",
        "agreement_vs_fp32",
        "pixel_accuracy_at_best_mismatch",
        "mean_iou_at_best_mismatch",
        "clear_iou_at_best_mismatch",
        "thick_cloud_iou_at_best_mismatch",
        "thin_cloud_iou_at_best_mismatch",
        "cloud_shadow_iou_at_best_mismatch",
        "best_iou_epoch",
        "best_mean_iou",
        "mismatch_at_best_iou",
        "soft_teacher",
        "hard_teacher",
        "ground_truth",
        "learning_rate",
        "epochs",
        "result_file",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(by_mismatch)

    print()
    print("=" * 72)
    print("FINAL QAT SWEEP RESULT")
    print("=" * 72)
    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print()
    print("Recommended version:", comparison[
        "recommended_version"
    ])

    print("Comparison JSON:", json_path)
    print("Comparison CSV:", csv_path)
    print("Individual logs:", LOG_DIRECTORY)


if __name__ == "__main__":
    main()
