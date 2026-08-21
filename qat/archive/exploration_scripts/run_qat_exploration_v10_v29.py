#!/usr/bin/env python3
"""Generate, run, log, and compare 20 all-INT8 QAT experiments from V9."""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
SOURCE = ROOT / "train_qat_int8_v9.py"
START_CHECKPOINT = ROOT / "qat/checkpoints/qat_int8_v9_best.pth"
RESULTS = ROOT / "qat/results/exploration_v10_v29"
LOGS = ROOT / "qat/logs/exploration_v10_v29"
SCRIPTS = ROOT / "qat/generated_trainers"
CHECKPOINTS = ROOT / "qat/checkpoints"
for directory in (RESULTS, LOGS, SCRIPTS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

# All configurations remain fully fake-INT8 across every Conv layer.
EXPERIMENTS = [
    # version, soft teacher, hard teacher, labels, learning rate, epochs
    ("v10", .60, .35, .05, "5e-9", 10),
    ("v11", .65, .30, .05, "5e-9", 10),
    ("v12", .70, .25, .05, "5e-9", 10),
    ("v13", .60, .35, .05, "1e-8", 10),
    ("v14", .70, .25, .05, "1e-8", 10),
    ("v15", .75, .20, .05, "1e-8", 10),
    ("v16", .55, .40, .05, "2e-8", 8),
    ("v17", .60, .35, .05, "2e-8", 8),
    ("v18", .65, .30, .05, "2e-8", 8),
    ("v19", .70, .25, .05, "2e-8", 8),
    ("v20", .75, .20, .05, "2e-8", 8),
    ("v21", .55, .40, .05, "3e-8", 8),
    ("v22", .60, .35, .05, "3e-8", 8),
    ("v23", .65, .30, .05, "3e-8", 8),
    ("v24", .70, .25, .05, "3e-8", 8),
    ("v25", .75, .20, .05, "3e-8", 8),
    ("v26", .55, .40, .05, "5e-8", 6),
    ("v27", .60, .35, .05, "5e-8", 6),
    ("v28", .65, .30, .05, "5e-8", 6),
    ("v29", .70, .25, .05, "5e-8", 6),
]


def make_trainer(version, soft, hard, labels, learning_rate, epochs):
    text = SOURCE.read_text(encoding="utf-8-sig")
    text = re.sub(r"lr=[0-9.eE+-]+,", f"lr={learning_rate},", text, count=1)
    text = re.sub(r"epochs\s*=\s*\d+", f"epochs = {epochs}", text, count=1)

    loss_pattern = re.compile(
        r"loss\s*=\s*\(\s*"
        r"[0-9.]+\s*\*\s*distillation_loss\s*\+\s*"
        r"[0-9.]+\s*\*\s*teacher_class_loss\s*\+\s*"
        r"[0-9.]+\s*\*\s*ground_truth_loss\s*\)",
        re.MULTILINE,
    )
    replacement = (
        "loss = (\n"
        f"                {soft:.2f} * distillation_loss\n"
        f"                + {hard:.2f} * teacher_class_loss\n"
        f"                + {labels:.2f} * ground_truth_loss\n"
        "            )"
    )
    text, count = loss_pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch loss block for {version}")

    # Every new experiment starts from V9, while preserving V9 itself.
    text = text.replace("qat/checkpoints/qat_int8_v4_best.pth", "qat/checkpoints/qat_int8_v9_best.pth")
    text = text.replace("qat/checkpoints/qat_int8_v9_best.pth", f"qat/checkpoints/qat_int8_{version}_best.pth")

    # Restore the first checkpoint path occurrence as the V9 input.
    output_path = f"qat/checkpoints/qat_int8_{version}_best.pth"
    first = text.find(output_path)
    if first == -1:
        raise RuntimeError(f"Could not locate checkpoint path for {version}")
    text = text[:first] + "qat/checkpoints/qat_int8_v9_best.pth" + text[first + len(output_path):]

    text = text.replace("qat/results/qat_int8_v9_training.json", f"qat/results/exploration_v10_v29/qat_int8_{version}_training.json")

    script_path = SCRIPTS / f"train_qat_int8_{version}.py"
    compile(text, str(script_path), "exec")
    script_path.write_text(text, encoding="utf-8")
    return script_path


def run_and_log(version, script_path):
    log_path = LOGS / f"qat_int8_{version}.log"
    print(f"\n{'=' * 68}\nRUNNING {version.upper()}\n{'=' * 68}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{version} failed. See {log_path}")


def summarize(version, config):
    path = RESULTS / f"qat_int8_{version}_training.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    epochs = data["epochs"]
    best_mismatch = min(epochs, key=lambda row: row["mismatch_vs_fp32_teacher"])
    best_iou = max(epochs, key=lambda row: row["mean_iou"])
    soft, hard, labels, lr, total_epochs = config
    return {
        "version": version,
        "soft_teacher": soft,
        "hard_teacher": hard,
        "ground_truth": labels,
        "learning_rate": lr,
        "planned_epochs": total_epochs,
        "best_mismatch_epoch": best_mismatch["epoch"],
        "mismatch_vs_fp32": best_mismatch["mismatch_vs_fp32_teacher"],
        "agreement_vs_fp32": 1.0 - best_mismatch["mismatch_vs_fp32_teacher"],
        "pixel_accuracy": best_mismatch["pixel_accuracy"],
        "mean_iou": best_mismatch["mean_iou"],
        "clear_iou": best_mismatch["clear_iou"],
        "thick_cloud_iou": best_mismatch["thick_cloud_iou"],
        "thin_cloud_iou": best_mismatch["thin_cloud_iou"],
        "cloud_shadow_iou": best_mismatch["cloud_shadow_iou"],
        "best_iou_epoch": best_iou["epoch"],
        "best_mean_iou": best_iou["mean_iou"],
        "result_file": str(path),
        "log_file": str(LOGS / f"qat_int8_{version}.log"),
    }


def save_plots(rows):
    import matplotlib.pyplot as plt
    versions = [row["version"].upper() for row in rows]
    agreement = [100 * row["agreement_vs_fp32"] for row in rows]
    mean_iou = [100 * row["mean_iou"] for row in rows]
    thin = [100 * row["thin_cloud_iou"] for row in rows]
    shadow = [100 * row["cloud_shadow_iou"] for row in rows]
    best_index = max(range(len(rows)), key=lambda i: rows[i]["agreement_vs_fp32"])
    colors = ["#7f8c8d"] * len(rows)
    colors[best_index] = "#2e86de"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].bar(versions, agreement, color=colors)
    axes[0, 0].set_title("Agreement with FP32 Teacher")
    axes[0, 0].set_ylabel("Agreement (%)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(versions, [100 - x for x in agreement], color=colors)
    axes[0, 1].set_title("Mismatch vs FP32 (Lower Is Better)")
    axes[0, 1].set_ylabel("Mismatch (%)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].plot(versions, mean_iou, marker="o")
    axes[1, 0].set_title("Mean IoU")
    axes[1, 0].set_ylabel("Mean IoU (%)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=.3)

    axes[1, 1].plot(versions, thin, marker="o", label="Thin cloud")
    axes[1, 1].plot(versions, shadow, marker="o", label="Cloud shadow")
    axes[1, 1].set_title("Sensitive-Class IoU")
    axes[1, 1].set_ylabel("IoU (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=.3)

    fig.suptitle("CloudSEN12 All-INT8 QAT Exploration V10-V29")
    fig.tight_layout(rect=(0, 0, 1, .96))
    plot_path = RESULTS / "qat_exploration_v10_v29.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}")
    if not START_CHECKPOINT.exists():
        raise SystemExit(f"Missing {START_CHECKPOINT}")

    rows = []
    manifest = []
    for version, soft, hard, labels, lr, epochs in EXPERIMENTS:
        config = (soft, hard, labels, lr, epochs)
        manifest.append({
            "version": version,
            "soft_teacher": soft,
            "hard_teacher": hard,
            "ground_truth": labels,
            "learning_rate": lr,
            "epochs": epochs,
            "start_checkpoint": str(START_CHECKPOINT),
            "all_conv_layers_fake_int8": True,
        })
        script = make_trainer(version, soft, hard, labels, lr, epochs)
        run_and_log(version, script)
        rows.append(summarize(version, config))

    ranking = sorted(rows, key=lambda row: (row["mismatch_vs_fp32"], -row["mean_iou"]))
    comparison = {
        "purpose": "All-INT8 QAT hyperparameter exploration",
        "selection_rule": "Lowest mismatch versus FP32, then highest mean IoU",
        "recommended_version": ranking[0]["version"],
        "ranking": ranking,
        "experiment_manifest": manifest,
    }

    json_path = RESULTS / "qat_exploration_v10_v29_summary.json"
    csv_path = RESULTS / "qat_exploration_v10_v29_summary.csv"
    manifest_path = RESULTS / "qat_exploration_v10_v29_manifest.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ranking[0].keys())
        writer.writeheader()
        writer.writerows(ranking)

    plot_path = save_plots(rows)
    print("\nEXPLORATION COMPLETE")
    print("Recommended version:", ranking[0]["version"])
    print("Summary:", json_path)
    print("CSV:", csv_path)
    print("Manifest:", manifest_path)
    print("Plot:", plot_path)
    print("Logs:", LOGS)


if __name__ == "__main__":
    main()
