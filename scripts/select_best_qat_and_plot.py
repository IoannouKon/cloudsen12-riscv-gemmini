import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


root = Path.cwd()
results_root = root / "qat" / "results"
output_dir = results_root / "final_comparison"
output_dir.mkdir(parents=True, exist_ok=True)


def get_version(path):
    match = re.search(
        r"qat_int8_v(\d+)_training\.json$",
        path.name,
    )

    if match is None:
        return None, None

    number = int(match.group(1))

    return f"v{number}", number


def summarize(path):
    version, number = get_version(path)

    if version is None:
        return None

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    epochs = data.get("epochs", [])

    if not epochs:
        return None

    best_mismatch = min(
        epochs,
        key=lambda row:
            row["mismatch_vs_fp32_teacher"],
    )

    best_iou = max(
        epochs,
        key=lambda row:
            row["mean_iou"],
    )

    mismatch = float(
        best_mismatch[
            "mismatch_vs_fp32_teacher"
        ]
    )

    checkpoint = (
        root
        / "qat"
        / "checkpoints"
        / f"qat_int8_{version}_best.pth"
    )

    return {
        "version": version,
        "number": number,
        "best_mismatch_epoch": int(
            best_mismatch["epoch"]
        ),
        "mismatch_vs_fp32": mismatch,
        "agreement_vs_fp32": 1.0 - mismatch,
        "pixel_accuracy": float(
            best_mismatch["pixel_accuracy"]
        ),
        "mean_iou": float(
            best_mismatch["mean_iou"]
        ),
        "clear_iou": float(
            best_mismatch["clear_iou"]
        ),
        "thick_cloud_iou": float(
            best_mismatch["thick_cloud_iou"]
        ),
        "thin_cloud_iou": float(
            best_mismatch["thin_cloud_iou"]
        ),
        "cloud_shadow_iou": float(
            best_mismatch["cloud_shadow_iou"]
        ),
        "best_iou_epoch": int(
            best_iou["epoch"]
        ),
        "best_mean_iou": float(
            best_iou["mean_iou"]
        ),
        "checkpoint": str(
            checkpoint.relative_to(root)
        ),
        "checkpoint_exists": checkpoint.exists(),
        "result_file": str(
            path.relative_to(root)
        ),
    }


result_files = sorted(
    results_root.rglob(
        "qat_int8_v*_training.json"
    )
)

versions = {}

for result_file in result_files:
    summary = summarize(result_file)

    if summary is None:
        continue

    name = summary["version"]
    previous = versions.get(name)

    if (
        previous is None
        or summary["mismatch_vs_fp32"]
        < previous["mismatch_vs_fp32"]
    ):
        versions[name] = summary


rows = sorted(
    versions.values(),
    key=lambda row: row["number"],
)

if not rows:
    raise SystemExit(
        "No completed QAT result files found."
    )


ranking = sorted(
    rows,
    key=lambda row: (
        row["mismatch_vs_fp32"],
        -row["mean_iou"],
    ),
)

winner = ranking[0]


summary_output = {
    "selection_rule": (
        "Lowest mismatch versus FP32, "
        "then highest mean IoU"
    ),
    "versions_found": len(rows),
    "recommended_version": winner["version"],
    "winner": winner,
    "ranking": ranking,
}


summary_path = (
    output_dir / "qat_all_versions_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary_output,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


csv_path = (
    output_dir / "qat_all_versions_ranking.csv"
)

csv_fields = [
    key
    for key in ranking[0].keys()
    if key != "number"
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
    writer.writerows(ranking)


text_path = output_dir / "qat_best_model.txt"

text_path.write_text(
    f"Recommended version: "
    f"{winner['version'].upper()}\n"
    f"Agreement vs FP32: "
    f"{winner['agreement_vs_fp32'] * 100:.4f}%\n"
    f"Mismatch vs FP32: "
    f"{winner['mismatch_vs_fp32'] * 100:.4f}%\n"
    f"Pixel accuracy: "
    f"{winner['pixel_accuracy'] * 100:.4f}%\n"
    f"Mean IoU: "
    f"{winner['mean_iou'] * 100:.4f}%\n"
    f"Clear IoU: "
    f"{winner['clear_iou'] * 100:.4f}%\n"
    f"Thick-cloud IoU: "
    f"{winner['thick_cloud_iou'] * 100:.4f}%\n"
    f"Thin-cloud IoU: "
    f"{winner['thin_cloud_iou'] * 100:.4f}%\n"
    f"Cloud-shadow IoU: "
    f"{winner['cloud_shadow_iou'] * 100:.4f}%\n"
    f"Checkpoint: {winner['checkpoint']}\n",
    encoding="utf-8",
)


labels = [
    row["version"].upper()
    for row in rows
]

agreement = [
    row["agreement_vs_fp32"] * 100
    for row in rows
]

mismatch = [
    row["mismatch_vs_fp32"] * 100
    for row in rows
]

mean_iou = [
    row["mean_iou"] * 100
    for row in rows
]

pixel_accuracy = [
    row["pixel_accuracy"] * 100
    for row in rows
]

thin_iou = [
    row["thin_cloud_iou"] * 100
    for row in rows
]

shadow_iou = [
    row["cloud_shadow_iou"] * 100
    for row in rows
]


winner_index = labels.index(
    winner["version"].upper()
)

colors = ["#A7B0B5"] * len(rows)
colors[winner_index] = "#1769AA"


figure, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
)


axes[0, 0].bar(
    labels,
    agreement,
    color=colors,
    edgecolor="white",
)

axes[0, 0].set_title(
    "Agreement with FP32 teacher"
)

axes[0, 0].set_ylabel(
    "Agreement (%)"
)


axes[0, 1].bar(
    labels,
    mismatch,
    color=colors,
    edgecolor="white",
)

axes[0, 1].set_title(
    "Mismatch versus FP32 teacher"
)

axes[0, 1].set_ylabel(
    "Mismatch, lower is better (%)"
)


axes[1, 0].plot(
    labels,
    mean_iou,
    marker="o",
    linewidth=2,
    label="Mean IoU",
)

axes[1, 0].plot(
    labels,
    pixel_accuracy,
    marker="o",
    linewidth=2,
    label="Pixel accuracy",
)

axes[1, 0].set_title(
    "Ground-truth segmentation performance"
)

axes[1, 0].set_ylabel(
    "Score (%)"
)

axes[1, 0].legend(
    frameon=False
)


axes[1, 1].plot(
    labels,
    thin_iou,
    marker="o",
    linewidth=2,
    label="Thin-cloud IoU",
)

axes[1, 1].plot(
    labels,
    shadow_iou,
    marker="o",
    linewidth=2,
    label="Cloud-shadow IoU",
)

axes[1, 1].set_title(
    "Quantization-sensitive classes"
)

axes[1, 1].set_ylabel(
    "IoU (%)"
)

axes[1, 1].legend(
    frameon=False
)


for axis in axes.flat:
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


figure.suptitle(
    "CloudSEN12 Fully Fake-INT8 QAT Exploration",
    fontsize=18,
    fontweight="bold",
)

figure.text(
    0.5,
    0.012,
    (
        f"Selected model: "
        f"{winner['version'].upper()} | "
        f"FP32 agreement "
        f"{winner['agreement_vs_fp32'] * 100:.3f}% | "
        f"Mean IoU "
        f"{winner['mean_iou'] * 100:.3f}%"
    ),
    ha="center",
    fontsize=11,
)

figure.tight_layout(
    rect=(0, 0.045, 1, 0.95)
)


plot_path = (
    output_dir
    / "qat_all_versions_paper_plot.png"
)

figure.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


print()
print("QAT MODEL SELECTION COMPLETE")
print("============================")
print("Versions found:", len(rows))
print(
    "Recommended version:",
    winner["version"].upper(),
)
print(
    "Agreement vs FP32:",
    f"{winner['agreement_vs_fp32'] * 100:.4f}%",
)
print(
    "Mismatch vs FP32:",
    f"{winner['mismatch_vs_fp32'] * 100:.4f}%",
)
print(
    "Mean IoU:",
    f"{winner['mean_iou'] * 100:.4f}%",
)
print(
    "Checkpoint:",
    winner["checkpoint"],
)
print("Plot:", plot_path)
print("Ranking CSV:", csv_path)
print("Summary JSON:", summary_path)
print("Short summary:", text_path)
