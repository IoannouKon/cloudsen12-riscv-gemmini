#!/usr/bin/env python3

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cloudsen12_models import cloudsen12
from torch.utils.data import DataLoader, Dataset


class CachedPatchDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(Path(directory).glob("*.npz"))

        if not self.files:
            raise RuntimeError(
                f"No NPZ patches found in {directory}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            image = data["image"].astype(
                np.float32
            ) / 10000.0

            target = data["target"].astype(
                np.int64
            )

        return (
            torch.from_numpy(image),
            torch.from_numpy(target),
        )


def calculate_class_weights(dataset):
    counts = np.zeros(4, dtype=np.int64)

    print("Calculating training class frequencies...")

    for index in range(len(dataset)):
        _, target = dataset[index]

        values, frequencies = np.unique(
            target.numpy(),
            return_counts=True,
        )

        for value, frequency in zip(
            values,
            frequencies,
        ):
            if 0 <= int(value) < 4:
                counts[int(value)] += int(frequency)

    frequencies = counts / counts.sum()

    weights = 1.0 / np.sqrt(
        np.maximum(frequencies, 1e-8)
    )

    weights = weights / weights.mean()

    print("Class pixel counts:", counts.tolist())
    print("Class frequencies:", frequencies.tolist())
    print("Class weights:", weights.tolist())

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


@torch.no_grad()
def evaluate(model, loader, device, class_weights):
    model.eval()

    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    intersections = torch.zeros(
        4,
        dtype=torch.int64,
    )

    unions = torch.zeros(
        4,
        dtype=torch.int64,
    )

    start_time = time.perf_counter()

    for batch_index, (images, targets) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_loss += (
            loss.item() * images.shape[0]
        )

        total_pixels += targets.numel()

        correct_pixels += (
            predictions == targets
        ).sum().item()

        predictions_cpu = predictions.cpu()
        targets_cpu = targets.cpu()

        for class_id in range(4):
            predicted_class = (
                predictions_cpu == class_id
            )

            target_class = (
                targets_cpu == class_id
            )

            intersections[class_id] += (
                predicted_class & target_class
            ).sum()

            unions[class_id] += (
                predicted_class | target_class
            ).sum()

        if batch_index % 5 == 0:
            print(
                f"  validation batch "
                f"{batch_index}/{len(loader)}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time

    class_iou = []

    for class_id in range(4):
        if unions[class_id] > 0:
            value = (
                intersections[class_id].item()
                / unions[class_id].item()
            )
        else:
            value = 1.0

        class_iou.append(value)

    return {
        "loss": total_loss / len(loader.dataset),
        "pixel_accuracy": (
            correct_pixels / total_pixels
        ),
        "mean_iou": float(
            sum(class_iou) / len(class_iou)
        ),
        "clear_iou": class_iou[0],
        "thick_cloud_iou": class_iou[1],
        "thin_cloud_iou": class_iou[2],
        "cloud_shadow_iou": class_iou[3],
        "seconds": elapsed,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    class_weights,
):
    model.train()

    total_loss = 0.0
    start_time = time.perf_counter()

    for batch_index, (images, targets) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            loss.item() * images.shape[0]
        )

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == len(loader)
        ):
            memory = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            print(
                f"  training batch "
                f"{batch_index}/{len(loader)} "
                f"loss={loss.item():.6f} "
                f"gpu_memory={memory:.3f} GiB",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time

    return {
        "loss": total_loss / len(loader.dataset),
        "seconds": elapsed,
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda")

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = CachedPatchDataset(
        "qat/data/train"
    )

    validation_dataset = CachedPatchDataset(
        "qat/data/validation"
    )

    print(
        "Training patches:",
        len(train_dataset),
    )

    print(
        "Validation patches:",
        len(validation_dataset),
    )

    # Use workers=0 first for maximum Windows reliability.
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    class_weights = calculate_class_weights(
        train_dataset
    ).to(device)

    wrapper = cloudsen12.load_model_by_name(
        name="UNetMobV2_V2",
        weights_folder="weights",
        device=device,
    )

    model = wrapper.model

    print("\nEvaluating original pretrained model...")

    pretrained_metrics = evaluate(
        model,
        validation_loader,
        device,
        class_weights,
    )

    print(
        json.dumps(
            pretrained_metrics,
            indent=2,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-4,
    )

    print("\nTraining one FP32 baseline epoch...")

    training_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        class_weights,
    )

    print(
        json.dumps(
            training_metrics,
            indent=2,
        )
    )

    print("\nEvaluating fine-tuned model...")

    finetuned_metrics = evaluate(
        model,
        validation_loader,
        device,
        class_weights,
    )

    print(
        json.dumps(
            finetuned_metrics,
            indent=2,
        )
    )

    checkpoint_path = Path(
        "qat/checkpoints/fp32_baseline_epoch1.pth"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "pretrained_metrics": (
                pretrained_metrics
            ),
            "training_metrics": (
                training_metrics
            ),
            "finetuned_metrics": (
                finetuned_metrics
            ),
            "class_weights": (
                class_weights.detach().cpu()
            ),
        },
        checkpoint_path,
    )

    results = {
        "pretrained": pretrained_metrics,
        "training_epoch_1": training_metrics,
        "finetuned_epoch_1": finetuned_metrics,
    }

    results_path = Path(
        "qat/results/fp32_baseline_epoch1.json"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nSaved checkpoint:", checkpoint_path)
    print("Saved results:", results_path)
    print("FP32 baseline training: PASS")


if __name__ == "__main__":
    main()
