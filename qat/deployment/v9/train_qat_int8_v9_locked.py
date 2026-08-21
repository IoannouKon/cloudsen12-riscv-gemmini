#!/usr/bin/env python3

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cloudsen12_models import cloudsen12
from torch.utils.data import DataLoader, Dataset


class CachedDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(Path(directory).glob("*.npz"))

        if not self.files:
            raise RuntimeError(f"No patches found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            image = data["image"].astype(np.float32) / 10000.0
            target = data["target"].astype(np.int64)

        return torch.from_numpy(image), torch.from_numpy(target)


class SymmetricFakeQuant(torch.nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.qmax = (1 << (bits - 1)) - 1

    def forward(self, tensor):
        maximum = tensor.detach().abs().amax()

        if maximum == 0:
            return tensor

        scale = maximum / self.qmax

        quantized = torch.clamp(
            torch.round(tensor / scale),
            -self.qmax,
            self.qmax,
        )

        return tensor + (quantized * scale - tensor).detach()


class QATConv2d(torch.nn.Module):
    def __init__(self, original):
        super().__init__()

        self.weight = original.weight

        if original.bias is not None:
            self.bias = original.bias
        else:
            self.bias = None

        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

        self.input_fake_quant = SymmetricFakeQuant()
        self.output_fake_quant = SymmetricFakeQuant()

    def quantize_weight_per_channel(self):
        weight = self.weight
        qmax = 127

        maximum = (
            weight.detach()
            .abs()
            .flatten(1)
            .amax(dim=1)
            .clamp_min(1e-12)
        )

        scale = maximum / qmax

        shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
        scale = scale.reshape(shape)

        quantized = torch.clamp(
            torch.round(weight / scale),
            -qmax,
            qmax,
        )

        return weight + (quantized * scale - weight).detach()

    def forward(self, tensor):
        tensor = self.input_fake_quant(tensor)
        weight = self.quantize_weight_per_channel()

        output = F.conv2d(
            tensor,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        return self.output_fake_quant(output)


def replace_convolutions(module):
    replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Conv2d):
            setattr(module, name, QATConv2d(child))
            replaced += 1
        else:
            replaced += replace_convolutions(child)

    return replaced


def calculate_class_weights(dataset):
    counts = np.zeros(4, dtype=np.int64)

    for index in range(len(dataset)):
        _, target = dataset[index]
        values, frequencies = np.unique(
            target.numpy(),
            return_counts=True,
        )

        for value, frequency in zip(values, frequencies):
            if 0 <= int(value) < 4:
                counts[int(value)] += int(frequency)

    fractions = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(fractions, 1e-8))
    weights /= weights.mean()

    print("Class weights:", weights.tolist())

    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(student, teacher, loader, device):
    student.eval()
    teacher.eval()

    intersections = torch.zeros(4, dtype=torch.int64)
    unions = torch.zeros(4, dtype=torch.int64)

    mismatch_pixels = 0
    total_pixels = 0
    correct_pixels = 0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        student_logits = student(images)
        teacher_logits = teacher(images)

        student_prediction = student_logits.argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)

        mismatch_pixels += (
            student_prediction != teacher_prediction
        ).sum().item()

        correct_pixels += (
            student_prediction == targets
        ).sum().item()

        total_pixels += targets.numel()

        student_cpu = student_prediction.cpu()
        target_cpu = targets.cpu()

        for class_id in range(4):
            predicted = student_cpu == class_id
            expected = target_cpu == class_id

            intersections[class_id] += (
                predicted & expected
            ).sum()

            unions[class_id] += (
                predicted | expected
            ).sum()

        if batch_index % 5 == 0:
            print(
                f"  validation batch {batch_index}/{len(loader)}",
                flush=True,
            )

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
        "pixel_accuracy": correct_pixels / total_pixels,
        "mismatch_vs_fp32_teacher": mismatch_pixels / total_pixels,
        "mean_iou": sum(class_iou) / 4,
        "clear_iou": class_iou[0],
        "thick_cloud_iou": class_iou[1],
        "thin_cloud_iou": class_iou[2],
        "cloud_shadow_iou": class_iou[3],
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_dataset = CachedDataset("qat/data/train")
    validation_dataset = CachedDataset("qat/data/validation")

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

    print("Loading original FP32 teacher...")

    teacher_wrapper = cloudsen12.load_model_by_name(
        name="UNetMobV2_V2",
        weights_folder="weights",
        device=device,
    )

    teacher = teacher_wrapper.model
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    print("Creating QAT student from original weights...")

    student = copy.deepcopy(teacher)

    for parameter in student.parameters():
        parameter.requires_grad = True

    replaced = replace_convolutions(student)

    print("QAT convolutions:", replaced)

    version_2_checkpoint = Path(
        "qat/checkpoints/qat_int8_v4_best.pth"
    )

    if not version_2_checkpoint.exists():
        raise RuntimeError(
            f"Missing Version 2 checkpoint: "
            f"{version_2_checkpoint}"
        )

    checkpoint = torch.load(
        version_2_checkpoint,
        map_location=device,
        weights_only=False,
    )

    student.load_state_dict(
        checkpoint["student_state_dict"]
    )

    print(
        "Loaded Version 2 checkpoint:",
        version_2_checkpoint,
    )

    frozen_batchnorm_count = 0

    for module in student.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

            if module.weight is not None:
                module.weight.requires_grad = False

            if module.bias is not None:
                module.bias.requires_grad = False

            frozen_batchnorm_count += 1

    print(
        "Frozen BatchNorm layers:",
        frozen_batchnorm_count,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-8,
        weight_decay=1e-5,
    )

    epochs = 8
    temperature = 2.0
    best_mean_iou = -1.0

    results = []

    print("Evaluating QAT student before training...")

    initial_metrics = evaluate(
        student,
        teacher,
        validation_loader,
        device,
    )

    print(json.dumps(initial_metrics, indent=2))

    for epoch in range(1, epochs + 1):
        student.train()

        for module in student.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0
        start_time = time.perf_counter()

        for batch_index, (images, targets) in enumerate(
            train_loader,
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

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            ground_truth_loss = F.cross_entropy(
                student_logits,
                targets,
                weight=class_weights,
            )

            student_log_probabilities = F.log_softmax(
                student_logits / temperature,
                dim=1,
            )

            teacher_probabilities = F.softmax(
                teacher_logits / temperature,
                dim=1,
            )

            # Average KL divergence across batch and spatial pixels.
            distillation_loss = F.kl_div(
                student_log_probabilities,
                teacher_probabilities,
                reduction="none",
            ).sum(dim=1).mean() * (temperature * temperature)

            # Preserve teacher behavior while still learning ground truth.
            teacher_predictions = torch.argmax(
                teacher_logits,
                dim=1,
            )

            teacher_class_loss = F.cross_entropy(
                student_logits,
                teacher_predictions,
            )

            loss = (
                0.65 * distillation_loss
                + 0.30 * teacher_class_loss
                + 0.05 * ground_truth_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += loss.item()

            if batch_index % 10 == 0:
                print(
                    f"epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{len(train_loader)} "
                    f"loss={loss.item():.6f} "
                    f"gpu={torch.cuda.memory_allocated() / 1024**3:.3f} GiB",
                    flush=True,
                )

        training_seconds = time.perf_counter() - start_time

        metrics = evaluate(
            student,
            teacher,
            validation_loader,
            device,
        )

        metrics["epoch"] = epoch
        metrics["training_loss"] = total_loss / len(train_loader)
        metrics["training_seconds"] = training_seconds

        results.append(metrics)

        print("Epoch result:")
        print(json.dumps(metrics, indent=2))

        if metrics["mean_iou"] > best_mean_iou:
            best_mean_iou = metrics["mean_iou"]

            checkpoint = {
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "metrics": metrics,
                "qat_conv_count": replaced,
            }

            torch.save(
                checkpoint,
                "qat/checkpoints/qat_int8_v9_best.pth",
            )

            print(
                "Saved best checkpoint: "
                "qat/checkpoints/qat_int8_v9_best.pth"
            )

    output = {
        "initial_fake_int8": initial_metrics,
        "epochs": results,
        "best_mean_iou": best_mean_iou,
        "qat_conv_count": replaced,
    }

    Path("qat/results/qat_int8_v9_training.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("Saved: qat/results/qat_int8_v9_training.json")
    print("QAT training complete.")


if __name__ == "__main__":
    main()
