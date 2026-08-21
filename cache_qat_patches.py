#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import rasterio as rio
import tacoreader.v1 as tacoreader
from rasterio.windows import Window


def choose_window(mask, patch_size, rng, attempts=30):
    height, width = mask.shape

    best_window = None
    best_target = None
    best_score = -1.0

    for _ in range(attempts):
        x = rng.randint(0, width - patch_size)
        y = rng.randint(0, height - patch_size)

        target = mask[
            y:y + patch_size,
            x:x + patch_size,
        ]

        classes, counts = np.unique(
            target,
            return_counts=True,
        )

        fractions = counts.astype(np.float64) / target.size
        diversity_score = len(classes)

        rare_class_score = sum(
            float(fractions[index])
            for index, class_id in enumerate(classes)
            if int(class_id) in (2, 3)
        )

        score = diversity_score + 2.0 * rare_class_score

        if score > best_score:
            best_score = score
            best_target = target.copy()
            best_window = Window(
                x,
                y,
                patch_size,
                patch_size,
            )

        if diversity_score >= 3 and rare_class_score >= 0.05:
            break

    return best_window, best_target


def write_split(
    dataset,
    sample_indices,
    patch_count,
    output_directory,
    patch_size,
    seed,
):
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(seed)
    records = []

    created = 0
    attempts = 0

    while created < patch_count:
        attempts += 1
        sample_index = rng.choice(sample_indices)

        try:
            sample = dataset.read(sample_index)

            target_path = sample.read(1)
            image_path = sample.read(0)

            with rio.open(target_path) as target_source:
                target_full = target_source.read(1)

            height, width = target_full.shape

            if height < patch_size or width < patch_size:
                print(
                    f"Skip sample {sample_index}: "
                    f"target is smaller than patch size",
                    flush=True,
                )
                continue

            window, target = choose_window(
                target_full,
                patch_size,
                rng,
            )

            with rio.open(image_path) as image_source:
                image = image_source.read(window=window)

            if image.shape != (
                13,
                patch_size,
                patch_size,
            ):
                print(
                    f"Skip sample {sample_index}: "
                    f"unexpected image shape {image.shape}",
                    flush=True,
                )
                continue

            image = image.astype(
                np.uint16,
                copy=False,
            )

            target = target.astype(
                np.uint8,
                copy=False,
            )

            output_path = (
                output_directory
                / f"patch_{created:05d}.npz"
            )

            np.savez_compressed(
                output_path,
                image=image,
                target=target,
            )

            classes, counts = np.unique(
                target,
                return_counts=True,
            )

            records.append(
                {
                    "file": output_path.name,
                    "sample_index": int(sample_index),
                    "x": int(window.col_off),
                    "y": int(window.row_off),
                    "classes": {
                        str(int(class_id)): int(count)
                        for class_id, count
                        in zip(classes, counts)
                    },
                }
            )

            created += 1

            print(
                f"[{output_directory.name}] "
                f"{created}/{patch_count}: "
                f"sample={sample_index} "
                f"classes={classes.tolist()}",
                flush=True,
            )

        except Exception as error:
            print(
                f"Skip sample {sample_index}: {error}",
                flush=True,
            )

            if attempts > patch_count * 30:
                raise RuntimeError(
                    "Too many failed patch attempts."
                ) from error

    manifest_path = output_directory / "manifest.json"

    manifest_path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved manifest: {manifest_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-count",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--validation-count",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    root = Path.cwd()

    train_directory = root / "qat/data/train"
    validation_directory = root / "qat/data/validation"

    print("Loading CloudSEN12 dataset...", flush=True)

    dataset = tacoreader.load(
        "tacofoundation:cloudsen12-l1c"
    )

    print("Dataset loaded.", flush=True)

    # Use different source scenes for training and validation.
    training_indices = list(range(0, 80))
    validation_indices = list(range(80, 100))

    print(
        f"Creating {args.train_count} training patches...",
        flush=True,
    )

    write_split(
        dataset=dataset,
        sample_indices=training_indices,
        patch_count=args.train_count,
        output_directory=train_directory,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    print(
        f"Creating {args.validation_count} validation patches...",
        flush=True,
    )

    write_split(
        dataset=dataset,
        sample_indices=validation_indices,
        patch_count=args.validation_count,
        output_directory=validation_directory,
        patch_size=args.patch_size,
        seed=args.seed + 1,
    )

    print("Patch cache complete.", flush=True)


if __name__ == "__main__":
    main()
