from pathlib import Path

source = Path("train_qat_int8_v2.py")
destination = Path("train_qat_int8_v4.py")

text = source.read_text(encoding="utf-8-sig")

# Version 4 uses a very small continuation learning rate.
text = text.replace(
    "lr=5e-7,",
    "lr=5e-8,",
)

# Run only six controlled continuation epochs.
text = text.replace(
    "epochs = 10",
    "epochs = 6",
)

# Replace the existing Version 2 loss.
old_loss = """            loss = (
                0.25 * ground_truth_loss
                + 0.75 * distillation_loss
            )"""

new_loss = """            teacher_predictions = torch.argmax(
                teacher_logits,
                dim=1,
            )

            teacher_class_loss = F.cross_entropy(
                student_logits,
                teacher_predictions,
            )

            loss = (
                0.60 * distillation_loss
                + 0.35 * teacher_class_loss
                + 0.05 * ground_truth_loss
            )"""

if old_loss not in text:
    raise RuntimeError(
        "Could not find the Version 2 loss block."
    )

text = text.replace(old_loss, new_loss, 1)

# Locate the checkpoint-loading block.
start = text.find(
    '    previous_checkpoint = Path('
)

if start == -1:
    raise RuntimeError(
        "Could not find the Version 2 checkpoint-loading block."
    )

end = text.find(
    "    optimizer = torch.optim.AdamW(",
    start,
)

if end == -1:
    raise RuntimeError(
        "Could not find the optimizer after checkpoint loading."
    )

old_checkpoint_block = text[start:end]

new_checkpoint_block = '''    version_2_checkpoint = Path(
        "qat/checkpoints/qat_int8_v2_best.pth"
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

'''

text = (
    text[:start]
    + new_checkpoint_block
    + text[end:]
)

# Keep BatchNorm frozen after each student.train() call.
old_train = """        student.train()

        total_loss = 0.0"""

new_train = """        student.train()

        for module in student.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0"""

if old_train not in text:
    raise RuntimeError(
        "Could not find the student.train() block."
    )

text = text.replace(old_train, new_train, 1)

# Give Version 4 separate output filenames.
text = text.replace(
    '"qat/checkpoints/qat_int8_v2_best.pth"',
    '"qat/checkpoints/qat_int8_v4_best.pth"',
)

# Restore the Version 2 input checkpoint after the global replacement.
text = text.replace(
    '''version_2_checkpoint = Path(
        "qat/checkpoints/qat_int8_v4_best.pth"
    )''',
    '''version_2_checkpoint = Path(
        "qat/checkpoints/qat_int8_v2_best.pth"
    )''',
    1,
)

text = text.replace(
    "qat/results/qat_int8_v2_training.json",
    "qat/results/qat_int8_v4_training.json",
)

destination.write_text(
    text,
    encoding="utf-8",
)

print("Created:", destination)
print("Version 2 input checkpoint preserved.")
print("Version 4 output checkpoint separated.")
print("All 63 Conv layers remain fake-INT8.")
print("Soft teacher loss: 60%")
print("Hard teacher-class loss: 35%")
print("Ground-truth loss: 5%")
print("Learning rate: 5e-8")
print("Epochs: 6")
print("BatchNorm: frozen")
