from pathlib import Path

import tacoreader.v1 as tacoreader

from cache_qat_patches import write_split


dataset = tacoreader.load(
    "tacofoundation:cloudsen12-l1c"
)

# Training used scenes 0-79.
# Validation used scenes 80-99.
# Final test uses previously unseen scenes 100-119.
test_indices = list(range(100, 120))

write_split(
    dataset=dataset,
    sample_indices=test_indices,
    patch_count=200,
    output_directory=Path("qat/data/test"),
    patch_size=256,
    seed=2026,
)

print("Unseen test cache complete.")
