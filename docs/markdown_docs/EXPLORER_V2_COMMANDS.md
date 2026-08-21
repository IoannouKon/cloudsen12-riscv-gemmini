# Partial Forward v2 Generator

This generator supports Conv, Clip, Relu, Add, Concat, and Resize.

Run inside `~/cloudsen12_pure_c`:

```bash
TARGET_SUPPORTED_OPS=100 python scripts/generate_partial_forward_v2.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude src/test_partial_forward_v2.c src/conv2d.c src/clip.c src/relu.c src/add.c src/concat.c src/resize.c -lm -o build/test_partial_forward_v2

./build/test_partial_forward_v2
```

Then try `TARGET_SUPPORTED_OPS=120`.
