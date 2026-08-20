# Generated partial pure C forward

Run inside `~/cloudsen12_pure_c`:

```bash
TARGET_SUPPORTED_OPS=20 python scripts/generate_partial_forward.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude src/test_partial_forward.c src/conv2d.c src/clip.c src/relu.c src/add.c -lm -o build/test_partial_forward

./build/test_partial_forward
```

If this passes, increase the number:

```bash
TARGET_SUPPORTED_OPS=40 python scripts/generate_partial_forward.py
```

Then recompile and rerun.
