# Multi-operator pure C validation commands

Run inside `~/cloudsen12_pure_c`.

## Add

```bash
python scripts/export_first_add_test.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude \
  src/test_first_add.c src/add.c \
  -lm -o build/test_first_add

./build/test_first_add
```

## Concat

```bash
python scripts/export_first_concat_test.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude \
  src/test_first_concat.c src/concat.c \
  -lm -o build/test_first_concat

./build/test_first_concat
```

## Resize

```bash
python scripts/export_first_resize_test.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude \
  src/test_first_resize.c src/resize.c \
  -lm -o build/test_first_resize

./build/test_first_resize
```

Expected: mean/max error close to 0. If Resize error is not close to 0, the ONNX Resize mode is not the naive nearest mapping and we will adjust the C kernel.
