# Fixed Concat and Resize commands

Run inside `~/cloudsen12_pure_c`.

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

If resize error is high, send the output; we will adjust coordinate mapping.
