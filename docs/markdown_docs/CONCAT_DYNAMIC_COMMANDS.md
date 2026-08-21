# Dynamic-shape Concat test

Run inside `~/cloudsen12_pure_c`:

```bash
python scripts/export_first_concat_test.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude src/test_first_concat.c src/concat.c -lm -o build/test_first_concat

./build/test_first_concat
```

Compile only after `export_first_concat_test.py` succeeds and creates `include/first_concat_meta.h`.
