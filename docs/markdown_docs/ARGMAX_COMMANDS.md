# Final Argmax pure C test

Run inside `~/cloudsen12_pure_c`:

```bash
python scripts/export_final_logits_argmax_test.py

gcc -O2 -Wall -Wextra -std=c11 -Iinclude src/test_final_argmax.c src/argmax.c -lm -o build/test_final_argmax

./build/test_final_argmax
```

Open output image from WSL:

```bash
explorer.exe outputs/predictions/pure_c_argmax_pred.pgm
```
