# Full Pure C Forward: Next Step

Run inside `~/cloudsen12_pure_c`.

```bash
python scripts/export_runtime_shapes.py
python scripts/make_execution_plan.py
python scripts/generate_forward_todo.py
```

Then inspect:

```bash
head -n 80 c_execution_plan.md
cat c_execution_plan.md | grep -A20 "Unsupported operators"
ls -lh src/cloudsen12_forward_todo.c include/cloudsen12_forward.h
```

Send the output of:

```bash
cat c_execution_plan.md | grep -A20 "Unsupported operators"
grep -n "Node 0" -A40 src/cloudsen12_forward_todo.c
```

After this, we will turn the TODO skeleton into real `cloudsen12_forward.c` calls layer by layer.
