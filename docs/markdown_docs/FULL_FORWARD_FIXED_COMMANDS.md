# Fixed full forward planning step

The previous script failed because it declared Shape node outputs as float. Shape outputs are int64, so this fixed version only exports floating-point operator outputs.

Run inside `~/cloudsen12_pure_c`:

```bash
python scripts/export_runtime_shapes.py
python scripts/make_execution_plan.py
python scripts/generate_forward_todo.py
```

Then inspect:

```bash
cat c_execution_plan.md | grep -A20 "Unsupported operators"
grep -n "Node 0" -A50 src/cloudsen12_forward_todo.c
ls -lh config/runtime_shapes.json config/c_execution_plan.json c_execution_plan.md
```
