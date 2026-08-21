import json
from collections import Counter

MANIFEST = "config/model_manifest.json"
SHAPES = "config/runtime_shapes.json"
OUT_MD = "c_execution_plan.md"
OUT_JSON = "config/c_execution_plan.json"

with open(MANIFEST) as f:
    manifest = json.load(f)

with open(SHAPES) as f:
    shapes = json.load(f)

nodes = manifest["nodes"]
initializers = {x["name"]: x for x in manifest["initializers"]}

supported_base = {"Conv", "Clip", "Relu", "Add", "Resize"}
ignored_base = {"Constant", "Shape", "Slice"}

plan = []
unsupported = []
counts = Counter(n["op_type"] for n in nodes)

for n in nodes:
    op = n["op_type"]

    if op in supported_base:
        status = "supported"
    elif op == "Concat":
        axis = n["attrs"].get("axis", 0)
        if axis == 1:
            status = "supported"
        else:
            status = "ignored"
    elif op in ignored_base:
        status = "ignored"
    else:
        status = "unsupported"

    out_shapes = [shapes.get(o, {}).get("shape") for o in n["outputs"]]

    in_shapes = []
    for inp in n["inputs"]:
        if inp in shapes:
            in_shapes.append(shapes[inp]["shape"])
        elif inp in initializers:
            in_shapes.append(initializers[inp]["shape"])
        elif inp == "":
            in_shapes.append([])
        else:
            in_shapes.append(None)

    entry = {
        "index": n["index"],
        "op_type": op,
        "name": n["name"],
        "inputs": n["inputs"],
        "outputs": n["outputs"],
        "input_shapes": in_shapes,
        "output_shapes": out_shapes,
        "attrs": n["attrs"],
        "status": status
    }

    plan.append(entry)

    if status == "unsupported":
        unsupported.append(entry)

with open(OUT_JSON, "w") as f:
    json.dump(plan, f, indent=2)

with open(OUT_MD, "w") as f:
    f.write("# CloudSEN12 Pure C Execution Plan\n\n")

    f.write("## Operator counts\n\n")
    for op, c in counts.most_common():
        f.write(f"- {op}: {c}\n")

    f.write("\n## Unsupported operators\n\n")
    if not unsupported:
        f.write("None. All non-bookkeeping ops are covered by current pure C kernels.\n")
    else:
        for u in unsupported:
            f.write(f"- node {u['index']}: {u['op_type']} {u['name']}\n")

    f.write("\n## Notes\n\n")
    f.write("Constant, Shape, Slice, and Concat axis=0 nodes are bookkeeping nodes used for dynamic Resize shape construction. They are ignored in the first pure C plan. Their results are folded into known runtime tensor shapes.\n\n")

    f.write("## First 80 execution entries\n\n")
    for e in plan[:80]:
        f.write(f"### Node {e['index']}: {e['op_type']}\n")
        f.write(f"name: `{e['name']}`\n\n")
        f.write(f"status: `{e['status']}`\n\n")
        f.write(f"inputs: `{e['inputs']}`\n\n")
        f.write(f"input shapes: `{e['input_shapes']}`\n\n")
        f.write(f"outputs: `{e['outputs']}`\n\n")
        f.write(f"output shapes: `{e['output_shapes']}`\n\n")

print("Saved:", OUT_JSON)
print("Saved:", OUT_MD)
print("Unsupported operators:", len(unsupported))

if unsupported:
    for u in unsupported[:20]:
        print("  ", u["index"], u["op_type"], u["name"])
