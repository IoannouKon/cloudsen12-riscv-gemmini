import onnx
import numpy as np
import json
import os
from onnx import numpy_helper

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
weights_dir = "weights"
manifest_path = "config/model_manifest.json"

os.makedirs(weights_dir, exist_ok=True)

model = onnx.load(model_path)

manifest = {
    "model": model_path,
    "initializers": [],
    "nodes": []
}

print("Exporting initializers...")

for init in model.graph.initializer:
    arr = numpy_helper.to_array(init)
    safe_name = init.name.replace("/", "_").replace(":", "_").replace(".", "_")
    filename = safe_name + ".bin"
    filepath = os.path.join(weights_dir, filename)

    arr.astype(np.float32).tofile(filepath)

    manifest["initializers"].append({
        "name": init.name,
        "file": filepath,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size)
    })

print("Exported", len(manifest["initializers"]), "initializers.")

print("Exporting nodes...")

for idx, node in enumerate(model.graph.node):
    attrs = {}

    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = float(attr.f)
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8", errors="ignore")
        else:
            attrs[attr.name] = "UNSUPPORTED_ATTR_TYPE"

    manifest["nodes"].append({
        "index": idx,
        "op_type": node.op_type,
        "name": node.name,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attrs": attrs
    })

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("Saved:", manifest_path)

total_params = sum(x["numel"] for x in manifest["initializers"])
print("Total initializer parameters:", total_params)
print("Estimated FP32 MB:", total_params * 4 / (1024 * 1024))
