import onnx
import numpy as np
import onnxruntime as ort
import json
from onnx import helper, TensorProto

MODEL = "models/cloudsen12_unetmobv2_v2.onnx"
INPUT_BIN = "inputs/input_1x13x512x512_float32.bin"
OUT_JSON = "config/runtime_shapes.json"
TEMP_MODEL = "models/runtime_shapes_float_outputs.onnx"

FLOAT_OPS = {"Conv", "Clip", "Relu", "Add", "Resize"}

print("Loading ONNX model:", MODEL)
model = onnx.load(MODEL)

existing_outputs = set(o.name for o in model.graph.output)
added = 0

for node in model.graph.node:
    add_this_node = False

    if node.op_type in FLOAT_OPS:
        add_this_node = True

    elif node.op_type == "Concat":
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)

        # axis=1 is real NCHW feature-map concat.
        # axis=0 is usually shape bookkeeping for Resize and outputs int64.
        if axis == 1:
            add_this_node = True

    if not add_this_node:
        continue

    for out in node.output:
        if out and out not in existing_outputs:
            model.graph.output.append(
                helper.make_tensor_value_info(out, TensorProto.FLOAT, None)
            )
            existing_outputs.add(out)
            added += 1

onnx.save(model, TEMP_MODEL)

print("Temporary model saved:", TEMP_MODEL)
print("Added float graph outputs:", added)

x = np.fromfile(INPUT_BIN, dtype=np.float32).reshape(1, 13, 512, 512)

print("Creating ONNX Runtime session...")
sess = ort.InferenceSession(TEMP_MODEL, providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

print("Number of outputs to fetch:", len(output_names))
print("Running ONNX Runtime once to collect runtime shapes...")

outs = sess.run(output_names, {input_name: x})

shape_map = {}

for name, arr in zip(output_names, outs):
    shape_map[name] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size)
    }

shape_map[input_name] = {
    "shape": list(x.shape),
    "dtype": str(x.dtype),
    "numel": int(x.size)
}

with open(OUT_JSON, "w") as f:
    json.dump(shape_map, f, indent=2)

print("Saved:", OUT_JSON)
print("Collected tensors:", len(shape_map))
