import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"
os.makedirs("first_concat", exist_ok=True)

model = onnx.load(model_path)

# Select a real feature-map concat: axis=1 and exactly two inputs.
# We do NOT require static ONNX shape inference, because this graph has dynamic Resize/Concat shapes.
concat_node = None
axis = None
for node in model.graph.node:
    if node.op_type != "Concat":
        continue
    node_axis = 0
    for attr in node.attribute:
        if attr.name == "axis":
            node_axis = int(attr.i)
    if node_axis == 1 and len(node.input) == 2:
        concat_node = node
        axis = node_axis
        break

if concat_node is None:
    raise RuntimeError("No channel-axis Concat node with two inputs found")

print("Selected Concat node:", concat_node.name)
print("Axis:", axis)
print("Inputs:", list(concat_node.input))
print("Outputs:", list(concat_node.output))

out_name = concat_node.output[0]
input_names = list(concat_node.input)

# Add output and input tensors as graph outputs with unknown shapes.
# This avoids ONNX Runtime failures caused by bad static shape inference values like [0,0,0,0].
model.graph.output.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None))
for name in input_names:
    model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))

temp_model_path = "experiments/operator_tests/first_concat/temp_first_concat_io.onnx"
onnx.save(model, temp_model_path)

x = np.fromfile(input_path, dtype=np.float32).reshape(1,13,512,512)
sess = ort.InferenceSession(temp_model_path, providers=["CPUExecutionProvider"])
fetches = [out_name] + input_names
outs = sess.run(fetches, {"input": x})

ref = outs[0].astype(np.float32)
inputs = [o.astype(np.float32) for o in outs[1:]]

print("Runtime output shape:", ref.shape)
for i, arr in enumerate(inputs):
    print(f"Runtime input {i} shape:", arr.shape)

# Validate tensor rank and same N/H/W.
if len(ref.shape) != 4:
    raise RuntimeError(f"Concat output is not NCHW rank-4: {ref.shape}")
for i, arr in enumerate(inputs):
    if len(arr.shape) != 4:
        raise RuntimeError(f"Concat input {i} is not NCHW rank-4: {arr.shape}")
    if arr.shape[0] != ref.shape[0] or arr.shape[2] != ref.shape[2] or arr.shape[3] != ref.shape[3]:
        raise RuntimeError(f"Concat input {i} N/H/W does not match output")

expected_c = sum(arr.shape[1] for arr in inputs)
if expected_c != ref.shape[1]:
    raise RuntimeError(f"Channel mismatch: inputs total C={expected_c}, output C={ref.shape[1]}")

ref.tofile("experiments/operator_tests/first_concat/concat_ref.bin")
channels=[]
for i, arr in enumerate(inputs):
    arr.tofile(f"experiments/operator_tests/first_concat/concat_in_{i}.bin")
    channels.append(arr.shape[1])

n, c_out, h, w = ref.shape
with open("include/first_concat_meta.h", "w") as f:
    f.write("#ifndef FIRST_CONCAT_META_H\n#define FIRST_CONCAT_META_H\n\n")
    f.write(f"#define FCAT_NUM_INPUTS {len(inputs)}\n")
    f.write(f"#define FCAT_N {n}\n")
    f.write(f"#define FCAT_H {h}\n")
    f.write(f"#define FCAT_W {w}\n")
    f.write(f"#define FCAT_C_OUT {c_out}\n")
    for i,ch in enumerate(channels):
        f.write(f"#define FCAT_C_IN_{i} {ch}\n")
    f.write("#endif\n")

print("Saved:")
print("  experiments/operator_tests/first_concat/concat_ref.bin")
for i in range(len(inputs)):
    print(f"  experiments/operator_tests/first_concat/concat_in_{i}.bin")
print("  include/first_concat_meta.h")
