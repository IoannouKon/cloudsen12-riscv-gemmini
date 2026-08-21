import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto, shape_inference
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"

os.makedirs("first_add", exist_ok=True)

model = onnx.load(model_path)
inferred = shape_inference.infer_shapes(model)

shape_map = {}
for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
    tt = vi.type.tensor_type
    if tt.HasField("shape"):
        dims = []
        for d in tt.shape.dim:
            dims.append(d.dim_value if d.dim_value else 0)
        shape_map[vi.name] = dims

add_node = None
for node in model.graph.node:
    if node.op_type == "Add":
        add_node = node
        break

if add_node is None:
    raise RuntimeError("No Add node found")

print("First Add node:", add_node.name)
print("Inputs:", list(add_node.input))
print("Outputs:", list(add_node.output))

add_a_name = add_node.input[0]
add_b_name = add_node.input[1]
add_out_name = add_node.output[0]

shape_a = shape_map.get(add_a_name)
shape_b = shape_map.get(add_b_name)
shape_out = shape_map.get(add_out_name)

print("Add A shape:", shape_a)
print("Add B shape:", shape_b)
print("Add output shape:", shape_out)

if shape_a is None or shape_b is None or shape_out is None:
    raise RuntimeError("Could not infer Add tensor shapes")

model.graph.output.append(helper.make_tensor_value_info(add_a_name, TensorProto.FLOAT, shape_a))
model.graph.output.append(helper.make_tensor_value_info(add_b_name, TensorProto.FLOAT, shape_b))
model.graph.output.append(helper.make_tensor_value_info(add_out_name, TensorProto.FLOAT, shape_out))

temp_model_path = "experiments/operator_tests/first_add/temp_first_add_io.onnx"
onnx.save(model, temp_model_path)

x = np.fromfile(input_path, dtype=np.float32).reshape(1, 13, 512, 512)
sess = ort.InferenceSession(temp_model_path, providers=["CPUExecutionProvider"])
a, b, ref = sess.run([add_a_name, add_b_name, add_out_name], {"input": x})

a = a.astype(np.float32)
b = b.astype(np.float32)
ref = ref.astype(np.float32)

print("Reference A shape:", a.shape)
print("Reference B shape:", b.shape)
print("Reference output shape:", ref.shape)

a.tofile("experiments/operator_tests/first_add/add_a.bin")
b.tofile("experiments/operator_tests/first_add/add_b.bin")
ref.tofile("experiments/operator_tests/first_add/add_ref.bin")

count = int(np.prod(ref.shape))
with open("include/first_add_meta.h", "w") as f:
    f.write("#ifndef FIRST_ADD_META_H\n")
    f.write("#define FIRST_ADD_META_H\n\n")
    f.write(f"#define FA_COUNT {count}\n")
    f.write("#endif\n")

print("Saved:")
print("  experiments/operator_tests/first_add/add_a.bin")
print("  experiments/operator_tests/first_add/add_b.bin")
print("  experiments/operator_tests/first_add/add_ref.bin")
print("  include/first_add_meta.h")
