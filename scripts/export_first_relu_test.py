import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto, shape_inference
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"

os.makedirs("first_relu", exist_ok=True)

model = onnx.load(model_path)
inferred = shape_inference.infer_shapes(model)

# Build shape map
shape_map = {}
for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
    dims = []
    tt = vi.type.tensor_type
    if tt.HasField("shape"):
        for d in tt.shape.dim:
            dims.append(d.dim_value if d.dim_value else 0)
        shape_map[vi.name] = dims

relu_node = None
for node in model.graph.node:
    if node.op_type == "Relu":
        relu_node = node
        break

if relu_node is None:
    raise RuntimeError("No Relu node found")

print("First Relu node:", relu_node.name)
print("Inputs:", list(relu_node.input))
print("Outputs:", list(relu_node.output))

relu_input_name = relu_node.input[0]
relu_output_name = relu_node.output[0]

in_shape = shape_map.get(relu_input_name)
out_shape = shape_map.get(relu_output_name)

print("Relu input shape:", in_shape)
print("Relu output shape:", out_shape)

if in_shape is None or out_shape is None:
    raise RuntimeError("Could not infer Relu tensor shapes")

# Add Relu input and output as model outputs
model.graph.output.append(
    helper.make_tensor_value_info(relu_input_name, TensorProto.FLOAT, in_shape)
)
model.graph.output.append(
    helper.make_tensor_value_info(relu_output_name, TensorProto.FLOAT, out_shape)
)

temp_model_path = "experiments/operator_tests/first_relu/temp_first_relu_io.onnx"
onnx.save(model, temp_model_path)

x = np.fromfile(input_path, dtype=np.float32).reshape(1, 13, 512, 512)

sess = ort.InferenceSession(temp_model_path, providers=["CPUExecutionProvider"])
relu_in, relu_out = sess.run([relu_input_name, relu_output_name], {"input": x})

relu_in = relu_in.astype(np.float32)
relu_out = relu_out.astype(np.float32)

print("Reference input shape:", relu_in.shape)
print("Reference output shape:", relu_out.shape)

relu_in.tofile("experiments/operator_tests/first_relu/relu_input.bin")
relu_out.tofile("experiments/operator_tests/first_relu/relu_ref.bin")

count = int(np.prod(relu_out.shape))
with open("include/first_relu_meta.h", "w") as f:
    f.write("#ifndef FIRST_RELU_META_H\n")
    f.write("#define FIRST_RELU_META_H\n\n")
    f.write(f"#define FR_COUNT {count}\n")
    f.write("#endif\n")

print("Saved:")
print("  experiments/operator_tests/first_relu/relu_input.bin")
print("  experiments/operator_tests/first_relu/relu_ref.bin")
print("  include/first_relu_meta.h")
