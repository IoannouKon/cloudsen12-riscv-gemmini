import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto, shape_inference
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"

os.makedirs("first_clip", exist_ok=True)

model = onnx.load(model_path)
inferred = shape_inference.infer_shapes(model)

clip_node = None
for node in model.graph.node:
    if node.op_type == "Clip":
        clip_node = node
        break

if clip_node is None:
    raise RuntimeError("No Clip node found")

print("First Clip node:", clip_node.name)
print("Inputs:", list(clip_node.input))
print("Outputs:", list(clip_node.output))

output_name = clip_node.output[0]

out_shape = None
for vi in inferred.graph.value_info:
    if vi.name == output_name:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else 0)
        out_shape = dims
        break

print("First Clip output shape:", out_shape)

if out_shape is None:
    raise RuntimeError("Could not infer first Clip output shape")

model.graph.output.append(
    helper.make_tensor_value_info(output_name, TensorProto.FLOAT, out_shape)
)

temp_model_path = "experiments/operator_tests/first_clip/temp_first_clip_output.onnx"
onnx.save(model, temp_model_path)

x = np.fromfile(input_path, dtype=np.float32).reshape(1, 13, 512, 512)

sess = ort.InferenceSession(temp_model_path, providers=["CPUExecutionProvider"])
ref = sess.run([output_name], {"input": x})[0].astype(np.float32)

print("Reference Clip output shape:", ref.shape)

ref.tofile("experiments/operator_tests/first_clip/clip_ref.bin")

print("Saved: experiments/operator_tests/first_clip/clip_ref.bin")
