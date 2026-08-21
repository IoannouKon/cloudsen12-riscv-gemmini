import onnx
import numpy as np
import onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto, shape_inference
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"

os.makedirs("first_conv", exist_ok=True)

model = onnx.load(model_path)
inferred = shape_inference.infer_shapes(model)

# Find first Conv node
conv_node = None
for node in model.graph.node:
    if node.op_type == "Conv":
        conv_node = node
        break

if conv_node is None:
    raise RuntimeError("No Conv node found")

print("First Conv node:", conv_node.name)
print("Inputs:", list(conv_node.input))
print("Outputs:", list(conv_node.output))

input_name = conv_node.input[0]
weight_name = conv_node.input[1]
bias_name = conv_node.input[2] if len(conv_node.input) > 2 else None
output_name = conv_node.output[0]

initializers = {init.name: init for init in model.graph.initializer}

w = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
if bias_name is not None:
    b = numpy_helper.to_array(initializers[bias_name]).astype(np.float32)
else:
    b = np.zeros((w.shape[0],), dtype=np.float32)

attrs = {
    "pads": [0, 0, 0, 0],
    "strides": [1, 1],
    "dilations": [1, 1],
    "group": 1
}

for attr in conv_node.attribute:
    if attr.name == "pads":
        attrs["pads"] = list(attr.ints)
    elif attr.name == "strides":
        attrs["strides"] = list(attr.ints)
    elif attr.name == "dilations":
        attrs["dilations"] = list(attr.ints)
    elif attr.name == "group":
        attrs["group"] = int(attr.i)

print("Weight shape:", w.shape)
print("Bias shape:", b.shape)
print("Attrs:", attrs)

# Add first Conv output as model output
shape_dict = {}
for vi in inferred.graph.value_info:
    if vi.name == output_name:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else 0)
        shape_dict[output_name] = dims

out_shape = shape_dict.get(output_name)
print("First Conv output shape:", out_shape)

if out_shape is None:
    raise RuntimeError("Could not infer first Conv output shape")

# Append intermediate output
model.graph.output.append(
    helper.make_tensor_value_info(output_name, TensorProto.FLOAT, out_shape)
)

temp_model_path = "experiments/operator_tests/first_conv/temp_first_conv_output.onnx"
onnx.save(model, temp_model_path)

# Run ONNX Runtime and get first Conv output
x = np.fromfile(input_path, dtype=np.float32).reshape(1, 13, 512, 512)

sess = ort.InferenceSession(temp_model_path, providers=["CPUExecutionProvider"])
outputs = sess.run([output_name], {"input": x})
ref = outputs[0].astype(np.float32)

print("Reference output shape:", ref.shape)

# Save files
w.tofile("experiments/operator_tests/first_conv/conv_w.bin")
b.tofile("experiments/operator_tests/first_conv/conv_b.bin")
ref.tofile("experiments/operator_tests/first_conv/conv_ref.bin")

# Generate C header with metadata
N, C_IN, H_IN, W_IN = x.shape
C_OUT, C_PER_GROUP, KH, KW = w.shape
_, _, H_OUT, W_OUT = ref.shape

with open("include/first_conv_meta.h", "w") as f:
    f.write("#ifndef FIRST_CONV_META_H\n")
    f.write("#define FIRST_CONV_META_H\n\n")

    f.write(f"#define FC_N {N}\n")
    f.write(f"#define FC_C_IN {C_IN}\n")
    f.write(f"#define FC_H_IN {H_IN}\n")
    f.write(f"#define FC_W_IN {W_IN}\n")

    f.write(f"#define FC_C_OUT {C_OUT}\n")
    f.write(f"#define FC_KH {KH}\n")
    f.write(f"#define FC_KW {KW}\n")

    f.write(f"#define FC_H_OUT {H_OUT}\n")
    f.write(f"#define FC_W_OUT {W_OUT}\n")

    f.write(f"#define FC_PAD_TOP {attrs['pads'][0]}\n")
    f.write(f"#define FC_PAD_LEFT {attrs['pads'][1]}\n")
    f.write(f"#define FC_PAD_BOTTOM {attrs['pads'][2]}\n")
    f.write(f"#define FC_PAD_RIGHT {attrs['pads'][3]}\n")

    f.write(f"#define FC_STRIDE_H {attrs['strides'][0]}\n")
    f.write(f"#define FC_STRIDE_W {attrs['strides'][1]}\n")

    f.write(f"#define FC_DILATION_H {attrs['dilations'][0]}\n")
    f.write(f"#define FC_DILATION_W {attrs['dilations'][1]}\n")

    f.write(f"#define FC_GROUP {attrs['group']}\n\n")
    f.write("#endif\n")

print("Saved:")
print("  experiments/operator_tests/first_conv/conv_w.bin")
print("  experiments/operator_tests/first_conv/conv_b.bin")
print("  experiments/operator_tests/first_conv/conv_ref.bin")
print("  include/first_conv_meta.h")
