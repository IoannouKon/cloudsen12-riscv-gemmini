import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto, shape_inference
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"
os.makedirs("first_resize", exist_ok=True)

model = onnx.load(model_path)
inferred = shape_inference.infer_shapes(model)

shape_map = {}
for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
    tt = vi.type.tensor_type
    if tt.HasField("shape"):
        dims=[]
        for d in tt.shape.dim:
            dims.append(d.dim_value if d.dim_value else 0)
        shape_map[vi.name]=dims

resize_node=None
for node in model.graph.node:
    if node.op_type == "Resize":
        resize_node=node
        break
if resize_node is None:
    raise RuntimeError("No Resize node found")

attrs={}
for attr in resize_node.attribute:
    if attr.type == onnx.AttributeProto.STRING:
        attrs[attr.name]=attr.s.decode()
    elif attr.type == onnx.AttributeProto.INT:
        attrs[attr.name]=int(attr.i)
    elif attr.type == onnx.AttributeProto.FLOAT:
        attrs[attr.name]=float(attr.f)
    elif attr.type == onnx.AttributeProto.INTS:
        attrs[attr.name]=list(attr.ints)
print("Selected Resize node:", resize_node.name)
print("Inputs:", list(resize_node.input))
print("Outputs:", list(resize_node.output))
print("Attrs:", attrs)

inp_name = resize_node.input[0]
out_name = resize_node.output[0]
in_shape = shape_map.get(inp_name)
print("Resize input inferred shape:", in_shape)

if in_shape is None or len(in_shape) != 4:
    raise RuntimeError("Could not infer Resize input shape")

# IMPORTANT: do not declare output shape because this resize gets size dynamically.
model.graph.output.append(helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, in_shape))
model.graph.output.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None))

temp_model_path="experiments/operator_tests/first_resize/temp_first_resize_io.onnx"
onnx.save(model,temp_model_path)

x=np.fromfile(input_path,dtype=np.float32).reshape(1,13,512,512)
sess=ort.InferenceSession(temp_model_path,providers=["CPUExecutionProvider"])
resize_in, ref=sess.run([inp_name,out_name], {"input": x})
resize_in=resize_in.astype(np.float32)
ref=ref.astype(np.float32)
resize_in.tofile("experiments/operator_tests/first_resize/resize_input.bin")
ref.tofile("experiments/operator_tests/first_resize/resize_ref.bin")

print("Resize runtime input shape:", resize_in.shape)
print("Resize runtime output shape:", ref.shape)

n,c,h_in,w_in=resize_in.shape
_,_,h_out,w_out=ref.shape
with open("include/first_resize_meta.h","w") as f:
    f.write("#ifndef FIRST_RESIZE_META_H\n#define FIRST_RESIZE_META_H\n\n")
    f.write(f"#define FRS_N {n}\n#define FRS_C {c}\n#define FRS_H_IN {h_in}\n#define FRS_W_IN {w_in}\n")
    f.write(f"#define FRS_H_OUT {h_out}\n#define FRS_W_OUT {w_out}\n")
    f.write("#endif\n")
print("Saved first_resize files and include/first_resize_meta.h")
