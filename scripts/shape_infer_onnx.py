import onnx
from onnx import shape_inference

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
out_path = "models/cloudsen12_unetmobv2_v2_shapes.onnx"

print("Loading:", model_path)
model = onnx.load(model_path)

print("Running ONNX shape inference...")
inferred = shape_inference.infer_shapes(model)

onnx.save(inferred, out_path)
print("Saved:", out_path)

def shape_to_list(value_info):
    t = value_info.type.tensor_type
    dims = []
    for d in t.shape.dim:
        if d.dim_value:
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims

print("\n=== ValueInfo Tensor Shapes ===")
for vi in inferred.graph.value_info:
    print(vi.name, shape_to_list(vi))

print("\n=== Output Shapes ===")
for vi in inferred.graph.output:
    print(vi.name, shape_to_list(vi))
