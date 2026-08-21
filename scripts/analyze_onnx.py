import onnx
from collections import Counter

model_path = "models/cloudsen12_unetmobv2_v2.onnx"

print("Loading:", model_path)
model = onnx.load(model_path)

print("\n=== Basic Info ===")
print("IR version:", model.ir_version)
print("Producer:", model.producer_name)

print("Opset imports:")
for opset in model.opset_import:
    print("  domain:", opset.domain, "version:", opset.version)

nodes = list(model.graph.node)
initializers = {init.name: init for init in model.graph.initializer}

print("\n=== Node Count ===")
print("Total nodes:", len(nodes))

ops = Counter(node.op_type for node in nodes)

print("\n=== Operator Counts ===")
for op, count in ops.most_common():
    print(f"{op}: {count}")

print("\n=== Graph Inputs ===")
for inp in model.graph.input:
    print(inp.name)

print("\n=== Graph Outputs ===")
for out in model.graph.output:
    print(out.name)

print("\n=== Initializers / Weights ===")
total_weight_bytes = 0

for init in model.graph.initializer:
    numel = 1
    for d in init.dims:
        numel *= d
    total_weight_bytes += numel * 4

print("Number of initializers:", len(model.graph.initializer))
print("Approx FP32 weight bytes:", total_weight_bytes)
print("Approx FP32 weight MB:", total_weight_bytes / (1024 * 1024))

print("\n=== First 120 Nodes ===")

for i, node in enumerate(nodes[:120]):
    print(f"\n[{i}] {node.op_type}  name={node.name}")
    print("  inputs:")

    for x in node.input:
        tag = "WEIGHT" if x in initializers else "TENSOR"
        print(f"    {x}  ({tag})")

    print("  outputs:")
    for x in node.output:
        print(f"    {x}")

print("\n=== Conv Nodes Summary ===")

conv_count = 0

for i, node in enumerate(nodes):
    if node.op_type == "Conv":
        conv_count += 1
        print(f"\nConv #{conv_count} node_index={i} name={node.name}")
        print("  inputs:", list(node.input))
        print("  outputs:", list(node.output))

        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                print("  attr", attr.name, list(attr.ints))
            elif attr.type == onnx.AttributeProto.INT:
                print("  attr", attr.name, attr.i)
            elif attr.type == onnx.AttributeProto.FLOAT:
                print("  attr", attr.name, attr.f)
            elif attr.type == onnx.AttributeProto.STRING:
                print("  attr", attr.name, attr.s)

print("\nDone.")
