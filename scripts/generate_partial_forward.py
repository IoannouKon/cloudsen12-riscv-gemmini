import json
from pathlib import Path
import os
import re
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

PLAN = "config/c_execution_plan.json"
MANIFEST = "config/model_manifest.json"
MODEL = "models/cloudsen12_unetmobv2_v2.onnx"
INPUT_BIN = "inputs/input_1x13x512x512_float32.bin"
OUT_C = "src/test_partial_forward.c"
OUT_META = "include/partial_forward_meta.h"
OUT_REF = "experiments/partial_forward/partial_forward/partial_ref.bin"
OUT_ONNX = "experiments/partial_forward/partial_forward/temp_partial_output.onnx"

# Number of compute nodes to generate. Keep this modest first.
# Equivalent to a multi-layer sanity check before full model generation.
TARGET_SUPPORTED_OPS = int(os.environ.get("TARGET_SUPPORTED_OPS", "20"))
SUPPORTED = {"Conv", "Clip", "Relu", "Add"}

os.makedirs("partial_forward", exist_ok=True)

with open(PLAN) as f:
    plan = json.load(f)
with open(MANIFEST) as f:
    manifest = json.load(f)

init_map = {x["name"]: x for x in manifest["initializers"]}

# Pick first TARGET_SUPPORTED_OPS among simple supported ops.
selected = []
for e in plan:
    if e["status"] == "supported" and e["op_type"] in SUPPORTED:
        selected.append(e)
    if len(selected) >= TARGET_SUPPORTED_OPS:
        break

if not selected:
    raise RuntimeError("No supported nodes selected")

last_tensor = selected[-1]["outputs"][0]
last_shape = selected[-1]["output_shapes"][0]
print("Selected supported ops:", len(selected))
print("Last selected node:", selected[-1]["index"], selected[-1]["op_type"], selected[-1]["name"])
print("Last tensor:", last_tensor)
print("Last shape:", last_shape)

if last_shape is None or any(x is None for x in last_shape):
    raise RuntimeError("Last tensor has unknown shape")

# Export ONNX Runtime reference for last tensor.
model = onnx.load(MODEL)
model.graph.output.append(helper.make_tensor_value_info(last_tensor, TensorProto.FLOAT, last_shape))
onnx.save(model, OUT_ONNX)

x = np.fromfile(INPUT_BIN, dtype=np.float32).reshape(1,13,512,512)
sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
ref = sess.run([last_tensor], {"input": x})[0].astype(np.float32)
ref.tofile(OUT_REF)
print("Saved reference:", OUT_REF, ref.shape)

# Utility for C names.
def cname(s):
    s = s.replace("/", "_").replace(".", "_").replace(":", "_").replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = s.strip("_")
    if not s:
        s = "tensor"
    if s[0].isdigit():
        s = "t_" + s
    return s

# Tensor variable names and shapes.
tensor_vars = {"input": "input"}
tensor_shapes = {"input": [1,13,512,512]}
allocs = []
code_lines = []
weight_loads = []
cleanup = []

# Create output buffers and op calls.
for e in selected:
    op = e["op_type"]
    out_name = e["outputs"][0]
    out_shape = e["output_shapes"][0]
    out_var = cname(out_name)
    tensor_vars[out_name] = out_var
    tensor_shapes[out_name] = out_shape
    out_count_expr = " * ".join(str(int(x)) for x in out_shape)
    allocs.append(f"    float *{out_var} = (float *)malloc((size_t)({out_count_expr}) * sizeof(float));")
    cleanup.append(f"    free({out_var});")

    if op == "Conv":
        inp_name, w_name, b_name = e["inputs"][:3]
        inp_var = tensor_vars[inp_name]
        in_shape = e["input_shapes"][0]
        w_shape = e["input_shapes"][1]
        b_shape = e["input_shapes"][2]
        attrs = e["attrs"]
        pads = attrs.get("pads", [0,0,0,0])
        strides = attrs.get("strides", [1,1])
        dilations = attrs.get("dilations", [1,1])
        groups = attrs.get("group", 1)
        w_info = init_map[w_name]
        b_info = init_map[b_name]
        w_var = cname(w_name)
        b_var = cname(b_name)
        w_count = " * ".join(str(int(x)) for x in w_shape)
        b_count = " * ".join(str(int(x)) for x in b_shape)
        weight_loads.append(f"    float *{w_var} = (float *)malloc((size_t)({w_count}) * sizeof(float));")
        weight_loads.append(f"    float *{b_var} = (float *)malloc((size_t)({b_count}) * sizeof(float));")
        weight_loads.append(f"    if (!read_float_file(\"{w_info['file']}\", {w_var}, (size_t)({w_count}))) {{ printf(\"Failed weight {w_name}\\n\"); return 1; }}")
        weight_loads.append(f"    if (!read_float_file(\"{b_info['file']}\", {b_var}, (size_t)({b_count}))) {{ printf(\"Failed bias {b_name}\\n\"); return 1; }}")
        cleanup.append(f"    free({w_var});")
        cleanup.append(f"    free({b_var});")
        n,c_in,h_in,w_in = in_shape
        c_out,_,kh,kw = w_shape
        _,_,h_out,w_out = out_shape
        code_lines.append(f"    // Node {e['index']}: Conv")
        code_lines.append(f"    conv2d_nchw({inp_var}, {w_var}, {b_var}, {out_var}, {n}, {c_in}, {h_in}, {w_in}, {c_out}, {kh}, {kw}, {h_out}, {w_out}, {pads[0]}, {pads[1]}, {strides[0]}, {strides[1]}, {dilations[0]}, {dilations[1]}, {groups});")
    elif op == "Clip":
        inp_var = tensor_vars[e["inputs"][0]]
        count = " * ".join(str(int(x)) for x in out_shape)
        code_lines.append(f"    // Node {e['index']}: Clip/ReLU6")
        code_lines.append(f"    clip_float({inp_var}, {out_var}, (int)({count}), 0.0f, 6.0f);")
    elif op == "Relu":
        inp_var = tensor_vars[e["inputs"][0]]
        count = " * ".join(str(int(x)) for x in out_shape)
        code_lines.append(f"    // Node {e['index']}: Relu")
        code_lines.append(f"    relu_float({inp_var}, {out_var}, (int)({count}));")
    elif op == "Add":
        a_var = tensor_vars[e["inputs"][0]]
        b_var = tensor_vars[e["inputs"][1]]
        count = " * ".join(str(int(x)) for x in out_shape)
        code_lines.append(f"    // Node {e['index']}: Add")
        code_lines.append(f"    add_float({a_var}, {b_var}, {out_var}, (int)({count}));")

last_var = tensor_vars[last_tensor]
last_count = int(np.prod(last_shape))
with open(OUT_META, "w") as f:
    f.write("#ifndef PARTIAL_FORWARD_META_H\n#define PARTIAL_FORWARD_META_H\n\n")
    f.write(f"#define PF_LAST_COUNT {last_count}\n")
    f.write("#endif\n")

c = []
c.append('#include "conv2d.h"')
c.append('#include "clip.h"')
c.append('#include "relu.h"')
c.append('#include "add.h"')
c.append('#include "partial_forward_meta.h"')
c.append('#include <stdio.h>')
c.append('#include <stdlib.h>')
c.append('#include <math.h>')
c.append('')
c.append('static int read_float_file(const char *path, float *data, size_t count) {')
c.append('    FILE *f = fopen(path, "rb");')
c.append('    if (!f) return 0;')
c.append('    size_t n = fread(data, sizeof(float), count, f);')
c.append('    fclose(f);')
c.append('    return n == count;')
c.append('}')
c.append('')
c.append('int main(void) {')
c.append('    size_t input_count = (size_t)1 * 13 * 512 * 512;')
c.append('    float *input = (float *)malloc(input_count * sizeof(float));')
c.append('    float *ref = (float *)malloc((size_t)PF_LAST_COUNT * sizeof(float));')
c.append('    if (!input || !ref) { printf("Allocation failed\\n"); return 1; }')
c.append('    if (!read_float_file("inputs/input_1x13x512x512_float32.bin", input, input_count)) { printf("Failed input\\n"); return 1; }')
c.append(f'    if (!read_float_file("{OUT_REF}", ref, (size_t)PF_LAST_COUNT)) {{ printf("Failed partial ref\\n"); return 1; }}')
c.extend(weight_loads)
c.extend(allocs)
c.append('    printf("Running generated partial pure C forward...\\n");')
c.extend(code_lines)
c.append('    double max_abs = 0.0;')
c.append('    double mean_abs = 0.0;')
c.append(f'    for (size_t i = 0; i < (size_t)PF_LAST_COUNT; i++) {{')
c.append(f'        double diff = fabs((double){last_var}[i] - (double)ref[i]);')
c.append('        if (diff > max_abs) max_abs = diff;')
c.append('        mean_abs += diff;')
c.append('    }')
c.append('    mean_abs /= (double)PF_LAST_COUNT;')
c.append('    printf("Partial output count: %d\\n", PF_LAST_COUNT);')
c.append('    printf("Mean abs error vs ONNX Runtime: %.10f\\n", mean_abs);')
c.append('    printf("Max abs error vs ONNX Runtime: %.10f\\n", max_abs);')
for line in cleanup:
    c.append(line)
c.append('    free(input);')
c.append('    free(ref);')
c.append('    return 0;')
c.append('}')

Path(OUT_C).write_text("\n".join(c))
print("Saved:", OUT_C)
print("Saved:", OUT_META)
print("Compile with:")
print("gcc -O2 -Wall -Wextra -std=c11 -Iinclude src/test_partial_forward.c src/conv2d.c src/clip.c src/relu.c src/add.c -lm -o build/test_partial_forward")
