
#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

PLAN = "config/c_execution_plan.json"
MANIFEST = "config/model_manifest.json"
MODEL = "models/cloudsen12_unetmobv2_v2.onnx"
INPUT_BIN = "inputs/input_1x13x512x512_float32.bin"
OUT_C = "backends/pure_c_fp32/src/test_partial_forward_v2.c"
OUT_META = "backends/pure_c_fp32/include/partial_forward_v2_meta.h"
OUT_REF = "experiments/partial_forward/partial_forward_v2/partial_ref.bin"
OUT_ONNX = "experiments/partial_forward/partial_forward_v2/temp_partial_output.onnx"
TARGET_SUPPORTED_OPS = int(os.environ.get("TARGET_SUPPORTED_OPS", "100"))
SUPPORTED = {"Conv", "Clip", "Relu", "Add", "Concat", "Resize"}
Path("partial_forward_v2").mkdir(exist_ok=True)

with open(PLAN) as f:
    plan = json.load(f)
with open(MANIFEST) as f:
    manifest = json.load(f)
init_map = {x["name"]: x for x in manifest["initializers"]}

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

model = onnx.load(MODEL)
model.graph.output.append(helper.make_tensor_value_info(last_tensor, TensorProto.FLOAT, last_shape))
onnx.save(model, OUT_ONNX)
x = np.fromfile(INPUT_BIN, dtype=np.float32).reshape(1, 13, 512, 512)
sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
ref = sess.run([last_tensor], {"input": x})[0].astype(np.float32)
ref.tofile(OUT_REF)
print("Saved reference:", OUT_REF, ref.shape)

def cname(s):
    s = s.replace("/", "_").replace(".", "_").replace(":", "_").replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_")
    if not s:
        s = "tensor"
    if s[0].isdigit():
        s = "t_" + s
    return s

def count_expr(shape):
    return " * ".join(str(int(x)) for x in shape)

def numel(shape):
    v = 1
    for x in shape:
        v *= int(x)
    return v

tensor_vars = {"input": "input"}
allocs, weight_loads, code_lines, cleanup = [], [], [], []

for e in selected:
    op = e["op_type"]
    out_name = e["outputs"][0]
    out_shape = e["output_shapes"][0]
    if out_shape is None or any(x is None for x in out_shape):
        raise RuntimeError(f"Unknown output shape for node {e['index']} {op}")
    out_var = cname(out_name)
    tensor_vars[out_name] = out_var
    allocs.append(f"    float *{out_var} = (float *)malloc((size_t)({count_expr(out_shape)}) * sizeof(float));")
    cleanup.append(f"    free({out_var});")

    if op == "Conv":
        inp_name, w_name, b_name = e["inputs"][:3]
        if inp_name not in tensor_vars:
            raise RuntimeError(f"Missing tensor for Conv input {inp_name} at node {e['index']}")
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
        w_count = count_expr(w_shape)
        b_count = count_expr(b_shape)
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
        inp = tensor_vars.get(e["inputs"][0])
        if inp is None:
            raise RuntimeError(f"Missing tensor for Clip input {e['inputs'][0]} at node {e['index']}")
        code_lines.append(f"    // Node {e['index']}: Clip/ReLU6")
        code_lines.append(f"    clip_float({inp}, {out_var}, (int)({count_expr(out_shape)}), 0.0f, 6.0f);")
    elif op == "Relu":
        inp = tensor_vars.get(e["inputs"][0])
        if inp is None:
            raise RuntimeError(f"Missing tensor for Relu input {e['inputs'][0]} at node {e['index']}")
        code_lines.append(f"    // Node {e['index']}: Relu")
        code_lines.append(f"    relu_float({inp}, {out_var}, (int)({count_expr(out_shape)}));")
    elif op == "Add":
        a = tensor_vars.get(e["inputs"][0])
        b = tensor_vars.get(e["inputs"][1])
        if a is None or b is None:
            raise RuntimeError(f"Missing tensor for Add inputs {e['inputs']} at node {e['index']}")
        code_lines.append(f"    // Node {e['index']}: Add")
        code_lines.append(f"    add_float({a}, {b}, {out_var}, (int)({count_expr(out_shape)}));")
    elif op == "Concat":
        axis = e["attrs"].get("axis", 0)
        if axis != 1:
            raise RuntimeError(f"Concat axis not supported in compute path: axis={axis} node {e['index']}")
        input_vars, channels = [], []
        for name, shape in zip(e["inputs"], e["input_shapes"]):
            if name not in tensor_vars:
                raise RuntimeError(f"Missing tensor for Concat input {name} at node {e['index']}")
            input_vars.append(tensor_vars[name])
            channels.append(int(shape[1]))
        n,c_out,h,w = out_shape
        arr_name = f"concat_inputs_{e['index']}"
        ch_name = f"concat_channels_{e['index']}"
        code_lines.append(f"    // Node {e['index']}: Concat axis=1")
        code_lines.append(f"    const float *{arr_name}[] = {{{', '.join(input_vars)}}};")
        code_lines.append(f"    const int {ch_name}[] = {{{', '.join(str(c) for c in channels)}}};")
        code_lines.append(f"    concat_channel_nchw({arr_name}, {ch_name}, {len(input_vars)}, {out_var}, {n}, {h}, {w});")
    elif op == "Resize":
        inp_name = e["inputs"][0]
        inp = tensor_vars.get(inp_name)
        if inp is None:
            raise RuntimeError(f"Missing tensor for Resize input {inp_name} at node {e['index']}")
        in_shape = e["input_shapes"][0]
        n,c,h_in,w_in = in_shape
        _,_,h_out,w_out = out_shape
        code_lines.append(f"    // Node {e['index']}: Resize nearest")
        code_lines.append(f"    resize_nearest_nchw({inp}, {out_var}, {n}, {c}, {h_in}, {w_in}, {h_out}, {w_out});")

last_var = tensor_vars[last_tensor]
last_count = numel(last_shape)
with open(OUT_META, "w") as f:
    f.write("#ifndef PARTIAL_FORWARD_V2_META_H\n#define PARTIAL_FORWARD_V2_META_H\n\n")
    f.write(f"#define PFV2_LAST_COUNT {last_count}\n")
    f.write("#endif\n")

c = []
c.extend([
    '#include "conv2d.h"', '#include "clip.h"', '#include "relu.h"', '#include "add.h"',
    '#include "concat.h"', '#include "resize.h"', '#include "partial_forward_v2_meta.h"',
    '#include <stdio.h>', '#include <stdlib.h>', '#include <math.h>', '',
    'static int read_float_file(const char *path, float *data, size_t count) {',
    '    FILE *f = fopen(path, "rb");', '    if (!f) return 0;',
    '    size_t n = fread(data, sizeof(float), count, f);', '    fclose(f);',
    '    return n == count;', '}', '', 'int main(void) {',
    '    size_t input_count = (size_t)1 * 13 * 512 * 512;',
    '    float *input = (float *)malloc(input_count * sizeof(float));',
    '    float *ref = (float *)malloc((size_t)PFV2_LAST_COUNT * sizeof(float));',
    '    if (!input || !ref) { printf("Allocation failed\\n"); return 1; }',
    '    if (!read_float_file("inputs/input_1x13x512x512_float32.bin", input, input_count)) { printf("Failed input\\n"); return 1; }',
    f'    if (!read_float_file("{OUT_REF}", ref, (size_t)PFV2_LAST_COUNT)) {{ printf("Failed partial ref\\n"); return 1; }}'
])
c.extend(weight_loads)
c.extend(allocs)
c.append('    printf("Running generated partial pure C forward v2...\\n");')
c.extend(code_lines)
c.extend([
    '    double max_abs = 0.0;', '    double mean_abs = 0.0;',
    '    for (size_t i = 0; i < (size_t)PFV2_LAST_COUNT; i++) {',
    f'        double diff = fabs((double){last_var}[i] - (double)ref[i]);',
    '        if (diff > max_abs) max_abs = diff;', '        mean_abs += diff;', '    }',
    '    mean_abs /= (double)PFV2_LAST_COUNT;',
    '    printf("Partial output count: %d\\n", PFV2_LAST_COUNT);',
    '    printf("Mean abs error vs ONNX Runtime: %.10f\\n", mean_abs);',
    '    printf("Max abs error vs ONNX Runtime: %.10f\\n", max_abs);'
])
c.extend(cleanup)
c.extend(['    free(input);', '    free(ref);', '    return 0;', '}'])
Path(OUT_C).write_text("\n".join(c))
print("Saved:", OUT_C)
print("Saved:", OUT_META)
print("Compile with:")
print("gcc -O2 -Wall -Wextra -std=c11 -Iinclude src/test_partial_forward_v2.c src/conv2d.c src/clip.c src/relu.c src/add.c src/concat.c src/resize.c -lm -o build/test_partial_forward_v2")
