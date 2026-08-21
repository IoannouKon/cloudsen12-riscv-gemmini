import json

PLAN = "config/c_execution_plan.json"
OUT_C = "src/cloudsen12_forward_todo.c"
OUT_H = "include/cloudsen12_forward.h"

with open(PLAN) as f:
    plan = json.load(f)

with open(OUT_H, "w") as f:
    f.write('#ifndef CLOUDSEN12_FORWARD_H\n')
    f.write('#define CLOUDSEN12_FORWARD_H\n\n')
    f.write('void cloudsen12_forward_todo(const float *input, float *logits);\n\n')
    f.write('#endif\n')

with open(OUT_C, "w") as f:
    f.write('#include "cloudsen12_forward.h"\n')
    f.write('#include "conv2d.h"\n')
    f.write('#include "clip.h"\n')
    f.write('#include "relu.h"\n')
    f.write('#include "add.h"\n')
    f.write('#include "concat.h"\n')
    f.write('#include "resize.h"\n\n')
    f.write('/*\n')
    f.write('  Auto-generated TODO skeleton.\n')
    f.write('  This is NOT full inference yet. It lists the exact ONNX node order.\n')
    f.write('  Next step is to add tensor allocation/loading and replace TODO comments with kernel calls.\n')
    f.write('*/\n\n')
    f.write('void cloudsen12_forward_todo(const float *input, float *logits) {\n')
    f.write('    (void)input;\n')
    f.write('    (void)logits;\n\n')
    for e in plan:
        op=e['op_type']
        idx=e['index']
        name=e['name'].replace('*/','')
        f.write(f'    // Node {idx}: {op} - {name}\n')
        f.write(f'    // status: {e["status"]}\n')
        f.write(f'    // inputs: {e["inputs"]}\n')
        f.write(f'    // input_shapes: {e["input_shapes"]}\n')
        f.write(f'    // outputs: {e["outputs"]}\n')
        f.write(f'    // output_shapes: {e["output_shapes"]}\n')
        if e['status'] == 'supported':
            f.write(f'    // TODO: call pure C {op} kernel here.\n\n')
        else:
            f.write('    // IGNORE/TODO: bookkeeping node or unsupported node.\n\n')
    f.write('}\n')

print("Saved:", OUT_H)
print("Saved:", OUT_C)
