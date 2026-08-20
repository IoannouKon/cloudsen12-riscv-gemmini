# CloudSEN12 Pure C Execution Plan

## Operator counts

- Constant: 90
- Conv: 63
- Clip: 35
- Add: 10
- Relu: 10
- Concat: 9
- Shape: 5
- Slice: 5
- Resize: 5

## Unsupported operators

None. All non-bookkeeping ops are covered by current pure C kernels.

## Notes

Constant, Shape, Slice, and Concat axis=0 nodes are bookkeeping nodes used for dynamic Resize shape construction. They are ignored in the first pure C plan. Their results are folded into known runtime tensor shapes.

## First 80 execution entries

### Node 0: Conv
name: `/encoder/features.0/features.0.0/Conv`

status: `supported`

inputs: `['input', 'onnx::Conv_715', 'onnx::Conv_716']`

input shapes: `[[1, 13, 512, 512], [32, 13, 3, 3], [32]]`

outputs: `['/encoder/features.0/features.0.0/Conv_output_0']`

output shapes: `[[1, 32, 256, 256]]`

### Node 1: Constant
name: `/encoder/features.0/features.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.0/features.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 2: Constant
name: `/encoder/features.0/features.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.0/features.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 3: Clip
name: `/encoder/features.0/features.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.0/features.0.0/Conv_output_0', '/encoder/features.0/features.0.2/Constant_output_0', '/encoder/features.0/features.0.2/Constant_1_output_0']`

input shapes: `[[1, 32, 256, 256], None, None]`

outputs: `['/encoder/features.0/features.0.2/Clip_output_0']`

output shapes: `[[1, 32, 256, 256]]`

### Node 4: Conv
name: `/encoder/features.1/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.0/features.0.2/Clip_output_0', 'onnx::Conv_718', 'onnx::Conv_719']`

input shapes: `[[1, 32, 256, 256], [32, 1, 3, 3], [32]]`

outputs: `['/encoder/features.1/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 32, 256, 256]]`

### Node 5: Constant
name: `/encoder/features.1/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.1/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 6: Constant
name: `/encoder/features.1/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.1/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 7: Clip
name: `/encoder/features.1/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.1/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.1/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.1/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 32, 256, 256], None, None]`

outputs: `['/encoder/features.1/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 32, 256, 256]]`

### Node 8: Conv
name: `/encoder/features.1/conv/conv.1/Conv`

status: `supported`

inputs: `['/encoder/features.1/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_721', 'onnx::Conv_722']`

input shapes: `[[1, 32, 256, 256], [16, 32, 1, 1], [16]]`

outputs: `['/encoder/features.1/conv/conv.1/Conv_output_0']`

output shapes: `[[1, 16, 256, 256]]`

### Node 9: Conv
name: `/encoder/features.2/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.1/conv/conv.1/Conv_output_0', 'onnx::Conv_724', 'onnx::Conv_725']`

input shapes: `[[1, 16, 256, 256], [96, 16, 1, 1], [96]]`

outputs: `['/encoder/features.2/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 96, 256, 256]]`

### Node 10: Constant
name: `/encoder/features.2/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.2/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 11: Constant
name: `/encoder/features.2/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.2/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 12: Clip
name: `/encoder/features.2/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.2/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.2/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.2/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 96, 256, 256], None, None]`

outputs: `['/encoder/features.2/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 96, 256, 256]]`

### Node 13: Conv
name: `/encoder/features.2/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.2/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_727', 'onnx::Conv_728']`

input shapes: `[[1, 96, 256, 256], [96, 1, 3, 3], [96]]`

outputs: `['/encoder/features.2/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 96, 128, 128]]`

### Node 14: Constant
name: `/encoder/features.2/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.2/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 15: Constant
name: `/encoder/features.2/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.2/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 16: Clip
name: `/encoder/features.2/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.2/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.2/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.2/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 96, 128, 128], None, None]`

outputs: `['/encoder/features.2/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 96, 128, 128]]`

### Node 17: Conv
name: `/encoder/features.2/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.2/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_730', 'onnx::Conv_731']`

input shapes: `[[1, 96, 128, 128], [24, 96, 1, 1], [24]]`

outputs: `['/encoder/features.2/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 24, 128, 128]]`

### Node 18: Conv
name: `/encoder/features.3/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.2/conv/conv.2/Conv_output_0', 'onnx::Conv_733', 'onnx::Conv_734']`

input shapes: `[[1, 24, 128, 128], [144, 24, 1, 1], [144]]`

outputs: `['/encoder/features.3/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 144, 128, 128]]`

### Node 19: Constant
name: `/encoder/features.3/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.3/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 20: Constant
name: `/encoder/features.3/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.3/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 21: Clip
name: `/encoder/features.3/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.3/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.3/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.3/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 144, 128, 128], None, None]`

outputs: `['/encoder/features.3/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 144, 128, 128]]`

### Node 22: Conv
name: `/encoder/features.3/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.3/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_736', 'onnx::Conv_737']`

input shapes: `[[1, 144, 128, 128], [144, 1, 3, 3], [144]]`

outputs: `['/encoder/features.3/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 144, 128, 128]]`

### Node 23: Constant
name: `/encoder/features.3/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.3/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 24: Constant
name: `/encoder/features.3/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.3/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 25: Clip
name: `/encoder/features.3/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.3/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.3/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.3/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 144, 128, 128], None, None]`

outputs: `['/encoder/features.3/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 144, 128, 128]]`

### Node 26: Conv
name: `/encoder/features.3/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.3/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_739', 'onnx::Conv_740']`

input shapes: `[[1, 144, 128, 128], [24, 144, 1, 1], [24]]`

outputs: `['/encoder/features.3/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 24, 128, 128]]`

### Node 27: Add
name: `/encoder/features.3/Add`

status: `supported`

inputs: `['/encoder/features.2/conv/conv.2/Conv_output_0', '/encoder/features.3/conv/conv.2/Conv_output_0']`

input shapes: `[[1, 24, 128, 128], [1, 24, 128, 128]]`

outputs: `['/encoder/features.3/Add_output_0']`

output shapes: `[[1, 24, 128, 128]]`

### Node 28: Conv
name: `/encoder/features.4/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.3/Add_output_0', 'onnx::Conv_742', 'onnx::Conv_743']`

input shapes: `[[1, 24, 128, 128], [144, 24, 1, 1], [144]]`

outputs: `['/encoder/features.4/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 144, 128, 128]]`

### Node 29: Constant
name: `/encoder/features.4/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.4/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 30: Constant
name: `/encoder/features.4/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.4/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 31: Clip
name: `/encoder/features.4/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.4/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.4/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.4/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 144, 128, 128], None, None]`

outputs: `['/encoder/features.4/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 144, 128, 128]]`

### Node 32: Conv
name: `/encoder/features.4/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.4/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_745', 'onnx::Conv_746']`

input shapes: `[[1, 144, 128, 128], [144, 1, 3, 3], [144]]`

outputs: `['/encoder/features.4/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 144, 64, 64]]`

### Node 33: Constant
name: `/encoder/features.4/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.4/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 34: Constant
name: `/encoder/features.4/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.4/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 35: Clip
name: `/encoder/features.4/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.4/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.4/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.4/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 144, 64, 64], None, None]`

outputs: `['/encoder/features.4/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 144, 64, 64]]`

### Node 36: Conv
name: `/encoder/features.4/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.4/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_748', 'onnx::Conv_749']`

input shapes: `[[1, 144, 64, 64], [32, 144, 1, 1], [32]]`

outputs: `['/encoder/features.4/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 32, 64, 64]]`

### Node 37: Conv
name: `/encoder/features.5/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.4/conv/conv.2/Conv_output_0', 'onnx::Conv_751', 'onnx::Conv_752']`

input shapes: `[[1, 32, 64, 64], [192, 32, 1, 1], [192]]`

outputs: `['/encoder/features.5/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 38: Constant
name: `/encoder/features.5/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.5/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 39: Constant
name: `/encoder/features.5/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.5/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 40: Clip
name: `/encoder/features.5/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.5/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.5/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.5/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 192, 64, 64], None, None]`

outputs: `['/encoder/features.5/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 41: Conv
name: `/encoder/features.5/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.5/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_754', 'onnx::Conv_755']`

input shapes: `[[1, 192, 64, 64], [192, 1, 3, 3], [192]]`

outputs: `['/encoder/features.5/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 42: Constant
name: `/encoder/features.5/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.5/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 43: Constant
name: `/encoder/features.5/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.5/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 44: Clip
name: `/encoder/features.5/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.5/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.5/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.5/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 192, 64, 64], None, None]`

outputs: `['/encoder/features.5/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 45: Conv
name: `/encoder/features.5/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.5/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_757', 'onnx::Conv_758']`

input shapes: `[[1, 192, 64, 64], [32, 192, 1, 1], [32]]`

outputs: `['/encoder/features.5/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 32, 64, 64]]`

### Node 46: Add
name: `/encoder/features.5/Add`

status: `supported`

inputs: `['/encoder/features.4/conv/conv.2/Conv_output_0', '/encoder/features.5/conv/conv.2/Conv_output_0']`

input shapes: `[[1, 32, 64, 64], [1, 32, 64, 64]]`

outputs: `['/encoder/features.5/Add_output_0']`

output shapes: `[[1, 32, 64, 64]]`

### Node 47: Conv
name: `/encoder/features.6/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.5/Add_output_0', 'onnx::Conv_760', 'onnx::Conv_761']`

input shapes: `[[1, 32, 64, 64], [192, 32, 1, 1], [192]]`

outputs: `['/encoder/features.6/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 48: Constant
name: `/encoder/features.6/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.6/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 49: Constant
name: `/encoder/features.6/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.6/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 50: Clip
name: `/encoder/features.6/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.6/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.6/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.6/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 192, 64, 64], None, None]`

outputs: `['/encoder/features.6/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 51: Conv
name: `/encoder/features.6/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.6/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_763', 'onnx::Conv_764']`

input shapes: `[[1, 192, 64, 64], [192, 1, 3, 3], [192]]`

outputs: `['/encoder/features.6/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 52: Constant
name: `/encoder/features.6/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.6/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 53: Constant
name: `/encoder/features.6/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.6/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 54: Clip
name: `/encoder/features.6/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.6/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.6/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.6/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 192, 64, 64], None, None]`

outputs: `['/encoder/features.6/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 55: Conv
name: `/encoder/features.6/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.6/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_766', 'onnx::Conv_767']`

input shapes: `[[1, 192, 64, 64], [32, 192, 1, 1], [32]]`

outputs: `['/encoder/features.6/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 32, 64, 64]]`

### Node 56: Add
name: `/encoder/features.6/Add`

status: `supported`

inputs: `['/encoder/features.5/Add_output_0', '/encoder/features.6/conv/conv.2/Conv_output_0']`

input shapes: `[[1, 32, 64, 64], [1, 32, 64, 64]]`

outputs: `['/encoder/features.6/Add_output_0']`

output shapes: `[[1, 32, 64, 64]]`

### Node 57: Conv
name: `/encoder/features.7/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.6/Add_output_0', 'onnx::Conv_769', 'onnx::Conv_770']`

input shapes: `[[1, 32, 64, 64], [192, 32, 1, 1], [192]]`

outputs: `['/encoder/features.7/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 58: Constant
name: `/encoder/features.7/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.7/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 59: Constant
name: `/encoder/features.7/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.7/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 60: Clip
name: `/encoder/features.7/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.7/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.7/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.7/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 192, 64, 64], None, None]`

outputs: `['/encoder/features.7/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 192, 64, 64]]`

### Node 61: Conv
name: `/encoder/features.7/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.7/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_772', 'onnx::Conv_773']`

input shapes: `[[1, 192, 64, 64], [192, 1, 3, 3], [192]]`

outputs: `['/encoder/features.7/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 192, 32, 32]]`

### Node 62: Constant
name: `/encoder/features.7/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.7/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 63: Constant
name: `/encoder/features.7/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.7/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 64: Clip
name: `/encoder/features.7/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.7/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.7/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.7/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 192, 32, 32], None, None]`

outputs: `['/encoder/features.7/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 192, 32, 32]]`

### Node 65: Conv
name: `/encoder/features.7/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.7/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_775', 'onnx::Conv_776']`

input shapes: `[[1, 192, 32, 32], [64, 192, 1, 1], [64]]`

outputs: `['/encoder/features.7/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 64, 32, 32]]`

### Node 66: Conv
name: `/encoder/features.8/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.7/conv/conv.2/Conv_output_0', 'onnx::Conv_778', 'onnx::Conv_779']`

input shapes: `[[1, 64, 32, 32], [384, 64, 1, 1], [384]]`

outputs: `['/encoder/features.8/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 384, 32, 32]]`

### Node 67: Constant
name: `/encoder/features.8/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.8/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 68: Constant
name: `/encoder/features.8/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.8/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 69: Clip
name: `/encoder/features.8/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.8/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.8/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.8/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 384, 32, 32], None, None]`

outputs: `['/encoder/features.8/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 384, 32, 32]]`

### Node 70: Conv
name: `/encoder/features.8/conv/conv.1/conv.1.0/Conv`

status: `supported`

inputs: `['/encoder/features.8/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_781', 'onnx::Conv_782']`

input shapes: `[[1, 384, 32, 32], [384, 1, 3, 3], [384]]`

outputs: `['/encoder/features.8/conv/conv.1/conv.1.0/Conv_output_0']`

output shapes: `[[1, 384, 32, 32]]`

### Node 71: Constant
name: `/encoder/features.8/conv/conv.1/conv.1.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.8/conv/conv.1/conv.1.2/Constant_output_0']`

output shapes: `[None]`

### Node 72: Constant
name: `/encoder/features.8/conv/conv.1/conv.1.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.8/conv/conv.1/conv.1.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 73: Clip
name: `/encoder/features.8/conv/conv.1/conv.1.2/Clip`

status: `supported`

inputs: `['/encoder/features.8/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.8/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.8/conv/conv.1/conv.1.2/Constant_1_output_0']`

input shapes: `[[1, 384, 32, 32], None, None]`

outputs: `['/encoder/features.8/conv/conv.1/conv.1.2/Clip_output_0']`

output shapes: `[[1, 384, 32, 32]]`

### Node 74: Conv
name: `/encoder/features.8/conv/conv.2/Conv`

status: `supported`

inputs: `['/encoder/features.8/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_784', 'onnx::Conv_785']`

input shapes: `[[1, 384, 32, 32], [64, 384, 1, 1], [64]]`

outputs: `['/encoder/features.8/conv/conv.2/Conv_output_0']`

output shapes: `[[1, 64, 32, 32]]`

### Node 75: Add
name: `/encoder/features.8/Add`

status: `supported`

inputs: `['/encoder/features.7/conv/conv.2/Conv_output_0', '/encoder/features.8/conv/conv.2/Conv_output_0']`

input shapes: `[[1, 64, 32, 32], [1, 64, 32, 32]]`

outputs: `['/encoder/features.8/Add_output_0']`

output shapes: `[[1, 64, 32, 32]]`

### Node 76: Conv
name: `/encoder/features.9/conv/conv.0/conv.0.0/Conv`

status: `supported`

inputs: `['/encoder/features.8/Add_output_0', 'onnx::Conv_787', 'onnx::Conv_788']`

input shapes: `[[1, 64, 32, 32], [384, 64, 1, 1], [384]]`

outputs: `['/encoder/features.9/conv/conv.0/conv.0.0/Conv_output_0']`

output shapes: `[[1, 384, 32, 32]]`

### Node 77: Constant
name: `/encoder/features.9/conv/conv.0/conv.0.2/Constant`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.9/conv/conv.0/conv.0.2/Constant_output_0']`

output shapes: `[None]`

### Node 78: Constant
name: `/encoder/features.9/conv/conv.0/conv.0.2/Constant_1`

status: `ignored`

inputs: `[]`

input shapes: `[]`

outputs: `['/encoder/features.9/conv/conv.0/conv.0.2/Constant_1_output_0']`

output shapes: `[None]`

### Node 79: Clip
name: `/encoder/features.9/conv/conv.0/conv.0.2/Clip`

status: `supported`

inputs: `['/encoder/features.9/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.9/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.9/conv/conv.0/conv.0.2/Constant_1_output_0']`

input shapes: `[[1, 384, 32, 32], None, None]`

outputs: `['/encoder/features.9/conv/conv.0/conv.0.2/Clip_output_0']`

output shapes: `[[1, 384, 32, 32]]`

