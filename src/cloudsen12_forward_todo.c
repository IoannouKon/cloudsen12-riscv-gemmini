#include "cloudsen12_forward.h"
#include "conv2d.h"
#include "clip.h"
#include "relu.h"
#include "add.h"
#include "concat.h"
#include "resize.h"

/*
  Auto-generated TODO skeleton.
  This is NOT full inference yet. It lists the exact ONNX node order.
  Next step is to add tensor allocation/loading and replace TODO comments with kernel calls.
*/

void cloudsen12_forward_todo(const float *input, float *logits) {
    (void)input;
    (void)logits;

    // Node 0: Conv - /encoder/features.0/features.0.0/Conv
    // status: supported
    // inputs: ['input', 'onnx::Conv_715', 'onnx::Conv_716']
    // input_shapes: [[1, 13, 512, 512], [32, 13, 3, 3], [32]]
    // outputs: ['/encoder/features.0/features.0.0/Conv_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Conv kernel here.

    // Node 1: Constant - /encoder/features.0/features.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.0/features.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 2: Constant - /encoder/features.0/features.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.0/features.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 3: Clip - /encoder/features.0/features.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.0/features.0.0/Conv_output_0', '/encoder/features.0/features.0.2/Constant_output_0', '/encoder/features.0/features.0.2/Constant_1_output_0']
    // input_shapes: [[1, 32, 256, 256], None, None]
    // outputs: ['/encoder/features.0/features.0.2/Clip_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Clip kernel here.

    // Node 4: Conv - /encoder/features.1/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.0/features.0.2/Clip_output_0', 'onnx::Conv_718', 'onnx::Conv_719']
    // input_shapes: [[1, 32, 256, 256], [32, 1, 3, 3], [32]]
    // outputs: ['/encoder/features.1/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Conv kernel here.

    // Node 5: Constant - /encoder/features.1/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.1/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 6: Constant - /encoder/features.1/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.1/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 7: Clip - /encoder/features.1/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.1/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.1/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.1/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 32, 256, 256], None, None]
    // outputs: ['/encoder/features.1/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Clip kernel here.

    // Node 8: Conv - /encoder/features.1/conv/conv.1/Conv
    // status: supported
    // inputs: ['/encoder/features.1/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_721', 'onnx::Conv_722']
    // input_shapes: [[1, 32, 256, 256], [16, 32, 1, 1], [16]]
    // outputs: ['/encoder/features.1/conv/conv.1/Conv_output_0']
    // output_shapes: [[1, 16, 256, 256]]
    // TODO: call pure C Conv kernel here.

    // Node 9: Conv - /encoder/features.2/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.1/conv/conv.1/Conv_output_0', 'onnx::Conv_724', 'onnx::Conv_725']
    // input_shapes: [[1, 16, 256, 256], [96, 16, 1, 1], [96]]
    // outputs: ['/encoder/features.2/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 96, 256, 256]]
    // TODO: call pure C Conv kernel here.

    // Node 10: Constant - /encoder/features.2/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.2/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 11: Constant - /encoder/features.2/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.2/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 12: Clip - /encoder/features.2/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.2/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.2/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.2/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 96, 256, 256], None, None]
    // outputs: ['/encoder/features.2/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 96, 256, 256]]
    // TODO: call pure C Clip kernel here.

    // Node 13: Conv - /encoder/features.2/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.2/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_727', 'onnx::Conv_728']
    // input_shapes: [[1, 96, 256, 256], [96, 1, 3, 3], [96]]
    // outputs: ['/encoder/features.2/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 96, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 14: Constant - /encoder/features.2/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.2/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 15: Constant - /encoder/features.2/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.2/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 16: Clip - /encoder/features.2/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.2/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.2/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.2/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 96, 128, 128], None, None]
    // outputs: ['/encoder/features.2/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 96, 128, 128]]
    // TODO: call pure C Clip kernel here.

    // Node 17: Conv - /encoder/features.2/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.2/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_730', 'onnx::Conv_731']
    // input_shapes: [[1, 96, 128, 128], [24, 96, 1, 1], [24]]
    // outputs: ['/encoder/features.2/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 24, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 18: Conv - /encoder/features.3/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.2/conv/conv.2/Conv_output_0', 'onnx::Conv_733', 'onnx::Conv_734']
    // input_shapes: [[1, 24, 128, 128], [144, 24, 1, 1], [144]]
    // outputs: ['/encoder/features.3/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 144, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 19: Constant - /encoder/features.3/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.3/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 20: Constant - /encoder/features.3/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.3/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 21: Clip - /encoder/features.3/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.3/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.3/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.3/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 144, 128, 128], None, None]
    // outputs: ['/encoder/features.3/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 144, 128, 128]]
    // TODO: call pure C Clip kernel here.

    // Node 22: Conv - /encoder/features.3/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.3/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_736', 'onnx::Conv_737']
    // input_shapes: [[1, 144, 128, 128], [144, 1, 3, 3], [144]]
    // outputs: ['/encoder/features.3/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 144, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 23: Constant - /encoder/features.3/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.3/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 24: Constant - /encoder/features.3/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.3/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 25: Clip - /encoder/features.3/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.3/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.3/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.3/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 144, 128, 128], None, None]
    // outputs: ['/encoder/features.3/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 144, 128, 128]]
    // TODO: call pure C Clip kernel here.

    // Node 26: Conv - /encoder/features.3/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.3/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_739', 'onnx::Conv_740']
    // input_shapes: [[1, 144, 128, 128], [24, 144, 1, 1], [24]]
    // outputs: ['/encoder/features.3/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 24, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 27: Add - /encoder/features.3/Add
    // status: supported
    // inputs: ['/encoder/features.2/conv/conv.2/Conv_output_0', '/encoder/features.3/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 24, 128, 128], [1, 24, 128, 128]]
    // outputs: ['/encoder/features.3/Add_output_0']
    // output_shapes: [[1, 24, 128, 128]]
    // TODO: call pure C Add kernel here.

    // Node 28: Conv - /encoder/features.4/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.3/Add_output_0', 'onnx::Conv_742', 'onnx::Conv_743']
    // input_shapes: [[1, 24, 128, 128], [144, 24, 1, 1], [144]]
    // outputs: ['/encoder/features.4/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 144, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 29: Constant - /encoder/features.4/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.4/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 30: Constant - /encoder/features.4/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.4/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 31: Clip - /encoder/features.4/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.4/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.4/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.4/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 144, 128, 128], None, None]
    // outputs: ['/encoder/features.4/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 144, 128, 128]]
    // TODO: call pure C Clip kernel here.

    // Node 32: Conv - /encoder/features.4/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.4/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_745', 'onnx::Conv_746']
    // input_shapes: [[1, 144, 128, 128], [144, 1, 3, 3], [144]]
    // outputs: ['/encoder/features.4/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 144, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 33: Constant - /encoder/features.4/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.4/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 34: Constant - /encoder/features.4/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.4/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 35: Clip - /encoder/features.4/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.4/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.4/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.4/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 144, 64, 64], None, None]
    // outputs: ['/encoder/features.4/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 144, 64, 64]]
    // TODO: call pure C Clip kernel here.

    // Node 36: Conv - /encoder/features.4/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.4/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_748', 'onnx::Conv_749']
    // input_shapes: [[1, 144, 64, 64], [32, 144, 1, 1], [32]]
    // outputs: ['/encoder/features.4/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 32, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 37: Conv - /encoder/features.5/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.4/conv/conv.2/Conv_output_0', 'onnx::Conv_751', 'onnx::Conv_752']
    // input_shapes: [[1, 32, 64, 64], [192, 32, 1, 1], [192]]
    // outputs: ['/encoder/features.5/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 38: Constant - /encoder/features.5/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.5/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 39: Constant - /encoder/features.5/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.5/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 40: Clip - /encoder/features.5/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.5/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.5/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.5/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 192, 64, 64], None, None]
    // outputs: ['/encoder/features.5/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Clip kernel here.

    // Node 41: Conv - /encoder/features.5/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.5/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_754', 'onnx::Conv_755']
    // input_shapes: [[1, 192, 64, 64], [192, 1, 3, 3], [192]]
    // outputs: ['/encoder/features.5/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 42: Constant - /encoder/features.5/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.5/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 43: Constant - /encoder/features.5/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.5/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 44: Clip - /encoder/features.5/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.5/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.5/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.5/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 192, 64, 64], None, None]
    // outputs: ['/encoder/features.5/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Clip kernel here.

    // Node 45: Conv - /encoder/features.5/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.5/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_757', 'onnx::Conv_758']
    // input_shapes: [[1, 192, 64, 64], [32, 192, 1, 1], [32]]
    // outputs: ['/encoder/features.5/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 32, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 46: Add - /encoder/features.5/Add
    // status: supported
    // inputs: ['/encoder/features.4/conv/conv.2/Conv_output_0', '/encoder/features.5/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 32, 64, 64], [1, 32, 64, 64]]
    // outputs: ['/encoder/features.5/Add_output_0']
    // output_shapes: [[1, 32, 64, 64]]
    // TODO: call pure C Add kernel here.

    // Node 47: Conv - /encoder/features.6/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.5/Add_output_0', 'onnx::Conv_760', 'onnx::Conv_761']
    // input_shapes: [[1, 32, 64, 64], [192, 32, 1, 1], [192]]
    // outputs: ['/encoder/features.6/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 48: Constant - /encoder/features.6/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.6/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 49: Constant - /encoder/features.6/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.6/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 50: Clip - /encoder/features.6/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.6/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.6/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.6/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 192, 64, 64], None, None]
    // outputs: ['/encoder/features.6/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Clip kernel here.

    // Node 51: Conv - /encoder/features.6/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.6/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_763', 'onnx::Conv_764']
    // input_shapes: [[1, 192, 64, 64], [192, 1, 3, 3], [192]]
    // outputs: ['/encoder/features.6/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 52: Constant - /encoder/features.6/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.6/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 53: Constant - /encoder/features.6/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.6/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 54: Clip - /encoder/features.6/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.6/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.6/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.6/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 192, 64, 64], None, None]
    // outputs: ['/encoder/features.6/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Clip kernel here.

    // Node 55: Conv - /encoder/features.6/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.6/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_766', 'onnx::Conv_767']
    // input_shapes: [[1, 192, 64, 64], [32, 192, 1, 1], [32]]
    // outputs: ['/encoder/features.6/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 32, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 56: Add - /encoder/features.6/Add
    // status: supported
    // inputs: ['/encoder/features.5/Add_output_0', '/encoder/features.6/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 32, 64, 64], [1, 32, 64, 64]]
    // outputs: ['/encoder/features.6/Add_output_0']
    // output_shapes: [[1, 32, 64, 64]]
    // TODO: call pure C Add kernel here.

    // Node 57: Conv - /encoder/features.7/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.6/Add_output_0', 'onnx::Conv_769', 'onnx::Conv_770']
    // input_shapes: [[1, 32, 64, 64], [192, 32, 1, 1], [192]]
    // outputs: ['/encoder/features.7/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 58: Constant - /encoder/features.7/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.7/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 59: Constant - /encoder/features.7/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.7/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 60: Clip - /encoder/features.7/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.7/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.7/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.7/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 192, 64, 64], None, None]
    // outputs: ['/encoder/features.7/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 192, 64, 64]]
    // TODO: call pure C Clip kernel here.

    // Node 61: Conv - /encoder/features.7/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.7/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_772', 'onnx::Conv_773']
    // input_shapes: [[1, 192, 64, 64], [192, 1, 3, 3], [192]]
    // outputs: ['/encoder/features.7/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 192, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 62: Constant - /encoder/features.7/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.7/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 63: Constant - /encoder/features.7/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.7/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 64: Clip - /encoder/features.7/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.7/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.7/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.7/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 192, 32, 32], None, None]
    // outputs: ['/encoder/features.7/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 192, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 65: Conv - /encoder/features.7/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.7/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_775', 'onnx::Conv_776']
    // input_shapes: [[1, 192, 32, 32], [64, 192, 1, 1], [64]]
    // outputs: ['/encoder/features.7/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 66: Conv - /encoder/features.8/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.7/conv/conv.2/Conv_output_0', 'onnx::Conv_778', 'onnx::Conv_779']
    // input_shapes: [[1, 64, 32, 32], [384, 64, 1, 1], [384]]
    // outputs: ['/encoder/features.8/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 67: Constant - /encoder/features.8/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.8/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 68: Constant - /encoder/features.8/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.8/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 69: Clip - /encoder/features.8/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.8/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.8/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.8/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.8/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 70: Conv - /encoder/features.8/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.8/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_781', 'onnx::Conv_782']
    // input_shapes: [[1, 384, 32, 32], [384, 1, 3, 3], [384]]
    // outputs: ['/encoder/features.8/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 71: Constant - /encoder/features.8/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.8/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 72: Constant - /encoder/features.8/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.8/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 73: Clip - /encoder/features.8/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.8/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.8/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.8/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.8/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 74: Conv - /encoder/features.8/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.8/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_784', 'onnx::Conv_785']
    // input_shapes: [[1, 384, 32, 32], [64, 384, 1, 1], [64]]
    // outputs: ['/encoder/features.8/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 75: Add - /encoder/features.8/Add
    // status: supported
    // inputs: ['/encoder/features.7/conv/conv.2/Conv_output_0', '/encoder/features.8/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 64, 32, 32], [1, 64, 32, 32]]
    // outputs: ['/encoder/features.8/Add_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Add kernel here.

    // Node 76: Conv - /encoder/features.9/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.8/Add_output_0', 'onnx::Conv_787', 'onnx::Conv_788']
    // input_shapes: [[1, 64, 32, 32], [384, 64, 1, 1], [384]]
    // outputs: ['/encoder/features.9/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 77: Constant - /encoder/features.9/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.9/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 78: Constant - /encoder/features.9/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.9/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 79: Clip - /encoder/features.9/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.9/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.9/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.9/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.9/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 80: Conv - /encoder/features.9/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.9/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_790', 'onnx::Conv_791']
    // input_shapes: [[1, 384, 32, 32], [384, 1, 3, 3], [384]]
    // outputs: ['/encoder/features.9/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 81: Constant - /encoder/features.9/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.9/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 82: Constant - /encoder/features.9/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.9/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 83: Clip - /encoder/features.9/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.9/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.9/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.9/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.9/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 84: Conv - /encoder/features.9/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.9/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_793', 'onnx::Conv_794']
    // input_shapes: [[1, 384, 32, 32], [64, 384, 1, 1], [64]]
    // outputs: ['/encoder/features.9/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 85: Add - /encoder/features.9/Add
    // status: supported
    // inputs: ['/encoder/features.8/Add_output_0', '/encoder/features.9/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 64, 32, 32], [1, 64, 32, 32]]
    // outputs: ['/encoder/features.9/Add_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Add kernel here.

    // Node 86: Conv - /encoder/features.10/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.9/Add_output_0', 'onnx::Conv_796', 'onnx::Conv_797']
    // input_shapes: [[1, 64, 32, 32], [384, 64, 1, 1], [384]]
    // outputs: ['/encoder/features.10/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 87: Constant - /encoder/features.10/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.10/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 88: Constant - /encoder/features.10/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.10/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 89: Clip - /encoder/features.10/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.10/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.10/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.10/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.10/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 90: Conv - /encoder/features.10/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.10/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_799', 'onnx::Conv_800']
    // input_shapes: [[1, 384, 32, 32], [384, 1, 3, 3], [384]]
    // outputs: ['/encoder/features.10/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 91: Constant - /encoder/features.10/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.10/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 92: Constant - /encoder/features.10/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.10/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 93: Clip - /encoder/features.10/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.10/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.10/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.10/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.10/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 94: Conv - /encoder/features.10/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.10/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_802', 'onnx::Conv_803']
    // input_shapes: [[1, 384, 32, 32], [64, 384, 1, 1], [64]]
    // outputs: ['/encoder/features.10/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 95: Add - /encoder/features.10/Add
    // status: supported
    // inputs: ['/encoder/features.9/Add_output_0', '/encoder/features.10/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 64, 32, 32], [1, 64, 32, 32]]
    // outputs: ['/encoder/features.10/Add_output_0']
    // output_shapes: [[1, 64, 32, 32]]
    // TODO: call pure C Add kernel here.

    // Node 96: Conv - /encoder/features.11/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.10/Add_output_0', 'onnx::Conv_805', 'onnx::Conv_806']
    // input_shapes: [[1, 64, 32, 32], [384, 64, 1, 1], [384]]
    // outputs: ['/encoder/features.11/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 97: Constant - /encoder/features.11/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.11/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 98: Constant - /encoder/features.11/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.11/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 99: Clip - /encoder/features.11/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.11/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.11/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.11/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.11/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 100: Conv - /encoder/features.11/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.11/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_808', 'onnx::Conv_809']
    // input_shapes: [[1, 384, 32, 32], [384, 1, 3, 3], [384]]
    // outputs: ['/encoder/features.11/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 101: Constant - /encoder/features.11/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.11/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 102: Constant - /encoder/features.11/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.11/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 103: Clip - /encoder/features.11/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.11/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.11/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.11/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 384, 32, 32], None, None]
    // outputs: ['/encoder/features.11/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 384, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 104: Conv - /encoder/features.11/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.11/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_811', 'onnx::Conv_812']
    // input_shapes: [[1, 384, 32, 32], [96, 384, 1, 1], [96]]
    // outputs: ['/encoder/features.11/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 96, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 105: Conv - /encoder/features.12/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.11/conv/conv.2/Conv_output_0', 'onnx::Conv_814', 'onnx::Conv_815']
    // input_shapes: [[1, 96, 32, 32], [576, 96, 1, 1], [576]]
    // outputs: ['/encoder/features.12/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 106: Constant - /encoder/features.12/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.12/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 107: Constant - /encoder/features.12/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.12/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 108: Clip - /encoder/features.12/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.12/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.12/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.12/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 576, 32, 32], None, None]
    // outputs: ['/encoder/features.12/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 109: Conv - /encoder/features.12/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.12/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_817', 'onnx::Conv_818']
    // input_shapes: [[1, 576, 32, 32], [576, 1, 3, 3], [576]]
    // outputs: ['/encoder/features.12/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 110: Constant - /encoder/features.12/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.12/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 111: Constant - /encoder/features.12/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.12/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 112: Clip - /encoder/features.12/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.12/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.12/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.12/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 576, 32, 32], None, None]
    // outputs: ['/encoder/features.12/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 113: Conv - /encoder/features.12/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.12/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_820', 'onnx::Conv_821']
    // input_shapes: [[1, 576, 32, 32], [96, 576, 1, 1], [96]]
    // outputs: ['/encoder/features.12/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 96, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 114: Add - /encoder/features.12/Add
    // status: supported
    // inputs: ['/encoder/features.11/conv/conv.2/Conv_output_0', '/encoder/features.12/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 96, 32, 32], [1, 96, 32, 32]]
    // outputs: ['/encoder/features.12/Add_output_0']
    // output_shapes: [[1, 96, 32, 32]]
    // TODO: call pure C Add kernel here.

    // Node 115: Conv - /encoder/features.13/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.12/Add_output_0', 'onnx::Conv_823', 'onnx::Conv_824']
    // input_shapes: [[1, 96, 32, 32], [576, 96, 1, 1], [576]]
    // outputs: ['/encoder/features.13/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 116: Constant - /encoder/features.13/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.13/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 117: Constant - /encoder/features.13/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.13/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 118: Clip - /encoder/features.13/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.13/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.13/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.13/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 576, 32, 32], None, None]
    // outputs: ['/encoder/features.13/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 119: Conv - /encoder/features.13/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.13/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_826', 'onnx::Conv_827']
    // input_shapes: [[1, 576, 32, 32], [576, 1, 3, 3], [576]]
    // outputs: ['/encoder/features.13/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 120: Constant - /encoder/features.13/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.13/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 121: Constant - /encoder/features.13/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.13/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 122: Clip - /encoder/features.13/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.13/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.13/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.13/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 576, 32, 32], None, None]
    // outputs: ['/encoder/features.13/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 123: Conv - /encoder/features.13/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.13/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_829', 'onnx::Conv_830']
    // input_shapes: [[1, 576, 32, 32], [96, 576, 1, 1], [96]]
    // outputs: ['/encoder/features.13/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 96, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 124: Add - /encoder/features.13/Add
    // status: supported
    // inputs: ['/encoder/features.12/Add_output_0', '/encoder/features.13/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 96, 32, 32], [1, 96, 32, 32]]
    // outputs: ['/encoder/features.13/Add_output_0']
    // output_shapes: [[1, 96, 32, 32]]
    // TODO: call pure C Add kernel here.

    // Node 125: Conv - /encoder/features.14/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.13/Add_output_0', 'onnx::Conv_832', 'onnx::Conv_833']
    // input_shapes: [[1, 96, 32, 32], [576, 96, 1, 1], [576]]
    // outputs: ['/encoder/features.14/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 126: Constant - /encoder/features.14/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.14/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 127: Constant - /encoder/features.14/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.14/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 128: Clip - /encoder/features.14/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.14/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.14/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.14/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 576, 32, 32], None, None]
    // outputs: ['/encoder/features.14/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 576, 32, 32]]
    // TODO: call pure C Clip kernel here.

    // Node 129: Conv - /encoder/features.14/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.14/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_835', 'onnx::Conv_836']
    // input_shapes: [[1, 576, 32, 32], [576, 1, 3, 3], [576]]
    // outputs: ['/encoder/features.14/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 576, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 130: Constant - /encoder/features.14/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.14/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 131: Constant - /encoder/features.14/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.14/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 132: Clip - /encoder/features.14/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.14/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.14/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.14/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 576, 16, 16], None, None]
    // outputs: ['/encoder/features.14/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 576, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 133: Conv - /encoder/features.14/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.14/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_838', 'onnx::Conv_839']
    // input_shapes: [[1, 576, 16, 16], [160, 576, 1, 1], [160]]
    // outputs: ['/encoder/features.14/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 160, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 134: Conv - /encoder/features.15/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.14/conv/conv.2/Conv_output_0', 'onnx::Conv_841', 'onnx::Conv_842']
    // input_shapes: [[1, 160, 16, 16], [960, 160, 1, 1], [960]]
    // outputs: ['/encoder/features.15/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 135: Constant - /encoder/features.15/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.15/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 136: Constant - /encoder/features.15/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.15/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 137: Clip - /encoder/features.15/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.15/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.15/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.15/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 960, 16, 16], None, None]
    // outputs: ['/encoder/features.15/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 138: Conv - /encoder/features.15/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.15/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_844', 'onnx::Conv_845']
    // input_shapes: [[1, 960, 16, 16], [960, 1, 3, 3], [960]]
    // outputs: ['/encoder/features.15/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 139: Constant - /encoder/features.15/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.15/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 140: Constant - /encoder/features.15/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.15/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 141: Clip - /encoder/features.15/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.15/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.15/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.15/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 960, 16, 16], None, None]
    // outputs: ['/encoder/features.15/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 142: Conv - /encoder/features.15/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.15/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_847', 'onnx::Conv_848']
    // input_shapes: [[1, 960, 16, 16], [160, 960, 1, 1], [160]]
    // outputs: ['/encoder/features.15/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 160, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 143: Add - /encoder/features.15/Add
    // status: supported
    // inputs: ['/encoder/features.14/conv/conv.2/Conv_output_0', '/encoder/features.15/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 160, 16, 16], [1, 160, 16, 16]]
    // outputs: ['/encoder/features.15/Add_output_0']
    // output_shapes: [[1, 160, 16, 16]]
    // TODO: call pure C Add kernel here.

    // Node 144: Conv - /encoder/features.16/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.15/Add_output_0', 'onnx::Conv_850', 'onnx::Conv_851']
    // input_shapes: [[1, 160, 16, 16], [960, 160, 1, 1], [960]]
    // outputs: ['/encoder/features.16/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 145: Constant - /encoder/features.16/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.16/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 146: Constant - /encoder/features.16/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.16/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 147: Clip - /encoder/features.16/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.16/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.16/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.16/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 960, 16, 16], None, None]
    // outputs: ['/encoder/features.16/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 148: Conv - /encoder/features.16/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.16/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_853', 'onnx::Conv_854']
    // input_shapes: [[1, 960, 16, 16], [960, 1, 3, 3], [960]]
    // outputs: ['/encoder/features.16/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 149: Constant - /encoder/features.16/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.16/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 150: Constant - /encoder/features.16/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.16/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 151: Clip - /encoder/features.16/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.16/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.16/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.16/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 960, 16, 16], None, None]
    // outputs: ['/encoder/features.16/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 152: Conv - /encoder/features.16/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.16/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_856', 'onnx::Conv_857']
    // input_shapes: [[1, 960, 16, 16], [160, 960, 1, 1], [160]]
    // outputs: ['/encoder/features.16/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 160, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 153: Add - /encoder/features.16/Add
    // status: supported
    // inputs: ['/encoder/features.15/Add_output_0', '/encoder/features.16/conv/conv.2/Conv_output_0']
    // input_shapes: [[1, 160, 16, 16], [1, 160, 16, 16]]
    // outputs: ['/encoder/features.16/Add_output_0']
    // output_shapes: [[1, 160, 16, 16]]
    // TODO: call pure C Add kernel here.

    // Node 154: Conv - /encoder/features.17/conv/conv.0/conv.0.0/Conv
    // status: supported
    // inputs: ['/encoder/features.16/Add_output_0', 'onnx::Conv_859', 'onnx::Conv_860']
    // input_shapes: [[1, 160, 16, 16], [960, 160, 1, 1], [960]]
    // outputs: ['/encoder/features.17/conv/conv.0/conv.0.0/Conv_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 155: Constant - /encoder/features.17/conv/conv.0/conv.0.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.17/conv/conv.0/conv.0.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 156: Constant - /encoder/features.17/conv/conv.0/conv.0.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.17/conv/conv.0/conv.0.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 157: Clip - /encoder/features.17/conv/conv.0/conv.0.2/Clip
    // status: supported
    // inputs: ['/encoder/features.17/conv/conv.0/conv.0.0/Conv_output_0', '/encoder/features.17/conv/conv.0/conv.0.2/Constant_output_0', '/encoder/features.17/conv/conv.0/conv.0.2/Constant_1_output_0']
    // input_shapes: [[1, 960, 16, 16], None, None]
    // outputs: ['/encoder/features.17/conv/conv.0/conv.0.2/Clip_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 158: Conv - /encoder/features.17/conv/conv.1/conv.1.0/Conv
    // status: supported
    // inputs: ['/encoder/features.17/conv/conv.0/conv.0.2/Clip_output_0', 'onnx::Conv_862', 'onnx::Conv_863']
    // input_shapes: [[1, 960, 16, 16], [960, 1, 3, 3], [960]]
    // outputs: ['/encoder/features.17/conv/conv.1/conv.1.0/Conv_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 159: Constant - /encoder/features.17/conv/conv.1/conv.1.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.17/conv/conv.1/conv.1.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 160: Constant - /encoder/features.17/conv/conv.1/conv.1.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.17/conv/conv.1/conv.1.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 161: Clip - /encoder/features.17/conv/conv.1/conv.1.2/Clip
    // status: supported
    // inputs: ['/encoder/features.17/conv/conv.1/conv.1.0/Conv_output_0', '/encoder/features.17/conv/conv.1/conv.1.2/Constant_output_0', '/encoder/features.17/conv/conv.1/conv.1.2/Constant_1_output_0']
    // input_shapes: [[1, 960, 16, 16], None, None]
    // outputs: ['/encoder/features.17/conv/conv.1/conv.1.2/Clip_output_0']
    // output_shapes: [[1, 960, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 162: Conv - /encoder/features.17/conv/conv.2/Conv
    // status: supported
    // inputs: ['/encoder/features.17/conv/conv.1/conv.1.2/Clip_output_0', 'onnx::Conv_865', 'onnx::Conv_866']
    // input_shapes: [[1, 960, 16, 16], [320, 960, 1, 1], [320]]
    // outputs: ['/encoder/features.17/conv/conv.2/Conv_output_0']
    // output_shapes: [[1, 320, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 163: Conv - /encoder/features.18/features.18.0/Conv
    // status: supported
    // inputs: ['/encoder/features.17/conv/conv.2/Conv_output_0', 'onnx::Conv_868', 'onnx::Conv_869']
    // input_shapes: [[1, 320, 16, 16], [1280, 320, 1, 1], [1280]]
    // outputs: ['/encoder/features.18/features.18.0/Conv_output_0']
    // output_shapes: [[1, 1280, 16, 16]]
    // TODO: call pure C Conv kernel here.

    // Node 164: Constant - /encoder/features.18/features.18.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.18/features.18.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 165: Constant - /encoder/features.18/features.18.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/encoder/features.18/features.18.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 166: Clip - /encoder/features.18/features.18.2/Clip
    // status: supported
    // inputs: ['/encoder/features.18/features.18.0/Conv_output_0', '/encoder/features.18/features.18.2/Constant_output_0', '/encoder/features.18/features.18.2/Constant_1_output_0']
    // input_shapes: [[1, 1280, 16, 16], None, None]
    // outputs: ['/encoder/features.18/features.18.2/Clip_output_0']
    // output_shapes: [[1, 1280, 16, 16]]
    // TODO: call pure C Clip kernel here.

    // Node 167: Shape - /decoder/blocks.0/Shape
    // status: ignored
    // inputs: ['/encoder/features.18/features.18.2/Clip_output_0']
    // input_shapes: [[1, 1280, 16, 16]]
    // outputs: ['/decoder/blocks.0/Shape_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 168: Constant - /decoder/blocks.0/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.0/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 169: Constant - /decoder/blocks.0/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.0/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 170: Constant - /decoder/blocks.0/Constant_2
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.0/Constant_2_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 171: Slice - /decoder/blocks.0/Slice
    // status: ignored
    // inputs: ['/decoder/blocks.0/Shape_output_0', '/decoder/blocks.0/Constant_1_output_0', '/decoder/blocks.0/Constant_2_output_0', '/decoder/blocks.0/Constant_output_0']
    // input_shapes: [None, None, None, None]
    // outputs: ['/decoder/blocks.0/Slice_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 172: Constant - /decoder/blocks.0/Constant_3
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.0/Constant_3_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 173: Concat - /decoder/blocks.0/Concat
    // status: ignored
    // inputs: ['/decoder/blocks.0/Slice_output_0', '/decoder/blocks.0/Constant_3_output_0']
    // input_shapes: [None, None]
    // outputs: ['/decoder/blocks.0/Concat_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 174: Resize - /decoder/blocks.0/Resize
    // status: supported
    // inputs: ['/encoder/features.18/features.18.2/Clip_output_0', '', '', '/decoder/blocks.0/Concat_output_0']
    // input_shapes: [[1, 1280, 16, 16], [], [], None]
    // outputs: ['/decoder/blocks.0/Resize_output_0']
    // output_shapes: [[1, 1280, 32, 32]]
    // TODO: call pure C Resize kernel here.

    // Node 175: Concat - /decoder/blocks.0/Concat_1
    // status: supported
    // inputs: ['/decoder/blocks.0/Resize_output_0', '/encoder/features.13/Add_output_0']
    // input_shapes: [[1, 1280, 32, 32], [1, 96, 32, 32]]
    // outputs: ['/decoder/blocks.0/Concat_1_output_0']
    // output_shapes: [[1, 1376, 32, 32]]
    // TODO: call pure C Concat kernel here.

    // Node 176: Conv - /decoder/blocks.0/conv1/conv1.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.0/Concat_1_output_0', 'onnx::Conv_871', 'onnx::Conv_872']
    // input_shapes: [[1, 1376, 32, 32], [256, 1376, 3, 3], [256]]
    // outputs: ['/decoder/blocks.0/conv1/conv1.0/Conv_output_0']
    // output_shapes: [[1, 256, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 177: Relu - /decoder/blocks.0/conv1/conv1.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.0/conv1/conv1.0/Conv_output_0']
    // input_shapes: [[1, 256, 32, 32]]
    // outputs: ['/decoder/blocks.0/conv1/conv1.2/Relu_output_0']
    // output_shapes: [[1, 256, 32, 32]]
    // TODO: call pure C Relu kernel here.

    // Node 178: Conv - /decoder/blocks.0/conv2/conv2.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.0/conv1/conv1.2/Relu_output_0', 'onnx::Conv_874', 'onnx::Conv_875']
    // input_shapes: [[1, 256, 32, 32], [256, 256, 3, 3], [256]]
    // outputs: ['/decoder/blocks.0/conv2/conv2.0/Conv_output_0']
    // output_shapes: [[1, 256, 32, 32]]
    // TODO: call pure C Conv kernel here.

    // Node 179: Relu - /decoder/blocks.0/conv2/conv2.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.0/conv2/conv2.0/Conv_output_0']
    // input_shapes: [[1, 256, 32, 32]]
    // outputs: ['/decoder/blocks.0/conv2/conv2.2/Relu_output_0']
    // output_shapes: [[1, 256, 32, 32]]
    // TODO: call pure C Relu kernel here.

    // Node 180: Shape - /decoder/blocks.1/Shape
    // status: ignored
    // inputs: ['/decoder/blocks.0/conv2/conv2.2/Relu_output_0']
    // input_shapes: [[1, 256, 32, 32]]
    // outputs: ['/decoder/blocks.1/Shape_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 181: Constant - /decoder/blocks.1/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.1/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 182: Constant - /decoder/blocks.1/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.1/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 183: Constant - /decoder/blocks.1/Constant_2
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.1/Constant_2_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 184: Slice - /decoder/blocks.1/Slice
    // status: ignored
    // inputs: ['/decoder/blocks.1/Shape_output_0', '/decoder/blocks.1/Constant_1_output_0', '/decoder/blocks.1/Constant_2_output_0', '/decoder/blocks.1/Constant_output_0']
    // input_shapes: [None, None, None, None]
    // outputs: ['/decoder/blocks.1/Slice_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 185: Constant - /decoder/blocks.1/Constant_3
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.1/Constant_3_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 186: Concat - /decoder/blocks.1/Concat
    // status: ignored
    // inputs: ['/decoder/blocks.1/Slice_output_0', '/decoder/blocks.1/Constant_3_output_0']
    // input_shapes: [None, None]
    // outputs: ['/decoder/blocks.1/Concat_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 187: Resize - /decoder/blocks.1/Resize
    // status: supported
    // inputs: ['/decoder/blocks.0/conv2/conv2.2/Relu_output_0', '', '', '/decoder/blocks.1/Concat_output_0']
    // input_shapes: [[1, 256, 32, 32], [], [], None]
    // outputs: ['/decoder/blocks.1/Resize_output_0']
    // output_shapes: [[1, 256, 64, 64]]
    // TODO: call pure C Resize kernel here.

    // Node 188: Concat - /decoder/blocks.1/Concat_1
    // status: supported
    // inputs: ['/decoder/blocks.1/Resize_output_0', '/encoder/features.6/Add_output_0']
    // input_shapes: [[1, 256, 64, 64], [1, 32, 64, 64]]
    // outputs: ['/decoder/blocks.1/Concat_1_output_0']
    // output_shapes: [[1, 288, 64, 64]]
    // TODO: call pure C Concat kernel here.

    // Node 189: Conv - /decoder/blocks.1/conv1/conv1.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.1/Concat_1_output_0', 'onnx::Conv_877', 'onnx::Conv_878']
    // input_shapes: [[1, 288, 64, 64], [128, 288, 3, 3], [128]]
    // outputs: ['/decoder/blocks.1/conv1/conv1.0/Conv_output_0']
    // output_shapes: [[1, 128, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 190: Relu - /decoder/blocks.1/conv1/conv1.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.1/conv1/conv1.0/Conv_output_0']
    // input_shapes: [[1, 128, 64, 64]]
    // outputs: ['/decoder/blocks.1/conv1/conv1.2/Relu_output_0']
    // output_shapes: [[1, 128, 64, 64]]
    // TODO: call pure C Relu kernel here.

    // Node 191: Conv - /decoder/blocks.1/conv2/conv2.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.1/conv1/conv1.2/Relu_output_0', 'onnx::Conv_880', 'onnx::Conv_881']
    // input_shapes: [[1, 128, 64, 64], [128, 128, 3, 3], [128]]
    // outputs: ['/decoder/blocks.1/conv2/conv2.0/Conv_output_0']
    // output_shapes: [[1, 128, 64, 64]]
    // TODO: call pure C Conv kernel here.

    // Node 192: Relu - /decoder/blocks.1/conv2/conv2.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.1/conv2/conv2.0/Conv_output_0']
    // input_shapes: [[1, 128, 64, 64]]
    // outputs: ['/decoder/blocks.1/conv2/conv2.2/Relu_output_0']
    // output_shapes: [[1, 128, 64, 64]]
    // TODO: call pure C Relu kernel here.

    // Node 193: Shape - /decoder/blocks.2/Shape
    // status: ignored
    // inputs: ['/decoder/blocks.1/conv2/conv2.2/Relu_output_0']
    // input_shapes: [[1, 128, 64, 64]]
    // outputs: ['/decoder/blocks.2/Shape_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 194: Constant - /decoder/blocks.2/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.2/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 195: Constant - /decoder/blocks.2/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.2/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 196: Constant - /decoder/blocks.2/Constant_2
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.2/Constant_2_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 197: Slice - /decoder/blocks.2/Slice
    // status: ignored
    // inputs: ['/decoder/blocks.2/Shape_output_0', '/decoder/blocks.2/Constant_1_output_0', '/decoder/blocks.2/Constant_2_output_0', '/decoder/blocks.2/Constant_output_0']
    // input_shapes: [None, None, None, None]
    // outputs: ['/decoder/blocks.2/Slice_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 198: Constant - /decoder/blocks.2/Constant_3
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.2/Constant_3_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 199: Concat - /decoder/blocks.2/Concat
    // status: ignored
    // inputs: ['/decoder/blocks.2/Slice_output_0', '/decoder/blocks.2/Constant_3_output_0']
    // input_shapes: [None, None]
    // outputs: ['/decoder/blocks.2/Concat_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 200: Resize - /decoder/blocks.2/Resize
    // status: supported
    // inputs: ['/decoder/blocks.1/conv2/conv2.2/Relu_output_0', '', '', '/decoder/blocks.2/Concat_output_0']
    // input_shapes: [[1, 128, 64, 64], [], [], None]
    // outputs: ['/decoder/blocks.2/Resize_output_0']
    // output_shapes: [[1, 128, 128, 128]]
    // TODO: call pure C Resize kernel here.

    // Node 201: Concat - /decoder/blocks.2/Concat_1
    // status: supported
    // inputs: ['/decoder/blocks.2/Resize_output_0', '/encoder/features.3/Add_output_0']
    // input_shapes: [[1, 128, 128, 128], [1, 24, 128, 128]]
    // outputs: ['/decoder/blocks.2/Concat_1_output_0']
    // output_shapes: [[1, 152, 128, 128]]
    // TODO: call pure C Concat kernel here.

    // Node 202: Conv - /decoder/blocks.2/conv1/conv1.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.2/Concat_1_output_0', 'onnx::Conv_883', 'onnx::Conv_884']
    // input_shapes: [[1, 152, 128, 128], [64, 152, 3, 3], [64]]
    // outputs: ['/decoder/blocks.2/conv1/conv1.0/Conv_output_0']
    // output_shapes: [[1, 64, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 203: Relu - /decoder/blocks.2/conv1/conv1.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.2/conv1/conv1.0/Conv_output_0']
    // input_shapes: [[1, 64, 128, 128]]
    // outputs: ['/decoder/blocks.2/conv1/conv1.2/Relu_output_0']
    // output_shapes: [[1, 64, 128, 128]]
    // TODO: call pure C Relu kernel here.

    // Node 204: Conv - /decoder/blocks.2/conv2/conv2.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.2/conv1/conv1.2/Relu_output_0', 'onnx::Conv_886', 'onnx::Conv_887']
    // input_shapes: [[1, 64, 128, 128], [64, 64, 3, 3], [64]]
    // outputs: ['/decoder/blocks.2/conv2/conv2.0/Conv_output_0']
    // output_shapes: [[1, 64, 128, 128]]
    // TODO: call pure C Conv kernel here.

    // Node 205: Relu - /decoder/blocks.2/conv2/conv2.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.2/conv2/conv2.0/Conv_output_0']
    // input_shapes: [[1, 64, 128, 128]]
    // outputs: ['/decoder/blocks.2/conv2/conv2.2/Relu_output_0']
    // output_shapes: [[1, 64, 128, 128]]
    // TODO: call pure C Relu kernel here.

    // Node 206: Shape - /decoder/blocks.3/Shape
    // status: ignored
    // inputs: ['/decoder/blocks.2/conv2/conv2.2/Relu_output_0']
    // input_shapes: [[1, 64, 128, 128]]
    // outputs: ['/decoder/blocks.3/Shape_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 207: Constant - /decoder/blocks.3/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.3/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 208: Constant - /decoder/blocks.3/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.3/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 209: Constant - /decoder/blocks.3/Constant_2
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.3/Constant_2_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 210: Slice - /decoder/blocks.3/Slice
    // status: ignored
    // inputs: ['/decoder/blocks.3/Shape_output_0', '/decoder/blocks.3/Constant_1_output_0', '/decoder/blocks.3/Constant_2_output_0', '/decoder/blocks.3/Constant_output_0']
    // input_shapes: [None, None, None, None]
    // outputs: ['/decoder/blocks.3/Slice_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 211: Constant - /decoder/blocks.3/Constant_3
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.3/Constant_3_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 212: Concat - /decoder/blocks.3/Concat
    // status: ignored
    // inputs: ['/decoder/blocks.3/Slice_output_0', '/decoder/blocks.3/Constant_3_output_0']
    // input_shapes: [None, None]
    // outputs: ['/decoder/blocks.3/Concat_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 213: Resize - /decoder/blocks.3/Resize
    // status: supported
    // inputs: ['/decoder/blocks.2/conv2/conv2.2/Relu_output_0', '', '', '/decoder/blocks.3/Concat_output_0']
    // input_shapes: [[1, 64, 128, 128], [], [], None]
    // outputs: ['/decoder/blocks.3/Resize_output_0']
    // output_shapes: [[1, 64, 256, 256]]
    // TODO: call pure C Resize kernel here.

    // Node 214: Concat - /decoder/blocks.3/Concat_1
    // status: supported
    // inputs: ['/decoder/blocks.3/Resize_output_0', '/encoder/features.1/conv/conv.1/Conv_output_0']
    // input_shapes: [[1, 64, 256, 256], [1, 16, 256, 256]]
    // outputs: ['/decoder/blocks.3/Concat_1_output_0']
    // output_shapes: [[1, 80, 256, 256]]
    // TODO: call pure C Concat kernel here.

    // Node 215: Conv - /decoder/blocks.3/conv1/conv1.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.3/Concat_1_output_0', 'onnx::Conv_889', 'onnx::Conv_890']
    // input_shapes: [[1, 80, 256, 256], [32, 80, 3, 3], [32]]
    // outputs: ['/decoder/blocks.3/conv1/conv1.0/Conv_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Conv kernel here.

    // Node 216: Relu - /decoder/blocks.3/conv1/conv1.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.3/conv1/conv1.0/Conv_output_0']
    // input_shapes: [[1, 32, 256, 256]]
    // outputs: ['/decoder/blocks.3/conv1/conv1.2/Relu_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Relu kernel here.

    // Node 217: Conv - /decoder/blocks.3/conv2/conv2.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.3/conv1/conv1.2/Relu_output_0', 'onnx::Conv_892', 'onnx::Conv_893']
    // input_shapes: [[1, 32, 256, 256], [32, 32, 3, 3], [32]]
    // outputs: ['/decoder/blocks.3/conv2/conv2.0/Conv_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Conv kernel here.

    // Node 218: Relu - /decoder/blocks.3/conv2/conv2.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.3/conv2/conv2.0/Conv_output_0']
    // input_shapes: [[1, 32, 256, 256]]
    // outputs: ['/decoder/blocks.3/conv2/conv2.2/Relu_output_0']
    // output_shapes: [[1, 32, 256, 256]]
    // TODO: call pure C Relu kernel here.

    // Node 219: Shape - /decoder/blocks.4/Shape
    // status: ignored
    // inputs: ['/decoder/blocks.3/conv2/conv2.2/Relu_output_0']
    // input_shapes: [[1, 32, 256, 256]]
    // outputs: ['/decoder/blocks.4/Shape_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 220: Constant - /decoder/blocks.4/Constant
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.4/Constant_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 221: Constant - /decoder/blocks.4/Constant_1
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.4/Constant_1_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 222: Constant - /decoder/blocks.4/Constant_2
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.4/Constant_2_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 223: Slice - /decoder/blocks.4/Slice
    // status: ignored
    // inputs: ['/decoder/blocks.4/Shape_output_0', '/decoder/blocks.4/Constant_1_output_0', '/decoder/blocks.4/Constant_2_output_0', '/decoder/blocks.4/Constant_output_0']
    // input_shapes: [None, None, None, None]
    // outputs: ['/decoder/blocks.4/Slice_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 224: Constant - /decoder/blocks.4/Constant_3
    // status: ignored
    // inputs: []
    // input_shapes: []
    // outputs: ['/decoder/blocks.4/Constant_3_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 225: Concat - /decoder/blocks.4/Concat
    // status: ignored
    // inputs: ['/decoder/blocks.4/Slice_output_0', '/decoder/blocks.4/Constant_3_output_0']
    // input_shapes: [None, None]
    // outputs: ['/decoder/blocks.4/Concat_output_0']
    // output_shapes: [None]
    // IGNORE/TODO: bookkeeping node or unsupported node.

    // Node 226: Resize - /decoder/blocks.4/Resize
    // status: supported
    // inputs: ['/decoder/blocks.3/conv2/conv2.2/Relu_output_0', '', '', '/decoder/blocks.4/Concat_output_0']
    // input_shapes: [[1, 32, 256, 256], [], [], None]
    // outputs: ['/decoder/blocks.4/Resize_output_0']
    // output_shapes: [[1, 32, 512, 512]]
    // TODO: call pure C Resize kernel here.

    // Node 227: Conv - /decoder/blocks.4/conv1/conv1.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.4/Resize_output_0', 'onnx::Conv_895', 'onnx::Conv_896']
    // input_shapes: [[1, 32, 512, 512], [16, 32, 3, 3], [16]]
    // outputs: ['/decoder/blocks.4/conv1/conv1.0/Conv_output_0']
    // output_shapes: [[1, 16, 512, 512]]
    // TODO: call pure C Conv kernel here.

    // Node 228: Relu - /decoder/blocks.4/conv1/conv1.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.4/conv1/conv1.0/Conv_output_0']
    // input_shapes: [[1, 16, 512, 512]]
    // outputs: ['/decoder/blocks.4/conv1/conv1.2/Relu_output_0']
    // output_shapes: [[1, 16, 512, 512]]
    // TODO: call pure C Relu kernel here.

    // Node 229: Conv - /decoder/blocks.4/conv2/conv2.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.4/conv1/conv1.2/Relu_output_0', 'onnx::Conv_898', 'onnx::Conv_899']
    // input_shapes: [[1, 16, 512, 512], [16, 16, 3, 3], [16]]
    // outputs: ['/decoder/blocks.4/conv2/conv2.0/Conv_output_0']
    // output_shapes: [[1, 16, 512, 512]]
    // TODO: call pure C Conv kernel here.

    // Node 230: Relu - /decoder/blocks.4/conv2/conv2.2/Relu
    // status: supported
    // inputs: ['/decoder/blocks.4/conv2/conv2.0/Conv_output_0']
    // input_shapes: [[1, 16, 512, 512]]
    // outputs: ['/decoder/blocks.4/conv2/conv2.2/Relu_output_0']
    // output_shapes: [[1, 16, 512, 512]]
    // TODO: call pure C Relu kernel here.

    // Node 231: Conv - /segmentation_head/segmentation_head.0/Conv
    // status: supported
    // inputs: ['/decoder/blocks.4/conv2/conv2.2/Relu_output_0', 'segmentation_head.0.weight', 'segmentation_head.0.bias']
    // input_shapes: [[1, 16, 512, 512], [4, 16, 3, 3], [4]]
    // outputs: ['logits']
    // output_shapes: [[1, 4, 512, 512]]
    // TODO: call pure C Conv kernel here.

}
