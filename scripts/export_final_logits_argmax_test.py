import numpy as np
import onnxruntime as ort
import os

model_path = "models/cloudsen12_unetmobv2_v2.onnx"
input_path = "inputs/input_1x13x512x512_float32.bin"
ref_pred_path = "inputs/pytorch_pred_512x512_uint8.bin"

os.makedirs("final_argmax", exist_ok=True)

x = np.fromfile(input_path, dtype=np.float32).reshape(1, 13, 512, 512)
py_ref = np.fromfile(ref_pred_path, dtype=np.uint8).reshape(512, 512)

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print("ONNX input:", input_name)
print("ONNX output:", output_name)

logits = sess.run([output_name], {input_name: x})[0].astype(np.float32)
print("Logits shape:", logits.shape)

onnx_pred = np.argmax(logits, axis=1).squeeze(0).astype(np.uint8)
mismatch = np.mean(onnx_pred != py_ref)
print("ONNX argmax classes:", np.unique(onnx_pred))
print("Mismatch ONNX argmax vs PyTorch ref:", mismatch)

logits.tofile("experiments/operator_tests/final_argmax/final_logits_1x4x512x512_float32.bin")
onnx_pred.tofile("experiments/operator_tests/final_argmax/onnx_argmax_512x512_uint8.bin")

with open("include/final_argmax_meta.h", "w") as f:
    f.write("#ifndef FINAL_ARGMAX_META_H\n")
    f.write("#define FINAL_ARGMAX_META_H\n\n")
    f.write("#define FARG_N 1\n")
    f.write("#define FARG_CLASSES 4\n")
    f.write("#define FARG_H 512\n")
    f.write("#define FARG_W 512\n")
    f.write("#endif\n")

print("Saved:")
print("  experiments/operator_tests/final_argmax/final_logits_1x4x512x512_float32.bin")
print("  experiments/operator_tests/final_argmax/onnx_argmax_512x512_uint8.bin")
print("  include/final_argmax_meta.h")
