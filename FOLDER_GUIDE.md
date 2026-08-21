# Folder Guide

Quick purpose guide for the main repository folders:

- `backends/` - grouped inference backend entry points (Python, C++ ONNX, pure C FP32, pure C INT8).
- `benchmark/` - benchmark scripts, input subsets, logs, and timing/accuracy outputs.
- `config/` - configuration files used by training/export/benchmark helpers.
- `docs/` - notes, command references, and workflow documentation.
- `experiments/` - one-off experiments and intermediate research artifacts.
- `include/` - FP32 pure-C headers.
- `src/` - FP32 pure-C implementation and test programs.
- `pure_c_int8/` - source package for the INT8 pure-C backend (now also exposed at `backends/pure_c_int8/`).
- `qat/` - quantization-aware training data, checkpoints, and deployment artifacts.
- `scripts/` - helper scripts for automation and reproducible runs.
- `weights/` - model weights and generated parameter blobs used by pure-C inference.
- `final_four_backend_benchmark/`, `v9_*` - archived experiment snapshots and fixed-run artifacts.
