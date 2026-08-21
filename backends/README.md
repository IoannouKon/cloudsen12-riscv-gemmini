# Backend Directory Guide

This folder groups all inference backends used in the project:

- `python_original/` - PyTorch reference backend and dataset preparation helpers.
- `cpp_onnx/` - C++ backend that runs the ONNX model with OpenCV DNN.
- `pure_c/` - FP32 pure-C backend entry point (links to root `src/`, `include/`, `weights/`).
- `pure_c_int8/` - INT8 pure-C backend entry point (links to root `pure_c_int8/` package).
