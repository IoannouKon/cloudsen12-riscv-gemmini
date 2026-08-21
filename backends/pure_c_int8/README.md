# Pure C INT8 backend

This backend entry point groups the INT8 pure-C inference package with the rest of the backends.

The source package remains in `pure_c_int8/` at repository root. This folder exposes stable links to the same files.

Typical commands (run from repository root):

```bash
bash backends/pure_c_int8/build_and_validate_wsl.sh
bash backends/pure_c_int8/benchmark_int8_wsl.sh 10
```
