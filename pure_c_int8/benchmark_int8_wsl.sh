#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runs=${1:-10};mkdir -p pure_c_int8/results
: > pure_c_int8/results/int8_times.csv
echo run,seconds,max_rss_kb >> pure_c_int8/results/int8_times.csv
for i in $(seq 1 "$runs");do /usr/bin/time -f '%e,%M' -o pure_c_int8/results/one.time ./build/cloudsen12_v9_int8 pure_c_int8/test_input_int8.bin qat/deployment/v9/approved_pure_c pure_c_int8/results/c_mask.bin; echo "$i,$(cat pure_c_int8/results/one.time)" >> pure_c_int8/results/int8_times.csv;done
cat pure_c_int8/results/int8_times.csv
