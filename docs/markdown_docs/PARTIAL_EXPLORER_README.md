# Partial Forward Exploration Script

This script tests multiple `TARGET_SUPPORTED_OPS` values automatically.

## Install/use

Copy into your project:

```bash
cd ~/cloudsen12_pure_c
cp /mnt/c/Users/k.ioannou/Downloads/pure_c_partial_explorer.zip .
unzip -o pure_c_partial_explorer.zip
```

## Run default exploration

```bash
python scripts/explore_partial_forward.py
```

## Run custom exploration

```bash
python scripts/explore_partial_forward.py --targets 20 40 60 80 100 120 140
```

## Stop when something fails

```bash
python scripts/explore_partial_forward.py --targets 20 40 60 80 100 120 140 --stop-on-fail
```

## Outputs

```text
experiments/partial_forward/partial_forward_exploration.csv
experiments/partial_forward/partial_forward_logs/target_XXXX_generate.log
experiments/partial_forward/partial_forward_logs/target_XXXX_compile.log
experiments/partial_forward/partial_forward_logs/target_XXXX_run.log
```

If a target fails, inspect the relevant log file.
