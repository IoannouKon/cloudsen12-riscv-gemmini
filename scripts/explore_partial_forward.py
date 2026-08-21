#!/usr/bin/env python3
"""
Explore generated partial pure-C forward depth for CloudSEN12.

Run from inside ~/cloudsen12_pure_c:

    python scripts/explore_partial_forward.py

Optional:

    python scripts/explore_partial_forward.py --targets 20 40 60 80 100 120 140
    python scripts/explore_partial_forward.py --timeout 300

Outputs:
    experiments/partial_forward/partial_forward_exploration.csv
    experiments/partial_forward/partial_forward_logs/target_<N>_*.log
"""

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
LOG_DIR = ROOT / "partial_forward_logs"
CSV_PATH = ROOT / "experiments/partial_forward/partial_forward_exploration.csv"

GEN_SCRIPT = ROOT / "scripts" / "generate_partial_forward.py"
SRC = [
    "src/test_partial_forward.c",
    "src/conv2d.c",
    "src/clip.c",
    "src/relu.c",
    "src/add.c",
]
OUT_BIN = "build/test_partial_forward"

COMPILE_CMD = [
    "gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-Iinclude",
    *SRC,
    "-lm", "-o", OUT_BIN,
]

MEAN_RE = re.compile(r"Mean abs error vs ONNX Runtime:\s*([0-9eE+\-.]+)")
MAX_RE = re.compile(r"Max abs error vs ONNX Runtime:\s*([0-9eE+\-.]+)")
COUNT_RE = re.compile(r"Partial output count:\s*(\d+)")
LAST_NODE_RE = re.compile(r"Last selected node:\s*(.*)")
LAST_TENSOR_RE = re.compile(r"Last tensor:\s*(.*)")
LAST_SHAPE_RE = re.compile(r"Last shape:\s*(.*)")


def run_cmd(cmd, *, env=None, timeout=300, log_path=None):
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    try:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        out += f"\n[TIMEOUT] Command timed out after {timeout} seconds\n"
        rc = 124

    if log_path:
        log_path.write_text(f"$ {printable}\n\n{out}", encoding="utf-8")

    return rc, out


def match_one(regex, text):
    m = regex.search(text)
    return m.group(1).strip() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--targets",
        nargs="+",
        type=int,
        default=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        help="TARGET_SUPPORTED_OPS values to test",
    )
    ap.add_argument("--timeout", type=int, default=300, help="Timeout per command in seconds")
    ap.add_argument("--stop-on-fail", action="store_true", help="Stop exploration at first failure")
    args = ap.parse_args()

    if not GEN_SCRIPT.exists():
        print(f"ERROR: missing {GEN_SCRIPT}")
        print("Run this from ~/cloudsen12_pure_c and make sure scripts/generate_partial_forward.py exists.")
        return 1

    LOG_DIR.mkdir(exist_ok=True)
    rows = []

    print("CloudSEN12 partial pure-C forward exploration")
    print("Root:", ROOT)
    print("Targets:", args.targets)
    print("CSV:", CSV_PATH)
    print("Logs:", LOG_DIR)
    print()

    for target in args.targets:
        print(f"=== TARGET_SUPPORTED_OPS={target} ===")
        env = os.environ.copy()
        env["TARGET_SUPPORTED_OPS"] = str(target)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "target_supported_ops": target,
            "generate_status": "",
            "compile_status": "",
            "run_status": "",
            "partial_output_count": "",
            "mean_abs_error": "",
            "max_abs_error": "",
            "last_selected_node": "",
            "last_tensor": "",
            "last_shape": "",
            "notes": "",
        }

        gen_log = LOG_DIR / f"target_{target:04d}_generate.log"
        rc, out = run_cmd([sys.executable, str(GEN_SCRIPT)], env=env, timeout=args.timeout, log_path=gen_log)
        row["last_selected_node"] = match_one(LAST_NODE_RE, out)
        row["last_tensor"] = match_one(LAST_TENSOR_RE, out)
        row["last_shape"] = match_one(LAST_SHAPE_RE, out)
        if rc != 0:
            row["generate_status"] = f"FAIL({rc})"
            row["notes"] = f"generation failed, see {gen_log}"
            rows.append(row)
            print("generate: FAIL", rc)
            print("see:", gen_log)
            if args.stop_on_fail:
                break
            continue
        row["generate_status"] = "OK"
        print("generate: OK")
        print("last node:", row["last_selected_node"])

        comp_log = LOG_DIR / f"target_{target:04d}_compile.log"
        rc, out = run_cmd(COMPILE_CMD, timeout=args.timeout, log_path=comp_log)
        if rc != 0:
            row["compile_status"] = f"FAIL({rc})"
            row["run_status"] = "SKIP"
            row["notes"] = f"compile failed, see {comp_log}"
            rows.append(row)
            print("compile: FAIL", rc)
            print("see:", comp_log)
            if args.stop_on_fail:
                break
            continue
        row["compile_status"] = "OK"
        print("compile: OK")

        run_log = LOG_DIR / f"target_{target:04d}_run.log"
        rc, out = run_cmd([f"./{OUT_BIN}"], timeout=args.timeout, log_path=run_log)
        row["partial_output_count"] = match_one(COUNT_RE, out)
        row["mean_abs_error"] = match_one(MEAN_RE, out)
        row["max_abs_error"] = match_one(MAX_RE, out)
        if rc != 0:
            row["run_status"] = f"FAIL({rc})"
            row["notes"] = f"run failed, see {run_log}"
            rows.append(row)
            print("run: FAIL", rc)
            print("see:", run_log)
            if args.stop_on_fail:
                break
            continue
        row["run_status"] = "OK"
        rows.append(row)
        print("run: OK")
        print("count:", row["partial_output_count"])
        print("mean:", row["mean_abs_error"])
        print("max:", row["max_abs_error"])
        print()

    fieldnames = [
        "timestamp",
        "target_supported_ops",
        "generate_status",
        "compile_status",
        "run_status",
        "partial_output_count",
        "mean_abs_error",
        "max_abs_error",
        "last_selected_node",
        "last_tensor",
        "last_shape",
        "notes",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Exploration complete.")
    print("Summary CSV:", CSV_PATH)
    print()
    print("Quick view:")
    for r in rows:
        print(
            f"target={r['target_supported_ops']:<4} "
            f"gen={r['generate_status']:<8} "
            f"compile={r['compile_status']:<8} "
            f"run={r['run_status']:<8} "
            f"mean={r['mean_abs_error'] or '-'} "
            f"max={r['max_abs_error'] or '-'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
