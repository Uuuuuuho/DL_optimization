#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate horizontal fusion application using 01_horizontal_fusion.py.

Steps
1) Generate fusion hints and external dependency edges (07_generate_fusion_hints.py)
2) Optionally run 01_horizontal_fusion.py with those artifacts

Note: This script does not import 04/05/06 files (their filenames are not importable).
      It simply consumes any reports they produced if present (via 07). Use --show-cmd
      to print the exact command and run it manually if you prefer.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Apply horizontal fusion with generated hints/edges")
    p.add_argument("--in", dest="in_model", required=True, help="Input ONNX model path")
    p.add_argument("--out", dest="out_model", required=True, help="Output ONNX model path")
    p.add_argument("--work-dir", default="out", help="Directory to store intermediate artifacts")
    p.add_argument("--min-group-size", type=int, default=2, help="Minimum group size for hints")
    p.add_argument("--no-matmul", action="store_true", help="Disable MatMul grouping for hints")
    p.add_argument("--no-gemm", action="store_true", help="Disable Gemm grouping for hints")
    p.add_argument("--run", action="store_true", help="Actually run 01_horizontal_fusion.py after generating artifacts")
    p.add_argument("--show-cmd", action="store_true", help="Print the command that would be run for 01")
    return p.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    hints = work_dir / "hfusion_hints.json"
    edges = work_dir / "edges.json"

    # Step 1: generate hints and edges
    gen_cmd = [
        sys.executable,
        str(Path(__file__).with_name("07_generate_fusion_hints.py")),
        "--model", args.in_model,
        "--hints", str(hints),
        "--edges", str(edges),
        "--min-group-size", str(args.min_group_size),
    ]
    if args.no_matmul:
        gen_cmd.append("--no-matmul")
    if args.no_gemm:
        gen_cmd.append("--no-gemm")

    print("[1/2] Generating hints and edges...")
    try:
        subprocess.check_call(gen_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate hints/edges: {e}")
        sys.exit(1)

    # Step 2: build command for 01
    run_cmd = [
        sys.executable,
        str(Path(__file__).with_name("01_horizontal_fusion.py")),
        "--in", args.in_model,
        "--out", args.out_model,
        "--dep-source", "edges-json",
        "--dep-file", str(edges),
        "--hints-file", str(hints),
    ]

    if args.show_cmd or not args.run:
        print("[2/2] Command to run 01_horizontal_fusion:")
        print(" ".join(run_cmd))
        if not args.run:
            return

    print("[2/2] Running 01_horizontal_fusion.py ...")
    try:
        subprocess.check_call(run_cmd)
        print(f"Done. Wrote fused model to: {args.out_model}")
    except subprocess.CalledProcessError as e:
        print(f"01_horizontal_fusion.py failed: {e}")
        sys.exit(e.returncode if isinstance(e, subprocess.CalledProcessError) else 1)


if __name__ == "__main__":
    main()
