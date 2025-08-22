#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict

# Local import relative to repo root
try:
    from src.tuner.profiler import load_times_json, summarize_times
except Exception:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.tuner.profiler import load_times_json, summarize_times  # type: ignore


def main():
    p = argparse.ArgumentParser(description="Compare two trtexec exportTimes logs and compute speedup")
    p.add_argument("baseline", help="Path to baseline *_times.json")
    p.add_argument("candidate", help="Path to candidate *_times.json")
    p.add_argument("--use", choices=["meanLatency", "trimmedMean_p10", "medianLatency"], default="trimmedMean_p10",
                   help="Metric to compare")
    args = p.parse_args()

    base = summarize_times(load_times_json(args.baseline))
    cand = summarize_times(load_times_json(args.candidate))

    key = args.use if args.use in base and args.use in cand else "meanLatency"
    b = float(base.get(key, float("nan")))
    c = float(cand.get(key, float("nan")))
    speedup = b / c if (b and c and b > 0 and c > 0) else float("nan")

    print({
        "metric": key,
        "baseline": b,
        "candidate": c,
        "speedup": speedup,
        "baseline_summary": base,
        "candidate_summary": cand,
    })


if __name__ == "__main__":
    main()
