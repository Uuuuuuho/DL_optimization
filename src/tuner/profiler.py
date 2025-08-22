from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def load_times_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return math.nan
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


def _trimmed_mean(values: Sequence[float], trim: float = 0.1) -> float:
    if not values:
        return math.nan
    n = len(values)
    if n < 2:
        return float(values[0])
    t = max(0, min(0.5, trim))
    k = int(n * t)
    trimmed = values[k:n - k] if k > 0 else values
    return sum(trimmed) / len(trimmed)


def summarize_times(times_json: Any) -> Dict[str, float]:
    """
    Summarize trtexec --exportTimes output.

    Supports:
    - Summary dicts that already contain meanLatency/percentiles.
    - Arrays of per-iteration objects with fields like latencyMs/computeMs.
    """
    out: Dict[str, float] = {}

    # Case 1: dict with standard keys
    if isinstance(times_json, Mapping):
        key_map = {
            "meanLatency": "meanLatency",
            "medianLatency": "medianLatency",
            "percentile90": "percentile90",
            "percentile95": "percentile95",
            "percentile99": "percentile99",
            "throughput": "throughput",
        }
        for k, v in key_map.items():
            if v in times_json:  # type: ignore[index]
                try:
                    out[k] = float(times_json[v])  # type: ignore[index]
                except Exception:
                    pass
        if out:
            return out

    # Case 2: list of per-iteration records
    if isinstance(times_json, list) and times_json and isinstance(times_json[0], Mapping):
        latencies: List[float] = []
        computes: List[float] = []
        for rec in times_json:
            # Prefer latencyMs; fallback to computeMs if present
            val = rec.get("latencyMs")
            if isinstance(val, (int, float)):
                latencies.append(float(val))
            val_c = rec.get("computeMs")
            if isinstance(val_c, (int, float)):
                computes.append(float(val_c))

        latencies.sort()
        computes.sort()
        use_vals = latencies if latencies else computes
        if not use_vals:
            return out

        out["count"] = float(len(use_vals))
        out["meanLatency"] = sum(use_vals) / len(use_vals)
        out["medianLatency"] = _percentile(use_vals, 50)
        out["percentile90"] = _percentile(use_vals, 90)
        out["percentile95"] = _percentile(use_vals, 95)
        out["percentile99"] = _percentile(use_vals, 99)
        out["trimmedMean_p10"] = _trimmed_mean(use_vals, 0.1)
        # Approx throughput (1 / seconds per iter), with latency in ms
        if out["meanLatency"] > 0:
            out["throughput"] = 1000.0 / out["meanLatency"]
        # Also export computeMs aggregates if present
        if computes:
            out["meanComputeMs"] = sum(computes) / len(computes)
            out["medianComputeMs"] = _percentile(computes, 50)
            out["p95ComputeMs"] = _percentile(computes, 95)
        return out

    return out


__all__ = ["load_times_json", "summarize_times"]
