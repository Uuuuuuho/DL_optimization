from __future__ import annotations

from typing import Any, Dict, List


def pick_best(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the best row by meanLatency (or mean_ms fallback), then build_time_s."""
    if not metrics:
        return {}

    def key(m: Dict[str, Any]):
        ml = m.get("meanLatency")
        if ml is None:
            ml = m.get("mean_ms", float("inf"))
        bt = m.get("build_time_s", float("inf"))
        return (ml if ml is not None else float("inf"), bt)

    return sorted(metrics, key=key)[0]


__all__ = ["pick_best"]
