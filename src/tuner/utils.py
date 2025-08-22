from __future__ import annotations

from typing import Dict, List, Tuple
import onnx


def get_first_input_and_outputs(path: str) -> Tuple[str, List[str]]:
    m = onnx.load(path)
    g = m.graph
    in_name = g.input[0].name
    outs = [o.name for o in g.output]
    return in_name, outs


def shapes_str(name: str, dims: List[int]) -> str:
    # Convert dims (with -1 as 1) to "d1xd2x..."
    d = [str(x if x and x > 0 else 1) for x in dims]
    return f"{name}:{'x'.join(d)}"


__all__ = ["get_first_input_and_outputs", "shapes_str"]
