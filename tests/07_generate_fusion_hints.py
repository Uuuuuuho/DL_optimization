#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate horizontal fusion hints and dependency edges for 01_horizontal_fusion.py

What this does
- Loads an ONNX model
- Finds MatMul/Gemm nodes that share the same left-hand input (A)
- Groups candidates by A (and key attrs for Gemm)
- Outputs:
  1) hints JSON (schema expected by 01_horizontal_fusion.py)
  2) edges JSON (simple external dependency edges: [[src_name, dst_name], ...])

Optional external signals
- If TVM/MLIR/Comprehensive analysis reports are present, they inform ranking and grouping
  (but they are not required).

Usage
  python 07_generate_fusion_hints.py \
    --model onnx_out/parallel_matmul.onnx \
    --hints out/hfusion_hints.json \
    --edges out/edges.json \
    --min-group-size 2

Then run 01 with:
  python 01_horizontal_fusion.py \
    --in onnx_out/parallel_matmul.onnx \
    --out onnx_out/parallel_matmul_fused.onnx \
    --dep-source edges-json --dep-file out/edges.json \
    --hints-file out/hfusion_hints.json

Notes
- Node names are required for external edges and hints. Nodes without names are skipped.
- Hints only propose groups; 01 will still validate independence and shapes before fusing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import onnx
    from onnx import numpy_helper
except Exception:
    onnx = None  # type: ignore
    numpy_helper = None  # type: ignore


# ------------------------------
# Helpers
# ------------------------------
def _load_onnx(model_path: str):
    if onnx is None:
        raise RuntimeError("onnx is not installed. Please install it to run this tool.")
    return onnx.load(model_path)


def _initializers_dict(model) -> Dict[str, object]:
    inits: Dict[str, object] = {}
    for t in model.graph.initializer:
        inits[t.name] = t
    return inits


def _tensor_shape_from_init(init) -> Optional[Tuple[int, ...]]:
    try:
        arr = numpy_helper.to_array(init)
        return tuple(arr.shape)
    except Exception:
        # fallback to dims
        if hasattr(init, "dims"):
            return tuple(int(d) for d in init.dims)
        return None


def _get_attr_i(node, name: str, default: int) -> int:
    for a in node.attribute:
        if hasattr(a, "name") and a.name == name and hasattr(a, "i"):
            try:
                return int(a.i)
            except Exception:
                pass
    return default


def _safe_name(node) -> Optional[str]:
    n = getattr(node, "name", None)
    if n is None or n == "":
        return None
    return n


def _build_edges(model) -> List[Tuple[str, str]]:
    """Build (producer_name, consumer_name) edges; skip nodes without names."""
    # Map tensor -> producer node name
    prod: Dict[str, Optional[str]] = {}
    for n in model.graph.node:
        nname = _safe_name(n)
        for o in n.output:
            prod[o] = nname

    edges: List[Tuple[str, str]] = []
    for n in model.graph.node:
        cname = _safe_name(n)
        if not cname:
            continue
        for i in n.input:
            p = prod.get(i)
            if p and p != cname:
                edges.append((p, cname))
    return edges


# ------------------------------
# Candidate discovery (MatMul/Gemm)
# ------------------------------
def _discover_matmul_groups(model, min_group_size: int = 2):
    inits = _initializers_dict(model)
    groups: Dict[Tuple[str, int, Optional[str]], List[str]] = defaultdict(list)
    # key: (A_tensor_name, K, dtype) where dtype is optional string if available

    for n in model.graph.node:
        if n.op_type != "MatMul":
            continue
        name = _safe_name(n)
        if not name:
            continue
        if len(n.input) < 2:
            continue
        A, B = n.input[0], n.input[1]
        if B not in inits:
            continue
        ishape = _tensor_shape_from_init(inits[B])
        if not ishape or len(ishape) != 2:
            continue
        K = int(ishape[0])  # ONNX MatMul: A[..., K] x B[K, M]
        # dtype via data_type not easily accessible without numpy array; try numpy
        dtype_name: Optional[str] = None
        try:
            arr = numpy_helper.to_array(inits[B])
            dtype_name = str(arr.dtype)
        except Exception:
            dtype_name = None
        key = (A, K, dtype_name)
        groups[key].append(name)

    # filter by min size
    groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}
    return groups


def _discover_gemm_groups(model, min_group_size: int = 2):
    inits = _initializers_dict(model)
    groups: Dict[Tuple[str, int, int, int, Optional[str]], List[str]] = defaultdict(list)
    # key: (A_tensor_name, K, transA, transB, dtype)

    for n in model.graph.node:
        if n.op_type != "Gemm":
            continue
        name = _safe_name(n)
        if not name:
            continue
        if len(n.input) < 2:
            continue
        A, B = n.input[0], n.input[1]
        # C optional; if provided and not initializer, still OK
        if B not in inits:
            continue
        tA = _get_attr_i(n, "transA", 0)
        tB = _get_attr_i(n, "transB", 0)
        ishape = _tensor_shape_from_init(inits[B])
        if not ishape or len(ishape) != 2:
            continue
        # For Gemm, if transB=0, B[K, M]; else B[M, K]
        K = int(ishape[0] if tB == 0 else ishape[1])
        dtype_name: Optional[str] = None
        try:
            arr = numpy_helper.to_array(inits[B])
            dtype_name = str(arr.dtype)
        except Exception:
            dtype_name = None
        key = (A, K, tA, tB, dtype_name)
        groups[key].append(name)

    groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}
    return groups


# ------------------------------
# Optional: integrate external reports if present
# ------------------------------
def _load_optional_mlir_report(path: str = "onnx_mlir_optimization.json") -> Dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _load_optional_tvm_report(path: str = "optimization_analysis.json") -> Dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _load_optional_comprehensive_report(path: str = "optimization_results/comprehensive_optimization.json") -> Dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _rank_groups(group_nodes: List[str], bottlenecks: List[Dict]) -> float:
    """Simple score: count of nodes that appear in bottlenecks list (if available)."""
    try:
        bn_names = {b.get("node_name") for b in bottlenecks if isinstance(b, dict)}
    except Exception:
        bn_names = set()
    return float(sum(1 for n in group_nodes if n in bn_names))


# ------------------------------
# Emit hints and edges files
# ------------------------------
def build_hints_and_edges(model_path: str,
                          hints_out: str,
                          edges_out: str,
                          min_group_size: int = 2,
                          include_gemm: bool = True,
                          include_matmul: bool = True) -> Tuple[str, str]:
    model = _load_onnx(model_path)

    # Edges first
    edges = _build_edges(model)
    os.makedirs(os.path.dirname(edges_out) or ".", exist_ok=True)
    with open(edges_out, "w") as f:
        json.dump({"edges": edges}, f, indent=2)

    # Discover groups
    matmul_groups = _discover_matmul_groups(model, min_group_size) if include_matmul else {}
    gemm_groups = _discover_gemm_groups(model, min_group_size) if include_gemm else {}

    # Optional reports
    mlir = _load_optional_mlir_report()
    tvm = _load_optional_tvm_report()
    comp = _load_optional_comprehensive_report()
    bottlenecks = comp.get("bottlenecks", []) if isinstance(comp, dict) else []

    # Flatten into hint groups
    horizontal_groups: List[Dict] = []

    def add_group(op: str, nodes: List[str], force: bool = False):
        # Only include when all nodes have non-empty names
        if all(isinstance(n, str) and n for n in nodes):
            horizontal_groups.append({"op": op, "nodes": nodes, "force": force})

    # Use simple ranking to put likely hot groups first
    ranked: List[Tuple[float, str, List[str]]] = []
    for (_, _K, _dtype), nodes in matmul_groups.items():
        ranked.append((_rank_groups(nodes, bottlenecks), "MatMul", nodes))
    for (_, _K, _tA, _tB, _dtype), nodes in gemm_groups.items():
        ranked.append((_rank_groups(nodes, bottlenecks), "Gemm", nodes))
    ranked.sort(key=lambda x: (-x[0], -len(x[2])))

    # If MLIR finds MatMul+Bias fusion patterns, it's a hint that bias exists; we still keep independence checks in 01
    mlir_patterns = set()
    if isinstance(mlir, dict):
        try:
            mlir_patterns = {p.get("pattern") for p in mlir.get("pattern_matches", [])}
        except Exception:
            mlir_patterns = set()
    prefer_bias_fusion = any("MatMul+Bias" in (p or "") for p in mlir_patterns)

    for score, op, nodes in ranked:
        add_group(op, nodes, force=False if not prefer_bias_fusion else False)

    hints = {
        "blacklist": [],
        "horizontal_groups": horizontal_groups,
        "require_independence": True,
        "min_group_size": int(min_group_size),
    }

    os.makedirs(os.path.dirname(hints_out) or ".", exist_ok=True)
    with open(hints_out, "w") as f:
        json.dump(hints, f, indent=2)

    return hints_out, edges_out


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate fusion hints and edges for 01_horizontal_fusion.py")
    p.add_argument("--model", required=True, help="Path to input ONNX model")
    p.add_argument("--hints", required=True, help="Path to write hints JSON")
    p.add_argument("--edges", required=True, help="Path to write edges JSON")
    p.add_argument("--min-group-size", type=int, default=2, help="Minimum group size to hint")
    p.add_argument("--no-matmul", action="store_true", help="Disable MatMul grouping")
    p.add_argument("--no-gemm", action="store_true", help="Disable Gemm grouping")
    return p.parse_args()


def main():
    args = parse_args()
    include_matmul = not args.no_matmul
    include_gemm = not args.no_gemm
    hints_path, edges_path = build_hints_and_edges(
        model_path=args.model,
        hints_out=args.hints,
        edges_out=args.edges,
        min_group_size=args.min_group_size,
        include_gemm=include_gemm,
        include_matmul=include_matmul,
    )
    print(f"Wrote hints to: {hints_path}")
    print(f"Wrote edges to: {edges_path}")
    print("Next: run 01_horizontal_fusion.py with --dep-source edges-json --dep-file <edges> --hints-file <hints>.")


if __name__ == "__main__":
    main()
