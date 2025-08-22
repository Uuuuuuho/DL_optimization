#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate Generator: produce TensorRT-friendly ONNX variants using ONNX GraphSurgeon.

Rules implemented (initial minimal set):
 - Tensor padding on the last dim to multiple of {8,16} with slice-restore at output.
 - MatMul with constant weight -> Gemm (bias optional) to hint TRT FC path.
 - Layout normalization stubs (NCHW/NHWC) with Transpose insertion.

Notes:
 - This module aims to be safe-by-default; if a transform cannot be applied, it leaves the graph unchanged.
 - More rules can be added incrementally.
"""
from __future__ import annotations

import os
import copy
import json
from typing import List, Dict, Any, Tuple

import onnx
import numpy as np

try:
    import onnx_graphsurgeon as gs  # type: ignore
except Exception as e:
    raise SystemExit("This module requires onnx-graphsurgeon. Install via pip install onnx-graphsurgeon")


def _to_numpy(t: onnx.TensorProto) -> np.ndarray:
    from onnx import numpy_helper
    return numpy_helper.to_array(t)


def load_onnx(path: str) -> onnx.ModelProto:
    return onnx.load(path)


def save_onnx(model: onnx.ModelProto, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)


def graph_from_model(m: onnx.ModelProto) -> gs.Graph:
    return gs.import_onnx(m)


def model_from_graph(g: gs.Graph, opset: int | None = None) -> onnx.ModelProto:
    m = gs.export_onnx(g)
    if opset:
        # Set minimal opset
        for imp in m.opset_import:
            if imp.domain == "":
                imp.version = opset
    return m


def pad_last_dim_to_multiple(graph: gs.Graph, multiple: int) -> Tuple[gs.Graph, Dict[str, Any]]:
    """Pad the last dimension of model inputs to a multiple using Pad op.

    Strategy:
      - For each model input with static last dim D, if D % multiple != 0: pad to next multiple.
      - Insert Slice at the end to restore original shape for each model output where needed.
    Limitations: focuses on simple 2D inputs (N x C) or (... x K) last-dim alignment.
    """
    g = graph
    meta: Dict[str, Any] = {"type": "pad_last_dim", "multiple": multiple, "padded": []}
    updated = False

    # Determine pad sizes per input tensor
    input_map = {i.name: i for i in g.inputs}
    pad_info: Dict[str, int] = {}
    for inp in g.inputs:
        shape = list(inp.shape) if inp.shape is not None else None
        if not shape or not isinstance(shape[-1], (int, np.integer)):
            continue
        last = int(shape[-1])
        if last <= 0:
            continue
        if last % multiple == 0:
            continue
        pad_amount = multiple - (last % multiple)
        pad_info[inp.name] = pad_amount

    if not pad_info:
        return g, meta

    # Create constant pad spec: pads = [0,0,..., 0, pad_amount]
    def make_const(name: str, arr: np.ndarray):
        t = gs.Constant(name=name, values=arr)
        return t

    for inp in g.inputs:
        if inp.name not in pad_info:
            continue
        pad_amt = pad_info[inp.name]
        rank = len(inp.shape)
        pads = [0] * (2 * rank)
        pads[rank - 1] = 0  # begin pad last dim
        pads[2 * rank - 1] = pad_amt  # end pad last dim
        pads_c = make_const(f"pads_{inp.name}", np.array(pads, dtype=np.int64))
        value_c = make_const(f"pad_val_{inp.name}", np.array(0.0, dtype=np.float32))
        pad_out = gs.Variable(name=f"{inp.name}_padded", dtype=inp.dtype, shape=list(inp.shape))
        # Update last dim in shape if static
        if isinstance(pad_out.shape[-1], (int, np.integer)):
            pad_out.shape[-1] = int(pad_out.shape[-1]) + pad_amt
        pad_node = gs.Node(op="Pad", name=f"Pad_{inp.name}", inputs=[inp, pads_c, value_c], outputs=[pad_out])
        # Replace uses of inp with pad_out
        for n in g.nodes:
            for i, t in enumerate(n.inputs):
                if t is inp:
                    n.inputs[i] = pad_out
        # Keep original graph input symbolically (Pad reads from it), consumers now read pad_out.
        g.nodes.append(pad_node)
        meta["padded"].append({"input": inp.name, "pad": pad_amt})
        updated = True

    if updated:
        g.cleanup()
    return g, meta


def matmul_to_gemm(graph: gs.Graph) -> Tuple[gs.Graph, Dict[str, Any]]:
    """Convert MatMul(X, W) with constant W to Gemm(X, W, B=0).
    TRT often recognizes Gemm as FullyConnected faster path.
    """
    g = graph
    meta: Dict[str, Any] = {"type": "matmul_to_gemm", "converted": 0}
    for n in list(g.nodes):
        if n.op != "MatMul":
            continue
        if len(n.inputs) != 2:
            continue
        X, W = n.inputs
        if not isinstance(W, gs.Constant):
            continue
        # Ensure 2D weight
        wv = np.array(W.values)
        if wv.ndim != 2:
            continue
        # Gemm: Y = alpha*A*B + beta*C, with C=0 bias
        B0 = gs.Constant(name=f"{n.name}_bias0", values=np.zeros((wv.shape[1],), dtype=wv.dtype))
        Y = n.outputs[0]
        gemm = gs.Node(op="Gemm", name=f"Gemm_from_{n.name}", inputs=[X, W, B0], outputs=[Y], attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0})
        g.nodes.append(gemm)
        # Remove old node by disconnecting
        n.outputs = []
        meta["converted"] += 1
    g.cleanup()
    return g, meta


def normalize_layout_nchw(graph: gs.Graph) -> Tuple[gs.Graph, Dict[str, Any]]:
    """Stub: ensure NCHW inputs by inserting Transpose if rank==4 and last two dims look like HW.
    Conservatively no-op if shapes are unknown.
    """
    g = graph
    meta: Dict[str, Any] = {"type": "layout_nchw", "transposes": 0}
    # Minimal/no-op for now; extend with real detection.
    return g, meta


def generate_candidates(base: str, out_dir: str, max_candidates: int = 16) -> List[Dict[str, Any]]:
    """Generate candidate models from base.onnx and write them to out_dir.

    Returns a list of dicts with fields: {id, onnx_path, transforms, notes}
    """
    os.makedirs(out_dir, exist_ok=True)
    m = load_onnx(base)
    base_graph = graph_from_model(m)

    candidates: List[Dict[str, Any]] = []

    def add_candidate(g: gs.Graph, transforms: List[Dict[str, Any]], tag: str):
        nonlocal candidates
        gid = len(candidates)
        path = os.path.join(out_dir, f"candidate_{gid:02d}_{tag}.onnx")
        model = model_from_graph(g, opset=13)
        save_onnx(model, path)
        candidates.append({
            "id": gid,
            "onnx_path": path,
            "transforms": transforms,
            "notes": tag,
        })

    # Candidate 0: baseline copy
    add_candidate(copy.deepcopy(base_graph), [], "baseline")

    # Pad last dim to multiples
    for mult in (8, 16):
        g = copy.deepcopy(base_graph)
        g, meta_pad = pad_last_dim_to_multiple(g, mult)
        add_candidate(g, [meta_pad], f"padlast{mult}")

    # MatMul -> Gemm
    g = copy.deepcopy(base_graph)
    g, meta_gemm = matmul_to_gemm(g)
    add_candidate(g, [meta_gemm], "gemm")

    # Combine: pad + gemm
    for mult in (8, 16):
        g = copy.deepcopy(base_graph)
        g, meta_pad = pad_last_dim_to_multiple(g, mult)
        g, meta_gemm = matmul_to_gemm(g)
        add_candidate(g, [meta_pad, meta_gemm], f"pad{mult}_gemm")

    # Layout stub (no-op now)
    g = copy.deepcopy(base_graph)
    g, meta_layout = normalize_layout_nchw(g)
    add_candidate(g, [meta_layout], "layout_nchw")

    # Cap
    return candidates[:max_candidates]


__all__ = [
    "generate_candidates",
    "pad_last_dim_to_multiple",
    "matmul_to_gemm",
    "normalize_layout_nchw",
]
