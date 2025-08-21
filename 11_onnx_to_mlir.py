#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export ONNX model to textual MLIR (ONNX dialect) with optional optimizations.

Defaults: no optimization. You can enable simple ONNX graph optimizations via --opt-level
without requiring onnx-mlir, and optionally run an external mlir-opt pass pipeline
on the emitted MLIR if available.

Usage:
    # No optimization (default)
    python 11_onnx_to_mlir.py --in onnx_out/parallel_matmul.onnx --out onnx_out/parallel_matmul.mlir

    # With ONNX graph optimization level O2 (requires 'onnxoptimizer')
    python 11_onnx_to_mlir.py --in model.onnx --out model.mlir --opt-level O2

    # Optionally post-process MLIR with mlir-opt pass pipeline
    python 11_onnx_to_mlir.py --in model.onnx --out model.mlir --mlir-opt /usr/bin/mlir-opt \
        --mlir-opt-pipeline 'canonicalize,cse'
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Tuple

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except Exception:
    raise SystemExit("This tool requires 'onnx' package. Install it via pip install onnx")

# Optional ONNX graph optimization (pure Python)
try:
    import onnxoptimizer  # type: ignore
except Exception:
    onnxoptimizer = None  # noqa: N816


def onnx_dtype_to_mlir_token(elem_type: int) -> str:
    m = {
        onnx.TensorProto.FLOAT16: "f16",
        onnx.TensorProto.FLOAT: "f32",
        onnx.TensorProto.DOUBLE: "f64",
        onnx.TensorProto.INT8: "i8",
        onnx.TensorProto.INT16: "i16",
        onnx.TensorProto.INT32: "i32",
        onnx.TensorProto.INT64: "i64",
        onnx.TensorProto.UINT8: "ui8",
        onnx.TensorProto.UINT16: "ui16",
        onnx.TensorProto.UINT32: "ui32",
        onnx.TensorProto.UINT64: "ui64",
        onnx.TensorProto.BOOL: "i1",  # fallback; 09 parser treats unknown as f32 by default
    }
    return m.get(elem_type, "f32")


def tensor_type_str(dims: List[int], elem_type: int) -> str:
    dims_tokens: List[str] = []
    for d in dims:
        if d is None or d < 0:
            dims_tokens.append("?")
        else:
            dims_tokens.append(str(int(d)))
    dtok = onnx_dtype_to_mlir_token(elem_type)
    return f"tensor<{'x'.join(dims_tokens)}x{dtok}>" if dims_tokens else f"tensor<{dtok}>"


def value_info_to_type(vi: onnx.ValueInfoProto) -> str:
    ttype = vi.type.tensor_type
    elem_type = ttype.elem_type or onnx.TensorProto.FLOAT
    dims = [d.dim_value if d.HasField('dim_value') else -1 for d in ttype.shape.dim]
    return tensor_type_str(dims, elem_type)


def sanitize_name(name: str) -> str:
    if not name:
        return ""
    # Replace invalid characters with underscore
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def emit_dense_for_initializer(t: onnx.TensorProto) -> str:
    arr = numpy_helper.to_array(t)
    # Use Python nested lists to be compatible with 09 parser's ast.literal_eval
    return f"dense<{arr.tolist()}>"


def build_mlir(model: onnx.ModelProto) -> str:
    g = model.graph
    # Try to infer shapes to get more precise types (no optimization)
    try:
        model = shape_inference.infer_shapes(model)
        g = model.graph
    except Exception:
        pass

    lines: List[str] = []
    lines.append("module {")

    # Map ONNX value name -> SSA token (e.g., %arg0, %init_w, %v1)
    ssa: Dict[str, str] = {}
    constants_emitted: Dict[str, bool] = {}

    # Build function signature
    arg_parts: List[str] = []
    for i, inp in enumerate(g.input):
        nm = sanitize_name(inp.name) or f"input_{i}"
        ssa_token = f"%arg{i}"
        ssa[inp.name] = ssa_token
        arg_parts.append(f"{ssa_token}: {value_info_to_type(inp)}")

    # Outputs types
    ret_types: List[str] = []
    for out in g.output:
        ret_types.append(value_info_to_type(out))

    if len(ret_types) == 1:
        ret_sig = ret_types[0]
    else:
        ret_sig = f"({', '.join(ret_types)})"

    lines.append(f"  func.func @main({', '.join(arg_parts)}) -> {ret_sig} {{")

    # Emit Constants for initializers when first used
    init_map: Dict[str, onnx.TensorProto] = {init.name: init for init in g.initializer}

    def ensure_const(name: str) -> str:
        if name not in ssa:
            ssa[name] = f"%{sanitize_name(name) or 'init'}"
        if not constants_emitted.get(name) and name in init_map:
            t = init_map[name]
            dense = emit_dense_for_initializer(t)
            ttype = tensor_type_str(list(t.dims), t.data_type)
            lines.append(f"    {ssa[name]} = onnx.Constant {{value = {dense}}} : {ttype}")
            constants_emitted[name] = True
        return ssa[name]

    # Helper to get type of a value if available from inputs/outputs (fallback unknown)
    vi_types: Dict[str, str] = {vi.name: value_info_to_type(vi) for vi in list(g.input) + list(g.value_info) + list(g.output)}

    def get_type(name: str) -> str:
        t = vi_types.get(name)
        if t:
            return t
        if name in init_map:
            init = init_map[name]
            return tensor_type_str(list(init.dims), init.data_type)
        # unknown
        return "tensor<?xf32>"

    # Emit ops (assume single output for round-trip with 09 parser)
    tmp_id = 0
    for node in g.node:
        # Determine output name (first output only)
        out_name = node.output[0] if node.output else f"v{tmp_id}"
        if not out_name:
            out_name = f"v{tmp_id}"
        tmp_id += 1
        ssa_out = f"%{sanitize_name(out_name)}"
        # Record mapping to this SSA
        ssa[out_name] = ssa_out

        # Prepare operands
        ops_ssa: List[str] = []
        in_types: List[str] = []
        for inp in node.input:
            if inp in init_map:
                ops_ssa.append(ensure_const(inp))
            elif inp in ssa:
                ops_ssa.append(ssa[inp])
            else:
                # Unseen dynamic input (not a graph input?), synthesize token
                tok = f"%{sanitize_name(inp) or 'v'}"
                ssa[inp] = tok
                ops_ssa.append(tok)
            in_types.append(get_type(inp))

        out_type = get_type(out_name)
        # Build op line with optional attributes
        opname = node.op_type
        operands = ", ".join(ops_ssa)
        types_part = ", ".join(in_types) + f" -> {out_type}"
        # Convert attributes to MLIR-like literal map (subset)
        attrs_kv: List[str] = []
        for a in node.attribute:
            k = a.name
            v_str: Optional[str] = None
            if a.type == onnx.AttributeProto.INT:
                v_str = str(a.i)
            elif a.type == onnx.AttributeProto.FLOAT:
                v_str = ("%g" % a.f)
            elif a.type == onnx.AttributeProto.STRING:
                try:
                    v = a.s.decode("utf-8") if isinstance(a.s, (bytes, bytearray)) else str(a.s)
                except Exception:
                    v = str(a.s)
                v_str = f'"{v}"'
            elif a.type == onnx.AttributeProto.INTS:
                v_str = "[" + ", ".join(str(i) for i in a.ints) + "]"
            elif a.type == onnx.AttributeProto.FLOATS:
                v_str = "[" + ", ".join("%g" % f for f in a.floats) + "]"
            # Tensors/Graphs/SparseTensors are not emitted in this minimal emitter
            if v_str is not None:
                attrs_kv.append(f"{k}={v_str}")
        if attrs_kv:
            attrs_str = " {" + ", ".join(attrs_kv) + "}"
        else:
            attrs_str = ""
        lines.append(f"    {ssa_out} = onnx.{opname}{attrs_str} {operands} : {types_part}")

    # Return statement (use graph outputs)
    if len(g.output) == 1:
        out_vi = g.output[0]
        o_name = out_vi.name
        o_ssa = ssa.get(o_name, f"%{sanitize_name(o_name) or 'out'}")
        lines.append(f"    return {o_ssa} : {value_info_to_type(out_vi)}")
    else:
        parts: List[str] = []
        for out_vi in g.output:
            o_name = out_vi.name
            o_ssa = ssa.get(o_name, f"%{sanitize_name(o_name) or 'out'}")
            parts.append(f"{o_ssa} : {value_info_to_type(out_vi)}")
        lines.append(f"    return {', '.join(parts)}")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def optimize_onnx_model(model: onnx.ModelProto, level: str, passes: Optional[List[str]] = None) -> onnx.ModelProto:
    """Optionally optimize ONNX graph using onnxoptimizer.

    level: 'O0' (no-op), 'O1', 'O2', 'O3'
    passes: explicit pass list to override defaults for a given level.
    """
    lvl = (level or "O0").upper()
    if lvl == "O0":
        return model
    if onnxoptimizer is None:
        print("[warn] onnxoptimizer not installed; skipping ONNX optimizations.")
        return model

    # Define conservative pass sets by level
    default_sets: Dict[str, List[str]] = {
        "O1": [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "extract_constant_to_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
            "nop",
        ],
        "O2": [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "eliminate_nop_dropout",
            "extract_constant_to_initializer",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
            "eliminate_unused_initializer",
        ],
        "O3": [
            # O2 + extra folds
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "eliminate_nop_dropout",
            "extract_constant_to_initializer",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
            "eliminate_unused_initializer",
            "fold_constant",
        ],
    }
    pass_list = passes if passes is not None else default_sets.get(lvl, [])
    if not pass_list:
        return model
    try:
        # Some optimizers may require names on all nodes
        optimized = onnxoptimizer.optimize(model, pass_list)
        return optimized
    except Exception as e:
        print(f"[warn] onnxoptimizer failed ({e}); returning original model.")
        return model


def run_mlir_opt(mlir_text: str, mlir_opt_bin: str, pipeline: str) -> str:
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mlir", delete=True) as tmp:
        tmp.write(mlir_text)
        tmp.flush()
        # Use pass pipeline; fallback to canonicalize if pipeline empty
        args = [mlir_opt_bin]
        if pipeline:
            args.extend(["--pass-pipeline", pipeline])
        else:
            args.append("-canonicalize")
        args.append(tmp.name)
        try:
            res = subprocess.run(args, check=True, capture_output=True, text=True)
            return res.stdout if res.stdout else mlir_text
        except Exception as e:
            print(f"[warn] mlir-opt failed ({e}); using unoptimized MLIR.")
            return mlir_text


def parse_args():
    p = argparse.ArgumentParser(description="Export ONNX to textual MLIR (ONNX dialect) without optimization")
    p.add_argument("--in", dest="in_onnx", default="onnx_out/parallel_matmul.onnx", help="Input ONNX model path")
    p.add_argument("--out", dest="out_mlir", default="onnx_out/parallel_matmul.mlir", help="Output MLIR text path")
    p.add_argument("--opt-level", choices=["O0", "O1", "O2", "O3"], default="O0", help="ONNX graph optimization level (requires onnxoptimizer for O1+)")
    p.add_argument("--opt-passes", nargs="*", default=None, help="Explicit onnxoptimizer pass list (overrides --opt-level presets)")
    p.add_argument("--mlir-opt", dest="mlir_opt_bin", default=None, help="Path to mlir-opt binary to post-process MLIR (optional)")
    p.add_argument("--mlir-opt-pipeline", dest="mlir_opt_pipeline", default="", help="mlir-opt pass pipeline string, e.g. 'canonicalize,cse'")
    return p.parse_args()


def main():
    args = parse_args()
    model = onnx.load(args.in_onnx)
    # Optional ONNX graph optimization (pure Python)
    model = optimize_onnx_model(model, args.opt_level, args.opt_passes)
    mlir_text = build_mlir(model)
    # Optional mlir-opt post-pass
    if args.mlir_opt_bin:
        mlir_text = run_mlir_opt(mlir_text, args.mlir_opt_bin, args.mlir_opt_pipeline)
    with open(args.out_mlir, "w") as f:
        f.write(mlir_text)
    print(f"Saved MLIR to {args.out_mlir}")


if __name__ == "__main__":
    main()
