#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert MLIR (ONNX dialect, textual) to an ONNX (protobuf) model.

Scope
- Parses a subset of MLIR produced by onnx-mlir --EmitMLIR (onnx.<Op> dialect)
- Supports common ops with 1 output: Constant, MatMul, Gemm, Add, Relu, Conv, Mul, Sub, Div, Sigmoid, Tanh
- Infers graph inputs from function arguments and SSA values with no producer
- Infers graph outputs from `return` operands

Notes
- This is a best-effort textual parser intended for pipelines where you want to round-trip
  MLIR (ONNX dialect) back to ONNX for further graph-level tooling.
- For complex graphs or non-ONNX dialect MLIR, prefer a dedicated converter if available.

Usage
  python 09_mlir_to_onnx.py --in model.mlir --out model.onnx
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import onnx
    from onnx import helper, TensorProto
except Exception as e:
    raise SystemExit("This tool requires 'onnx' package. Install it via pip install onnx")


# ------------------------------
# Utilities
# ------------------------------
_DTYPE_MAP = {
    "f16": (TensorProto.FLOAT16, np.float16),
    "f32": (TensorProto.FLOAT, np.float32),
    "f64": (TensorProto.DOUBLE, np.float64),
    "i8": (TensorProto.INT8, np.int8),
    "i16": (TensorProto.INT16, np.int16),
    "i32": (TensorProto.INT32, np.int32),
    "i64": (TensorProto.INT64, np.int64),
    "ui8": (TensorProto.UINT8, np.uint8),
    "ui16": (TensorProto.UINT16, np.uint16),
    "ui32": (TensorProto.UINT32, np.uint32),
    "ui64": (TensorProto.UINT64, np.uint64),
}


def _parse_tensor_type(tensor_type: str) -> Tuple[List[int], int, type]:
    """Parse MLIR tensor type 'tensor<d1xd2x...xDT>' to (shape, onnx_dtype, np_dtype)."""
    m = re.match(r"tensor<([^>]+)>", tensor_type.strip())
    if not m:
        return ([], TensorProto.UNDEFINED, np.float32)
    inner = m.group(1)
    # last token is dtype (e.g., f32)
    parts = inner.split("x")
    dtok = parts[-1]
    onnx_dt, np_dt = _DTYPE_MAP.get(dtok, (TensorProto.UNDEFINED, np.float32))
    shape: List[int] = []
    for p in parts[:-1]:
        if p == "?" or p == "*":
            shape.append(-1)
        else:
            try:
                shape.append(int(p))
            except Exception:
                shape.append(-1)
    return (shape, onnx_dt, np_dt)


def _mlir_dense_to_numpy(dense_str: str, tensor_type: str, np_dt: type) -> np.ndarray:
    """Parse MLIR dense<...> literal into numpy array with shape from tensor_type."""
    # Normalize: dense<[...]> or dense<...>
    inner = dense_str.strip()
    if inner.startswith("dense<"):
        inner = inner[len("dense<"):]
    if inner.endswith(">"):
        inner = inner[:-1]
    inner = inner.strip()
    # Try to parse as Python literal list
    try:
        # MLIR often uses [] nesting compatible with Python lists
        data = ast.literal_eval(inner.replace("x", ","))
        arr = np.array(data, dtype=np_dt)
    except Exception:
        # Fallback: space/comma separated numbers
        nums = re.split(r"[\s,]+", inner.strip("[] {}"))
        nums = [n for n in nums if n]
        arr = np.array([float(n) for n in nums], dtype=np_dt)
    shape, _, _ = _parse_tensor_type(tensor_type)
    if shape and all(d > 0 for d in shape):
        try:
            arr = arr.reshape(shape)
        except Exception:
            pass
    return arr


@dataclass
class SSAValue:
    name: str
    tensor_type: Optional[str] = None
    producer: Optional[str] = None  # node name producing it


# ------------------------------
# Core converter
# ------------------------------
class MLIRToONNXConverter:
    def __init__(self, mlir_text: str, opset: int = 13) -> None:
        self.text = mlir_text
        self.opset = opset
        self.ssa: Dict[str, SSAValue] = {}
        self.initializers: Dict[str, np.ndarray] = {}
        self.nodes: List[onnx.NodeProto] = []
        self.func_args: Dict[str, str] = {}  # %argN -> tensor<...>
        self.graph_outputs: List[str] = []

    def convert(self) -> onnx.ModelProto:
        self._parse_function_signature()
        self._parse_operations()
        inputs, value_infos = self._build_inputs_and_values()
        outputs = self._build_outputs()

        graph = helper.make_graph(
            self.nodes,
            name="mlir_converted",
            inputs=inputs,
            outputs=outputs,
            initializer=[helper.make_tensor(k, self._np_to_onnx_dtype(v.dtype), v.shape, v.flatten().tolist())
                         for k, v in self.initializers.items()],
            value_info=value_infos,
        )
        model = helper.make_model(graph, producer_name="mlir_to_onnx", opset_imports=[helper.make_opsetid("", self.opset)])
        onnx.checker.check_model(model)
        return model

    # -------- Parsing --------
    def _parse_function_signature(self) -> None:
        # Example: func.func @main(%arg0: tensor<1x3xf32>, %arg1: tensor<...>) -> ...
        func_re = re.compile(r"func\.func\s+@([^(]+)\(([^)]*)\)")
        m = func_re.search(self.text)
        if not m:
            return
        args = m.group(2).strip()
        if not args:
            return
        for part in self._split_top_level(args, ","):
            part = part.strip()
            am = re.match(r"(%[a-zA-Z0-9_]+)\s*:\s*(tensor<[^>]+>)", part)
            if am:
                self.func_args[am.group(1)] = am.group(2)
                self.ssa[am.group(1)] = SSAValue(name=am.group(1)[1:], tensor_type=am.group(2), producer=None)

    def _parse_operations(self) -> None:
        # Return line: return %x, %y
        ret_m = re.search(r"\breturn\b\s+([^\n}]+)", self.text)
        if ret_m:
            returns = [s.strip() for s in ret_m.group(1).split(",")]
            for r in returns:
                # keep only the SSA id (e.g., '%y' from '%y : tensor<...>')
                m = re.match(r"(%[\w\d_]+)", r)
                if m:
                    self.graph_outputs.append(m.group(1))
        # Iterate over onnx ops
        for line in self.text.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # Constant
            m_const = re.match(r"(%[\w\d_]+)\s*=\s*onnx\.Constant\s*\{[^}]*value\s*=\s*([^}]*)\}\s*:\s*(tensor<[^>]+>)", line)
            if m_const:
                res, dense, ttype = m_const.group(1), m_const.group(2).strip(), m_const.group(3)
                _, _, np_dt = _parse_tensor_type(ttype)
                arr = _mlir_dense_to_numpy(dense, ttype, np_dt)
                name = res[1:]
                self.initializers[name] = arr
                self.ssa[res] = SSAValue(name=name, tensor_type=ttype, producer=None)
                continue
            # Generic one-output op with optional attrs
            m_op = re.match(r"(%[\w\d_]+)\s*=\s*onnx\.([A-Za-z0-9_]+)\s*(\{[^}]*\})?\s*([^:]*)\s*:\s*(.*)", line)
            if m_op:
                res = m_op.group(1)
                op = m_op.group(2)
                attrs_raw = m_op.group(3) or ""
                operands_part = m_op.group(4).strip()
                type_part = m_op.group(5).strip()
                inputs = [tok.strip() for tok in operands_part.split(",") if tok.strip().startswith("%")]
                attr_dict = self._parse_attrs(attrs_raw)
                # Fallbacks for required attributes if missing (e.g., Concat.axis)
                if op == "Concat" and "axis" not in attr_dict:
                    axis = self._infer_concat_axis_from_types(type_part)
                    attr_dict["axis"] = axis
                out_name = res[1:]
                node = helper.make_node(op_type=op, inputs=[self._ssa_to_name(i) for i in inputs], outputs=[out_name], name=out_name, **attr_dict)
                self.nodes.append(node)
                # Record SSA value and type (try to get output type from the tail of type_part)
                out_type_match = re.search(r"->\s*(tensor<[^>]+>)", type_part)
                out_ttype = out_type_match.group(1) if out_type_match else None
                self.ssa[res] = SSAValue(name=out_name, tensor_type=out_ttype, producer=node.name)
                continue
            # Ignore other lines

    def _parse_attrs(self, attrs_raw: str) -> Dict:
        attrs: Dict = {}
        if not attrs_raw:
            return attrs
        # Remove braces { ... }
        s = attrs_raw.strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        # Split by commas at top level
        for kv in self._split_top_level(s, ","):
            if not kv.strip():
                continue
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Remove MLIR type suffix like ": f64" or ": i64"
            v = re.sub(r"\s*:\s*[a-zA-Z0-9_<>]+", "", v)
            # Try parse to python literal
            val: object
            if v.startswith("[") or v.startswith("{"):
                try:
                    val = ast.literal_eval(v.replace("{", "[").replace("}", "]"))
                except Exception:
                    val = v
            else:
                if re.match(r"^[+-]?\d+\.?\d*$", v):
                    if "." in v:
                        val = float(v)
                    else:
                        try:
                            val = int(v)
                        except Exception:
                            val = v
                elif v.lower() in ("true", "false"):
                    val = v.lower() == "true"
                else:
                    val = v.strip('"')
            attrs[k] = val
        return attrs

    def _split_top_level(self, s: str, sep: str) -> List[str]:
        parts: List[str] = []
        depth = 0
        current = []
        i = 0
        while i < len(s):
            c = s[i]
            if c in "({[":
                depth += 1
            elif c in ")}]":
                depth = max(0, depth - 1)
            if c == sep and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(c)
            i += 1
        if current:
            parts.append("".join(current))
        return parts

    def _ssa_to_name(self, ssa: str) -> str:
        if ssa in self.ssa:
            return self.ssa[ssa].name
        # create an input placeholder
        name = ssa[1:]
        self.ssa[ssa] = SSAValue(name=name, tensor_type=None, producer=None)
        return name

    def _extract_types_from_signature(self, type_part: str) -> Tuple[List[str], Optional[str]]:
        """Extract input tensor<...> types and output type from an op signature tail.
        Example: "tensor<1x2xf32>, tensor<2x3xf32> -> tensor<1x3xf32>" -> ([...inputs...], output)
        """
        toks = re.findall(r"tensor<[^>]+>", type_part)
        if not toks:
            return ([], None)
        if "->" in type_part and len(toks) >= 1:
            return (toks[:-1], toks[-1]) if len(toks) > 1 else ([], toks[-1])
        # If arrow not found, assume last is output as a best-effort
        return (toks[:-1], toks[-1]) if len(toks) > 1 else (toks, None)

    def _infer_concat_axis_from_types(self, type_part: str) -> int:
        """Infer axis for Concat from input/output shapes when possible; default to 1.
        Strategy: find dimension index where input dims differ and (if output known) sum equals output dim.
        """
        in_types, out_type = self._extract_types_from_signature(type_part)
        in_shapes = [ _parse_tensor_type(t)[0] for t in in_types ]
        out_shape = _parse_tensor_type(out_type)[0] if out_type else []
        # Determine max rank among known shapes
        rank = 0
        for s in in_shapes:
            rank = max(rank, len(s))
        rank = max(rank, len(out_shape))
        if rank == 0:
            return 1
        # Try to find axis satisfying sum rule
        for ax in range(rank):
            same_others = True
            candidate_dims = []
            for s in in_shapes:
                if len(s) != rank:
                    same_others = False
                    break
                candidate_dims.append(s[ax])
                # check other dims equal (or unknown) across inputs
            if not same_others:
                continue
            ok_others = True
            for d_i in range(rank):
                if d_i == ax:
                    continue
                vals = [s[d_i] for s in in_shapes if len(s) == rank]
                known = [v for v in vals if v is not None and v >= 0]
                if len(set(known)) > 1:
                    ok_others = False
                    break
            if not ok_others:
                continue
            if out_shape and len(out_shape) == rank:
                out_d = out_shape[ax]
                known_in = [v for v in candidate_dims if v is not None and v >= 0]
                if out_d is not None and out_d >= 0 and known_in:
                    if sum(known_in) == out_d:
                        return ax
                else:
                    # can't verify, but candidate ok
                    return ax
            else:
                return ax
        # Fallback: pick first index where inputs differ, else 1
        for ax in range(rank):
            vals = [s[ax] for s in in_shapes if len(s) == rank]
            if len(set([v for v in vals if v is not None and v >= 0])) > 1:
                return ax
        return 1

    # -------- Graph assembly --------
    def _build_inputs_and_values(self):
        inputs: List[onnx.ValueInfoProto] = []
        value_infos: List[onnx.ValueInfoProto] = []
        produced = {v.name for v in self.ssa.values() if v.producer is not None or v.name in self.initializers}

        # Function args become inputs (unless they are initializers)
        for ssa_arg, ttype in self.func_args.items():
            name = self.ssa[ssa_arg].name if ssa_arg in self.ssa else ssa_arg[1:]
            shape, onnx_dt, _ = _parse_tensor_type(ttype)
            # Always provide a shape proto (empty dims if unknown) to satisfy checker
            dims = [d if d > 0 else None for d in shape] if shape else []
            vi = helper.make_tensor_value_info(name, onnx_dt, dims)
            inputs.append(vi)

        # Add additional SSA values used as inputs without producers as graph inputs
        for ssa_name, ssa_val in self.ssa.items():
            if ssa_name in self.func_args:
                continue
            if ssa_val.name in self.initializers:
                continue
            if ssa_val.producer is None:
                # treat as dynamic input
                shape, onnx_dt, _ = _parse_tensor_type(ssa_val.tensor_type or "tensor<?>xf32")
                # default dtype fallback to FLOAT, ensure shape exists (empty list if unknown)
                if onnx_dt == TensorProto.UNDEFINED:
                    onnx_dt = TensorProto.FLOAT
                dims = [d if d > 0 else None for d in shape] if shape else []
                vi = helper.make_tensor_value_info(ssa_val.name, onnx_dt, dims)
                if all(i.name != vi.name for i in inputs):
                    inputs.append(vi)

        # Optional: add value_info for produced tensors with known types
        for ssa_val in self.ssa.values():
            if ssa_val.producer is not None and ssa_val.tensor_type:
                shape, onnx_dt, _ = _parse_tensor_type(ssa_val.tensor_type)
                if onnx_dt == TensorProto.UNDEFINED:
                    onnx_dt = TensorProto.FLOAT
                dims = [d if d > 0 else None for d in shape] if shape else []
                value_infos.append(helper.make_tensor_value_info(ssa_val.name, onnx_dt, dims))

        return inputs, value_infos

    def _build_outputs(self) -> List[onnx.ValueInfoProto]:
        outs: List[onnx.ValueInfoProto] = []
        if not self.graph_outputs:
            # fallback: last node's output with safe defaults (FLOAT, empty shape)
            if self.nodes:
                last = self.nodes[-1].output[0]
                outs.append(helper.make_tensor_value_info(last, TensorProto.FLOAT, []))
            return outs
        for ssa_out in self.graph_outputs:
            v = self.ssa.get(ssa_out)
            if v and v.tensor_type:
                shape, onnx_dt, _ = _parse_tensor_type(v.tensor_type)
            else:
                shape, onnx_dt = [], TensorProto.FLOAT
            dims = [d if d > 0 else None for d in shape] if shape else []
            outs.append(helper.make_tensor_value_info(v.name if v else ssa_out[1:], onnx_dt, dims))
        return outs

    # -------- Helpers --------
    def _np_to_onnx_dtype(self, np_dtype) -> int:
        for k, (_, npdt) in _DTYPE_MAP.items():
            if np_dtype == npdt:
                return _DTYPE_MAP[k][0]
        # default
        if np_dtype == np.bool_:
            return TensorProto.BOOL
        return TensorProto.UNDEFINED


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert MLIR (ONNX dialect) to ONNX ModelProto")
    p.add_argument("--in", dest="in_mlir", default="onnx_out/parallel_matmul_opt.onnx.mlir", help="Path to MLIR textual file")
    p.add_argument("--out", dest="out_onnx", default="onnx_out/parallel_matmul_opt.onnx", help="Path to output ONNX file")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset to set in the model")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.in_mlir, "r") as f:
        mlir_text = f.read()
    conv = MLIRToONNXConverter(mlir_text, opset=args.opset)
    model = conv.convert()
    onnx.save(model, args.out_onnx)
    print(f"Saved ONNX model to {args.out_onnx}")


if __name__ == "__main__":
    main()
