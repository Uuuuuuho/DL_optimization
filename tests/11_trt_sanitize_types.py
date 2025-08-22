#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanitize ONNX model dtypes for TensorRT compatibility:
- Ensure ElementWise-like ops (Add/Sub/Mul/Div/Min/Max/Pow/Where) have matching input dtypes.
- Prefer promoting integer constants to the floating dtype of the paired input.
- Ensure Expand's shape input is INT32 (insert Cast if needed).

Usage:
  python 11_trt_sanitize_types.py --in onnx_out/motionnet_v1_fp32_opt.onnx \
    --out onnx_out/motionnet_v1_fp32_trtfix.onnx
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference

ELEMENTWISE_OPS = {
    "Add", "Sub", "Mul", "Div", "Min", "Max", "Pow", "Where",
}


def get_value_info_dtypes(model: onnx.ModelProto) -> Dict[str, int]:
    dtypes: Dict[str, int] = {}
    def add_vi(vi: onnx.ValueInfoProto):
        if vi.type and vi.type.tensor_type:
            et = vi.type.tensor_type.elem_type
            if et:
                dtypes[vi.name] = et
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        add_vi(vi)
    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type
    # Also record Constant node output dtypes where available (cheap type enrichment)
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        out = node.output[0]
        # Prefer the standard 'value' tensor attribute
        attr_map = {a.name: a for a in node.attribute}
        if "value" in attr_map and attr_map["value"].t is not None:
            dtypes[out] = attr_map["value"].t.data_type
            continue
        # Fallback to scalar/list forms
        if "value_float" in attr_map or "value_floats" in attr_map:
            dtypes[out] = TensorProto.FLOAT
        elif "value_int" in attr_map or "value_ints" in attr_map:
            # ONNX uses INT64 for generic integer tensors
            dtypes[out] = TensorProto.INT64
        elif "value_string" in attr_map or "value_strings" in attr_map:
            # Strings aren't used in math ops; skip
            pass
    return dtypes


def upcast_initializer_to(model: onnx.ModelProto, name: str, target_dtype: int) -> None:
    inits = model.graph.initializer
    for i, init in enumerate(inits):
        if init.name == name:
            arr = numpy_helper.to_array(init)
            # Map ONNX dtype to numpy dtype via a small table
            np_dtype = {
                TensorProto.FLOAT16: "float16",
                TensorProto.FLOAT: "float32",
                TensorProto.DOUBLE: "float64",
                TensorProto.INT8: "int8",
                TensorProto.INT16: "int16",
                TensorProto.INT32: "int32",
                TensorProto.INT64: "int64",
                TensorProto.UINT8: "uint8",
                TensorProto.UINT16: "uint16",
                TensorProto.UINT32: "uint32",
                TensorProto.UINT64: "uint64",
                TensorProto.BOOL: "bool_",
            }.get(target_dtype, "float32")
            new_arr = arr.astype(np_dtype)  # type: ignore[name-defined]
            inits[i].CopyFrom(numpy_helper.from_array(new_arr, name=name))
            return


def insert_cast(model: onnx.ModelProto, tensor_name: str, to_dtype: int) -> str:
    # Create a Cast node right before consumers; new tensor name suffixed with _cast
    cast_out = tensor_name + f"_cast_{to_dtype}"
    cast_node = helper.make_node(
        "Cast",
        inputs=[tensor_name],
        outputs=[cast_out],
        name=f"Cast_{tensor_name}_{to_dtype}",
        to=to_dtype,
    )
    model.graph.node.insert(0, cast_node)
    return cast_out


def _find_initializer(model: onnx.ModelProto, name: str) -> Optional[Tuple[int, onnx.TensorProto]]:
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            return i, init
    return None


def sanitize_elementwise_types(
    model: onnx.ModelProto,
    prefer_cast: bool = False,
    upcast_max_elems: int = 100_000,
) -> bool:
    dtypes = get_value_info_dtypes(model)
    changed = False
    for node in model.graph.node:
        if node.op_type not in ELEMENTWISE_OPS:
            continue
        if len(node.input) < 2:
            continue
        a, b = node.input[0], node.input[1]
        da, db = dtypes.get(a), dtypes.get(b)
        # If types match or unavailable, skip
        if da == db or (da is None or db is None):
            continue
        # If one is float and the other is integer, prefer float
        def is_float(dt: int) -> bool:
            return dt in (TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE)
        def is_int(dt: int) -> bool:
            return dt in (TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
                          TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64)
        if is_float(da) and is_int(db):
            # Try to upcast initializer, else insert cast
            if any(init.name == b for init in model.graph.initializer) and not prefer_cast:
                # Only upcast if small enough to avoid OOM
                found = _find_initializer(model, b)
                if found is not None:
                    _, init = found
                    numel = 1
                    for d in init.dims:
                        numel *= int(d) if d is not None else 0
                    if numel <= upcast_max_elems:
                        upcast_initializer_to(model, b, da)
                        dtypes[b] = da
                    else:
                        node.input[1] = insert_cast(model, b, da)
                        dtypes[node.input[1]] = da
                else:
                    node.input[1] = insert_cast(model, b, da)
                    dtypes[node.input[1]] = da
            else:
                node.input[1] = insert_cast(model, b, da)
                dtypes[node.input[1]] = da
            changed = True
        elif is_float(db) and is_int(da):
            if any(init.name == a for init in model.graph.initializer) and not prefer_cast:
                found = _find_initializer(model, a)
                if found is not None:
                    _, init = found
                    numel = 1
                    for d in init.dims:
                        numel *= int(d) if d is not None else 0
                    if numel <= upcast_max_elems:
                        upcast_initializer_to(model, a, db)
                        dtypes[a] = db
                    else:
                        node.input[0] = insert_cast(model, a, db)
                        dtypes[node.input[0]] = db
                else:
                    node.input[0] = insert_cast(model, a, db)
                    dtypes[node.input[0]] = db
            else:
                node.input[0] = insert_cast(model, a, db)
                dtypes[node.input[0]] = db
            changed = True
    return changed


def sanitize_expand_shape(
    model: onnx.ModelProto,
    prefer_cast: bool = False,
    upcast_max_elems: int = 100_000,
) -> bool:
    changed = False
    for node in model.graph.node:
        if node.op_type != "Expand" or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        # Force shape to INT32 for TRT
        target = TensorProto.INT32
        # If initializer, convert; else insert cast
        if any(init.name == shape_name for init in model.graph.initializer) and not prefer_cast:
            found = _find_initializer(model, shape_name)
            if found is not None:
                _, init = found
                # Expand shapes are usually small; still guard by size
                numel = 1
                for d in init.dims:
                    numel *= int(d) if d is not None else 0
                if numel <= upcast_max_elems:
                    upcast_initializer_to(model, shape_name, target)
                    changed = True
                else:
                    node.input[1] = insert_cast(model, shape_name, target)
                    changed = True
            else:
                node.input[1] = insert_cast(model, shape_name, target)
                changed = True
        else:
            node.input[1] = insert_cast(model, shape_name, target)
            changed = True
    return changed


def normalize_constant_nodes(model: onnx.ModelProto) -> bool:
    """Ensure Constant nodes use the 'value' TensorProto attribute.
    Converts value_float/value_int/value_floats/value_ints into a 'value' tensor.
    """
    changed = False
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        has_value = any(a.name == "value" for a in node.attribute)
        if has_value:
            continue
        # Collect alternative attributes
        alt = {a.name: a for a in node.attribute}
        tensor = None
        if "value_float" in alt:
            v = alt["value_float"].f
            tensor = numpy_helper.from_array(numpy_helper.to_array(helper.make_tensor("t", TensorProto.FLOAT, [1], [v])))
        elif "value_int" in alt:
            v = alt["value_int"].i
            tensor = helper.make_tensor("t", TensorProto.INT64, [1], [int(v)])
        elif "value_floats" in alt:
            vals = list(alt["value_floats"].floats)
            tensor = helper.make_tensor("t", TensorProto.FLOAT, [len(vals)], vals)
        elif "value_ints" in alt:
            vals = [int(i) for i in alt["value_ints"].ints]
            tensor = helper.make_tensor("t", TensorProto.INT64, [len(vals)], vals)
        elif "value_string" in alt or "value_strings" in alt:
            # Not typical for TRT paths; skip conversion
            continue
        if tensor is not None:
            # Remove old attrs and set 'value'
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute("value", tensor)])
            changed = True
    return changed


def parse_args():
    p = argparse.ArgumentParser(description="Sanitize ONNX dtypes for TensorRT")
    p.add_argument("--in", dest="in_onnx", required=True)
    p.add_argument("--out", dest="out_onnx", required=True)
    p.add_argument("--infer", action="store_true", help="Run ONNX shape inference (can be memory-heavy)")
    p.add_argument("--prefer-cast", action="store_true", help="Prefer inserting Cast over upcasting initializers")
    p.add_argument(
        "--upcast-max-elems",
        type=int,
        default=100_000,
        help="Only upcast initializers with element count <= this; else insert Cast",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model = onnx.load(args.in_onnx)
    # Optionally enrich value_info types (can be memory heavy)
    if args.infer:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            # Proceed without inferred shapes
            pass
    changed = False
    changed |= normalize_constant_nodes(model)
    changed |= sanitize_elementwise_types(
        model,
        prefer_cast=args.prefer_cast,
        upcast_max_elems=args.upcast_max_elems,
    )
    changed |= sanitize_expand_shape(
        model,
        prefer_cast=args.prefer_cast,
        upcast_max_elems=args.upcast_max_elems,
    )
    onnx.save(model, args.out_onnx)
    print(f"Saved sanitized model to {args.out_onnx} (changed={changed})")


if __name__ == "__main__":
    main()
