from __future__ import annotations

from typing import Dict, List, Tuple
import tempfile
import os

import numpy as np
import onnx
import onnxruntime as rt


def _load_io(model_path: str) -> Tuple[List[Tuple[str, List[int], int]], List[Tuple[str, List[int], int]]]:
    m = onnx.load(model_path)
    ins = [(i.name, [d.dim_value if d.HasField('dim_value') else -1 for d in i.type.tensor_type.shape.dim], i.type.tensor_type.elem_type) for i in m.graph.input]
    outs = [(o.name, [d.dim_value if d.HasField('dim_value') else -1 for d in o.type.tensor_type.shape.dim], o.type.tensor_type.elem_type) for o in m.graph.output]
    return ins, outs


def _elem_to_np_dtype(elem_type: int):
    et = int(elem_type)
    if et == onnx.TensorProto.FLOAT16:
        return np.float16
    if et == onnx.TensorProto.FLOAT:
        return np.float32
    if et == onnx.TensorProto.DOUBLE:
        return np.float64
    if et == onnx.TensorProto.INT32:
        return np.int32
    if et == onnx.TensorProto.INT64:
        return np.int64
    if et == onnx.TensorProto.BOOL:
        return np.bool_
    return np.float32


def _make_dummy(shape: List[int], dtype) -> np.ndarray:
    shp = [d if d and d > 0 else 1 for d in shape] or [1]
    if dtype in (np.float16, np.float32, np.float64):
        return np.random.randn(*shp).astype(dtype)
    if dtype in (np.int32, np.int64):
        return np.random.randint(0, 10, size=shp, dtype=dtype)
    if dtype is np.bool_:
        return np.random.randint(0, 2, size=shp).astype(np.bool_)
    return np.random.randn(*shp).astype(np.float32)


def ort_run(model_path: str, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = rt.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    ys = sess.run(outs, feed)
    return [np.asarray(y) for y in ys]


def validate_against_baseline(
    baseline: str,
    candidate: str,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Tuple[bool, str]:
    # structural check (force IR<=10 for this environment)
    def _load_downgrade(path: str) -> onnx.ModelProto:
        m = onnx.load(path)
        try:
            m.ir_version = min(getattr(m, "ir_version", 10) or 10, 10)
        except Exception:
            pass
        return m
    try:
        mb = _load_downgrade(baseline)
        mc = _load_downgrade(candidate)
        onnx.checker.check_model(mb)
        onnx.checker.check_model(mc)
    except Exception as e:
        return False, f"checker_invalid: {e}"

    # Persist downgraded models to temp files for ORT which only accepts file paths
    tmp_b = tempfile.NamedTemporaryFile(delete=False, suffix="_ir10.onnx")
    tmp_c = tempfile.NamedTemporaryFile(delete=False, suffix="_ir10.onnx")
    tmp_b.close(); tmp_c.close()
    try:
        onnx.save(mb, tmp_b.name)
        onnx.save(mc, tmp_c.name)
    except Exception as e:
        return False, f"save_ir10_failed: {e}"

    # build unified dummy input set from baseline inputs
    ins_b, _ = _load_io(tmp_b.name)
    ins_c, _ = _load_io(tmp_c.name)
    # map by position; if names differ, use candidate names on its feed
    feed_b: Dict[str, np.ndarray] = {}
    feed_c: Dict[str, np.ndarray] = {}
    for idx, (name_b, shape_b, elem_b) in enumerate(ins_b):
        np_dtype = _elem_to_np_dtype(int(elem_b))
        x = _make_dummy(shape_b, np_dtype)
        feed_b[name_b] = x
        # candidate input name
        name_c = ins_c[idx][0] if idx < len(ins_c) else name_b
        feed_c[name_c] = x

    try:
        yb = ort_run(tmp_b.name, feed_b)
        yc = ort_run(tmp_c.name, feed_c)
    except Exception as e:
        # Cleanup temp files before returning
        try:
            os.unlink(tmp_b.name)
            os.unlink(tmp_c.name)
        except Exception:
            pass
        return False, f"ort_error: {e}"

    if len(yb) != len(yc):
        return False, f"Output count mismatch: baseline={len(yb)}, cand={len(yc)}"

    for i, (a, b) in enumerate(zip(yb, yc)):
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            diff = float(np.max(np.abs(a - b)))
            try:
                os.unlink(tmp_b.name)
                os.unlink(tmp_c.name)
            except Exception:
                pass
            return False, f"Output[{i}] mismatch max_abs={diff}"
    try:
        os.unlink(tmp_b.name)
        os.unlink(tmp_c.name)
    except Exception:
        pass
    return True, "OK"


__all__ = ["validate_against_baseline", "ort_run"]
