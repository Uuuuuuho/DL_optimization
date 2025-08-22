#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh


def make_decoder_block(b: int = 1, s: int = 128, h: int = 512, out_path: str = "onnx_out/decoder_block.onnx") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    X = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [b, s, h])

    def const_w(name: str):
        w = np.random.randn(h, h).astype(np.float32) / np.sqrt(h)
        return onh.from_array(w, name)

    def const_b(name: str):
        b = np.random.randn(h).astype(np.float32) * 0.01
        return onh.from_array(b, name)

    Wq = const_w("Wq"); bq = const_b("bq")
    Wk = const_w("Wk"); bk = const_b("bk")
    Wv = const_w("Wv"); bv = const_b("bv")

    # MatMul input -> output shapes: [B,S,H] x [H,H] -> [B,S,H]
    Yq = oh.make_node("MatMul", inputs=["input", "Wq"], outputs=["Yq"], name="MatMul_Q")
    Aq = oh.make_node("Add", inputs=["Yq", "bq"], outputs=["Aq"], name="Add_Q")
    Yk = oh.make_node("MatMul", inputs=["input", "Wk"], outputs=["Yk"], name="MatMul_K")
    Ak = oh.make_node("Add", inputs=["Yk", "bk"], outputs=["Ak"], name="Add_K")
    Yv = oh.make_node("MatMul", inputs=["input", "Wv"], outputs=["Yv"], name="MatMul_V")
    Av = oh.make_node("Add", inputs=["Yv", "bv"], outputs=["Av"], name="Add_V")

    # Merge to a single output to ease TRT profiling
    Con = oh.make_node("Concat", inputs=["Aq", "Ak", "Av"], outputs=["Y"], name="Concat_Out", axis=-1)
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [b, s, h * 3])

    graph = oh.make_graph(
        nodes=[Yq, Aq, Yk, Ak, Yv, Av, Con],
        name="decoder_block",
        inputs=[X],
        outputs=[Y],
        initializer=[Wq, bq, Wk, bk, Wv, bv],
    )
    model = oh.make_model(graph, producer_name="gen_decoder_block", opset_imports=[oh.make_operatorsetid("", 13)])
    try:
        model.ir_version = min(getattr(model, "ir_version", 10) or 10, 10)
    except Exception:
        pass
    onnx.save(model, out_path)
    return out_path


if __name__ == "__main__":
    path = make_decoder_block()
    print(f"Wrote: {path}")
