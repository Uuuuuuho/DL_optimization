#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export example ONNX models for Horizontal Fusion experiments.

Models:
  1) parallel_matmul.onnx
     - Two bias-free MatMul ops that share the same input X.
  2) parallel_gemm.onnx
     - Two Linear(+bias) ops -> ONNX Gemm nodes that share the same input X.
  3) mixed_dependent.onnx
     - MatMul(X,W1), MatMul(X,W2), MatMul(X,W3) with an extra dependency path:
       Z = Relu(MatMul(X,W1)), and (optionally) Y_dep = MatMul(Z, Wd)
       (병렬 후보 + 의존 경로를 함께 갖는 그래프)

Run:
  python export_examples.py --outdir ./onnx_out --batch 4 --in-feat 128 --m1 64 --m2 96 --m3 32 --opset 13
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx.checker as oc


# -----------------------------
# 1) Parallel MatMul (no bias)
# -----------------------------
class ParallelMatMul(nn.Module):
    """
    Forward:
      Y1 = X @ W1   (W1: [K, M1])
      Y2 = X @ W2   (W2: [K, M2])
    Return: (Y1, Y2)
    """
    def __init__(self, in_features: int, m1: int, m2: int):
        super().__init__()
        # bias 없는 선형 연산을 'MatMul'로 내보내기 위해 torch.matmul 사용
        self.W1 = nn.Parameter(torch.randn(in_features, m1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(in_features, m2) * 0.02)

    def forward(self, x):
        y1 = torch.matmul(x, self.W1)  # -> ONNX MatMul
        y2 = torch.matmul(x, self.W2)  # -> ONNX MatMul
        return y1, y2


# -----------------------------
# 2) Parallel Gemm (Linear + bias)
# -----------------------------
class ParallelGemm(nn.Module):
    """
    Forward:
      Y1 = Linear(X) with bias -> typically ONNX Gemm
      Y2 = Linear(X) with bias -> typically ONNX Gemm
    """
    def __init__(self, in_features: int, m1: int, m2: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, m1, bias=True)
        self.fc2 = nn.Linear(in_features, m2, bias=True)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return y1, y2


# -----------------------------------------
# 3) Mixed graph with dependency path
# -----------------------------------------
class MixedDependent(nn.Module):
    """
    Forward:
      Y1 = X @ W1
      Z  = Relu(Y1)                    # dependency path from Y1
      Y2 = X @ W2                      # independent sibling w.r.t. Y3
      Y3 = X @ W3
      (optional) Y_dep = Z @ Wd        # adds more dependencies (kept internal)
    Returns: (Y1, Y2, Y3)  # 외부 출력은 병렬 타깃들 유지
    """
    def __init__(self, in_features: int, m1: int, m2: int, m3: int, add_dep_matmul: bool = True):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(in_features, m1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(in_features, m2) * 0.02)
        self.W3 = nn.Parameter(torch.randn(in_features, m3) * 0.02)
        self.add_dep_matmul = add_dep_matmul
        if add_dep_matmul:
            # Z:[N,m1] @ Wd:[m1, m3] -> [N, m3] (단지 의존 경로용. 출력으로 내보내진 않음)
            self.Wd = nn.Parameter(torch.randn(m1, m3) * 0.02)

    def forward(self, x):
        y1 = torch.matmul(x, self.W1)      # MatMul
        z  = torch.relu(y1)                # dependency path from y1
        if self.add_dep_matmul:
            _y_dep = torch.matmul(z, self.Wd)  # not returned; ensures path exists
        y2 = torch.matmul(x, self.W2)      # MatMul (independent of y1 path)
        y3 = torch.matmul(x, self.W3)      # MatMul (independent of y1 path)
        return y1, y2, y3


def export_onnx(model: nn.Module,
                example_input: torch.Tensor,
                out_path: str,
                opset: int = 13,
                dynamic: bool = True,
                names = None):
    model.eval()
    out_path = str(out_path)
    dynamic_axes = None
    input_names  = ["X"]
    if names is not None:
        output_names = names
    else:
        # fallback: y0, y1, ...
        with torch.no_grad():
            tmp = model(example_input)
        if isinstance(tmp, (tuple, list)):
            output_names = [f"Y{i}" for i in range(len(tmp))]
        else:
            output_names = ["Y"]

    if dynamic:
        # 첫 번째 차원(batch)을 동적으로
        dynamic_axes = {"X": {0: "N"}}
        for n in output_names:
            dynamic_axes[n] = {0: "N"}

    torch.onnx.export(
        model, example_input, out_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 간단 체크
    m = onnx.load(out_path)
    oc.check_model(m)
    print(f"[ok] exported: {out_path}")
    print(f"    inputs : {[i.name + str(i.type.tensor_type.shape.dim[0].dim_param or i.type.tensor_type.shape.dim[0].dim_value) for i in m.graph.input]}")
    print(f"    outputs: {[o.name for o in m.graph.output]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./onnx_out")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--in-feat", type=int, default=128)
    ap.add_argument("--m1", type=int, default=64)
    ap.add_argument("--m2", type=int, default=96)
    ap.add_argument("--m3", type=int, default=32)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dynamic", action="store_true", help="disable dynamic axes")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N, K = args.batch, args.in_feat
    x = torch.randn(N, K, dtype=torch.float32)

    # 1) parallel_matmul.onnx
    m1 = ParallelMatMul(in_features=K, m1=args.m1, m2=args.m2)
    export_onnx(
        m1, x, outdir / "parallel_matmul.onnx",
        opset=args.opset,
        dynamic=not args.no_dynamic,
        names=["Y1", "Y2"]
    )

    # 2) parallel_gemm.onnx
    m2 = ParallelGemm(in_features=K, m1=args.m1, m2=args.m2)
    export_onnx(
        m2, x, outdir / "parallel_gemm.onnx",
        opset=args.opset,
        dynamic=not args.no_dynamic,
        names=["Y1", "Y2"]
    )

    # 3) mixed_dependent.onnx
    m3 = MixedDependent(in_features=K, m1=args.m1, m2=args.m2, m3=args.m3, add_dep_matmul=True)
    export_onnx(
        m3, x, outdir / "mixed_dependent.onnx",
        opset=args.opset,
        dynamic=not args.no_dynamic,
        names=["O1", "O2", "O3"]
    )

    print("\n[hint] 이제 수평 병합 스크립트를 적용해보세요:")
    print("  python hfuse.py --in ./onnx_out/parallel_matmul.onnx --out ./onnx_out/parallel_matmul_fused.onnx")
    print("  python hfuse.py --in ./onnx_out/parallel_gemm.onnx   --out ./onnx_out/parallel_gemm_fused.onnx")
    print("  python hfuse.py --in ./onnx_out/mixed_dependent.onnx --out ./onnx_out/mixed_dependent_fused.onnx")


if __name__ == "__main__":
    main()

