#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporter for 09_mlir_to_onnx.py (no external onnx-mlir required)
- Builds small MLIR(ONNX dialect) snippets
- Converts to ONNX using MLIRToONNXConverter
- Writes both MLIR and ONNX files into onnx_out/

Run:
  python 10_test_mlir_to_onnx.py
Output:
  onnx_out/<case>.mlir, onnx_out/<case>.onnx
"""

import sys
from typing import List, Tuple
import onnx
from onnx import checker
import importlib.util, pathlib
from pathlib import Path

# Import the converter by file path (module name starts with digits)
p = pathlib.Path(__file__).with_name('09_mlir_to_onnx.py')
spec = importlib.util.spec_from_file_location('mlir2onnx', str(p))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
import sys as _sys
_sys.modules[spec.name] = mod  # ensure module registered for dataclasses/type resolution
spec.loader.exec_module(mod)  # type: ignore
MLIRToONNXConverter = getattr(mod, 'MLIRToONNXConverter')


def run_case(name: str, mlir: str, out_dir: Path) -> Tuple[str, bool, str, Path, Path]:
  mlir_path = out_dir / f"{name}.mlir"
  onnx_path = out_dir / f"{name}.onnx"
  try:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save MLIR
    mlir_path.write_text(mlir)
    # Convert and save ONNX
    conv = MLIRToONNXConverter(mlir_text=mlir, opset=13)
    model = conv.convert()
    onnx.save(model, str(onnx_path))
    # Best-effort validate
    try:
      checker.check_model(onnx.load(str(onnx_path)))
      msg = f"Exported and validated: {onnx_path.name}"
    except Exception as ve:
      msg = f"Exported (validation warn: {ve})"
    return (name, True, msg, mlir_path, onnx_path)
  except Exception as e:
    return (name, False, str(e), mlir_path, onnx_path)


def main():
    cases: List[Tuple[str, str]] = []

    cases.append((
        'matmul_basic',
        r"""
module {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %w = onnx.Constant {value = dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]>} : tensor<2x3xf32>
    %y = onnx.MatMul %arg0, %w : tensor<1x2xf32>, tensor<2x3xf32> -> tensor<1x3xf32>
    return %y : tensor<1x3xf32>
  }
}
"""
    ))

    cases.append((
        'gemm_bias',
        r"""
module {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %w = onnx.Constant {value = dense<[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]]>} : tensor<2x3xf32>
    %c = onnx.Constant {value = dense<[0.5, 0.0, -0.5]>} : tensor<3xf32>
    %y = onnx.Gemm %arg0, %w, %c {alpha=1.0, beta=1.0, transA=0, transB=0} : tensor<1x2xf32>, tensor<2x3xf32>, tensor<3xf32> -> tensor<1x3xf32>
    return %y : tensor<1x3xf32>
  }
}
"""
    ))

    cases.append((
        'add_relu_chain',
        r"""
module {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %w = onnx.Constant {value = dense<[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]]>} : tensor<2x3xf32>
    %b = onnx.Constant {value = dense<[1.0, 1.0, 1.0]>} : tensor<3xf32>
    %y = onnx.MatMul %arg0, %w : tensor<1x2xf32>, tensor<2x3xf32> -> tensor<1x3xf32>
    %z = onnx.Add %y, %b : tensor<1x3xf32>, tensor<3xf32> -> tensor<1x3xf32>
    %o = onnx.Relu %z : tensor<1x3xf32> -> tensor<1x3xf32>
    return %o : tensor<1x3xf32>
  }
}
"""
    ))

    cases.append((
        'multi_outputs',
        r"""
module {
  func.func @main(%arg0: tensor<1x2xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>) {
    %w = onnx.Constant {value = dense<[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]]>} : tensor<2x3xf32>
    %y = onnx.MatMul %arg0, %w : tensor<1x2xf32>, tensor<2x3xf32> -> tensor<1x3xf32>
    %o = onnx.Relu %y : tensor<1x3xf32> -> tensor<1x3xf32>
    return %y : tensor<1x3xf32>, %o : tensor<1x3xf32>
  }
}
"""
    ))

    out_dir = Path("onnx_out")
    results = [run_case(n, m, out_dir) for n, m in cases]
    for n, ok, msg, mlir_p, onnx_p in results:
        status = 'DONE' if ok else 'ERROR'
        print(f"[{status}] {n}: {msg}")
        print(f"  MLIR: {mlir_p}")
        print(f"  ONNX: {onnx_p}")
    # Do not exit non-zero; purpose is to export artifacts even without full tooling.


if __name__ == '__main__':
    main()
