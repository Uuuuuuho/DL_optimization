# !/usr/bin/bash

clear

python 11_onnx_to_mlir.py --in onnx_out/SymmetricFT.onnx --out onnx_out/SymmetricFT.mlir --opt-level O0
python 09_mlir_to_onnx.py --in onnx_out/SymmetricFT.mlir --out onnx_out/SymmetricFT_opt.onnx