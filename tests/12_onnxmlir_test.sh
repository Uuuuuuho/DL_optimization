#!/usr/bin/env bash

set -euo pipefail

# Always run relative to this script's directory
cd "$(dirname "$0")"

clear

# Round-trip conversion for parallel_matmul.onnx with ONNX graph optimizations
# 1) ONNX -> MLIR (with O2 optimizations via selected engine)
# 2) MLIR  -> ONNX (optimized)

IN_ONNX="onnx_out/parallel_matmul.onnx"
OUT_MLIR="onnx_out/parallel_matmul.mlir"
OUT_ONNX_OPT="onnx_out/parallel_matmul_opt.onnx"

echo "[1/2] Exporting MLIR from ${IN_ONNX} (O2) ..."
python 11_onnx_to_mlir.py \
	--in "${IN_ONNX}" \
	--out "${OUT_MLIR}" \
	--opt-level O2 \
	--opt-engine auto

echo "[2/2] Converting MLIR back to ONNX -> ${OUT_ONNX_OPT} ..."
python 09_mlir_to_onnx.py \
	--in "${OUT_MLIR}" \
	--out "${OUT_ONNX_OPT}"

echo "Done. Generated ${OUT_MLIR} and ${OUT_ONNX_OPT}"