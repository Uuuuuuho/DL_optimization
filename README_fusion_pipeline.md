# Horizontal Fusion Pipeline (Hints + External Edges)

This repository includes a horizontal fusion tool (`01_horizontal_fusion.py`).
To leverage analysis outputs and guide fusion safely, use the following workflow.

## 1) Prepare model
Place your ONNX model, e.g., `onnx_out/parallel_matmul.onnx`.

## 2) (Optional) Run analyses
- TVM analysis may generate `optimization_analysis.json` (04)
- ONNX-MLIR analysis may generate `onnx_mlir_optimization.json` (05)
- Comprehensive analysis may generate `optimization_results/comprehensive_optimization.json` (06)

These files are optional. If present, the hint generator can use them for prioritization.

## 3) Generate hints and dependency edges
```
python 07_generate_fusion_hints.py \
  --model onnx_out/parallel_matmul.onnx \
  --hints out/hfusion_hints.json \
  --edges out/edges.json \
  --min-group-size 2
```

Outputs:
- `out/hfusion_hints.json`: groups of MatMul/Gemm nodes to try fusing (by shared A)
- `out/edges.json`: external edges usable via `--dep-source edges-json`

## 4) Apply horizontal fusion
Option A: run 01 directly
```
python 01_horizontal_fusion.py \
  --in onnx_out/parallel_matmul.onnx \
  --out onnx_out/parallel_matmul_fused.onnx \
  --dep-source edges-json --dep-file out/edges.json \
  --hints-file out/hfusion_hints.json
```

Option B: via orchestrator (generates artifacts and runs 01)
```
python 08_apply_horizontal_fusion.py \
  --in onnx_out/parallel_matmul.onnx \
  --out onnx_out/parallel_matmul_fused.onnx \
  --work-dir out --min-group-size 2 --run
```

## Notes
- Node names must be populated for hinting and edges. 01 will still validate independence & shapes.
- You can disable MatMul or Gemm hinting using `--no-matmul`/`--no-gemm`.
- If you only want to preview the command, use `--show-cmd` with the orchestrator.

## Troubleshooting
- Install ONNX: `pip install onnx`
- If 01 fails due to unmet dependencies in the model graph, re-generate edges/hints and ensure nodes have names.
