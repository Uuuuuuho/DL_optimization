# TensorRT Frontend Optimization Auto-Tuner

A small, extensible toolkit that searches TensorRT-friendly frontend graph variants of an input ONNX and selects the best-performing candidate by benchmarking with TensorRT 8.6.

## Features (Phase 1)
- Candidate generator using ONNX GraphSurgeon:
  - K-dimension alignment for MatMul via zero-padding of A and B (K→multiple of {8,16,32}).
  - MatMul → Gemm rewrite when weights are constant.
  - Optional onnx-simplifier for graph cleanup.
  - Built-in Horizontal Fusion: parallel MatMul/Gemm with the same left input and constant weights are fused into a single op (weights concatenated along N, result split back). Safe/lossless within supported patterns.
- Validator: onnx.checker + ONNX Runtime-based numeric check against baseline.
- TensorRT engine builder via `trtexec` with timing cache reuse and shapes handling.
- Profiler: collects latency metrics from `--exportTimes` and summary logs.
- Selector: picks best candidate by mean latency (ties broken by p50/p95/build time).

## Quick start
1) Install dependencies in your venv (already used elsewhere in this repo):

```
pip install onnx onnxruntime-gpu onnx-graphsurgeon onnxsim numpy pandas pyyaml tqdm
```

2) Run the tuner:

```
python -m tuner.main \
  --onnx onnx_out/parallel_matmul.onnx \
  --outdir onnx_out/tune_run \
  --trtexec /mnt/e/Downloads/TensorRT-8.6.1.6/bin/trtexec \
  --precision FP16 \
  --shapes X:4x128
```

If your model has dynamic or unknown dims, provide `--shapes` for all inputs. Static fully-known shapes are auto-detected.

Environment requirements for TensorRT:
- Ensure CUDA libs and TensorRT libs are visible to the process, e.g.:
```
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/mnt/e/Downloads/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
```

3) Outputs in `--outdir`:
- `candidates/`: transformed ONNX files.
- `plans/`: optional TensorRT engine plans if `--save-plan` given.
- `logs/`: trtexec logs and JSONs.
- `metrics.csv`: per-candidate metrics.
- `best.onnx` and optionally `best.plan`.

### Horizontal Fusion options
- Enable internal horizontal fusion candidates:

```
--enable-hfusion
```

- Explore multiple minimum group sizes in one run using ranges/lists. Supported formats:
  - Single: `--hf-min-group 2`
  - Range inclusive: `--hf-min-group 2-5`  (2,3,4,5)
  - Range with step: `--hf-min-group 2-8:2`  (2,4,6,8)
  - Comma list: `--hf-min-group 2,4,8`

Example:

```
python -m tuner.main \
  --onnx onnx_out/parallel_matmul.onnx \
  --outdir onnx_out/tune_run \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --precision FP16 \
  --enable-hfusion \
  --hf-min-group 2-6:2
```

This generates separate h-fusion candidates per group size (e.g., g2, g4, g6) and benchmarks them.

## Notes
- This is an initial, safe set of transforms. Layout conversions and complex activation/norm rewrites are scaffolded for future phases.
- You can add your own rules under `tuner/candidate_generator.py` and register them in `build_candidate_space`.