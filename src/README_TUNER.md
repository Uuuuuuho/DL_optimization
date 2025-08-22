TensorRT Frontend Optimization Auto-Tuner
=========================================

Quick start
-----------

Prereqs:
- Python packages: onnx, onnxruntime, onnx-graphsurgeon
- TensorRT 8.6 installed with trtexec available (set TRTEXEC_BIN or ensure in PATH)
- CUDA libs available via LD_LIBRARY_PATH

Run:
- python -m tuner.main --base onnx_out/parallel_matmul.onnx \
  --out tuner_out \
  --shape X:4x128 \
  --fp16 \
  --timing-cache tuner_out/timing.cache \
  --lib-path /path/to/TensorRT/lib \
  --lib-path /usr/local/cuda/lib64

Outputs:
- tuner_out/candidates/candidate_*.onnx
- tuner_out/engines/*.plan and *_trtexec.log / *_trt_profile.json
- tuner_out/metrics/metrics.json + metrics.csv
- tuner_out/best.onnx and best.plan (symlink or copy)
