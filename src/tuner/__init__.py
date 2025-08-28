"""TensorRT Frontend Optimization Auto-Tuner package.

This package keeps the top-level import lightweight to avoid importing heavy
dependencies (e.g., onnxruntime) unless explicitly needed by a module.
"""

__all__ = [
    "candidate_generator",
    "validator",
    "trt_builder",
    "profiler",
    "selector",
    "utils",
]
