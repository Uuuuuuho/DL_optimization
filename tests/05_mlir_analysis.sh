# !/bin/bash
# 0) ONNX → MLIR (onnx dialect)
IN_ONNX_MODEL="onnx_out/parallel_matmul.onnx"
OUT_ONNX_MODEL="onnx_out/parallel_matmul_opt"
INTER_MLIR="onnx_out/parallel_matmul_inter"
onnx-mlir --EmitMLIR --O0 ${IN_ONNX_MODEL} -o ${OUT_ONNX_MODEL}

# # 1) 정적화 & 캐노니컬라이즈
# mlir-opt onnx_out/model.onnx.mlir \
#   --const-prop --cse --canonicalize \
#   -o onnx_out/model.clean.mlir

# # 2) 레이아웃/패딩/패턴 재작성
# mlir-opt model.clean.mlir \
#   --convert-nchw-to-nhwc \
#   --fuse-matmul-add-to-gemm \
#   --linear-bias-activation-fuse \
#   --pad-dims-to-multiples='multiples=16' \
#   -o model.trt_friendly.mlir

# # 3) 정밀도/양자화(QDQ 삽입 + 속성)
# mlir-opt model.trt_friendly.mlir \
#   --insert-qdq='per_channel=true method=smoothquant' \
#   --annotate-precision='fp16=true int8=candidate' \
#   -o model.trt_quant.mlir

# # 4) Attention/LN 특수화(있다면)
# mlir-opt model.trt_quant.mlir \
#   --rewrite-attention-to-trt-mha \
#   --lower-layernorm-to-scale-shift \
#   -o model.trt_special.mlir

# # 5) TRT 백엔드 친화적 로워링 or ONNX 재생성
# mlir-translate model.trt_special.mlir --mlir-to-onnx -o ${OUT_ONNX_MODEL}
# # → TensorRT 빌드/ONNX Runtime TRT-EP로 투입
