import subprocess
import re

def analyze_onnx_mlir_optimizations(model_path):
    """ONNX-MLIR 최적화 패스 분석"""
    
    optimization_report = {
        "applicable_passes": [],
        "pattern_matches": [],
        "optimization_stats": {}
    }
    
    # ONNX-MLIR 최적화 패스 실행 및 통계 수집
    passes_to_analyze = [
        ("--onnx-const-prop", "Constant Propagation"),
        ("--onnx-elide-constants", "Constant Elimination"),
        ("--onnx-shape-inference", "Shape Inference"),
        ("--onnx-decompose", "Operation Decomposition"),
        ("--onnx-rewrite", "Pattern Rewriting")
    ]
    
    for pass_flag, pass_name in passes_to_analyze:
        try:
            # 각 패스 실행 전후 비교
            cmd = f"onnx-mlir {pass_flag} --mlir-print-stats {model_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # 통계 파싱
            stats = parse_mlir_stats(result.stderr)
            if stats["modifications"] > 0:
                optimization_report["applicable_passes"].append({
                    "pass_name": pass_name,
                    "flag": pass_flag,
                    "modifications": stats["modifications"],
                    "benefit": stats["benefit"]
                })
        except Exception as e:
            print(f"Error analyzing {pass_name}: {e}")
    
    # 패턴 매칭 기회 분석
    pattern_analysis = analyze_pattern_matching(model_path)
    optimization_report["pattern_matches"] = pattern_analysis
    
    # 최적화 통계 요약
    optimization_report["optimization_stats"] = {
        "total_applicable_passes": len(optimization_report["applicable_passes"]),
        "estimated_op_reduction": calculate_op_reduction(optimization_report),
        "memory_optimization_potential": analyze_memory_patterns(model_path)
    }
    
    with open("onnx_mlir_optimization.json", "w") as f:
        json.dump(optimization_report, f, indent=2)
    
    return optimization_report

def parse_mlir_stats(output):
    """MLIR 통계 출력 파싱"""
    stats = {"modifications": 0, "benefit": ""}
    
    # 패턴별 통계 추출
    pattern = r"(\d+)\s+operations?\s+(?:replaced|eliminated|modified)"
    matches = re.findall(pattern, output)
    if matches:
        stats["modifications"] = sum(int(m) for m in matches)
        stats["benefit"] = f"Reduced {stats['modifications']} operations"
    
    return stats

def analyze_pattern_matching(model_path):
    """패턴 매칭 기회 분석"""
    patterns = []
    
    # MLIR 변환
    cmd = f"onnx-mlir --EmitMLIR {model_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    mlir_content = result.stdout
    
    # 일반적인 최적화 패턴 탐색
    optimization_patterns = {
        r"onnx\.Add.*onnx\.Relu": {
            "pattern": "Add+ReLU Fusion",
            "benefit": "Single fused operation"
        },
        r"onnx\.MatMul.*onnx\.Add": {
            "pattern": "MatMul+Bias Fusion",
            "benefit": "Integrated bias computation"
        },
        r"onnx\.Conv.*onnx\.BatchNormalization": {
            "pattern": "Conv+BN Fusion",
            "benefit": "Reduced memory access"
        },
        r"onnx\.Mul.*constant.*onnx\.Add.*constant": {
            "pattern": "Linear transformation folding",
            "benefit": "Compile-time computation"
        }
    }
    
    for pattern_regex, info in optimization_patterns.items():
        if re.search(pattern_regex, mlir_content):
            patterns.append(info)
    
    return patterns

def calculate_op_reduction(report):
    """연산 감소율 계산"""
    total_mods = sum(p["modifications"] for p in report["applicable_passes"])
    return f"{total_mods} operations can be eliminated/fused"

def analyze_memory_patterns(model_path):
    """메모리 최적화 패턴 분석"""
    memory_opts = []
    
    # Buffer 재사용 분석
    cmd = f"onnx-mlir --EmitLLVMIR {model_path} 2>&1 | grep -i 'alloc\\|buffer'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    alloc_count = result.stdout.count("alloc")
    if alloc_count > 10:
        memory_opts.append({
            "issue": "High allocation count",
            "count": alloc_count,
            "suggestion": "Enable buffer pooling"
        })
    
    return memory_opts
