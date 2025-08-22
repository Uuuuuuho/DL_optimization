class ComprehensiveOptimizationAnalyzer:
    """TVM과 ONNX-MLIR을 모두 활용한 종합 최적화 분석"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.optimization_insights = {
            "model_info": {},
            "bottlenecks": [],
            "optimization_opportunities": [],
            "platform_specific": {},
            "estimated_improvements": {}
        }
    
    def analyze_computational_bottlenecks(self):
        """계산 병목 지점 분석"""
        import numpy as np
        
        bottlenecks = []
        
        # FLOPS 계산
        for node in self.model.graph.node:
            flops = self._estimate_node_flops(node)
            if flops > 1e9:  # 1 GFLOP 이상
                bottlenecks.append({
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "estimated_flops": f"{flops/1e9:.2f} GFLOPS",
                    "optimization_suggestion": self._get_optimization_suggestion(node)
                })
        
        self.optimization_insights["bottlenecks"] = sorted(
            bottlenecks, 
            key=lambda x: float(x["estimated_flops"].split()[0]), 
            reverse=True
        )
    
    def _estimate_node_flops(self, node):
        """노드별 FLOPS 추정"""
        flops = 0
        
        if node.op_type == "Conv":
            # Conv FLOPS 계산 (간단한 추정)
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_size = np.prod(attr.ints)
                    # 입력/출력 채널 수 등을 고려한 계산 필요
                    flops = kernel_size * 1e8  # 예시 값
        
        elif node.op_type == "MatMul" or node.op_type == "Gemm":
            flops = 2e9  # 예시 값, 실제로는 shape 기반 계산 필요
        
        return flops
    
    def _get_optimization_suggestion(self, node):
        """노드별 최적화 제안"""
        suggestions = {
            "Conv": "Consider Winograd convolution or FFT-based convolution for large kernels",
            "MatMul": "Use optimized BLAS libraries or tensor core acceleration",
            "Gemm": "Enable GEMM packing and tiling optimizations",
            "BatchNormalization": "Fuse with preceding Conv layer",
            "Add": "Check for broadcast optimization opportunities"
        }
        return suggestions.get(node.op_type, "Profile for optimization opportunities")
    
    def analyze_dataflow_optimization(self):
        """데이터 흐름 최적화 분석"""
        dataflow_opts = []
        
        # 중간 텐서 재사용 분석
        tensor_usage = {}
        for node in self.model.graph.node:
            for output in node.output:
                tensor_usage[output] = tensor_usage.get(output, 0) + 1
        
        # 재사용 가능한 버퍼 찾기
        reusable_buffers = [t for t, count in tensor_usage.items() if count == 1]
        if reusable_buffers:
            dataflow_opts.append({
                "optimization": "Buffer reuse",
                "reusable_count": len(reusable_buffers),
                "memory_saving": f"{len(reusable_buffers) * 4}MB (estimated)",
                "implementation": "Enable in-place operations"
            })
        
        self.optimization_insights["optimization_opportunities"].extend(dataflow_opts)
    
    def analyze_precision_optimization(self):
        """정밀도 최적화 분석"""
        precision_opts = []
        
        # Mixed precision 기회 탐색
        fp32_ops = []
        for node in self.model.graph.node:
            # FP32로 실행되지만 낮은 정밀도 가능한 연산
            if node.op_type in ["Conv", "MatMul", "Add", "Mul"]:
                fp32_ops.append({
                    "node": node.name,
                    "op": node.op_type,
                    "current": "FP32",
                    "suggested": "FP16/BF16",
                    "speedup": "2x on GPU/TPU"
                })
        
        if fp32_ops:
            precision_opts.append({
                "optimization": "Mixed Precision",
                "applicable_ops": len(fp32_ops),
                "details": fp32_ops[:10],  # 처음 10개만
                "expected_speedup": "1.5-3x on modern hardware"
            })
        
        self.optimization_insights["optimization_opportunities"].extend(precision_opts)
    
    def analyze_platform_specific(self, target_platforms=["cpu", "gpu", "edge"]):
        """플랫폼별 최적화 분석"""
        
        for platform in target_platforms:
            platform_opts = []
            
            if platform == "cpu":
                platform_opts.extend([
                    {
                        "technique": "Vectorization",
                        "applicable": "All element-wise operations",
                        "flag": "-march=native",
                        "expected_speedup": "2-4x"
                    },
                    {
                        "technique": "OpenMP parallelization",
                        "applicable": "Batch processing",
                        "flag": "-fopenmp",
                        "expected_speedup": "Linear with cores"
                    }
                ])
            
            elif platform == "gpu":
                platform_opts.extend([
                    {
                        "technique": "Tensor Core utilization",
                        "applicable": "GEMM operations",
                        "requirement": "FP16/TF32 precision",
                        "expected_speedup": "2-10x"
                    },
                    {
                        "technique": "CUDNN fusion",
                        "applicable": "Conv-BN-ReLU patterns",
                        "requirement": "CUDNN 8.0+",
                        "expected_speedup": "20-30%"
                    }
                ])
            
            elif platform == "edge":
                platform_opts.extend([
                    {
                        "technique": "INT8 quantization",
                        "applicable": "Most layers",
                        "tool": "TVM quantization",
                        "expected_speedup": "2-4x",
                        "memory_reduction": "75%"
                    },
                    {
                        "technique": "Model pruning",
                        "applicable": "Dense layers",
                        "sparsity_target": "50-90%",
                        "expected_speedup": "2x at 50% sparsity"
                    }
                ])
            
            self.optimization_insights["platform_specific"][platform] = platform_opts
    
    def generate_optimization_report(self, output_dir="optimization_results"):
        """종합 최적화 리포트 생성"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 모든 분석 실행
        self.analyze_computational_bottlenecks()
        self.analyze_dataflow_optimization()
        self.analyze_precision_optimization()
        self.analyze_platform_specific()
        
        # 개선 예상치 계산
        self._calculate_expected_improvements()
        
        # JSON 리포트
        with open(f"{output_dir}/comprehensive_optimization.json", "w") as f:
            json.dump(self.optimization_insights, f, indent=2)
        
        # Markdown 리포트 생성
        self._generate_markdown_report(output_dir)
        
        # 실행 가능한 최적화 스크립트 생성
        self._generate_optimization_script(output_dir)
        
        print(f"Optimization analysis complete. Results in {output_dir}/")
        return self.optimization_insights
    
    def _calculate_expected_improvements(self):
        """전체 개선 예상치 계산"""
        improvements = {
            "latency_reduction": "30-50%",
            "memory_reduction": "40-60%",
            "throughput_increase": "2-3x",
            "power_efficiency": "40% better FLOPS/Watt"
        }
        
        # 적용 가능한 최적화 수에 따라 조정
        num_opts = len(self.optimization_insights["optimization_opportunities"])
        if num_opts > 10:
            improvements["confidence"] = "High"
        elif num_opts > 5:
            improvements["confidence"] = "Medium"
        else:
            improvements["confidence"] = "Low"
        
        self.optimization_insights["estimated_improvements"] = improvements
    
    def _generate_markdown_report(self, output_dir):
        """읽기 쉬운 Markdown 리포트 생성"""
        with open(f"{output_dir}/optimization_report.md", "w") as f:
            f.write("# Model Optimization Analysis Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Bottlenecks Found**: {len(self.optimization_insights['bottlenecks'])}\n")
            f.write(f"- **Optimization Opportunities**: {len(self.optimization_insights['optimization_opportunities'])}\n")
            f.write(f"- **Expected Improvements**: {self.optimization_insights['estimated_improvements']['latency_reduction']} latency reduction\n\n")
            
            f.write("## Top Computational Bottlenecks\n\n")
            for i, bottleneck in enumerate(self.optimization_insights['bottlenecks'][:5], 1):
                f.write(f"{i}. **{bottleneck['op_type']}** - {bottleneck['estimated_flops']}\n")
                f.write(f"   -
