import tvm
from tvm import relay
import json
import onnx

class OptimizationAnalyzer:
    def __init__(self, model_path):
        self.onnx_model = onnx.load(model_path)
        self.shape_dict = self._get_shape_dict()
        self.mod, self.params = relay.frontend.from_onnx(
            self.onnx_model, self.shape_dict
        )
        self.optimization_report = {
            "fusion_opportunities": [],
            "constant_folding": [],
            "dead_code": [],
            "layout_optimization": [],
            "quantization_candidates": [],
            "memory_optimization": []
        }
    
    def _get_shape_dict(self):
        return {inp.name: [d.dim_value for d in inp.type.tensor_type.shape.dim] 
                for inp in self.onnx_model.graph.input}
    
    def analyze_fusion_opportunities(self):
        """연산자 융합 기회 분석"""
        
        @tvm.ir.transform.module_pass(opt_level=0)
        def detect_fusion(mod, ctx):
            fusion_patterns = []
            
            class FusionDetector(relay.ExprVisitor):
                def visit_call(self, call):
                    # Conv + BN + ReLU 패턴 탐지
                    if call.op.name == "nn.conv2d":
                        for user in call.users:
                            if hasattr(user, 'op') and user.op.name == "nn.batch_norm":
                                for bn_user in user.users:
                                    if hasattr(bn_user, 'op') and bn_user.op.name == "nn.relu":
                                        fusion_patterns.append({
                                            "pattern": "Conv-BN-ReLU",
                                            "ops": ["conv2d", "batch_norm", "relu"],
                                            "benefit": "Reduce memory bandwidth by 30-40%"
                                        })
                    
                    # MatMul + Add 패턴 (Bias addition)
                    if call.op.name == "nn.dense":
                        for user in call.users:
                            if hasattr(user, 'op') and user.op.name == "add":
                                fusion_patterns.append({
                                    "pattern": "Dense-Add",
                                    "ops": ["dense", "add"],
                                    "benefit": "Fuse bias into dense operation"
                                })
                    
                    super().visit_call(call)
            
            FusionDetector().visit(mod["main"])
            self.optimization_report["fusion_opportunities"] = fusion_patterns
            return mod
        
        with tvm.transform.PassContext(opt_level=0):
            detect_fusion(self.mod)
    
    def analyze_constant_folding(self):
        """상수 폴딩 기회 분석"""
        
        class ConstantAnalyzer(relay.ExprVisitor):
            def __init__(self):
                super().__init__()
                self.constant_ops = []
            
            def visit_call(self, call):
                # 모든 입력이 상수인 연산 찾기
                all_constant = True
                for arg in call.args:
                    if not isinstance(arg, relay.Constant):
                        all_constant = False
                        break
                
                if all_constant and hasattr(call.op, 'name'):
                    self.constant_ops.append({
                        "op": call.op.name,
                        "can_precompute": True,
                        "benefit": "Eliminate runtime computation"
                    })
                
                super().visit_call(call)
        
        analyzer = ConstantAnalyzer()
        analyzer.visit(self.mod["main"])
        self.optimization_report["constant_folding"] = analyzer.constant_ops
    
    def analyze_layout_optimization(self):
        """레이아웃 최적화 기회 분석"""
        layout_issues = []
        
        class LayoutAnalyzer(relay.ExprVisitor):
            def __init__(self):
                super().__init__()
                self.current_layout = "NCHW"  # Default
            
            def visit_call(self, call):
                if hasattr(call.op, 'name'):
                    # Transpose 연산 탐지
                    if call.op.name == "transpose":
                        layout_issues.append({
                            "operation": "transpose",
                            "location": str(call.span) if call.span else "unknown",
                            "suggestion": "Consider layout transformation to avoid transpose",
                            "benefit": "Reduce memory copy overhead"
                        })
                    
                    # Conv2D 레이아웃 체크
                    if call.op.name == "nn.conv2d":
                        if hasattr(call.attrs, 'data_layout'):
                            if call.attrs.data_layout != "NCHW":
                                layout_issues.append({
                                    "operation": "conv2d",
                                    "current_layout": call.attrs.data_layout,
                                    "suggested_layout": "NCHW or NHWC based on target",
                                    "benefit": "Better cache utilization"
                                })
                
                super().visit_call(call)
        
        LayoutAnalyzer().visit(self.mod["main"])
        self.optimization_report["layout_optimization"] = layout_issues
    
    def analyze_quantization_opportunities(self):
        """양자화 가능 레이어 분석"""
        quantizable_ops = ["nn.conv2d", "nn.dense", "nn.batch_norm", "nn.relu"]
        quantization_candidates = []
        
        class QuantizationAnalyzer(relay.ExprVisitor):
            def visit_call(self, call):
                if hasattr(call.op, 'name') and call.op.name in quantizable_ops:
                    # 타입 체크
                    if hasattr(call, 'checked_type'):
                        dtype = call.checked_type.dtype if hasattr(call.checked_type, 'dtype') else 'float32'
                        if dtype == 'float32':
                            quantization_candidates.append({
                                "op": call.op.name,
                                "current_dtype": dtype,
                                "suggested_dtype": "int8",
                                "expected_speedup": "2-4x",
                                "memory_reduction": "75%"
                            })
                
                super().visit_call(call)
        
        QuantizationAnalyzer().visit(self.mod["main"])
        self.optimization_report["quantization_candidates"] = quantization_candidates
    
    def analyze_memory_optimization(self):
        """메모리 최적화 기회 분석"""
        memory_opts = []
        
        # 메모리 풀 재사용 분석
        with tvm.transform.PassContext(opt_level=3):
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(self.mod, target="llvm", params=self.params)
                
                # 메모리 할당 패턴 분석
                memory_plan = json.loads(graph)
                if "attrs" in memory_plan and "storage_id" in memory_plan["attrs"]:
                    storage_info = memory_plan["attrs"]["storage_id"]
                    reuse_opportunities = len(set(storage_info[1])) / len(storage_info[1])
                    
                    memory_opts.append({
                        "optimization": "memory_pooling",
                        "current_reuse_ratio": f"{reuse_opportunities:.2%}",
                        "suggestion": "Enable memory pool optimization",
                        "benefit": f"Reduce memory usage by {(1-reuse_opportunities)*100:.0f}%"
                    })
        
        self.optimization_report["memory_optimization"] = memory_opts
    
    def generate_report(self, output_file="optimization_report.json"):
        """전체 최적화 리포트 생성"""
        # 모든 분석 실행
        self.analyze_fusion_opportunities()
        self.analyze_constant_folding()
        self.analyze_layout_optimization()
        self.analyze_quantization_opportunities()
        self.analyze_memory_optimization()
        
        # 요약 정보 추가
        summary = {
            "total_fusion_opportunities": len(self.optimization_report["fusion_opportunities"]),
            "total_constant_folding": len(self.optimization_report["constant_folding"]),
            "total_layout_issues": len(self.optimization_report["layout_optimization"]),
            "total_quantization_candidates": len(self.optimization_report["quantization_candidates"]),
            "estimated_speedup": self._estimate_speedup(),
            "estimated_memory_reduction": self._estimate_memory_reduction()
        }
        
        final_report = {
            "summary": summary,
            "detailed_analysis": self.optimization_report,
            "recommendations": self._generate_recommendations()
        }
        
        # JSON 파일로 저장
        with open(output_file, "w") as f:
            json.dump(final_report, f, indent=2)
        
        # 읽기 쉬운 텍스트 리포트도 생성
        self._generate_text_report(final_report, output_file.replace('.json', '.txt'))
        
        return final_report
    
    def _estimate_speedup(self):
        """예상 속도 향상 계산"""
        speedup = 1.0
        if self.optimization_report["fusion_opportunities"]:
            speedup *= 1.3  # 30% improvement from fusion
        if self.optimization_report["quantization_candidates"]:
            speedup *= 2.0  # 2x from quantization
        return f"{speedup:.1f}x"
    
    def _estimate_memory_reduction(self):
        """예상 메모리 감소 계산"""
        reduction = 0
        if self.optimization_report["quantization_candidates"]:
            reduction = 75  # 75% reduction from int8
        return f"{reduction}%"
    
    def _generate_recommendations(self):
        """최적화 권장사항 생성"""
        recommendations = []
        
        if self.optimization_report["fusion_opportunities"]:
            recommendations.append({
                "priority": "HIGH",
                "action": "Enable operator fusion",
                "command": "with tvm.transform.PassContext(opt_level=3, config={'relay.FuseOps.max_depth': 10})",
                "expected_benefit": "30-40% latency reduction"
            })
        
        if self.optimization_report["quantization_candidates"]:
            recommendations.append({
                "priority": "HIGH",
                "action": "Apply INT8 quantization",
                "command": "relay.quantize.quantize(mod, params)",
                "expected_benefit": "2-4x speedup, 75% memory reduction"
            })
        
        if self.optimization_report["layout_optimization"]:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Optimize data layout",
                "command": "relay.transform.ConvertLayout({'nn.conv2d': ['NHWC', 'default']})",
                "expected_benefit": "10-20% performance improvement"
            })
        
        return recommendations
    
    def _generate_text_report(self, report, output_file):
        """읽기 쉬운 텍스트 리포트 생성"""
        with open(output_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("TVM OPTIMIZATION ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*30 + "\n")
            for key, value in report["summary"].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*30 + "\n")
            for i, rec in enumerate(report["recommendations"], 1):
                f.write(f"\n{i}. [{rec['priority']}] {rec['action']}\n")
                f.write(f"   Command: {rec['command']}\n")
                f.write(f"   Expected: {rec['expected_benefit']}\n")
            
            f.write("\n\nDETAILED FINDINGS\n")
            f.write("-"*30 + "\n")
            for category, items in report["detailed_analysis"].items():
                if items:
                    f.write(f"\n{category.upper()}:\n")
                    for item in items[:5]:  # 처음 5개만 표시
                        f.write(f"  - {item}\n")

# 사용 예제
analyzer = OptimizationAnalyzer("model.onnx")
report = analyzer.generate_report("optimization_analysis.json")
print("Optimization report generated successfully!")
