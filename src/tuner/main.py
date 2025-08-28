from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

validate_against_baseline = None  # type: ignore[assignment]
try:
    # Prefer package-relative imports when run as a module
    from .candidate_generator import generate_candidates
    from .trt_builder import build_with_trtexec as trt_build
    from .profiler import load_times_json, summarize_times
    from .selector import pick_best
    from .utils import get_first_input_and_outputs, shapes_str
except Exception:
    # Fallback for running as a script from the same directory
    from candidate_generator import generate_candidates  # type: ignore
    from trt_builder import build_with_trtexec as trt_build  # type: ignore
    from profiler import load_times_json, summarize_times  # type: ignore
    from selector import pick_best  # type: ignore
    from utils import get_first_input_and_outputs, shapes_str  # type: ignore


def parse_kv_shapes(s: str) -> Dict[str, str]:
    # input spec: name:1x3x224x224,name2:1x80
    out: Dict[str, str] = {}
    if not s:
        return out
    for part in s.split(","):
        if not part:
            continue
        k, v = part.split(":", 1)
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description="TensorRT Frontend Optimization Auto-Tuner")
    p.add_argument("--onnx", required=True, help="Path to base ONNX model")
    p.add_argument("--outdir", required=True, help="Output directory for run artifacts")
    p.add_argument("--trtexec", required=False, default=os.environ.get("TRTEXEC_BIN", "/usr/src/tensorrt/bin/trtexec"), help="Path to trtexec binary (required if not using --skip-build)")
    p.add_argument("--precision", choices=["FP32", "FP16"], default="FP16")
    p.add_argument("--shapes", default="", help="Input shapes spec 'name:dimx...,name2:...' if dynamic or unknown")
    p.add_argument("--save-plan", action="store_true", help="Save TensorRT plan files")
    p.add_argument("--timing-cache", default="", help="Path to timing cache file to reuse")
    p.add_argument("--simplify", action="store_true", help="Run onnx-simplifier for candidates")
    p.add_argument("--seed", type=int, default=0)
    # Horizontal Fusion controls
    p.add_argument("--enable-hfusion", action="store_true", help="Try horizontal fusion via 08_apply_horizontal_fusion.py")
    p.add_argument("--hf-min-group", default="2", help="Min group size(s) for h-fusion, e.g. '2', '2-5', '2-8:2', or '2,4,8'")
    p.add_argument("--hf-no-matmul", action="store_true", help="Disable MatMul grouping for h-fusion hints")
    p.add_argument("--hf-no-gemm", action="store_true", help="Disable Gemm grouping for h-fusion hints")
    p.add_argument("--skip-build", action="store_true", help="Skip TensorRT build & profiling; only validate candidates")
    p.add_argument("--skip-validation", action="store_true", help="Skip ORT validation and proceed to TRT build")
    p.add_argument("--export-profile", action="store_true", help="Export TensorRT layer profile JSON")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cand_dir = os.path.join(args.outdir, "candidates")
    logs_dir = os.path.join(args.outdir, "logs")
    plans_dir = os.path.join(args.outdir, "plans")
    os.makedirs(cand_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    if args.save_plan:
        os.makedirs(plans_dir, exist_ok=True)

    # Helpers
    def _parse_int_ranges(spec: str) -> List[int]:
        vals: set[int] = set()
        if not spec:
            return [2]
        parts = [s.strip() for s in str(spec).split(",") if s.strip()]
        for tok in parts or [str(spec)]:
            if "-" in tok:
                # A-B[:S]
                try:
                    rng, *step = tok.split(":")
                    a_str, b_str = rng.split("-")
                    a = int(a_str); b = int(b_str)
                    s = int(step[0]) if step else 1
                    if s == 0:
                        continue
                    lo, hi = (a, b) if a <= b else (b, a)
                    vals.update(range(lo, hi + 1, abs(s)))
                except Exception:
                    continue
            else:
                try:
                    v = int(tok)
                    vals.add(v)
                except Exception:
                    continue
        out = sorted({v for v in vals if v and v > 0})
        return out or [2]

    # 1) Generate candidates
    candidates = generate_candidates(
        args.onnx, cand_dir,
        use_simplifier=args.simplify,
        seed=args.seed,
        enable_hfusion=args.enable_hfusion,
        min_group_sizes=_parse_int_ranges(args.hf_min_group),
        disable_matmul=args.hf_no_matmul,
        disable_gemm=args.hf_no_gemm,
    )

    # 2) Validate against baseline
    valid: List[str] = []
    # Lazy import validator only if we need it
    if not args.skip_validation and validate_against_baseline is None:
        try:
            from .validator import validate_against_baseline as _v  # type: ignore
        except Exception:
            try:
                from validator import validate_against_baseline as _v  # type: ignore
            except Exception as e:
                print(f"[warn] validator unavailable ({e}); skipping validation.")
                args.skip_validation = True
                _v = None  # type: ignore
        validate_against_baseline = _v  # type: ignore

    if args.skip_validation:
        valid = list(candidates)
    else:
        for c in candidates:
            ok, msg = validate_against_baseline(args.onnx, c)  # type: ignore[operator]
            if ok:
                valid.append(c)
            else:
                print(f"[drop] {os.path.basename(c)}: {msg}")
        if not valid:
            raise SystemExit("No valid candidates after validation.")

    # Prepare shapes and detect opset for implicit/explicit batch
    user_shapes = parse_kv_shapes(args.shapes)
    import onnx
    m = onnx.load(args.onnx)
    opset = 0
    for imp in m.opset_import:
        if imp.domain == "":
            opset = int(imp.version)
            break
    if not user_shapes and opset >= 11:
        t = m.graph.input[0].type.tensor_type
        dims = [d.dim_value if d.HasField('dim_value') else 1 for d in t.shape.dim]
        in_name = m.graph.input[0].name
        user_shapes = {in_name: "x".join(str(int(x)) for x in dims)}

    # 3) Build & profile with trtexec
    metrics_rows: List[Dict[str, object]] = []
    if args.skip_build:
        # Only validation; record candidates as ok without perf
        import onnx
        for c in valid:
            try:
                m_c = onnx.load(c)
                node_count = len(m_c.graph.node)
                mm_count = sum(1 for n in m_c.graph.node if n.op_type in ("MatMul", "Gemm"))
            except Exception:
                node_count = None
                mm_count = None
            metrics_rows.append({"candidate": c, "rc": None, "node_count": node_count, "mm_count": mm_count})
    else:
        trtexec_bin = args.trtexec
        if not trtexec_bin or not os.path.exists(trtexec_bin):
            raise SystemExit("trtexec binary not found. Provide --trtexec or set TRTEXEC_BIN, or use --skip-build.")
        for c in valid:
            base = os.path.splitext(os.path.basename(c))[0]
            times_json = os.path.join(logs_dir, f"{base}_times.json")
            save_engine = os.path.join(plans_dir, f"{base}.plan") if args.save_plan else None
            # For implicit-batch (common in opset<11), skip --shapes flags
            pass_shapes = opset >= 11
            extra_args = []
            if args.export_profile:
                extra_args += ["--dumpProfile", f"--exportProfile={os.path.join(logs_dir, base + '_profile.json')}"]
            rc, out = trt_build(
                onnx_path=c,
                trtexec_bin=trtexec_bin,
                out_log_json=times_json,
                shapes=user_shapes,
                precision=args.precision,
                workspace_mib=2048,
                timing_cache=args.timing_cache or None,
                save_engine=save_engine,
                extra_args=extra_args,
                pass_shapes=pass_shapes,
            )
            row: Dict[str, object] = {
                "candidate": c,
                "rc": rc,
            }
            if os.path.exists(times_json):
                tdata = load_times_json(times_json)
                row.update(summarize_times(tdata))
            metrics_rows.append(row)

    # 4) Select best
    best = pick_best(metrics_rows)
    if best:
        best_onx = best.get("candidate")
        if best_onx:
            dst = os.path.join(args.outdir, "best.onnx")
            try:
                import shutil
                shutil.copyfile(best_onx, dst)
            except Exception:
                pass
        with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in metrics_rows for k in r.keys()}))
            w.writeheader()
            for r in metrics_rows:
                w.writerow(r)
        with open(os.path.join(args.outdir, "best.json"), "w") as f:
            json.dump(best, f, indent=2)
        print("Best:", json.dumps(best, indent=2))
    else:
        print("No best candidate found.")


if __name__ == "__main__":
    main()
