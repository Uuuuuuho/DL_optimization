from __future__ import annotations

import os
import json
import time
import subprocess
from typing import Dict, Any, Optional, List, Tuple


def _env_with_libs(extra_lib_paths: Optional[list[str]] = None) -> dict:
    env = dict(os.environ)
    llp = env.get("LD_LIBRARY_PATH", "")
    add = ":".join(extra_lib_paths or [])
    env["LD_LIBRARY_PATH"] = f"{add}:{llp}" if add else llp
    return env


def build_with_trtexec(
    onnx_path: str,
    trtexec_bin: str,
    out_log_json: str,
    shapes: Dict[str, str],  # name -> "dimx..."
    precision: str = "FP16",
    workspace_mib: int = 2048,
    timing_cache: Optional[str] = None,
    save_engine: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    pass_shapes: bool = True,
) -> Tuple[int, str]:
    os.makedirs(os.path.dirname(out_log_json), exist_ok=True)
    args: List[str] = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--exportTimes={out_log_json}",
        f"--memPoolSize=workspace:{workspace_mib}",
        "--profilingVerbosity=detailed",
        "--avgRuns=100",
        "--iterations=1000",
        "--warmUp=200",
    ]
    if precision.upper() == "FP16":
        args.append("--fp16")
    elif precision.upper() == "FP32":
        pass
    if pass_shapes and shapes:
        pairs = [f"{k}:{v}" for k, v in shapes.items()]
        spec = ",".join(pairs)
        args.append(f"--shapes={spec}")
        args.append(f"--minShapes={spec}")
        args.append(f"--optShapes={spec}")
        args.append(f"--maxShapes={spec}")
    if timing_cache:
        args.append(f"--timingCacheFile={timing_cache}")
    if save_engine:
        os.makedirs(os.path.dirname(save_engine), exist_ok=True)
        args.append(f"--saveEngine={save_engine}")
    if extra_args:
        args.extend(extra_args)

    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout

def build_with_trtexec_legacy(
    onnx_path: str,
    input_name: str,
    shape: str,  # e.g., "X:4x128" or "input:1x3x224x224"
    out_dir: str,
    fp16: bool = True,
    timing_cache: Optional[str] = None,
    extra_libs: Optional[list[str]] = None,
    iterations: int = 1000,
    warmup_ms: int = 200,
    trtexec_bin: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, os.path.basename(onnx_path).replace(".onnx", ".plan"))
    profile_json = os.path.join(out_dir, os.path.basename(onnx_path).replace(".onnx", "_trt_profile.json"))
    log_path = os.path.join(out_dir, os.path.basename(onnx_path).replace(".onnx", "_trtexec.log"))

    trtexec = trtexec_bin or os.environ.get("TRTEXEC_BIN", "/usr/src/tensorrt/bin/trtexec")
    mem_pool = "--memPoolSize=workspace:2048"
    prec = "--fp16" if fp16 else ""
    # shape flags for explicit batch
    shape_flags = [f"--shapes={shape}", f"--minShapes={shape}", f"--optShapes={shape}", f"--maxShapes={shape}"]
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        *shape_flags,
        mem_pool,
        prec,
        f"--iterations={iterations}",
        f"--warmUp={warmup_ms}",
        "--avgRuns=100",
        "--useSpinWait",
        "--dumpProfile",
        "--profilingVerbosity=detailed",
        f"--exportProfile={profile_json}",
        f"--saveEngine={plan_path}",
    ]
    # timing cache support via environment
    if timing_cache:
        os.makedirs(os.path.dirname(timing_cache), exist_ok=True)
        # Newer trtexec may use --timingCacheFile; handle if present
        cmd.append(f"--timingCacheFile={timing_cache}")

    env = _env_with_libs(extra_libs)
    t0 = time.perf_counter()
    with open(log_path, "w") as lf:
        try:
            proc = subprocess.run([c for c in cmd if c], env=env, check=True, capture_output=True, text=True)
            lf.write(proc.stdout)
            lf.write(proc.stderr)
            ok = True
            msg = "ok"
        except subprocess.CalledProcessError as e:
            lf.write(e.stdout or "")
            lf.write(e.stderr or "")
            ok = False
            msg = f"trtexec failed: {e}"
    dur_s = time.perf_counter() - t0
    return {
        "ok": ok,
        "message": msg,
        "engine": plan_path if os.path.exists(plan_path) else None,
        "profile": profile_json if os.path.exists(profile_json) else None,
        "log": log_path,
        "build_time_s": dur_s,
    }


__all__ = ["build_with_trtexec", "build_with_trtexec_legacy"]
