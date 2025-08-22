from __future__ import annotations

import os
import copy
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

import onnx
import numpy as np

try:
    import onnx_graphsurgeon as gs  # type: ignore
except Exception as e:
    raise SystemExit("This tool requires 'onnx-graphsurgeon'. Install: pip install onnx-graphsurgeon")

try:
    from onnxsim import simplify as onnx_simplify  # type: ignore
except Exception:
    onnx_simplify = None  # optional


def _save_model(m: onnx.ModelProto, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        # Align with onnxruntime in this workspace (supports up to IR 10)
        m.ir_version = min(getattr(m, "ir_version", 10) or 10, 10)
    except Exception:
        pass
    onnx.save(m, path)


def _polish_toposort(m: onnx.ModelProto) -> onnx.ModelProto:
    """Round-trip via GraphSurgeon cleanup to ensure topological order and remove dangles."""
    try:
        import onnx_graphsurgeon as gs  # type: ignore
    except Exception:
        return m
    g = gs.import_onnx(m)
    g.cleanup()
    return gs.export_onnx(g)


def _is_const_tensor(t: gs.Variable) -> bool:
    return isinstance(t, gs.Constant)


def _align_k_for_matmul(graph: gs.Graph, multiples: List[int]) -> int:
    """Pad K dimension (shared dim) of MatMul inputs/weights to align to preferred multiples.
    Only when right input is a constant weight [K, N] and left input has shape [*, K].
    Returns number of rewrites performed.
    """
    cnt = 0
    for n in list(graph.nodes):
        if n.op != "MatMul":
            continue
        a, b = n.inputs
        # Support: B is constant weight
        if not isinstance(b, (gs.Constant,)):
            continue
        if not isinstance(a, gs.Variable):
            continue
        w: np.ndarray = b.values
        if w.ndim != 2:
            continue
        K, N = w.shape
        if K <= 0:
            continue
        # find next aligned K'
        def next_aligned(k: int) -> Optional[int]:
            best = None
            for m in multiples:
                k2 = int(np.ceil(k / m) * m)
                best = k2 if best is None else min(best, k2)
            return best

        K2 = next_aligned(K)
        if K2 is None or K2 == K:
            continue
        pad_k = K2 - K
        # Pad weight along K: [K,N] -> [K2,N] with zeros at tail rows
        w_pad = np.pad(w, ((0, pad_k), (0, 0)), mode="constant")
        b.values = w_pad

        # Insert Pad op on A's last dim (assume [..., K]) and Slice back after MatMul
        # A_pad = Pad(A, pads=[0,...,0, 0, pad_k]) in ONNX axes semantics
        # GraphSurgeon: create Pad with pads constant
        pads = np.array([0, 0], dtype=np.int64)  # we will use opset>=11 Pad with sizes attribute through inputs
        pads_v = gs.Constant(name=f"pads_k_{n.name}", values=np.array([0, 0, 0, pad_k], dtype=np.int64))
        # For safety and simplicity, emulate Pad by Concat zeros along last dim
        zeros = gs.Constant(name=f"zeros_k_{n.name}", values=np.zeros((pad_k,), dtype=np.float32))
        # We need to broadcast zeros to match A's batch dims: use Expand
        shape_node = gs.Node(op="Shape", inputs=[a], outputs=[gs.Variable(name=f"shape_{a.name}")])
        last_dim = gs.Node(op="Gather", attrs={"axis": 0}, inputs=[shape_node.outputs[0], gs.Constant(name="g_idx", values=np.array([-1], dtype=np.int64))], outputs=[gs.Variable(name=f"kdim_{a.name}")])
        # Build zeros tensor shape: [..., pad_k]
        padk_v = gs.Constant(name=f"padk_{n.name}", values=np.array([pad_k], dtype=np.int64))
        new_shape = gs.Node(op="Concat", attrs={"axis": 0}, inputs=[shape_node.outputs[0], padk_v], outputs=[gs.Variable(name=f"shapez_{a.name}")])
        zeros_expand = gs.Node(op="ConstantOfShape", inputs=[new_shape.outputs[0]], attrs={"value": onnx.helper.make_tensor("v", onnx.TensorProto.FLOAT, [], [0.0])}, outputs=[gs.Variable(name=f"zeros_{a.name}")])
        a_pad = gs.Variable(name=f"{a.name}_pad")
        cat = gs.Node(op="Concat", attrs={"axis": -1}, inputs=[a, zeros_expand.outputs[0]], outputs=[a_pad])
        graph.nodes.extend([shape_node, last_dim, new_shape, zeros_expand, cat])

        # Re-wire MatMul input
        n.inputs[0] = a_pad

        # After MatMul, slice back to original K result is not necessary because padding is along K contracted dim
        cnt += 1
    graph.cleanup()
    return cnt


def _matmul_to_gemm(graph: gs.Graph) -> int:
    """Rewrite MatMul(A, W) with constant W[K,N] to Gemm(A, W, B=0).
    Keeps semantics equivalent; TRT often prefers Gemm/FC over MatMul in some cases.
    """
    cnt = 0
    for n in list(graph.nodes):
        if n.op != "MatMul":
            continue
        a, b = n.inputs
        if not isinstance(b, gs.Constant):
            continue
        wv = np.array(b.values)
        if wv.ndim != 2:
            continue
        # Bias must be length-N vector
        N = int(wv.shape[1])
        bias = gs.Constant(name=f"bias_{n.name}", values=np.zeros((N,), dtype=wv.dtype))
        gemm = gs.Node(
            op="Gemm",
            inputs=[a, b, bias],
            outputs=n.outputs,
            attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
        )
        # insert Gemm at the position of original MatMul to preserve topo order
        try:
            idx = graph.nodes.index(n)
        except ValueError:
            idx = len(graph.nodes)
        graph.nodes.insert(idx, gemm)
        # detach old node
        n.outputs = []
        cnt += 1
    graph.cleanup()
    return cnt


def generate_candidates(
    onnx_path: str,
    out_dir: str,
    use_simplifier: bool = True,
    seed: int = 0,
    enable_hfusion: bool = False,
    min_group_size: int = 2,
    disable_matmul: bool = False,
    disable_gemm: bool = False,
) -> List[str]:
    """Generate a small set of candidate ONNX graphs from the baseline.

    Returns list of candidate file paths including the baseline copy.
    """
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    base = onnx.load(onnx_path)
    baseline_path = os.path.join(out_dir, "candidate_00_baseline.onnx")
    _save_model(base, baseline_path)
    paths = [baseline_path]

    # Candidate: simplified baseline (if available)
    if use_simplifier and onnx_simplify is not None:
        try:
            simp_model, ok = onnx_simplify(copy.deepcopy(base), check_n=3)
            if ok:
                simp_model = _polish_toposort(simp_model)
                p = os.path.join(out_dir, f"candidate_{len(paths):02d}_simplified.onnx")
                _save_model(simp_model, p)
                paths.append(p)
        except Exception:
            pass

    # Optional: Horizontal Fusion candidate using repo scripts
    if enable_hfusion:
        hf_out = os.path.join(out_dir, f"candidate_{len(paths):02d}_hfusion.onnx")
        ok = _try_horizontal_fusion(
            in_model=onnx_path,
            out_model=hf_out,
            work_dir=os.path.join(out_dir, "hfusion"),
            min_group_size=min_group_size,
            no_matmul=disable_matmul,
            no_gemm=disable_gemm,
        )
        if ok:
            paths.append(hf_out)

    # Candidate: MatMul->Gemm
    g = gs.import_onnx(copy.deepcopy(base))
    rew = _matmul_to_gemm(g)
    if rew > 0:
        m1 = gs.export_onnx(g)
        if use_simplifier and onnx_simplify is not None:
            try:
                m1, ok = onnx_simplify(m1, check_n=3)
            except Exception:
                ok = False
        # enforce topo sort and cleanup
        m1 = _polish_toposort(m1)
        p1 = os.path.join(out_dir, f"candidate_{len(paths):02d}_gemm.onnx")
        _save_model(m1, p1)
        paths.append(p1)

    # Candidate 2: K-align padding for MatMul (8)
    g2 = gs.import_onnx(copy.deepcopy(base))
    cnt2 = _align_k_for_matmul(g2, [8])
    if cnt2 > 0:
        m2 = gs.export_onnx(g2)
        if use_simplifier and onnx_simplify is not None:
            try:
                m2, ok = onnx_simplify(m2, check_n=3)
            except Exception:
                ok = False
        m2 = _polish_toposort(m2)
        p2 = os.path.join(out_dir, f"candidate_{len(paths):02d}_kalign8.onnx")
        _save_model(m2, p2)
        paths.append(p2)

    # Candidate 3: K-align padding (16)
    g3 = gs.import_onnx(copy.deepcopy(base))
    cnt3 = _align_k_for_matmul(g3, [16])
    if cnt3 > 0:
        m3 = gs.export_onnx(g3)
        if use_simplifier and onnx_simplify is not None:
            try:
                m3, ok = onnx_simplify(m3, check_n=3)
            except Exception:
                ok = False
        m3 = _polish_toposort(m3)
        p3 = os.path.join(out_dir, f"candidate_{len(paths):02d}_kalign16.onnx")
        _save_model(m3, p3)
        paths.append(p3)

    return paths


def _try_horizontal_fusion(
    in_model: str,
    out_model: str,
    work_dir: str,
    min_group_size: int,
    no_matmul: bool,
    no_gemm: bool,
) -> bool:
    """Invoke repo's 08_apply_horizontal_fusion.py to produce a fused model."""
    # Locate the script at repo root relative to this file: src/tuner/.. -> repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    script = os.path.join(repo_root, "08_apply_horizontal_fusion.py")
    if not os.path.exists(script):
        return False
    cmd = [
        sys.executable,
        script,
        "--in", in_model,
        "--out", out_model,
        "--work-dir", work_dir,
        "--min-group-size", str(min_group_size),
        "--run",
    ]
    if no_matmul:
        cmd.append("--no-matmul")
    if no_gemm:
        cmd.append("--no-gemm")
    try:
        subprocess.check_call(cmd)
        return os.path.exists(out_model)
    except subprocess.CalledProcessError:
        return False
