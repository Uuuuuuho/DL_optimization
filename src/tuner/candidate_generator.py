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


def _horizontal_fuse(graph: "gs.Graph", min_group: int = 2, allow_matmul: bool = True, allow_gemm: bool = True) -> int:
    """Horizontally fuse MatMul/Gemm that share the same canonical left input and constant right inputs.

    - MatMul(A, Wi) -> concat W along N, single MatMul(A, Wcat) + Split on last dim.
    - Gemm(A, Wi, Bi) with transA/transB==0 -> concat W and B similarly.
    Returns number of fused groups.
    """
    try:
        import onnx_graphsurgeon as gs  # type: ignore
    except Exception:
        return 0

    fused_groups = 0

    def _canonical_left_input(v: gs.Variable) -> gs.Variable:
        """Follow trivial Identity producers to a canonical source variable."""
        seen: set[int] = set()
        cur = v
        for _ in range(8):
            vid = id(cur)
            if vid in seen:
                break
            seen.add(vid)
            if getattr(cur, "inputs", None) and len(cur.inputs) == 1:
                prod = cur.inputs[0]
                if prod and getattr(prod, "op", None) == "Identity" and prod.inputs and isinstance(prod.inputs[0], gs.Variable):
                    cur = prod.inputs[0]
                    continue
            break
        return cur

    def fuse_by_op(op_name: str) -> None:
        nonlocal fused_groups
        groups: Dict[int, List[gs.Node]] = {}
        left_by_key: Dict[int, gs.Variable] = {}
        for n in list(graph.nodes):
            if n.op != op_name:
                continue
            if op_name == "MatMul":
                if len(n.inputs) != 2:
                    continue
                a, w = n.inputs
                if not isinstance(a, gs.Variable) or not isinstance(w, gs.Constant):
                    continue
                wv = np.array(w.values)
                if wv.ndim != 2:
                    continue
                a_can = _canonical_left_input(a)
                key = id(a_can)
                groups.setdefault(key, []).append(n)
                left_by_key[key] = a_can
            else:  # Gemm
                if len(n.inputs) < 2:
                    continue
                a, w = n.inputs[:2]
                if not isinstance(a, gs.Variable) or not isinstance(w, gs.Constant):
                    continue
                ta = int(n.attrs.get("transA", 0)); tb = int(n.attrs.get("transB", 0))
                if ta != 0 or tb != 0:
                    continue
                wv = np.array(w.values)
                if wv.ndim != 2:
                    continue
                a_can = _canonical_left_input(a)
                key = id(a_can)
                groups.setdefault(key, []).append(n)
                left_by_key[key] = a_can

        for key, nodes in list(groups.items()):
            if len(nodes) < max(1, int(min_group)):
                continue
            nodes = sorted(nodes, key=lambda nn: graph.nodes.index(nn))

            weights: List[np.ndarray] = []
            biases: List[np.ndarray] = []
            outs: List[gs.Variable] = []
            dtype = None
            K = None
            ok = True
            for nn in nodes:
                if op_name == "MatMul":
                    _, w = nn.inputs
                    wv = np.array(w.values)
                    if dtype is None:
                        dtype = wv.dtype
                    if dtype != wv.dtype:
                        ok = False; break
                    if K is None:
                        K = int(wv.shape[0])
                    if int(wv.shape[0]) != K:
                        ok = False; break
                    weights.append(wv)
                    outs.append(nn.outputs[0])
                else:
                    _, w = nn.inputs[:2]
                    bvec = None
                    if len(nn.inputs) >= 3 and isinstance(nn.inputs[2], gs.Constant):
                        bvec = np.array(nn.inputs[2].values)
                    wv = np.array(w.values)
                    if dtype is None:
                        dtype = wv.dtype
                    if dtype != wv.dtype:
                        ok = False; break
                    if K is None:
                        K = int(wv.shape[0])
                    if int(wv.shape[0]) != K:
                        ok = False; break
                    if bvec is None:
                        bvec = np.zeros((int(wv.shape[1]),), dtype=wv.dtype)
                    else:
                        bvec = bvec.reshape((-1,))
                    weights.append(wv)
                    biases.append(bvec)
                    outs.append(nn.outputs[0])

            if not ok or not weights:
                continue

            Ns = [int(w.shape[1]) for w in weights]
            try:
                Wcat = np.concatenate(weights, axis=1)
                if op_name == "Gemm":
                    Bcat = np.concatenate(biases, axis=0) if biases else np.zeros((sum(Ns),), dtype=dtype)
            except Exception:
                continue

            left_in = left_by_key.get(key, nodes[0].inputs[0])
            w_const = gs.Constant(name=f"{left_in.name}_{op_name}_Wcat", values=Wcat)
            y_fused = gs.Variable(name=f"{outs[0].name}_{op_name}_fused", dtype=outs[0].dtype, shape=None)

            if op_name == "MatMul":
                fused = gs.Node(op="MatMul", inputs=[left_in, w_const], outputs=[y_fused])
            else:
                b_const = gs.Constant(name=f"{left_in.name}_{op_name}_Bcat", values=Bcat)
                fused = gs.Node(op="Gemm", inputs=[left_in, w_const, b_const], outputs=[y_fused], attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0})

            split_sizes = gs.Constant(name=f"split_sizes_{y_fused.name}", values=np.array(Ns, dtype=np.int64))
            split = gs.Node(op="Split", inputs=[y_fused, split_sizes], outputs=outs, attrs={"axis": -1})

            try:
                idx0 = graph.nodes.index(nodes[0])
            except ValueError:
                idx0 = len(graph.nodes)
            graph.nodes.insert(idx0, fused)
            graph.nodes.insert(idx0 + 1, split)

            for nn in nodes:
                nn.outputs = []

            fused_groups += 1

    if allow_matmul:
        fuse_by_op("MatMul")
    if allow_gemm:
        fuse_by_op("Gemm")

    if fused_groups:
        graph.cleanup()
    return fused_groups


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
    min_group_sizes: List[int] | None = None,
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
        groups = sorted({g for g in (min_group_sizes or [2]) if g and g > 0})
        for gsz in groups:
            tag = f"hfusion_g{gsz}"
            hf_out = os.path.join(out_dir, f"candidate_{len(paths):02d}_{tag}.onnx")
            ok = _try_horizontal_fusion(
                in_model=onnx_path,
                out_model=hf_out,
                work_dir=os.path.join(out_dir, f"hfusion_g{gsz}"),
                min_group_size=gsz,
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
    """Perform horizontal fusion internally without calling external modules.

    Fuses groups of MatMul/Gemm sharing the same left input with constant weights.
    Returns True if a model was written (fused or passthrough when no groups).
    """
    try:
        m = onnx.load(in_model)
        # Import to GS graph
        g = gs.import_onnx(m)
        fused_cnt = _horizontal_fuse(
            g,
            min_group=min_group_size,
            allow_matmul=not no_matmul,
            allow_gemm=not no_gemm,
        )
        # Export back to ONNX
        m_out = gs.export_onnx(g)
        m_out = _polish_toposort(m_out)
        os.makedirs(os.path.dirname(out_model), exist_ok=True)
        _save_model(m_out, out_model)
        return os.path.exists(out_model)
    except Exception:
        return False
