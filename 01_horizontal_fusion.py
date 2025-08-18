#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizontal Fusion with Dependency Analysis for ONNX (MatMul/Gemm)
- Loads an ONNX model
- Builds a dataflow dependency graph (node -> downstream consumers)
- Finds independent sibling nodes (no reachability between them)
- Performs horizontal fusion for MatMul/Gemm groups that share the same input

"""

import argparse
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Union, cast

import numpy as np
import onnx
import onnx.checker as oc
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx_graphsurgeon as gs
import pdb

# ------------------------------
# Utilities
# ------------------------------
def is_const(t: gs.Tensor) -> bool:
    return isinstance(t, gs.Constant)

def get_const_array(t: gs.Tensor) -> np.ndarray:
    assert is_const(t), "Expected Constant tensor"
    return t.values

def same_variable(a: gs.Tensor, b: gs.Tensor) -> bool:
    return a is b  # same object identity is fine in GS graph


# ------------------------------
# Build dependency graph (dataflow DAG)
# ------------------------------
class DepGraph:
    """
    Simple node-level dependency graph based on data edges.
    Nodes are gs.Node objects; edges point from producer node -> consumer node.
    """
    def __init__(self, graph: gs.Graph):
        self.graph = graph
        # Maps using hashable keys
        self.producer_of = {}            # tensor_key -> producer node (or None)
        self.consumers_of = defaultdict(list)  # tensor_key -> [consumer nodes]
        self.downstream = defaultdict(set)     # node_key -> set(node_key)
        self.upstream = defaultdict(set)       # node_key -> set(node_key)
        self._index()

    @staticmethod
    def _tensor_key(t: gs.Tensor):
        # Prefer a stable name string; fall back to id if unnamed
        name = getattr(t, "name", None)
        return name if name is not None else id(t)

    @staticmethod
    def _node_key(n: gs.Node):
        name = getattr(n, "name", None)
        return name if name is not None else id(n)

    def _index(self):
        # Outputs -> producers
        for node in self.graph.nodes:
            for t in node.outputs:
                if isinstance(t, gs.Tensor):
                    self.producer_of[self._tensor_key(t)] = node
        # Inputs -> consumers
        for node in self.graph.nodes:
            for t in node.inputs:
                if isinstance(t, gs.Tensor):
                    self.consumers_of[self._tensor_key(t)].append(node)
        # Build node adjacency
        for node in self.graph.nodes:
            src_k = self._node_key(node)
            for out_t in node.outputs:
                if not isinstance(out_t, gs.Tensor):
                    continue
                t_k = self._tensor_key(out_t)
                for cons in self.consumers_of.get(t_k, []):
                    if cons is node:
                        continue
                    dst_k = self._node_key(cons)
                    self.downstream[src_k].add(dst_k)
                    self.upstream[dst_k].add(src_k)

    def has_path(self, src: gs.Node, dst: gs.Node) -> bool:
        if src is dst:
            return True
        src_k = self._node_key(src)
        dst_k = self._node_key(dst)
        q = deque([src_k])
        visited = {src_k}
        while q:
            cur = q.popleft()
            for nxt in self.downstream.get(cur, []):
                if nxt == dst_k:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return False

    def indegree(self, node: gs.Node) -> int:
        return len(self.upstream.get(self._node_key(node), []))

    def outdegree(self, node: gs.Node) -> int:
        return len(self.downstream.get(self._node_key(node), []))

    def dump_summary(self, limit: int = 20):
        print("[DepGraph] nodes:", len(self.graph.nodes))
        for n in self.graph.nodes[:limit]:
            print(
                f"  - {n.name or n.op}: op={n.op}, in={len(n.inputs)}, out={len(n.outputs)}, "
                f"deg=({self.indegree(n)},{self.outdegree(n)})"
            )
        if len(self.graph.nodes) > limit:
            print(f"  ... ({len(self.graph.nodes)-limit} more)")

# ------------------------------
# Candidate grouping (MatMul/Gemm) with independence check
# ------------------------------
def collect_matmul_groups(g: gs.Graph) -> Dict[object, Tuple[gs.Variable, List[gs.Node]]]:
    # Return mapping: key -> (A_variable, [nodes]) where key is A.name or id(A)
    nodes_map: Dict[object, List[gs.Node]] = defaultdict(list)
    key_to_A: Dict[object, gs.Variable] = {}
    for n in g.nodes:
        if n.op != "MatMul":
            continue
        if len(n.inputs) != 2:
            continue
        A, B = n.inputs
        if not isinstance(A, gs.Variable):
            continue
        if not is_const(B):
            continue
        Barr = get_const_array(B)
        if Barr.ndim != 2:
            continue
        key = getattr(A, "name", None) or id(A)
        key_to_A[key] = A
        nodes_map[key].append(n)
    return {k: (key_to_A[k], v) for k, v in nodes_map.items()}

def collect_gemm_groups(g: gs.Graph) -> Dict[object, Tuple[gs.Variable, List[gs.Node]]]:
    def _tensor_key(t: gs.Tensor):
        return getattr(t, "name", None) or id(t)

    # Build a lightweight producer map (tensor_key -> producer node)
    prod_by_t = {}
    for node in g.nodes:
        for t in node.outputs:
            if isinstance(t, gs.Tensor):
                prod_by_t[_tensor_key(t)] = node

    def _unwrap_identity(t: gs.Tensor) -> gs.Tensor:
        # Walk back through chains of Identity producers to find the canonical source tensor
        seen = 0
        cur = t
        while seen < 16 and isinstance(cur, gs.Tensor):
            p = prod_by_t.get(_tensor_key(cur))
            if p is None or p.op != "Identity" or not p.inputs:
                break
            src = p.inputs[0]
            if not isinstance(src, gs.Tensor):
                break
            cur = src
            seen += 1
        return cur
    
    def gemm_with_const_weights(n: gs.Node) -> bool:
        if n.op != "Gemm":
            return False
        if len(n.inputs) < 2:
            return False
        A, B = n.inputs[0], n.inputs[1]
        C = n.inputs[2] if len(n.inputs) >= 3 else None
        if not isinstance(A, gs.Variable):
            return False
        if not is_const(B):
            return False
        if C is not None and not is_const(C):
            return False
        return True

    nodes_map: Dict[object, List[gs.Node]] = defaultdict(list)
    key_to_A: Dict[object, gs.Variable] = {}
    for n in g.nodes:
        if gemm_with_const_weights(n):
            A = n.inputs[0]
            # Canonicalize A by unwrapping Identity chains
            A = _unwrap_identity(A)
            key = getattr(A, "name", None) or id(A)
            key_to_A[key] = A
            nodes_map[key].append(n)
    return {k: (key_to_A[k], v) for k, v in nodes_map.items()}


def filter_independent_siblings(nodes: List[gs.Node], dep: DepGraph) -> List[List[gs.Node]]:
    """
    Given a node list (same op, same LHS input), split into maximal subsets where nodes are pairwise independent:
    (no path i->j and no path j->i).
    """
    # Build graph of conflicts (edge if dependent)
    N = len(nodes)
    if N < 2:
        return []
    conflict = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a, b = nodes[i], nodes[j]
            dep_ab = dep.has_path(a, b)
            dep_ba = dep.has_path(b, a)
            if dep_ab or dep_ba:
                conflict[i][j] = conflict[j][i] = True

    # Greedy partition into independent groups
    unused = set(range(N))
    groups_idx: List[List[int]] = []
    while unused:
        i = unused.pop()
        group = [i]
        # try to add as many as possible that do not conflict with current group
        added = True
        while added:
            added = False
            for j in list(unused):
                if all(not conflict[j][k] for k in group):
                    group.append(j)
                    unused.remove(j)
                    added = True
        if len(group) >= 2:
            groups_idx.append(group)

    # Convert to node groups
    groups = [[nodes[i] for i in grp] for grp in groups_idx]
    return groups


# ------------------------------
# Fusion implementations (MatMul/Gemm)
# ------------------------------
def fuse_matmul_group(A: gs.Tensor, nodes: List[gs.Node], g: gs.Graph, verbose: bool = True) -> bool:
    """
    Fuse a group of MatMul nodes sharing the same A (Variable) with Constant weights [K, Mi]
    Create:
        W_cat = Concat(B_i, axis=1)
        Y_cat = MatMul(A, W_cat)
        [outs] = Split(Y_cat, split_sizes, axis=-1)
    """
    # dtype & K check
    B_list = [get_const_array(n.inputs[1]) for n in nodes]
    dtypes = {b.dtype for b in B_list}
    if len(dtypes) != 1:
        if verbose: print("  [MatMul] skip: mixed dtypes")
        return False
    Ks = {b.shape[0] for b in B_list}
    if len(Ks) != 1:
        if verbose: print("  [MatMul] skip: K mismatch")
        return False
    if any(b.ndim != 2 for b in B_list):
        if verbose: print("  [MatMul] skip: non-2D weight")
        return False

    # Concat weights
    W_cat = np.concatenate(B_list, axis=1)
    _Aname = getattr(A, "name", None) or "A"
    W_cat_c = gs.Constant(name=f"{_Aname}_Wcat", values=W_cat)

    # New MatMul
    Y_cat = gs.Variable(name=f"{_Aname}_Ycat_mm", dtype=B_list[0].dtype)
    mm_cat = gs.Node(op="MatMul", inputs=[A, W_cat_c], outputs=[Y_cat])

    # Split
    splits = [b.shape[1] for b in B_list]
    split_sizes_c = gs.Constant(name=f"{_Aname}_split_sizes_mm", values=np.array(splits, dtype=np.int64))
    split_outs = []
    for n in nodes:
        tgt = gs.Variable(name=f"{n.outputs[0].name}_fused", dtype=B_list[0].dtype, shape=n.outputs[0].shape)
        split_outs.append(tgt)
    split_node = gs.Node(op="Split", inputs=[Y_cat, split_sizes_c], outputs=split_outs, attrs={"axis": -1})

    # Insert & rewire
    # Helper: replace all consumer inputs from old tensor to new tensor
    def _rewire_tensor(old_t: gs.Tensor, new_t: gs.Tensor):
        # Rewire node inputs in place
        for cons in g.nodes:
            if cons.inputs:
                for i, inp in enumerate(cons.inputs):
                    if inp is old_t:
                        cons.inputs[i] = new_t
        # Rewire graph outputs in place
        if g.outputs:
            for i, out in enumerate(g.outputs):
                if out is old_t:
                    g.outputs[i] = new_t

    g.nodes += [mm_cat, split_node]
    for old_node, new_out in zip(nodes, split_outs):
        old_out = old_node.outputs[0]
        _rewire_tensor(old_out, new_out)
    # Explicitly remove only the fused MatMul nodes (preserve others like Relu)
    for old_node in nodes:
        try:
            if old_node in g.nodes:
                g.nodes.remove(old_node)
        except Exception:
            pass

    if verbose:
        names = [n.name or n.outputs[0].name for n in nodes]
        print(f"  [MatMul] fused {len(nodes)} nodes: {names}")
    return True


def fuse_gemm_group(A: gs.Tensor, nodes: List[gs.Node], g: gs.Graph, verbose: bool = True) -> bool:
    """
    Fuse Gemm(A, B_i, C_i) nodes with constant weights (and optional constant biases).
    Supports non-default alpha/beta by folding: B_i <- alpha_i * B_i, C_i <- beta_i * C_i.
    Requires transA/transB to be identical across the group.
    """
    if len(nodes) < 2:
        return False

    # Gather attrs and constants
    Bs, Cs, alphas, betas, tAs, tBs = [], [], [], [], [], []
    for n in nodes:
        B = get_const_array(n.inputs[1])
        C = get_const_array(n.inputs[2]) if len(n.inputs) >= 3 and is_const(n.inputs[2]) else None
        Bs.append(B)
        Cs.append(C)
        # attrs values may be typed as 'object' by graphsurgeon; cast to numeric-friendly types before conversion
        alphas.append(float(cast(Union[float, int, str], n.attrs.get("alpha", 1.0))))
        betas.append(float(cast(Union[float, int, str], n.attrs.get("beta", 1.0))))
        tAs.append(int(cast(Union[int, float, str], n.attrs.get("transA", 0))))
        tBs.append(int(cast(Union[int, float, str], n.attrs.get("transB", 0))))

    # Ensure consistent dtype and trans flags
    dtypes = {b.dtype for b in Bs}
    if len(dtypes) != 1:
        if verbose: print("  [Gemm] skip: mixed dtypes")
        return False
    if len(set(tAs)) != 1 or len(set(tBs)) != 1:
        if verbose: print("  [Gemm] skip: mixed trans flags")
        return False
    tA = tAs[0]
    tB = tBs[0]

    # Check shapes and fold alpha/beta
    B_eff_list = []
    C_eff_list = []
    K_vals = []
    M_vals = []
    for B, C, a, b in zip(Bs, Cs, alphas, betas):
        if B.ndim != 2:
            if verbose: print("  [Gemm] skip: non-2D B")
            return False
        if tB == 0:
            K, M = B.shape
            M_vals.append(M)
            K_vals.append(K)
            B_eff = B * a
        else:
            M, K = B.shape
            M_vals.append(M)
            K_vals.append(K)
            B_eff = B * a
        if C is not None:
            if C.ndim != 1 or C.shape[0] != M:
                if verbose: print("  [Gemm] skip: bad bias shape")
                return False
            C_eff = C * b
        else:
            # create zeros bias
            C_eff = np.zeros((M,), dtype=Bs[0].dtype)
        B_eff_list.append(B_eff)
        C_eff_list.append(C_eff)

    if len(set(K_vals)) != 1:
        if verbose: print("  [Gemm] skip: K mismatch")
        return False

    # Concatenate along output dimension M
    axis = 1 if tB == 0 else 0
    B_cat = np.concatenate(B_eff_list, axis=axis)
    C_cat = np.concatenate(C_eff_list, axis=0)

    _Aname = getattr(A, "name", None) or "A"
    B_cat_c = gs.Constant(name=f"{_Aname}_Bcat", values=B_cat)
    C_cat_c = gs.Constant(name=f"{_Aname}_Ccat", values=C_cat)
    # alpha=1, beta=1 after folding; keep trans flags
    Y_cat = gs.Variable(name=f"{_Aname}_Ycat_gemm", dtype=Bs[0].dtype)
    gemm_cat = gs.Node(op="Gemm", inputs=[A, B_cat_c, C_cat_c], outputs=[Y_cat],
                       attrs={"alpha": 1.0, "beta": 1.0, "transA": tA, "transB": tB})

    # Prepare Split sizes by M
    splits = M_vals
    split_sizes_c = gs.Constant(name=f"{_Aname}_split_sizes_gemm", values=np.array(splits, dtype=np.int64))
    split_outs = []
    for n in nodes:
        # Preserve original output shape so that graph outputs remain well-typed
        out_shape = getattr(n.outputs[0], "shape", None)
        tgt = gs.Variable(name=f"{n.outputs[0].name}_fused", dtype=Bs[0].dtype, shape=out_shape)
        split_outs.append(tgt)
    split_node = gs.Node(op="Split", inputs=[Y_cat, split_sizes_c], outputs=split_outs, attrs={"axis": -1})

    # Rewire
    def _rewire_tensor(old_t: gs.Tensor, new_t: gs.Tensor):
        # replace uses of old_t in node inputs (in place)
        for cons in g.nodes:
            if cons.inputs:
                for i, inp in enumerate(cons.inputs):
                    if inp is old_t:
                        cons.inputs[i] = new_t
        # also update graph outputs that may point to old_t
        if g.outputs:
            for i, out in enumerate(g.outputs):
                if out is old_t:
                    g.outputs[i] = new_t

    g.nodes += [gemm_cat, split_node]
    for old_node, new_out in zip(nodes, split_outs):
        old_out = old_node.outputs[0]
        _rewire_tensor(old_out, new_out)
    # Explicitly remove only the fused Gemm nodes
    for old_node in nodes:
        try:
            if old_node in g.nodes:
                g.nodes.remove(old_node)
        except Exception:
            pass

    if verbose:
        names = [n.name or n.outputs[0].name for n in nodes]
        print(f"  [Gemm] fused {len(nodes)} nodes: {names}")
    return True


# ------------------------------
# Main fusion pipeline
# ------------------------------
def horizontal_fusion_with_dependency(in_path: str,
                                      out_path: str,
                                      try_matmul: bool = True,
                                      try_gemm: bool = True,
                                      verbose: bool = True) -> None:
    model = onnx.load(in_path)
    graph = gs.import_onnx(model)

    if verbose:
        print(f"[load] nodes={len(graph.nodes)}, tensors={len(graph.tensors())}")

    # Operator histogram before changes
    def _op_hist(g: gs.Graph) -> Dict[str, int]:
        h: Dict[str, int] = defaultdict(int)
        for n in g.nodes:
            h[n.op] += 1
        return dict(h)
    ops_before = _op_hist(graph)
    if verbose:
        print("[ops] before:", ops_before)

    dep = DepGraph(graph)
    if verbose:
        dep.dump_summary(limit=15)

    changed = False

    # 1) MatMul groups by shared A
    if try_matmul:
        mm_groups = collect_matmul_groups(graph)
        print(f"mm_groups: {mm_groups}")
        for _, (A, nodes) in mm_groups.items():
            # Partition into independent subsets
            for group in filter_independent_siblings(nodes, dep):
                # Optional: further sanity checks can be added here
                changed |= fuse_matmul_group(A, group, graph, verbose=verbose)

    # Rebuild dep graph if changed, before Gemm (avoid cleanup to preserve non-contributing nodes)
    if changed:
        graph.toposort()
        dep = DepGraph(graph)

    # 2) Gemm groups by shared A
    if try_gemm:
        gm_groups = collect_gemm_groups(graph)
        print(f"gm_groups: {gm_groups}")
        for _, (A, nodes) in gm_groups.items():
            for group in filter_independent_siblings(nodes, dep):
                print(f"group: {group}")
                changed |= fuse_gemm_group(A, group, graph, verbose=verbose)

    # Finalize (avoid cleanup to preserve nodes like Relu that may be disconnected from outputs)
    graph.toposort()
    ops_after = _op_hist(graph)
    if verbose:
        print("[ops] after:", ops_after)
        # Print simple diff
        all_ops = sorted(set(list(ops_before.keys()) + list(ops_after.keys())))
        diff = {op: ops_after.get(op,0) - ops_before.get(op,0) for op in all_ops}
        print("[ops] delta:", {k:v for k,v in diff.items() if v!=0})
    out_model = gs.export_onnx(graph)
    oc.check_model(out_model)
    onnx.save(out_model, out_path)

    if verbose:
        print(f"[save] wrote: {out_path}")
        print(f"[summary] changed={changed}, nodes={len(graph.nodes)}, tensors={len(graph.tensors())}")




# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Horizontal Fusion with Dependency Analysis (MatMul/Gemm)")
    ap.add_argument("--in", dest="in_path", type=str, required=False, default="onnx_out/parallel_gemm.onnx")
    ap.add_argument("--out", dest="out_path", type=str, required=False, default="onnx_out/parallel_gemm_fused.onnx")
    ap.add_argument("--no-matmul", action="store_true", help="disable MatMul fusion")
    ap.add_argument("--no-gemm", action="store_true", help="disable Gemm fusion")
    return ap.parse_args()

def main():
    args = parse_args()

    horizontal_fusion_with_dependency(
        in_path=args.in_path,
        out_path=args.out_path,
        try_matmul=not args.no_matmul,
        try_gemm=not args.no_gemm,
        verbose=True,
    )

if __name__ == "__main__":
    main()