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
    # Return mapping: key -> (A_variable, [nodes]) where key is canonicalized A
    def _tensor_key(t: gs.Tensor):
        return getattr(t, "name", None) or id(t)

    # Build producer map to unwrap Identity chains
    prod_by_t = {}
    for node in g.nodes:
        for t in node.outputs:
            if isinstance(t, gs.Tensor):
                prod_by_t[_tensor_key(t)] = node

    def _unwrap_identity_or_split(t: gs.Tensor) -> gs.Tensor:
        seen = 0
        cur = t
        while seen < 16 and isinstance(cur, gs.Tensor):
            p = prod_by_t.get(_tensor_key(cur))
            if p is None or not p.inputs:
                break
            if p.op == "Identity":
                src = p.inputs[0]
                if not isinstance(src, gs.Tensor):
                    break
                cur = src
                seen += 1
                continue
            if p.op == "Split":
                # treat split outputs as sharing the same base input for grouping
                src = p.inputs[0]
                if not isinstance(src, gs.Tensor):
                    break
                cur = src
                seen += 1
                continue
            break
        return cur

    nodes_map: Dict[object, List[gs.Node]] = defaultdict(list)
    key_to_A: Dict[object, gs.Variable] = {}
    for n in g.nodes:
        if n.op != "MatMul":
            continue
        if len(n.inputs) != 2:
            continue
        A_in, B_in = n.inputs
        A_in = _unwrap_identity_or_split(A_in)
        B_in = _unwrap_identity_or_split(B_in)
        if not isinstance(A_in, gs.Variable):
            continue
        if not is_const(B_in):
            continue
        Barr = get_const_array(B_in)
        if Barr.ndim != 2:
            continue
        key = getattr(A_in, "name", None) or id(A_in)
        key_to_A[key] = A_in
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

    def _unwrap_identity_or_split(t: gs.Tensor) -> gs.Tensor:
        # Walk back through chains of Identity producers to find the canonical source tensor
        seen = 0
        cur = t
        while seen < 16 and isinstance(cur, gs.Tensor):
            p = prod_by_t.get(_tensor_key(cur))
            if p is None or not p.inputs:
                break
            if p.op == "Identity":
                src = p.inputs[0]
                if not isinstance(src, gs.Tensor):
                    break
                cur = src
                seen += 1
                continue
            if p.op == "Split":
                # Consider outputs of the same Split as sharing the input for grouping
                src = p.inputs[0]
                if not isinstance(src, gs.Tensor):
                    break
                cur = src
                seen += 1
                continue
            break
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
            # Canonicalize A by unwrapping Identity and Split chains
            A = _unwrap_identity_or_split(A)
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
            conflict[i][j] = dep.has_path(a, b)
            conflict[j][i] = dep.has_path(b, a)

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

    # Detect optional bias Adds that immediately consume each MatMul
    bias_list = []  # may contain None if no bias for that node
    add_nodes_map = {}  # old_matmul_node -> add_node (if exists)
    has_any_bias = False
    for n, B in zip(nodes, B_list):
        bias_arr = None
        add_node = None
        # find consumers of this matmul output
        mat_out = n.outputs[0]
        consumers = [nd for nd in g.nodes if nd.inputs and any(inp is mat_out for inp in nd.inputs)]
        if len(consumers) == 1 and consumers[0].op == "Add":
            cand = consumers[0]
            # identify constant input among Add inputs
            const_inp = None
            for inp in cand.inputs:
                if is_const(inp):
                    const_inp = inp
                    break
            if const_inp is not None:
                carr = get_const_array(const_inp)
                # accept 1D bias whose length matches output dim (M)
                M = B.shape[1]
                if carr.ndim == 1 and carr.shape[0] == M:
                    bias_arr = carr
                    add_node = cand
        bias_list.append(bias_arr)
        if add_node is not None:
            add_nodes_map[n] = add_node
            has_any_bias = True

    # New MatMul
    Y_cat = gs.Variable(name=f"{_Aname}_Ycat_mm", dtype=B_list[0].dtype)
    mm_cat = gs.Node(op="MatMul", inputs=[A, W_cat_c], outputs=[Y_cat])

    # Split
    splits = [b.shape[1] for b in B_list]
    split_sizes_c = gs.Constant(name=f"{_Aname}_split_sizes_mm", values=np.array(splits, dtype=np.int64))
    split_outs = []
    for idx, n in enumerate(nodes):
        tgt = gs.Variable(name=f"{n.outputs[0].name}_fused", dtype=B_list[0].dtype, shape=n.outputs[0].shape)
        split_outs.append(tgt)
    split_node = gs.Node(op="Split", inputs=[Y_cat, split_sizes_c], outputs=split_outs, attrs={"axis": -1})

    # If biases exist, create Add nodes that add per-split bias constants
    add_outs = []
    add_nodes = []
    if has_any_bias:
        for idx, b in enumerate(bias_list):
            if b is None:
                # no bias for this slot -> passthrough (use split_out directly)
                add_outs.append(split_outs[idx])
            else:
                bias_c = gs.Constant(name=f"{_Aname}_bias_{idx}", values=b)
                add_out = gs.Variable(name=f"{split_outs[idx].name}_with_bias", dtype=B_list[0].dtype, shape=split_outs[idx].shape)
                add_node = gs.Node(op="Add", inputs=[split_outs[idx], bias_c], outputs=[add_out])
                add_nodes.append((bias_c, add_node))
                add_outs.append(add_out)

    # Insert new nodes (do not append Constant tensors to g.nodes; only nodes)
    new_nodes = [mm_cat, split_node]
    if has_any_bias:
        for _, add_n in add_nodes:
            new_nodes.append(add_n)
    g.nodes += new_nodes

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

    # Rewire original outputs (may be from MatMul directly or from Add)
    for old_node, new_out in zip(nodes, add_outs if has_any_bias else split_outs):
        # determine the original public-facing tensor to replace
        orig_final = old_node.outputs[0]
        # if there was an Add consuming the matmul, use that Add's output as the original final
        add_n = add_nodes_map.get(old_node)
        if add_n is not None:
            orig_final = add_n.outputs[0]
        _rewire_tensor(orig_final, new_out)

    # Remove old MatMul and their Add nodes (if any)
    for old_node in nodes:
        try:
            if old_node in g.nodes:
                g.nodes.remove(old_node)
        except Exception:
            pass
        add_n = add_nodes_map.get(old_node)
        if add_n is not None:
            try:
                if add_n in g.nodes:
                    g.nodes.remove(add_n)
            except Exception:
                pass

    if verbose:
        names = [n.name or n.outputs[0].name for n in nodes]
        print(f"  [MatMul] fused {len(nodes)} nodes: {names}")
    return True


# ------------------------------
# Cleanup: remove trivially dead nodes
# ------------------------------
def prune_dead_nodes(g: gs.Graph, max_passes: int = 3, verbose: bool = False) -> None:
    """Remove nodes whose outputs are unused by any node and are not graph outputs.
    Runs a few passes to reach a small fixed point. Conservative and op-agnostic.
    """
    def count_consumers() -> Dict[gs.Tensor, int]:
        cnt: Dict[gs.Tensor, int] = defaultdict(int)
        for nd in g.nodes:
            for inp in (nd.inputs or []):
                if isinstance(inp, gs.Tensor):
                    cnt[inp] += 1
        for out in (g.outputs or []):
            if isinstance(out, gs.Tensor):
                cnt[out] += 1
        return cnt

    passes = 0
    while passes < max_passes:
        passes += 1
        cons = count_consumers()
        to_remove = []
        for nd in g.nodes:
            outs = [t for t in (nd.outputs or []) if isinstance(t, gs.Tensor)]
            if not outs:
                # nodes with no outputs are removable
                to_remove.append(nd)
                continue
            all_unused = True
            for t in outs:
                if t in (g.outputs or []):
                    all_unused = False; break
                if cons.get(t, 0) > 0:
                    all_unused = False; break
            if all_unused:
                to_remove.append(nd)
        if not to_remove:
            break
        for nd in to_remove:
            try:
                g.nodes.remove(nd)
                if verbose:
                    print(f"[prune] removed dead node: {nd.name or nd.op}")
            except Exception:
                pass
    # reorder after pruning
    try:
        g.toposort()
    except Exception:
        pass


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

    # Optional cleanup: remove orphan Split producers that fed the old nodes
    try:
        # Build quick maps
        prod_by_tensor = {}
        for nd in g.nodes:
            for t in nd.outputs or []:
                if isinstance(t, gs.Tensor):
                    prod_by_tensor[t] = nd
        # Count consumers for a tensor
        def _consumer_count(t: gs.Tensor) -> int:
            cnt = 0
            for nd in g.nodes:
                for inp in (nd.inputs or []):
                    if inp is t:
                        cnt += 1
            # graph outputs also count
            for out in (g.outputs or []):
                if out is t:
                    cnt += 1
            return cnt
        # Candidates: producers of each old A input
        split_cands = set()
        for old_node in nodes:
            if old_node.inputs:
                a_in = old_node.inputs[0]
                p = prod_by_tensor.get(a_in)
                if p is not None and p.op == "Split":
                    split_cands.add(p)
        # After rewiring, if Split outputs are unused, remove the Split
        for sp in list(split_cands):
            outs = list(sp.outputs or [])
            if outs and all(_consumer_count(t) == 0 for t in outs):
                try:
                    g.nodes.remove(sp)
                except Exception:
                    pass
    except Exception:
        pass

    if verbose:
        names = [n.name or n.outputs[0].name for n in nodes]
        print(f"  [Gemm] fused {len(nodes)} nodes: {names}")
    return True


# ------------------------------
# Pre-fusion rewrite: remove Split before GEMMs by zero-padding weights
# ------------------------------
def try_remove_split_before_gemm_fusion(A: gs.Tensor, nodes: List[gs.Node], g: gs.Graph, verbose: bool = False) -> bool:
    """
    If all GEMMs in nodes take as input different outputs of the same Split whose input is A,
    rewrite those GEMMs to consume A directly with zero-padded B so that Split becomes unnecessary.
    Return True if any change was made.
    """
    if not nodes:
        return False
    # Build producer map: tensor -> producer node
    prod_by_t = {}
    for nd in g.nodes:
        for t in nd.outputs or []:
            if isinstance(t, gs.Tensor):
                prod_by_t[t] = nd

    # Check all nodes' A input comes from the same Split with input A
    split = None
    split_offsets = None  # list of cumulative offsets per output index
    split_sizes = None    # list of sizes per output index
    def get_split_info(sp: gs.Node):
        nonlocal split_offsets, split_sizes
        # determine per-output sizes along the last dimension
        sizes = None
        # prefer explicit split sizes from second input constant
        if len(sp.inputs) >= 2 and is_const(sp.inputs[1]):
            arr = get_const_array(sp.inputs[1])
            if arr.ndim == 1:
                sizes = [int(x) for x in arr.tolist()]
        if sizes is None:
            # fallback: use output tensor shapes
            sizes = []
            for ot in sp.outputs or []:
                shp = getattr(ot, "shape", None)
                if isinstance(shp, (list, tuple)) and len(shp) >= 1 and isinstance(shp[-1], int):
                    sizes.append(int(shp[-1]))
                else:
                    # cannot determine
                    return False
        split_sizes = sizes
        # compute offsets
        offs = []
        acc = 0
        for s in sizes:
            offs.append(acc)
            acc += s
        split_offsets = offs
        return True

    a_input = A
    # verify gs.Tensor
    if not isinstance(a_input, gs.Tensor):
        return False

    # Collect mapping for each node
    per_node_idx = {}
    for n in nodes:
        a_in = n.inputs[0]
        p = prod_by_t.get(a_in)
        if p is None or p.op != "Split":
            return False
        # Split's input must be A
        if not p.inputs or p.inputs[0] is not a_input:
            return False
        if split is None:
            split = p
            if not get_split_info(split):
                return False
        elif split is not p:
            return False
        # find output index
        try:
            idx = list(split.outputs or []).index(a_in)
        except ValueError:
            return False
        per_node_idx[n] = idx

    # K_total is sum of sizes
    K_total = sum(split_sizes) if split_sizes else None
    if K_total is None:
        return False

    # Rewrite each node: pad B to K_total along K dimension and set input to A
    changed = False
    for n in nodes:
        idx = per_node_idx[n]
        off = split_offsets[idx]
        K_i = split_sizes[idx]
        # get attrs
        tB = int(cast(Union[int, float, str], n.attrs.get("transB", 0)))
        # get B and optional C
        B = get_const_array(n.inputs[1])
        if tB == 0:
            # B shape [K_i, M]
            M = B.shape[1]
            B_pad = np.zeros((K_total, M), dtype=B.dtype)
            B_pad[off:off+K_i, :] = B
        else:
            # B shape [M, K_i]
            M = B.shape[0]
            B_pad = np.zeros((M, K_total), dtype=B.dtype)
            B_pad[:, off:off+K_i] = B
        # replace B with padded constant
        B_new = gs.Constant(name=f"{getattr(a_input, 'name', 'A')}_Bpad_{idx}", values=B_pad)
        n.inputs[1] = B_new
        # set A input to pre-split tensor
        n.inputs[0] = a_input
        changed = True

    if changed:
        # If split outputs are now unused, remove the split
        try:
            def _consumer_count(t: gs.Tensor) -> int:
                cnt = 0
                for nd in g.nodes:
                    for inp in (nd.inputs or []):
                        if inp is t:
                            cnt += 1
                for out in (g.outputs or []):
                    if out is t:
                        cnt += 1
                return cnt
            if split is not None:
                outs = list(split.outputs or [])
                if outs and all(_consumer_count(t) == 0 for t in outs):
                    try:
                        g.nodes.remove(split)
                        if verbose:
                            print(f"  [Gemm] removed Split before fusion: {split.name or 'Split'}")
                    except Exception:
                        pass
        except Exception:
            pass
    return changed

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
    any_change = True
    iter_cnt = 0
    # Run passes to a fixed point so later groups also get considered after graph changes
    while any_change and iter_cnt < 6:
        any_change = False
        iter_cnt += 1
        if verbose:
            print(f"[pass] iteration {iter_cnt}")
        # Recompute dep each pass
        dep = DepGraph(graph)
        # 1) MatMul groups by shared A
        if try_matmul:
            mm_groups = collect_matmul_groups(graph)
            if verbose:
                print(f"mm_groups: { {k: len(v[1]) for k,v in mm_groups.items()} }")
            for _, (A, nodes) in mm_groups.items():
                for group in filter_independent_siblings(nodes, dep):
                    if fuse_matmul_group(A, group, graph, verbose=verbose):
                        any_change = True
                        changed = True
                        graph.toposort()
                        prune_dead_nodes(graph, verbose=False)
                        dep = DepGraph(graph)

        # 2) Gemm groups by shared A
        if try_gemm:
            gm_groups = collect_gemm_groups(graph)
            if verbose:
                print(f"gm_groups: { {k: len(v[1]) for k,v in gm_groups.items()} }")
            for _, (A, nodes) in gm_groups.items():
                # pre-rewrite: if nodes are fed by the same Split(A), remove Split by zero-padding B
                if try_remove_split_before_gemm_fusion(A, nodes, graph, verbose=verbose):
                    graph.toposort()
                    prune_dead_nodes(graph, verbose=False)
                    dep = DepGraph(graph)
                for group in filter_independent_siblings(nodes, dep):
                    if fuse_gemm_group(A, group, graph, verbose=verbose):
                        any_change = True
                        changed = True
                        graph.toposort()
                        prune_dead_nodes(graph, verbose=False)
                        dep = DepGraph(graph)

    # Finalize (avoid cleanup to preserve nodes like Relu that may be disconnected from outputs)
    graph.toposort()
    prune_dead_nodes(graph, verbose=False)
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
    ap.add_argument("--in", dest="in_path", type=str, required=False, default="onnx_out/parallel_matmul.onnx")
    ap.add_argument("--out", dest="out_path", type=str, required=False, default="onnx_out/parallel_matmul_fused.onnx")
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