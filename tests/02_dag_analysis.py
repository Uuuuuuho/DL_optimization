#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static DAG Analysis for ONNX models
- Loads an ONNX model
- Builds a dataflow dependency graph (node -> downstream consumers)
- Produces a compiler-style schedule:
    - Topological order
    - Levelized schedule (layers of parallel nodes)
    - Linear chains (maximal 1-in/1-out paths)
- Emits a human-readable DAG expression, JSON, and Graphviz DOT

This tool is read-only: it does not modify or fuse the graph.
"""

import argparse
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Union, cast, Any

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
    return cast(gs.Constant, t).values

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
        self.nodes_by_key: Dict[object, gs.Node] = {}
        self.tensors_by_key: Dict[object, gs.Tensor] = {}
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
                    tk = self._tensor_key(t)
                    self.producer_of[tk] = node
                    self.tensors_by_key[tk] = t
        # Inputs -> consumers
        for node in self.graph.nodes:
            self.nodes_by_key[self._node_key(node)] = node
            for t in node.inputs:
                if isinstance(t, gs.Tensor):
                    tk = self._tensor_key(t)
                    self.consumers_of[tk].append(node)
                    self.tensors_by_key[tk] = t
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

    # Public helpers
    def node_key(self, n: gs.Node) -> object:
        return self._node_key(n)

    def tensor_key(self, t: gs.Tensor) -> object:
        return self._tensor_key(t)

    def topo_order(self) -> List[gs.Node]:
        # Kahn’s algorithm on node graph
        indeg: Dict[object, int] = {k: len(self.upstream.get(k, [])) for k in self.nodes_by_key}
        q = deque([self.nodes_by_key[k] for k, d in indeg.items() if d == 0])
        order: List[gs.Node] = []
        # Build reverse map key->node for fast lookup
        while q:
            n = q.popleft()
            order.append(n)
            nk = self.node_key(n)
            for mkey in list(self.downstream.get(nk, [])):
                indeg[mkey] -= 1
                if indeg[mkey] == 0:
                    q.append(self.nodes_by_key[mkey])
        # If cycle (shouldn't happen in ONNX dataflow), fall back to original order
        if len(order) != len(self.nodes_by_key):
            return list(self.graph.nodes)
        return order

    def levels(self) -> List[List[gs.Node]]:
        # Longest-distance-from-sources levelization
        order = self.topo_order()
        lvl: Dict[object, int] = {}
        for n in order:
            nk = self.node_key(n)
            if not self.upstream.get(nk):
                lvl[nk] = 0
            else:
                lvl[nk] = 1 + max(lvl[pk] for pk in self.upstream[nk])
        maxlvl = max(lvl.values(), default=-1)
        buckets: List[List[gs.Node]] = [[] for _ in range(maxlvl + 1)]
        for n in order:
            buckets[lvl[self.node_key(n)]].append(n)
        return buckets

    def linear_chains(self) -> List[List[gs.Node]]:
        # Maximal chains where each internal node has indegree==1 and outdegree==1
        seen: Set[object] = set()
        chains: List[List[gs.Node]] = []
        for n in self.topo_order():
            nk = self.node_key(n)
            if nk in seen:
                continue
            # start only at nodes that are not strict continuations
            if self.indegree(n) > 1:
                pass
            # start of chain if indegree != 1 or predecessor has outdegree != 1
            start = n
            # extend backward to the earliest node with indegree != 1 or pred outdeg != 1
            while True:
                preds = list(self.upstream.get(self.node_key(start), []))
                if len(preds) != 1:
                    break
                p = self.nodes_by_key[preds[0]]
                if self.outdegree(p) != 1:
                    break
                start = p
            # Now go forward
            cur = start
            chain: List[gs.Node] = [cur]
            while True:
                seen.add(self.node_key(cur))
                succs = list(self.downstream.get(self.node_key(cur), []))
                if len(succs) != 1:
                    break
                nxt = self.nodes_by_key[succs[0]]
                if self.indegree(nxt) != 1:
                    break
                chain.append(nxt)
                cur = nxt
            if len(chain) >= 2:
                chains.append(chain)
        return chains

#############################
# DAG analysis and reporting #
#############################

def _op_hist(g: gs.Graph) -> Dict[str, int]:
    h: Dict[str, int] = defaultdict(int)
    for n in g.nodes:
        h[n.op] += 1
    return dict(h)

def build_edge_list(dep: DepGraph) -> List[Tuple[object, object, Optional[str]]]:
    edges: List[Tuple[object, object, Optional[str]]] = []
    for nk, node in dep.nodes_by_key.items():
        for t in node.outputs:
            if not isinstance(t, gs.Tensor):
                continue
            tk = dep.tensor_key(t)
            for cons in dep.consumers_of.get(tk, []):
                dst = dep.node_key(cons)
                edges.append((nk, dst, getattr(t, "name", None)))
    return edges

def dag_to_text(graph: gs.Graph, dep: DepGraph, hide_identity: bool = True) -> str:
    lines: List[str] = []
    lines.append("# DAG Analysis Report")
    lines.append(f"nodes={len(graph.nodes)} tensors={len(graph.tensors())}")
    # IO summary
    ins = ", ".join([getattr(t, "name", None) or f"<unnamed:{i}>" for i, t in enumerate(graph.inputs)])
    outs = ", ".join([getattr(t, "name", None) or f"<unnamed:{i}>" for i, t in enumerate(graph.outputs)])
    lines.append(f"inputs: {ins}")
    lines.append(f"outputs: {outs}")
    # Ops
    lines.append(f"ops: {_op_hist(graph)}")
    lines.append("")

    # Topological order as S-expressions
    lines.append("[topo]")
    for idx, n in enumerate(dep.topo_order()):
        if hide_identity and n.op == "Identity":
            continue
        in_s = ", ".join([(getattr(i, 'name', None) or i.__class__.__name__) for i in n.inputs])
        out_s = ", ".join([(getattr(o, 'name', None) or o.__class__.__name__) for o in n.outputs])
        lines.append(f"  {idx:04d}: {n.name or n.op} = {n.op}({in_s}) -> [{out_s}]")
    lines.append("")

    # Levels
    lines.append("[levels]")
    for li, bucket in enumerate(dep.levels()):
        if hide_identity:
            bucket = [n for n in bucket if n.op != "Identity"]
        labels = [f"{n.name or n.op}:{n.op}" for n in bucket]
        lines.append(f"  L{li}: {labels}")
    lines.append("")

    # Linear chains
    lines.append("[linear_chains]")
    chains = dep.linear_chains()
    for ci, chain in enumerate(chains):
        if hide_identity:
            chain = [n for n in chain if n.op != "Identity"]
        if not chain:
            continue
        labels = [n.name or n.op for n in chain]
        lines.append(f"  C{ci}: {labels}")
    if not chains:
        lines.append("  (none)")

    return "\n".join(lines)

def dag_to_json(graph: gs.Graph, dep: DepGraph, hide_identity: bool = True) -> Dict[str, Any]:
    nodes = []
    keymap: Dict[object, int] = {}
    for i, n in enumerate(dep.topo_order()):
        if hide_identity and n.op == "Identity":
            continue
        nk = dep.node_key(n)
        keymap[nk] = len(nodes)
        nodes.append({
            "key": str(nk),
            "name": n.name,
            "op": n.op,
            "inputs": [getattr(t, "name", None) for t in n.inputs],
            "outputs": [getattr(t, "name", None) for t in n.outputs],
        })
    edges_json = []
    for src, dst, tname in build_edge_list(dep):
        if src not in keymap or dst not in keymap:
            continue
        edges_json.append({"src": str(src), "dst": str(dst), "tensor": tname})
    levels = []
    for bucket in dep.levels():
        lvl = []
        for n in bucket:
            if hide_identity and n.op == "Identity":
                continue
            lvl.append(str(dep.node_key(n)))
        if lvl:
            levels.append(lvl)
    chains = []
    for chain in dep.linear_chains():
        filtered = [str(dep.node_key(n)) for n in chain if not (hide_identity and n.op == "Identity")]
        if filtered:
            chains.append(filtered)
    return {
        "summary": {
            "nodes": len(graph.nodes),
            "tensors": len(graph.tensors()),
            "ops": _op_hist(graph),
            "inputs": [getattr(t, "name", None) for t in graph.inputs],
            "outputs": [getattr(t, "name", None) for t in graph.outputs],
        },
        "nodes": nodes,
        "edges": edges_json,
        "levels": levels,
        "linear_chains": chains,
    }

def dag_to_dot(graph: gs.Graph, dep: DepGraph, hide_identity: bool = True) -> str:
    # Simple Graphviz DOT
    def label(n: gs.Node) -> str:
        nm = n.name or n.op
        return nm.replace('"', '\"')
    included: Set[object] = set()
    for n in dep.topo_order():
        if hide_identity and n.op == "Identity":
            continue
        included.add(dep.node_key(n))
    lines = ["digraph G {"]
    # nodes
    for nk in included:
        n = dep.nodes_by_key[nk]
        color = "lightgrey" if n.op == "Identity" else "white"
        lines.append(f"  \"{nk}\" [label=\"{label(n)}\\n({n.op})\", style=filled, fillcolor={color}];")
    # edges
    for src, dst, tname in build_edge_list(dep):
        if src not in included or dst not in included:
            continue
        elabel = f" [label=\"{tname}\"]" if tname else ""
        lines.append(f"  \"{src}\" -> \"{dst}\"{elabel};")
    lines.append("}")
    return "\n".join(lines)


def run_dag_analysis(in_path: str,
                     out_prefix: Optional[str] = None,
                     formats: Optional[List[str]] = None,
                     hide_identity: bool = True,
                     verbose: bool = True) -> None:
    model = onnx.load(in_path)
    graph = gs.import_onnx(model)
    if verbose:
        print(f"[load] nodes={len(graph.nodes)}, tensors={len(graph.tensors())}")
    dep = DepGraph(graph)
    if verbose:
        dep.dump_summary(limit=15)

    # Derive default out prefix from input path
    import os
    if not out_prefix:
        base, _ = os.path.splitext(in_path)
        out_prefix = base + "_dag"
    if not formats:
        formats = ["text", "json", "dot"]

    # Generate artifacts
    if "text" in formats:
        txt = dag_to_text(graph, dep, hide_identity=hide_identity)
        with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
            f.write(txt)
        if verbose:
            print(f"[write] {out_prefix}.txt")
    if "json" in formats:
        import json
        js = dag_to_json(graph, dep, hide_identity=hide_identity)
        with open(out_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2)
        if verbose:
            print(f"[write] {out_prefix}.json")
    if "dot" in formats:
        dot = dag_to_dot(graph, dep, hide_identity=hide_identity)
        with open(out_prefix + ".dot", "w", encoding="utf-8") as f:
            f.write(dot)
        if verbose:
            print(f"[write] {out_prefix}.dot")




# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="ONNX Static DAG Analysis")
    ap.add_argument("--in", dest="in_path", type=str, required=False, default="onnx_out/parallel_matmul.onnx",
                    help="Path to ONNX model to analyze")
    ap.add_argument("--out-prefix", dest="out_prefix", type=str, required=False, default=None,
                    help="Output prefix for report files (defaults to <in>_dag)")
    ap.add_argument("--formats", dest="formats", type=str, required=False, default="text,json,dot",
                    help="Comma-separated list of outputs: text,json,dot")
    ap.add_argument("--show-identity", dest="show_identity", action="store_true",
                    help="Include Identity ops in reports")
    return ap.parse_args()

def main():
    args = parse_args()
    formats = [s.strip() for s in (args.formats or "").split(",") if s.strip()]
    run_dag_analysis(
        in_path=args.in_path,
        out_prefix=args.out_prefix,
        formats=formats,
        hide_identity=not args.show_identity,
        verbose=True,
    )

if __name__ == "__main__":
    main()