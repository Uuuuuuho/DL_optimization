#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export example ONNX models for Horizontal Fusion experiments.

Models:
  1) parallel_matmul.onnx
     - Two bias-free MatMul ops that share the same input X.
  2) parallel_gemm.onnx
     - Two Linear(+bias) ops -> ONNX Gemm nodes that share the same input X.
  3) mixed_dependent.onnx
     - MatMul(X,W1), MatMul(X,W2), MatMul(X,W3) with an extra dependency path:
       Z = Relu(MatMul(X,W1)), and (optionally) Y_dep = MatMul(Z, Wd)
       (병렬 후보 + 의존 경로를 함께 갖는 그래프)

Run:
  python export_examples.py --outdir ./onnx_out --batch 4 --in-feat 128 --m1 64 --m2 96 --m3 32 --opset 13
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx.checker as oc
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from typing import Any, Dict, List, Tuple, Union, Optional
import pdb

# -----------------------------
# 0) Symmetric Transformer
# -----------------------------
class SftLayer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(SftLayer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        # d_div = 2
        # d_inter = max(16, min(d_model + d_model + d_edge, d_model)//d_div)
        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model, bias=False),
            # nn.Linear(d_model + d_model + d_edge, d_inter, bias=False),
            # nn.Linear(d_inter, d_model, bias=False),
            nn.LayerNorm(d_model, bias=False),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge, bias=False),
                nn.LayerNorm(d_edge, bias=False),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge, bias=False)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn, bias=False)
        # d_inter = max(16, min(d_model, d_ffn)//d_div)
        # self.linear1 = nn.Sequential(
        #     nn.Linear(d_model, d_inter, bias=False),
        #     nn.Linear(d_inter, d_ffn, bias=False)
        # )
        
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model, bias=False)
        # d_inter = max(16, min(d_ffn, d_model)//d_div)
        # self.linear2 = nn.Sequential(
        #     nn.Linear(d_ffn, d_inter, bias=False),
        #     nn.Linear(d_inter, d_model, bias=False)
        # )

        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.norm3 = nn.LayerNorm(d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        
        decoderlayer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoderlayer, 1)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)
                edge:       (N, N, d_model)
                edge_mask:  (N, N)
        '''
        # update node
        # pdb.set_trace()
        x, edge, memory = self._build_memory(node, edge)
        # print(f"node.shape/edge.shape: {node.shape}/{edge.shape}")
        # x = self.decoder(x, memory).squeeze() # 'edge_mask' is not considered
        
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge, None

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :param  (1, N, d_model)
                :param  (N, N, d_edge)
                :param  (N, N, d_model)
        '''
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model]
                mem:                [N, N, d_model]
                attn_mask:          [N, N]
                key_padding_mask:   [N, N]
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''
        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(SftLayer(device=device,
                                   d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            x: (N, d_model)
            edge: (d_model, N, N)
            edge_mask: (N, N)
        '''
        # attn_multilayer = []
        ##########################################
        # Only use save tensor API when exporting
        ##########################################
        # save_tensor_inputs_for_onnx([x, edge, edge_mask],
        #                     ["SftLayer_node.pt", "SftLayer_edge.pt", "SftLayer_edge_mask.pt"])
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)
            # attn_multilayer.append(attn)
        return x, None

# -----------------------------
# 0-b) Vanilla Transformer Decoder
# -----------------------------
class TransformerDecoderModel(nn.Module):
    """
    Thin wrapper around PyTorch nn.TransformerDecoder for ONNX export.
    Inputs use batch_first=False: shapes (T, N, E) and (S, N, E).
    Returns the decoded sequence (T, N, E).
    """
    def __init__(self,
                 d_model: int = 128,
                 n_head: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_first: bool = False):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        # Export-friendly forward without masks/kv padding for portability
        return self.decoder(tgt, memory)

# -----------------------------
# 1) Parallel MatMul (no bias)
# -----------------------------
class ParallelMatMul(nn.Module):
    """
    Forward:
      Y1 = X @ W1   (W1: [K, M1])
      Y2 = X @ W2   (W2: [K, M2])
    Return: (Y1, Y2)
    """
    def __init__(self, x: int, in_features: int, m1: int, m2: int):
        super().__init__()
        # bias 없는 선형 연산을 'MatMul'로 내보내기 위해 torch.matmul 사용
        self.W1 = nn.Parameter(torch.randn(in_features, m1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(in_features, m2) * 0.02)
        
        self.fc1 = nn.Sequential(
            nn.Linear(x, in_features),
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, in_features)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(x, in_features),
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x):
        # y1 = torch.matmul(x, self.W1)  # -> ONNX MatMul
        # y2 = torch.matmul(x, self.W2)  # -> ONNX MatMul
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return y1, y2


# -----------------------------
# 2) Parallel Gemm (Linear + bias)
# -----------------------------
class ParallelGemm(nn.Module):
    """
    Forward:
      Y1 = Linear(X) with bias -> typically ONNX Gemm
      Y2 = Linear(X) with bias -> typically ONNX Gemm
    """
    def __init__(self, in_features: int, m1: int, m2: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, m1, bias=True)
        self.fc2 = nn.Linear(in_features, m2, bias=True)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return y1, y2


# -----------------------------------------
# 3) Mixed graph with dependency path
# -----------------------------------------
class MixedDependent(nn.Module):
    """
    Forward:
      Y1 = X @ W1
      Z  = Relu(Y1)                    # dependency path from Y1
      Y2 = X @ W2                      # independent sibling w.r.t. Y3
      Y3 = X @ W3
      (optional) Y_dep = Z @ Wd        # adds more dependencies (kept internal)
    Returns: (Y1, Y2, Y3)  # 외부 출력은 병렬 타깃들 유지
    """
    def __init__(self, in_features: int, m1: int, m2: int, m3: int, add_dep_matmul: bool = True):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(in_features, m1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(in_features, m2) * 0.02)
        self.W3 = nn.Parameter(torch.randn(in_features, m3) * 0.02)
        self.add_dep_matmul = add_dep_matmul
        if add_dep_matmul:
            # Z:[N,m1] @ Wd:[m1, m3] -> [N, m3] (단지 의존 경로용. 출력으로 내보내진 않음)
            self.Wd = nn.Parameter(torch.randn(m1, m3) * 0.02)

    def forward(self, x):
        y1 = torch.matmul(x, self.W1)      # MatMul
        z  = torch.relu(y1)                # dependency path from y1
        if self.add_dep_matmul:
            _y_dep = torch.matmul(z, self.Wd)  # not returned; ensures path exists
        y2 = torch.matmul(x, self.W2)      # MatMul (independent of y1 path)
        y3 = torch.matmul(x, self.W3)      # MatMul (independent of y1 path)
        return y1, y2, y3


def export_onnx(model: nn.Module,
                example_input: torch.Tensor,
                out_path: str,
                opset: int = 13,
                dynamic: bool = True,
                names = None):
    model.eval()
    out_path = str(out_path)
    dynamic_axes = None
    input_names  = ["X"]
    if names is not None:
        output_names = names
    else:
        # fallback: y0, y1, ...
        with torch.no_grad():
            tmp = model(example_input)
        if isinstance(tmp, (tuple, list)):
            output_names = [f"Y{i}" for i in range(len(tmp))]
        else:
            output_names = ["Y"]

    if dynamic:
        # 첫 번째 차원(batch)을 동적으로
        dynamic_axes = {"X": {0: "N"}}
        for n in output_names:
            dynamic_axes[n] = {0: "N"}

    torch.onnx.export(
        model, example_input, out_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 간단 체크
    m = onnx.load(out_path)
    oc.check_model(m)
    print(f"[ok] exported: {out_path}")
    print(f"    inputs : {[i.name + str(i.type.tensor_type.shape.dim[0].dim_param or i.type.tensor_type.shape.dim[0].dim_value) for i in m.graph.input]}")
    print(f"    outputs: {[o.name for o in m.graph.output]}")

def export_transformer_decoder_onnx(model: nn.Module,
                                    tgt: Tensor,
                                    memory: Tensor,
                                    out_path: str,
                                    opset: int = 13,
                                    dynamic: bool = True):
    """Export a Transformer decoder with (tgt, memory) inputs to ONNX."""
    model.eval()
    out_path = str(out_path)
    input_names = ["tgt", "memory"]
    output_names = ["out"]
    dynamic_axes = None
    if dynamic:
        # Allow variable time dims (T for tgt, S for memory) and batch N
        dynamic_axes = {
            "tgt": {0: "T", 1: "N"},
            "memory": {0: "S", 1: "N"},
            "out": {0: "T", 1: "N"},
        }

    torch.onnx.export(
        model,
        (tgt, memory),
        out_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    m = onnx.load(out_path)
    oc.check_model(m)
    print(f"[ok] exported: {out_path}")
    print(f"    inputs : {[i.name for i in m.graph.input]}")
    print(f"    outputs: {[o.name for o in m.graph.output]}")

class SimplConverter():
    def __init__(self):
        super().__init__()

    def onnx_convert(self, model, filepath, N=300):
        model.to("cpu")
        model.eval()
        if isinstance(model, SymmetricFusionTransformer):
            st_in_tokens = torch.rand(N, 128, dtype=torch.float32)      # x: (N, d_model)
            st_in_edge = torch.rand(N, N, 128, dtype=torch.float32)     # edge: (N, N, d_edge)
            st_in_mask = torch.randint(0, 2, (N, N)).bool()             # edge_mask: (N, N)

            input_sample = (st_in_tokens, st_in_edge, st_in_mask)
            input_names = ['tokens', 'rpe', 'rpes']
            output_names = ['out']

            torch.onnx.export(
                model,
                input_sample,
                str(filepath),
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=True,
                export_params=True,
            )
            return
        raise NotImplementedError("SimplConverter only supports SymmetricFusionTransformer")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./onnx_out")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--in-feat", type=int, default=2048)
    ap.add_argument("--m1", type=int, default=2048)
    ap.add_argument("--m2", type=int, default=2048)
    ap.add_argument("--m3", type=int, default=32)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dynamic", action="store_true", help="disable dynamic axes")
    ap.add_argument("--tgt-len", type=int, default=16, help="target sequence length for decoder export")
    ap.add_argument("--src-len", type=int, default=16, help="memory/source sequence length for decoder export")
    ap.add_argument("--decoder-layers", type=int, default=2, help="number of decoder layers")
    ap.add_argument("--decoder-heads", type=int, default=8, help="number of attention heads (d_model must be divisible)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N, K = args.batch, args.in_feat
    x = torch.randn(N, K, dtype=torch.float32)

    # 0) SymmetricFT.onnx
    st_model = SymmetricFusionTransformer('cuda:0')
    converter = SimplConverter()
    converter.onnx_convert(st_model, str(outdir / "SymmetricFT.onnx"))
    
    # 0-b) transformer_decoder.onnx
    # d_model uses K to match feature size; ensure divisible by heads
    if (K % args.decoder_heads) != 0:
        print(f"[warn] in-feat ({K}) is not divisible by decoder-heads ({args.decoder_heads}); adjusting heads to 1 for export.")
        heads = 1
    else:
        heads = args.decoder_heads
    dec_model = TransformerDecoderModel(d_model=K, n_head=heads, num_layers=args.decoder_layers, dim_feedforward=max(4*K, 256))
    T, S, B = args.tgt_len, args.src_len, N
    tgt = torch.randn(T, B, K, dtype=torch.float32)
    mem = torch.randn(S, B, K, dtype=torch.float32)
    export_transformer_decoder_onnx(
        dec_model,
        tgt,
        mem,
        str(outdir / "transformer_decoder.onnx"),
        opset=args.opset,
        dynamic=not args.no_dynamic,
    )
    
    # 1) parallel_matmul.onnx
    m1 = ParallelMatMul(x=K, in_features=K, m1=args.m1, m2=args.m2)
    export_onnx(
        m1, x, str(outdir / "parallel_matmul.onnx"),
        opset=args.opset,
        dynamic=not args.no_dynamic,
        names=["Y1", "Y2"]
    )

    # # 2) parallel_gemm.onnx
    # m2 = ParallelGemm(in_features=K, m1=args.m1, m2=args.m2)
    # export_onnx(
    #     m2, x, outdir / "parallel_gemm.onnx",
    #     opset=args.opset,
    #     dynamic=not args.no_dynamic,
    #     names=["Y1", "Y2"]
    # )

    # # 3) mixed_dependent.onnx
    # m3 = MixedDependent(in_features=K, m1=args.m1, m2=args.m2, m3=args.m3, add_dep_matmul=True)
    # export_onnx(
    #     m3, x, outdir / "mixed_dependent.onnx",
    #     opset=args.opset,
    #     dynamic=not args.no_dynamic,
    #     names=["O1", "O2", "O3"]
    # )

    print("\n[hint] 이제 수평 병합 스크립트를 적용해보세요:")
    print("  python hfuse.py --in ./onnx_out/parallel_matmul.onnx --out ./onnx_out/parallel_matmul_fused.onnx")
    # print("  python hfuse.py --in ./onnx_out/parallel_gemm.onnx   --out ./onnx_out/parallel_gemm_fused.onnx")
    # print("  python hfuse.py --in ./onnx_out/mixed_dependent.onnx --out ./onnx_out/mixed_dependent_fused.onnx")


if __name__ == "__main__":
    main()

