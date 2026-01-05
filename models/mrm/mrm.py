from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, MambaBlock, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, ConvSwiGLU
from models.sparse_embedding import CastedSparseEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MoE(nn.Module):
    def __init__(self, experts, d_model, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(d_model, len(experts))
        self.top_k = top_k

    def forward(self, x):
        # x: (B, T, D)
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)

        out = torch.zeros_like(x)

        for k in range(self.top_k):
            idx = topk_idx[..., k]
            prob = topk_probs[..., k]

            for i, expert in enumerate(self.experts):
                mask = idx == i
                if mask.any():
                    out[mask] += expert(x[mask]) * prob[mask].unsqueeze(-1)

        return out


class GEGLU(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.gelu(b) * a)


def build_experts(
    d_model,
    d_hidden,
    num_conv,
    num_ffn,
    kernel_size=3
):
    experts = []

    for _ in range(num_conv):
        experts.append(
            ConvSwiGLU(
                d_hidden,
                d_model,
            )
        )

    for _ in range(num_ffn):
        experts.append(GEGLU(d_model, d_hidden))

    experts.append(nn.Identity())  # stability expert
    return experts


class MRMBlock(nn.Module):
    def __init__(
        self,
        d_model,
        experts,
        use_attention=False,
        attn_heads=8
    ):
        super().__init__()

        self.mamba = MambaBlock(d_model)
        self.moe = MoE(experts, d_model)

        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(
                d_model, attn_heads, batch_first=True
            )
            self.norm_attn = nn.LayerNorm(d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x

        x = self.mamba(x)
        x = self.moe(x)

        if self.use_attention:
            attn_out, _ = self.attn(x, x, x)
            x = self.norm_attn(x + attn_out)

        return self.norm(x + residual)


class MRM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        d_hidden=3072,
        n1=4,   # stage 1 depth
        n2=6,   # stage 2 depth
        n3=2    # stage 3 depth
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        # ===== Stage 1: Local Understanding =====
        experts_s1 = build_experts(
            d_model, d_hidden,
            num_conv=4,
            num_ffn=2
        )
        self.stage1 = nn.Sequential(*[
            MRMBlock(
                d_model,
                experts_s1,
                use_attention=False
            ) for _ in range(n1)
        ])

        # ===== Stage 2: Latent Refinement =====
        experts_s2 = build_experts(
            d_model, d_hidden,
            num_conv=3,
            num_ffn=3
        )
        self.stage2 = nn.Sequential(*[
            MRMBlock(
                d_model,
                experts_s2,
                use_attention=False
            ) for _ in range(n2)
        ])

        # ===== Stage 3: Global Reasoning =====
        experts_s3 = build_experts(
            d_model, d_hidden,
            num_conv=1,
            num_ffn=4
        )
        self.stage3 = nn.Sequential(*[
            MRMBlock(
                d_model,
                experts_s3,
                use_attention=True
            ) for _ in range(n3)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        return self.lm_head(x)
