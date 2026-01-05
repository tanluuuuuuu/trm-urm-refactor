import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.mrm.mrm import MoE, build_experts
import torch
import torch.nn as nn

# Define parameters (matching run_mamba.py)
batch_size = 2
d_model = 768
d_hidden = 3072
seq_len = 81

# Build experts: 4 ConvSwiGLU + 2 GEGLU + 1 Identity
experts = build_experts(
    d_model=d_model,
    d_hidden=d_hidden,
    num_conv=4,
    num_ffn=2
)

# Create MoE model
moe = MoE(experts=experts, d_model=d_model, top_k=2).to("cuda")

# Create input hidden states
hidden_states = torch.randn(batch_size, seq_len, d_model).to("cuda")

# Forward pass
output = moe(hidden_states)

print(f"Output shape: {output.shape}")
print(f"Number of experts: {len(experts)}")
