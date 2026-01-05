from mamba_ssm import Mamba
import torch

# Define parameters
batch_size = 2
hidden_size = 768
seq_len = 81

# d_model must match the input dimension (hidden_size)
mamba = Mamba(
    d_model=hidden_size,
    d_state=16,
    d_conv=4,
    expand=2,
).to("cuda")

hidden_states = torch.randn(batch_size, seq_len, hidden_size).to("cuda")
output = mamba(hidden_states)
print(f"Output shape: {output.shape}")
