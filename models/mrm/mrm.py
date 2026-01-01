from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, ConvSwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, MambaBlock
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class MRMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class NoOpRoPE(nn.Module):
    """No-op rotary embedding for architectures that don't use positional encoding."""
    def forward(self):
        return None


class MRMConfig(BaseModel):
    """
    Configuration for Masked Reconstruction Model (MRM).

    Args:
        batch_size: Batch size per device
        seq_len: Maximum sequence length
        puzzle_emb_ndim: Dimension of puzzle embedding (0 to disable)
        num_puzzle_identifiers: Number of unique puzzle IDs for sparse embedding
        vocab_size: Vocabulary size for token embeddings
        num_layers: Number of transformer/mamba layers
        hidden_size: Hidden dimension size
        expansion: FFN expansion factor
        num_heads: Number of attention heads (for attention-based variants)
        pos_encodings: Type of positional encoding ("rope" or "none")
        attn_dropout: Dropout rate for attention
        mlp_dropout: Dropout rate for MLP
        rms_norm_eps: Epsilon for RMS normalization
        rope_theta: Base for rotary positional encoding
        loops: Maximum number of recurrent loops
        L_cycles: Number of layer iterations per loop (gradient-enabled)
        H_cycles: Number of additional no-grad cycles before final L_cycles
                   (H_cycles-1 cycles run without gradients, 1 final cycle with gradients)
                   This enables deeper unrolled computation with reduced memory usage.
        forward_dtype: Data type for forward pass ("bfloat16" or "float32")
        halt_exploration_prob: Probability of extending computation during exploration
        halt_init_bias: Initial bias for Q-head halt logits (negative = favor continue)
        min_halt_steps: Minimum steps before allowing early halt during exploration
    """
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int
    L_cycles: int
    H_cycles: int
    forward_dtype: str = "bfloat16"
    halt_exploration_prob: float = 0.1
    halt_init_bias: float = -2.0
    min_halt_steps: int = 2


class MRMBlock(nn.Module):
    def __init__(self, config: MRMConfig) -> None:
        super().__init__()
        # Replace attention with Mamba
        self.mamba = MambaBlock(
            hidden_size=config.hidden_size,
            state_size=getattr(config, 'mamba_state_size', 16),
            expand=getattr(config, 'mamba_expand', 2),
            d_conv=getattr(config, 'mamba_d_conv', 4),
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self._rms_norm_eps: float = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Mamba doesn't use cos_sin (RoPE), but we keep the signature for compatibility
        mamba_output = self.mamba(hidden_states)
        hidden_states = rms_norm(hidden_states + mamba_output, variance_epsilon=self._rms_norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self._rms_norm_eps)
        return hidden_states


class MRM_Inner(nn.Module):
    def __init__(self, config: MRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # Note: RoPE is not used by Mamba (state space models), but kept for compatibility
        # with architectures that may use attention layers instead of Mamba.
        # To enable RoPE for attention-based variants, set use_rope=True in config.
        if getattr(self.config, 'use_rope', False):
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        else:
            # Use no-op RoPE since Mamba doesn't use positional encoding
            self.rotary_emb = NoOpRoPE()

        self.layers = nn.ModuleList([MRMBlock(self.config) for _ in range(self.config.num_layers)])

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(self.config.halt_init_bias)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> MRMCarry:
        return MRMCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: MRMCarry) -> MRMCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden
        )
        return replace(carry, current_hidden=new_hidden)

    def forward(
        self,
        carry: MRMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[MRMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states = hidden_states + input_embeddings
                        for layer in self.layers:
                            hidden_states = layer(hidden_states=hidden_states, **seq_info)

        for _ in range(self.config.L_cycles):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **seq_info)

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class MRM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = MRMConfig(**config_dict)
        self.inner = MRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MRMCarry:
        batch_size = batch["inputs"].shape[0]
        base = self.inner.empty_carry(batch_size)
        return MRMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: MRMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[MRMCarry, Dict[str, torch.Tensor]]:

        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        new_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                # Exploration: randomly extend computation for some samples to encourage learning deeper reasoning
                halt_exploration_prob = self.config.halt_exploration_prob
                min_halt_steps = (torch.rand_like(q_halt_logits) < halt_exploration_prob) * torch.randint_like(
                    new_steps, low=self.config.min_halt_steps, high=self.config.loops + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

        return (
            MRMCarry(
                current_hidden=new_carry.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )
