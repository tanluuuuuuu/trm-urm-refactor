import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Basic MoE


class BasicExpert(nn.Module):
    # expert can be a Linear layer or MLP layer or more complicated (activation function = swiglu)
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.linear(x)


class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                BasicExpert(feature_in, feature_out) for _ in range(expert_number)
            ]
        )
        # gate is to choose an expert
        self.gate = nn.Linear(feature_in, expert_number)

    def forward(self, x):
        # x shape: (batch, feature_in)
        expert_weight = self.gate(x)  # shape (batch, expert_number)
        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]  # each element shape (batch, )
        # concat (batch, expert_number, feature_out)
        expert_output = torch.cat(expert_out_list, dim=1)
        expert_weight = expert_weight.unsqueeze(1)  # (batch, 1, expert_number)
        output = expert_weight @ expert_output  # (batch, 1, feature_out)
        return output.squeeze()


def test_basic_moe():
    print("=" * 50)
    print("Running test_basic_moe()...")
    print("=" * 50)
    x = torch.rand(2, 4)
    basic_moe = BasicMOE(4, 3, 2)
    out = basic_moe(x)
    print(out)
    print(out.shape)
    print("test_basic_moe() completed.\n")

# 2. SparseMoE, for LLM (reference: mistral MoE)


class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)  # shape (b*s, expert_number)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        # shapes are (b*s, top_k)
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1)
        # normalization on expert weights
        router_weights = router_weights / \
            router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(
            hidden_states.device, hidden_states.dtype)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.expert_number
        )  # shape (b*s, top_k, expert_number)
        expert_mask = expert_mask.permute(
            2, 1, 0)  # (expert_number, top_k, b*s)
        return router_logits, router_weights, selected_experts, expert_mask, expert_mask


class MOEConfig:
    def __init__(self,
                 hidden_dim,
                 expert_number,
                 top_k,
                 shared_experts_number=2):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number


class SparseMOE(nn.Module):
    # each token goes to topk experts, each token gets hidden_embeddings
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(config.expert_number) for _ in range(self.expert_number)
            ]
        )
        self.router = MOERouter(
            self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        # x shape (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)  # shape (b*s, hidden_dim)

        # OLD: too many values to unpack (expected 4)
        # router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # NEW: router_result is a tuple of 4 elements
        router_result = self.router(hidden_states)
        router_logits, router_weights, selected_experts_indices = router_result[:3]
        # selected_experts_indices shape (b*s, top_k), expert_mask shape (expert_number, top_k, b*s)
        expert_mask = router_result[3] if len(router_result) > 3 else None

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape (top_k, b*s)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx, top_x both 1-dim tensor, idx = 0/1 (expert top1 or top2)
            # top_x = index of token in batch*seq_len
            # e.g. input: batch_size = 2, seq_len = 4, top_x in [0, 7], meaning 8 tokens, idx in [0, 1], meaning this token views current expert as its top1/top2 expert
            # hidden_states shape (b*s, hidden_dim)
            # top_x's hidden_states, (selected_token_number, hidden_dim)
            current_state = hidden_states.unsqueeze(
                0)[:, top_x, :].reshape(-1, hidden_dim)
            # router_weight shape (b*s, top_k)
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # (selected_token_number, 1), broadcast here

            # add current expert output to final_hidden_state
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        # final_hidden_states back to original shape
        final_hidden_states = final_hidden_states.reshape(
            batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits  # shape (b*s, expert_number)


def test_token_level_moe():
    print("=" * 50)
    print("Running test_token_level_moe()...")
    print("=" * 50)
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)
    print("test_token_level_moe() completed.\n")

# 3. ShareExpert SparseMOE (DeepSeek)


class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(config.hidden_dim, config.hidden_dim) for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        # x shape (b, s, hidden_dim)
        # MOE
        sparse_moe_out, router_logits = self.moe_model(x)
        # for each x, go through shared experts
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ]  # each expert output shape (b, s, hidden_dim)
        shared_experts_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)
        return sparse_moe_out + router_logits + shared_experts_out


def test_share_expert_moe():
    print("=" * 50)
    print("Running test_share_expert_moe()...")
    print("=" * 50)
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    share_expert_moe = ShareExpertMOE(config)
    out = share_expert_moe(x)
    print(out[0].shape, out[1].shape)
    print("test_share_expert_moe() completed.\n")


def switch_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Calculate Switch Transformers load_balance
    Args:
        router_logits: shape [batch_size * sequence_length, num_experts]
        num_experts: number of experts
    Returns:
        total_loss = auxiliary_loss + z_loss
    """
    router_probs = torch.softmax(router_logits, dim=-1)  # [b*s, num_experts]
    # best expert for each token
    _, selected_experts = torch.topk(router_probs, k=2, dim=-1)  # [b*s, 2]
    # one-hot matrix for selected experts
    mask = torch.nn.functional.one_hot(
        selected_experts, num_experts).float()  # [b*s, 2, num_experts]
    mask = mask.sum(dim=1)  # [b*s, num_experts] - combine top_k selections
    # each expert expected load, ideally 1/num_experts
    expected_load = torch.ones_like(router_probs) / num_experts
    # actual load is each expert's token / total tokens, take average on batch dimension
    actual_load = mask.mean(dim=0)  # [num_experts]
    # aux loss penalizes difference btw load balance distribution and expected load
    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts
    # z_loss penalizes big router logits
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001  # hyperparam
    total_loss = aux_loss + z_loss * z_loss_weight
    return total_loss


def test_moe_training():
    print("=" * 50)
    print("Running test_moe_training()...")
    print("=" * 50)
    batch_size = 32
    seq_len = 16
    hidden_dim = 32
    num_batches = 100

    config = MOEConfig(hidden_dim=hidden_dim,
                       expert_number=4,
                       top_k=2,
                       shared_experts_number=2)
    model = ShareExpertMOE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for batch in range(num_batches):
        # generate random input data
        x = torch.randn(batch_size, seq_len, hidden_dim)
        target = torch.randn(batch_size, seq_len, hidden_dim)
        # forward pass
        output, router_logits = model(x)
        # mse loss for prediction
        mse_loss = F.mse_loss(output, target)
        aux_loss = switch_load_balancing_loss(
            router_logits, config.expert_number)
        total_loss = mse_loss + aux_loss * 0.01
        # backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print(f"Batch {batch}, Loss: {total_loss.item():.4f}"
                  f"(MSE: {mse_loss.item():.4f}, Aux: {aux_loss.item():.4f})")
    print("test_moe_training() completed.\n")


if __name__ == "__main__":
    # test_basic_moe()
    test_token_level_moe()
    # test_share_expert_moe()
    # test_moe_training()
