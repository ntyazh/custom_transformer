from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TransformerConfig:
    n_layer: int
    n_head: int
    n_kv_head: int
    hidden_dim: int
    intermediate_dim: int  # feedforward hidden dim
    dropout: float = 0.1
    vocab_size: int = 1024
    max_seq_len: int = 128


model_configs = {
    "nano": TransformerConfig(
        n_layer=3, n_head=4, n_kv_head=2, hidden_dim=96, intermediate_dim=256
    ),
    "mini": TransformerConfig(
        n_layer=6, n_head=6, n_kv_head=3, hidden_dim=384, intermediate_dim=1024
    ),
    "small": TransformerConfig(
        n_layer=12, n_head=12, n_kv_head=6, hidden_dim=768, intermediate_dim=2048
    ),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """Root Mean Square Layer Normalization

        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Causal Self-Attention with support of
        Grouped-Query Attention and ALiBi for positional encoding
        """
        super().__init__()
        self.config = config
        assert self.config.hidden_dim % self.config.n_head == 0
        assert self.config.n_head % self.config.n_kv_head == 0
        self.head_dim = self.config.hidden_dim // self.config.n_head
        self.scale = self.head_dim**-0.5
        self.q_per_kv = self.config.n_head // self.config.n_kv_head

        self.q_proj = nn.Linear(
            self.config.hidden_dim, self.head_dim * self.config.n_head, bias=False
        )
        self.kv_proj = nn.Linear(
            self.config.hidden_dim,
            2 * self.head_dim * self.config.n_kv_head,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )

        self.attn_dropout = nn.Dropout(self.config.dropout)

        self.register_buffer(
            "causal_mask", self._create_causal_mask(self.config.max_seq_len)
        )
        self.register_buffer("alibi", self._build_alibi_bias(self.config.n_head))

    def _build_alibi_bias(self, num_heads: int) -> Tensor:
        """Build ALiBi for specified number of heads:

        Returns:
            Tensor with ALiBi biases, shape: [1, num heads, 1, 1]
        """
        x = (2**8) ** (1 / self.config.n_head)
        alibi_m = torch.tensor([1 / x ** (i + 1) for i in range(num_heads)]).view(
            1, num_heads, 1, 1
        )
        return alibi_m

    def _get_relative_positions(self, seq_len):
        return (
            torch.arange(seq_len).view(1, seq_len)
            - torch.arange(seq_len).view(seq_len, 1)
        ).to(DEVICE)

    def _create_causal_mask(self, max_seq_len: int) -> Tensor:
        """Create causal mask with ones where tokens can attend to each other.

        Returns:
            Tensor with causal mask, shape: [1, 1, seq len, seq len]
        """
        return torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len
        )

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Apply Self-Attention to input data with respect to pad tokens.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        q = self.q_proj(x)  # bs, seq_len, hidden_dim (hidden_dim = n_heads * head_dim)
        q = q.view(q.shape[0], q.shape[1], self.config.n_head, self.head_dim)

        k, v = self.kv_proj(x).chunk(2, dim=-1)  # bs, seq_len, n_kv_head * head_dim
        k = k.view(k.shape[0], k.shape[1], self.config.n_kv_head, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.config.n_kv_head, self.head_dim)

        k = torch.repeat_interleave(
            k, self.q_per_kv, dim=2
        )  # bs, seq_len, n_head, head_dim
        v = torch.repeat_interleave(v, self.q_per_kv, dim=2)
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # bs, n_head, seq_len, head_dim

        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_scores = attention_scores + self.alibi * self._get_relative_positions(
            k.shape[2]
        )
        if attention_mask is not None:
            attention_mask = torch.repeat_interleave(
                attention_mask.unsqueeze(1), k.shape[2], dim=1
            )  # bs, seq_len, seq_len
            attention_mask = torch.repeat_interleave(
                attention_mask.unsqueeze(1), k.shape[1], dim=1
            )  # bs, n_head, seq_len, seq_len
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float("-inf")
            )

        attention_matrix = attention_scores.softmax(dim=-1)
        attention_matrix = self.attn_dropout(attention_matrix)
        attention_matrix = attention_matrix @ v
        attention_matrix = (
            attention_matrix.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], x.shape[2])
        )
        return self.out_proj(attention_matrix)


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Gated Liner Unit with Swish Activation"""
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_dim, 2 * config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU to input data.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        u, gate = self.fc1(x).chunk(2, dim=-1)
        gate = gate * torch.sigmoid(gate)
        y = self.fc2(gate * u)
        return self.dropout(y)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        """Base Transformer Block
        - Causal Self-Attention and SwiGLU as main elements
        - Pre-normalization via RMSNorm
        - Regularization with dropouts before residuals
        """
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_dim)
        self.res_dropout_1 = nn.Dropout(config.dropout)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = RMSNorm(config.hidden_dim)
        self.res_dropout_2 = nn.Dropout(config.dropout)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Apply Transformer Block to input data.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        x = self.ln_1(x + self.res_dropout_1(self.attn(x, attention_mask)))
        x = self.ln_2(x + self.res_dropout_2(self.mlp(x)))
        return x


class TransformerForCausalLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: TransformerConfig):
        """Transformer model for Language Modeling"""
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_final = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.scale)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tensor:
        """Calculate logits for given input ids.

        Args:
            x: input tensor, shape [bs, seq len]
            attention_mask: mask with zeros for pad tokens, shape [bs, seq len]
        Returns:
            logits, shape [bs, seq len, hidden dim]
        """
        x = self.emb_dropout(self.token_emb(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens,
        eos_token_id,
        temperature=1.0,
        do_sample=False,
        top_k=None,
    ) -> Tensor:
        """Take a conditioning sequence of indices and complete the sequence max_new_tokens times,
        feeding the predictions back into the model each time.

        Args:
            idx: tensor with conditional tokens, shape [seq len]
            max_new_tokens: maximum number of new tokens
            eos_token_id: index of EOS token to stop generation
            temperature, do_sample, top_k: generation parameters
        Return:
            tensor with generated indexes
        """
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.shape[1] <= self.max_seq_len else idx[:, -self.max_seq_len :]
            )
            logits = self(idx_cond)

            # 1. Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # 2. Optionally crop the logits to only the top k options
            if top_k is not None:
                mask = logits < torch.topk(logits, top_k, dim=1).values[:, -1, None]
                logits[mask] = -float("inf")

            # 3. apply softmax to convert logits to probabilities
            probs = logits.softmax(dim=-1)

            # 4. Either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = probs.multinomial(num_samples=1, replacement=True)
            else:
                idx_next = torch.argmax(probs, dim=-1)[:, None]

            # 5. Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == eos_token_id:
                break
        return idx
