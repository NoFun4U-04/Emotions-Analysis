import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim//2]
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, heads, seq_len, dim]
        """
        seq_len = x.size(2)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated


class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.d_k)

    def split_heads(self, x):
        # [batch, seq_len, d_model] -> [batch, heads, seq_len, d_k]
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        return x

    def combine_heads(self, x):
        # [batch, heads, seq_len, d_k] -> [batch, seq_len, d_model]
        bsz = x.size(0)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return x

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape

        # QKV projections
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Split heads
        q = self.split_heads(q)  # [batch, heads, seq_len, d_k]
        k = self.split_heads(k)  # [batch, heads, seq_len, d_k]
        v = self.split_heads(v)  # [batch, heads, seq_len, d_k]

        # Apply rotary embeddings
        q = self.rope(q)
        k = self.rope(k)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Handle attention mask
        if mask is not None:
            # Reshape mask to match attention scores dimensions
            # mask shape: [batch, seq_len] -> [batch, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(2)
            mask = mask.expand(-1, self.n_heads, -1, seq_len)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Attention weights and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = self.combine_heads(context)
        
        return self.o_proj(context)
