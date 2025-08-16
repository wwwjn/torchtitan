# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn


# Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_positional_embeddings.py
class QwenRotaryEmbedding(nn.Module):
    """
    RoPE Embeddings used in the Qwen model.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim`` // ``num_heads`` or ``head_dim``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (float): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.outer(seq_idx, self.theta).float

        # We cache the cos and sin embeddings instead of the IDs. This helps
        # ensure we have correct behavior when training with bf16
        # Size: [max_seq_len, (dim * 2)]
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer("cache", cache, persistent=False)
    
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [bsz, num_attn_heads, seq_len, head_dim]

        Returns:
            cos (torch.Tensor): RoPE cosine embeddings
            sin (torch.Tensor): RoPE sine embeddings

        """
        # input tensor has shape [bsz, seq_len, num_attn_heads, head_dim]
        seq_len = x.shape[1]
        head_dim = x.shape[-1]

        rope_cache = self.cache
        # reshape for broadcast
        # tensor has shape [b, s, 1, h_d * 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d * 2]
        rope_cache = rope_cache.view(-1, seq_len, 1, head_dim * 2)

        # [b, s, 1, h_d]
        cos = rope_cache[..., :head_dim].to(dtype=x.dtype, device=x.device)
        sin = rope_cache[..., head_dim:].to(dtype=x.dtype, device=x.device)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        # cos: [b, s, 1, h_d]
        # x: [b, s, n_h, h_d]
        x_out = (x * cos) + (rotated * sin)
        return x_out.type_as(x)
