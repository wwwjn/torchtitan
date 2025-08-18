# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Qwen3ModelArgs

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
        idx_theta = torch.outer(seq_idx, self.theta).float()

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
                [bsz, seq_len, num_attn_heads, head_dim]

        Returns:
            cos (torch.Tensor): RoPE cosine embeddings
            sin (torch.Tensor): RoPE sine embeddings

        """
        # input tensor x has shape [bsz, seq_len, num_attn_heads, head_dim]
        seq_len = x.shape[1]
        head_dim = x.shape[-1]

        # reshape for broadcast
        rope_cache = self.cache.view(-1, seq_len, 1, head_dim * 2)

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


def print_tensor_stats(name, tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    print(
        f"{name} - Shape: {tensor.shape} Mean: {mean:.6f}, Min: {min_val:.6f}, Max: {max_val:.6f}, Std: {std:.6f}, First 10 values: {tensor.flatten()[:10].tolist()}"
    )

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: Qwen3ModelArgs, rotary_emb: QwenRotaryEmbedding):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim

        # RMSNorm added here to the here to include the q-k norm
        # This is one of the main differences between Llama3 and Qwen3
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        
        self.rotary_emb = rotary_emb

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            cos (torch.Tensor): Precomputed cos Rotary embedding tensor.
            sin (torch.Tensor): Precomputed cos Rotary embedding tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        print_tensor_stats("xq after metrics", xq)
        print_tensor_stats("xk after metrics", xk)
        print_tensor_stats("xv after metrics ", xv)

        # Adding the q_norm and k_norm here
        # Last layer of adding q-k norm
        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        print_tensor_stats("xq after q_norm", xq) # Shape: torch.Size([1, 4096, 16, 128])
        print_tensor_stats("xk after k_norm", xk)  # Shape: torch.Size([1, 4096, 8, 128])

        # print_tensor_stats("cos", cos)
        # print_tensor_stats("sin", sin)
        # xq, xk = apply_rotary_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)


        print_tensor_stats("keys", keys)
        print_tensor_stats("values", values)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Hidden dimension is directly added from the model argsS
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs, rotary_emb: QwenRotaryEmbedding):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args, rotary_emb)
        self.feed_forward = FeedForward(
            dim=model_args.dim, hidden_dim=model_args.hidden_dim
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        # init rotary embeddings only once in the Model
        self.rotary_emb = QwenRotaryEmbedding(dim=model_args.head_dim, max_seq_len=model_args.max_seq_len, base=model_args.rope_theta)
        
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args, self.rotary_emb)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
       
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        if self.model_args.enable_weight_tying:
            self.output.weight = self.tok_embeddings.weight
        
        self.init_weights()

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        
        if self.output is not None and not self.model_args.enable_weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        if self.rotary_emb:
            self.rotary_emb.rope_init()

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
        eos_id: int | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).
            eos_id (int | None): End-of-sequence token ID. If not provided, uses self.eos_id.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        if self.model_args.use_flex_attn:
            init_attention_mask(
                input_batch if input_batch is not None else tokens,
                eos_id=eos_id,
            )
    
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
