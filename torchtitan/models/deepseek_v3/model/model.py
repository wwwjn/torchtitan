# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from re import S
from typing import Tuple

import torch
from torch import nn
from torchtitan.models.attention import build_attention
from torchtitan.protocols.train_spec import ModelProtocol

from .args import DeepSeekV3ModelArgs
from .moe import FeedForward, MoE


def print_tensor_stats(name, tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    print(
        f"{name} - Shape: {tensor.shape} Mean: {mean:.6f}, Min: {min_val:.6f}, Max: {max_val:.6f}, Std: {std:.6f}, First 10 values: {tensor.flatten()[:10].tolist()}"
    )


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepSeekV3ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> Tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Basic RoPE frequency calculation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices
    t = torch.arange(seqlen)

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def apply_rotary_emb_with_permute(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # first half is real, second half is imaginary
    from torchtitan.models.llama3.model.model import reshape_for_broadcast

    xq_ = torch.complex(
        xq[..., : xq.shape[-1] // 2].float(), xq[..., xq.shape[-1] // 2 :].float()
    )
    xk_ = torch.complex(
        xk[..., : xk.shape[-1] // 2].float(), xk[..., xk.shape[-1] // 2 :].float()
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # added this
    xq_out = torch.cat([xq_out[..., ::2], xq_out[..., 1::2]], dim=-1)
    xk_out = torch.cat([xk_out[..., ::2], xk_out[..., 1::2]], dim=-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        self._init_rope()

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def _init_rope(self):
        from .deepseek_rotary_emb import (
            DeepseekV3RotaryEmbedding,
            DeepseekV3YarnRotaryEmbedding,
        )

        config = {
            "rope_scaling": {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            "max_position_embeddings": 163840,
            "rope_theta": 10000,
        }
        if config["rope_scaling"] is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=config["max_position_embeddings"],
                base=config["rope_theta"],
            )
        else:
            scaling_type = config["rope_scaling"]["type"]
            scaling_factor = config["rope_scaling"]["factor"]
            scaling_type = "yarn"

            if scaling_type == "yarn":
                kwargs = {
                    key: config["rope_scaling"][key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in config["rope_scaling"]
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=config["max_position_embeddings"],
                    scaling_factor=scaling_factor,
                    base=config["rope_theta"],
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # for i in range(0, 10):
        #     print("self.wkv_a.weight: ", self.wq_a.weight[0][i])

        bsz, seqlen, _ = x.size()

        print_tensor_stats("input: ", x)

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            print_tensor_stats("After wq_a: ", q)
            q = self.q_norm(q)
            print_tensor_stats("After q_norm: ", q)
            q = self.wq_b(q)
        print_tensor_stats("After wq_b: ", q)
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        print_tensor_stats("After wkv_a: ", kv)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # TODO(jiani): switch to HF rotary embedding implementation
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.kv_norm(kv)
        print_tensor_stats("After kv_norm: ", kv)
        kv = self.wkv_b(kv)  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        print_tensor_stats("After wkv_b: ", kv)
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # # NOTE(jianiw): Test using HF rotary embedding implementation
        # from .deepseek_rotary_emb import apply_rotary_pos_emb

        # kv_seq_len = v.shape[-3]
        # cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        # device = x.device
        # position_ids = torch.arange(
        #     0,
        #     kv_seq_len,  # TODO: Check this is correct
        #     dtype=torch.long,
        #     device=device,
        # )
        # position_ids = position_ids.unsqueeze(0)
        # k_pe = k_pe.view(bsz, seqlen, 1, self.qk_rope_head_dim).transpose(1, 2)
        # q_pe = q_pe.transpose(
        #     1, 2
        # )  # k_pe torch.Size([1, 1, 2048, 64]) q_pe: torch.Size([1, 128, 2048, 64])
        # HF: before apply rotary post emb: q_pe torch.Size([1, 128, 2048, 64]), k_pe torch.Size([1, 1, 2048, 64]), cos torch.Size([2048, 64]), sin torch.Size([2048, 64]), position_ids torch.Size([1, 2048])
        # titan: Before applying rotary emb, the shape is: k_pe torch.Size([1, 1, 2048, 64]) q_pe: torch.Size([1, 128, 2048, 64]), cos torch.Size([128, 64]), sin torch.Size([128, 64]), position_ids torch.Size([1, 128])

        # print(
        #     f"Before applying rotary emb, the shape is: k_pe {k_pe.shape} q_pe: {q_pe.shape}, cos {cos.shape}, sin {sin.shape}, position_ids {position_ids.shape}"
        # )

        # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        # q_pe, k_pe = q_pe.transpose(1, 2), k_pe.transpose(1, 2)

        print_tensor_stats("After k_pe apply_rotary_emb: ", k_pe)
        print_tensor_stats("After q_pe apply_rotary_emb: ", q_pe)
        

        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        print_tensor_stats("k: ", k)
        print_tensor_stats("v: ", v)
        print_tensor_stats("q: ", q)
        qk = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1))
        print_tensor_stats("After titan's rotary_emb, qk: ", qk)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)
        

        # TODO: Need to pass softmax_scale to sdpa() interface.
        # For mask, DeepseekV3 uses causal mask, so we can use the default mask in sdpa
        # https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L17
        output = self.sdpa(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        print_tensor_stats("After attention: ", output)
        output = output.reshape(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        print_tensor_stats("output after wo: ", output)
        return output

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):

        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            self.moe = MoE(model_args)
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        # Print statistics before and after each normalization

        # residual = hidden_states

        # print_tensor_stats("hidden_states before input_layernorm", hidden_states)

        # hidden_states = self.input_layernorm(hidden_states)

        # print_tensor_stats("hidden_states after input_layernorm", hidden_states)

        # # Self Attention
        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        #     **kwargs,
        # )
        # hidden_states = residual + hidden_states

        # print_tensor_stats("hidden_states after self attention", hidden_states)

        # # Fully Connected
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)

        # print_tensor_stats(
        #     "hidden_states after post attention layernorm", hidden_states
        # )
        # hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        # print_tensor_stats("hidden_states after mlp", hidden_states)
        ## Our implementation of DeepSeek-V3
        print_tensor_stats("input: ", x)
        attn_norm_out = self.attention_norm(x)
        print_tensor_stats("After attention_norm", attn_norm_out)

        h = x + self.attention(attn_norm_out, freqs_cis)

        print_tensor_stats("after x+attention", h)
        ffn_output = self.ffn_norm(h)
        print_tensor_stats("After ffn norm", ffn_output)

        if self.moe_enabled:
            out = h + self.moe(ffn_output)
        else:
            out = h + self.feed_forward(ffn_output)
        print_tensor_stats("After x+feed_forward", out)
        return out

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(nn.Module, ModelProtocol):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=True
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        # Reset hidden states collection
        self._hidden_states = []

        # Get embeddings
        input_embeds = self.tok_embeddings(tokens)
        # Store and print statistics for input embeddings
        self._hidden_states.append(input_embeds.detach())

        print_tensor_stats("input_embeds: ", input_embeds)
        h = input_embeds

        # Process through layers
        for i, layer in enumerate(self.layers.values()):
            # NOTE(jianiw): Reset the hidden states to be input embeddings for the each layer to avoid numerical difference accumualation
            # h = input_embeds
            h = layer(h, self.freqs_cis)

            print_tensor_stats(f"layer {i} output: ", h)
            # Store and print statistics for this layer
            self._hidden_states.append(h.detach())

        # Apply final normalization
        h = self.norm(h)

        print_tensor_stats("After norm: ", h)
        # Generate output logits
        output = self.output(h)
        print_tensor_stats("output: ", output)

        return output

    def get_hidden_states(self):
        """
        Returns the hidden states collected during the forward pass.

        Returns:
            list[torch.Tensor]: List of hidden states, where:
                - hidden_states[0] is the input embeddings
                - hidden_states[i] for i>0 is the output of the i-th layer
        """
        return self._hidden_states
