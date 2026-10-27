# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One input-projection GEMM per GDN layer in the generator.

``GDN.forward`` runs six projections of the same input (q, k, v, z, a, b), each a GEMM
plus a split-K reduce at decode sizes, and the vLLM adapter concatenates the three conv
weights on every call: ~13 kernels per GDN layer where vLLM runs one fused GEMM. Here the
generator keeps a concatenated copy of the six weights (and of the conv weights, and fp32
copies of A_log and dt_bias) per GDN layer, refreshed in place after every weight load so
CUDA graphs keep reading the same addresses, and runs one GEMM whose output columns are
split back into the six tensors and passed on as views.

Costs a second copy of the projection weights: 0.76 GiB per rank for Qwen3.5-35B-A3B
at TP2 (4 GiB for 27B, where it is not worth the KV cache). Inference only.
Opt-in with ``TORCHTITAN_FUSED_GDN_PROJ=1``.
"""

import os

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.tensor import DTensor

from torchtitan.models.common.attention.attention import local_head_split

from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input

ENABLED = os.environ.get("TORCHTITAN_FUSED_GDN_PROJ") == "1"
_PROJECTIONS = (
    "in_proj_q",
    "in_proj_k",
    "in_proj_v",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
)
_original_forward = GDN.forward


def _local(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def refresh(model: torch.nn.Module) -> None:
    """Copy each GDN layer's current projection and conv weights into its fused buffers."""
    if not ENABLED:
        return
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, GDN):
                continue
            linears = [getattr(module, name) for name in _PROJECTIONS]
            # Plain bias-free projections only: LoRA / quantized subclasses own _linear.
            if any(
                linear.bias is not None or type(linear)._linear is not Linear._linear
                for linear in linears
            ):
                continue
            weights = [_local(linear.weight).flatten(0, -2) for linear in linears]
            convs = [
                _local(conv.weight)
                for conv in (module.conv_q, module.conv_k, module.conv_v)
            ]
            A_log, dt_bias = _local(module.A_log), _local(module.dt_bias)
            fused = getattr(module, "_fused_in_proj_weight", None)
            shape = (sum(w.shape[0] for w in weights), weights[0].shape[1])
            if fused is None or fused.shape != shape:
                module._fused_in_proj_weight = torch.empty(
                    shape, dtype=weights[0].dtype, device=weights[0].device
                )
                module._fused_in_proj_splits = [w.shape[0] for w in weights]
                module._fused_conv_weight = torch.empty(
                    sum(c.shape[0] for c in convs),
                    *convs[0].shape[1:],
                    dtype=convs[0].dtype,
                    device=convs[0].device,
                )
                # The GDN kernels take fp32 decay parameters.
                module._fused_A_log = torch.empty_like(A_log, dtype=torch.float32)
                module._fused_dt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
            torch.cat(weights, dim=0, out=module._fused_in_proj_weight)
            torch.cat(convs, dim=0, out=module._fused_conv_weight)
            module._fused_A_log.copy_(A_log)
            module._fused_dt_bias.copy_(dt_bias)


def _fused_forward(self, x_TD, attention_metadata=None):
    fused = getattr(self, "_fused_in_proj_weight", None)
    if fused is None or torch.is_grad_enabled():
        return _original_forward(self, x_TD, attention_metadata)
    x_TD = maybe_gather_tp_input(self, x_TD)
    num_tokens = x_TD.shape[0]
    projected_TW = F.linear(x_TD, fused)
    query_TC, key_TC, value_TC, gate_TC, a_TH, b_TH = projected_TW.split(
        self._fused_in_proj_splits, dim=-1
    )
    # The conv and decode kernels read rows at any stride, so q / k / v (adjacent columns),
    # a and b stay views of the projection.
    output_THV = self.inner_attention(
        query_TC,
        key_TC,
        value_TC,
        a_TH,
        b_TH,
        self.conv_q.weight,
        self.conv_k.weight,
        self.conv_v.weight,
        self._fused_A_log,
        self._fused_dt_bias,
        key_head_dim=self.key_head_dim,
        value_head_dim=self.value_head_dim,
        attention_metadata=attention_metadata,
        fused_conv_weight=self._fused_conv_weight,
        mixed_qkv_TC=projected_TW[:, : sum(self._fused_in_proj_splits[:3])],
    )
    gate_THV = local_head_split(gate_TC, self.value_head_dim, cp_shard_dim=0)
    remat.recompute_needs_tensor(output_THV, gate_THV)
    output_THV = self.norm(output_THV, gate_THV)
    return self.out_proj(output_THV.reshape(num_tokens, -1))


def maybe_patch_gdn_forward() -> None:
    if ENABLED and GDN.forward is _original_forward:
        GDN.forward = _fused_forward
