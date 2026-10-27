# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed experts sharded by TP instead of EP in the generator, as vLLM and SGLang run them.

Under EP each rank computes only the picks routed to its own experts, so ranks finish a
MoE layer at different times and the faster one waits in the MoE output's all-reduce:
~19 us per layer on average (p90 ~55 us) for Qwen3.5-35B-A3B EP2 decode on A100, vs
7 us for the same all-reduce after a GDN layer. With every expert split across ranks
along its intermediate dimension, both ranks compute every pick on half the columns:
the same weight bytes per rank, no imbalance.

Weights still load and sync in the EP layout. After each full load, ``relayout`` swaps
half of every local expert's intermediate columns with the peer rank in place, so the
same storage (and the CUDA graphs reading it) then holds all experts' half-width weights:

    w13 [E/2, 2, F, D] (EP, own experts)  ->  [E, 2, F/2, D] (TP, all experts)
    w2  [E/2, D, F]                        ->  [E, D, F/2]

The model's state dict is not meaningful in between: only loads write it. Requires the
replicated-token MoE path (EP group = TP group of two ranks, DP = 1, no sequence
parallelism). Opt-in with ``TORCHTITAN_MOE_TP_EXPERTS=1``.
"""

import os

import torch
import torch.distributed as dist

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher

ENABLED = os.environ.get("TORCHTITAN_MOE_TP_EXPERTS") == "1"
_original_forward = GroupedLinear.forward


def _forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
    weight = getattr(self, "_tp_weight", None)
    if weight is None:
        return _original_forward(self, input_RI, offsets_E)
    output_RO = self._grouped_mm(
        input_RI=input_RI, weight_EOI=weight.flatten(1, -2), offsets_E=offsets_E
    )
    return output_RO.reshape(*output_RO.shape[:-1], *weight.shape[1:-1])


def _swap_halves(weight: torch.Tensor, dim: int, rank: int, group) -> torch.Tensor:
    """Re-lay ``weight`` ([E/2, ..., F at ``dim``, ...], this rank's experts) in place as
    [E, ..., F/2, ...]: this rank's half of every expert, own experts at ``rank * E/2``."""
    local_experts, width = weight.shape[0], weight.shape[dim]
    half = width // 2
    peer = 1 - rank
    keep = weight.narrow(dim, rank * half, half).contiguous()
    send = weight.narrow(dim, peer * half, half).contiguous()
    recv = torch.empty_like(send)
    for request in dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send, dist.get_global_rank(group, peer), group),
            dist.P2POp(dist.irecv, recv, dist.get_global_rank(group, peer), group),
        ]
    ):
        request.wait()
    shape = list(weight.shape)
    shape[0], shape[dim] = 2 * local_experts, half
    relaid = weight.view(shape)
    relaid[rank * local_experts : (rank + 1) * local_experts].copy_(keep)
    relaid[peer * local_experts : (peer + 1) * local_experts].copy_(recv)
    return relaid


def maybe_patch_grouped_linear() -> None:
    """Before parallelize, which binds each module's forward into its SPMD wrapper."""
    if ENABLED and GroupedLinear.forward is _original_forward:
        GroupedLinear.forward = _forward


def relayout(model: torch.nn.Module) -> None:
    """After a full weight load: switch every replicated-token MoE to TP-sharded experts."""
    if not ENABLED:
        return
    # The loaded weights are inference tensors, writable in place only in inference mode.
    with torch.inference_mode():
        for module in model.modules():
            if not isinstance(module, RoutedExperts):
                continue
            dispatcher = module.token_dispatcher
            if not (
                type(dispatcher) is AllToAllTokenDispatcher
                and dispatcher.replicated_tokens
                and dispatcher.ep_mesh is not None
                and dispatcher.ep_mesh.size() == 2
            ):
                continue
            group = dispatcher.ep_mesh.get_group()
            rank = dispatcher.ep_mesh.get_local_rank()
            w13, w2 = module.w13.weight, module.w2.weight
            assert w13.dim() == 4 and w2.dim() == 3, (w13.shape, w2.shape)
            # w13 [E/2, 2 (gate, up), F, D]; w2 [E/2, D, F].
            module.w13._tp_weight = _swap_halves(w13, 2, rank, group)
            module.w2._tp_weight = _swap_halves(w2, 2, rank, group)
            dispatcher.experts_tp_sharded = True
