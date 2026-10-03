# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-wide memory runtime for Dist-MoE experts in a generator.

``DistMoeRuntime`` plans for training: saved-activation slots, pipeline
liveness and weight gradients. A generator runs no backward and no pipeline, so
it needs only scratch memory, sized by the scheduler's token bound, plus one
rule enforced at the expert boundary: every expert-parallel rank passes the same
local token count (see ``padding.py``).

Shape suffixes in this file use ``T`` for local input tokens, ``K`` for selected
experts, and ``D`` for the model dimension.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dist_moe
import torch
import torch.distributed as dist

from torchtitan.config import Configurable
from torchtitan.models.common.dist_moe.padding import (
    pad_to_num_tokens,
    route_padding_to_local_experts,
)


if TYPE_CHECKING:
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.common.dist_moe.routed_experts import DistMoeRoutedExperts


logger = logging.getLogger(__name__)

__all__ = ["DistMoeInferenceRuntime"]


def _engine_step() -> object | None:
    """vLLM's per-step forward context, or None outside a vLLM forward."""
    try:
        from vllm.forward_context import get_forward_context

        return get_forward_context()
    except Exception:  # noqa: BLE001  (vLLM absent or not inside a forward)
        return None


class DistMoeInferenceRuntime(Configurable):
    """Own one scratch-only annex context shared by a generator's Dist-MoE experts.

    Build it after the model is parallelized and before any CUDA-graph capture.
    Expert modules keep a non-owning reference and read ``context`` in forward.

    Args:
        config: Scratch and VMM policy.
        model_parts: Final local model modules.
        parallelism_context: Final distributed mesh topology.
        device: CUDA device that owns the annex context and buffers.
        max_num_batched_tokens: Largest token batch one rank runs in a step
            (the scheduler's bound); the unsharded count used to derive the
            local routing input bound.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configure rank-wide Dist-MoE scratch policy for inference.

        Args:
            scratch_capacity_factor: Routing imbalance that must fit entirely
                in device-resident scratch. ``1.0`` covers balanced
                ``local_tokens * top_k`` routing. Routing beyond the capacity
                is an illegal access, not an error.
            vmm_capacity_factor: Optional total device-plus-host scratch
                capacity. ``None`` disables VMM.
            num_sms: Optional SM count for each Dist-MoE launch. ``None`` uses
                the annex default.
        """

        scratch_capacity_factor: float = 1.0
        vmm_capacity_factor: float | None = None
        num_sms: int | None = None

        def __post_init__(self) -> None:
            if (
                not math.isfinite(self.scratch_capacity_factor)
                or self.scratch_capacity_factor <= 0
            ):
                raise ValueError("scratch_capacity_factor must be finite and positive")
            if self.vmm_capacity_factor is not None and (
                not math.isfinite(self.vmm_capacity_factor)
                or self.vmm_capacity_factor <= 0
            ):
                raise ValueError("vmm_capacity_factor must be finite and positive")
            if self.num_sms is not None and self.num_sms <= 0:
                raise ValueError("num_sms must be positive")

    def __init__(
        self,
        config: Config,
        *,
        model_parts: Sequence[torch.nn.Module],
        parallelism_context: ParallelismContext,
        device: torch.device,
        max_num_batched_tokens: int,
    ) -> None:
        from .routed_experts import DistMoeRoutedExperts

        self.config = config
        self._closed = False
        self._modules = tuple(
            dict.fromkeys(
                module
                for model_part in model_parts
                for module in model_part.modules()
                if isinstance(module, DistMoeRoutedExperts)
            )
        )
        if not self._modules:
            raise ValueError(
                "Dist-MoE inference runtime requires at least one expert module"
            )
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
            raise ValueError("Dist-MoE requires an SM100-or-newer CUDA device")
        if parallelism_context.pp_enabled:
            raise ValueError("Dist-MoE inference does not support pipeline parallelism")

        ep_mesh = parallelism_context.get_optional_mesh(
            "ep", include_singleton_axes=True
        )
        if ep_mesh is None:
            raise RuntimeError("Dist-MoE requires an expert-parallel mesh")
        self.ep_pg = ep_mesh.get_group()

        num_token_shards = parallelism_context.cp * parallelism_context.tp
        if max_num_batched_tokens % num_token_shards:
            raise ValueError(
                "Dist-MoE input tokens must divide evenly across CP and TP"
            )
        self.max_local_input_tokens = max_num_batched_tokens // num_token_shards

        # Dist-MoE gives each EP rank a contiguous block of experts.
        ep_size = dist.get_world_size(self.ep_pg)
        num_experts = self._modules[0].num_experts
        if num_experts % ep_size:
            raise ValueError(
                f"{num_experts} experts do not divide over {ep_size} EP ranks"
            )
        self.num_local_experts = num_experts // ep_size
        self.first_local_expert = dist.get_rank(self.ep_pg) * self.num_local_experts
        self._token_count_group: dist.ProcessGroup | None = None
        self._token_count_cache: tuple[object, int, int] | None = None

        context_config = self._resolve_context_config(self._modules[0])
        for module in self._modules[1:]:
            if self._resolve_context_config(module) != context_config:
                raise ValueError(
                    "All local Dist-MoE layers must resolve one context configuration"
                )
        self.context = dist_moe.create_context(
            group=self.ep_pg,
            config=context_config,
            device=device,
        )
        for module in self._modules:
            module._runtime = self

    def _resolve_context_config(self, module: DistMoeRoutedExperts) -> dist_moe.Config:
        """Build the annex context configuration for one local expert module."""
        vmm = (
            None
            if self.config.vmm_capacity_factor is None
            else dist_moe.VmmConfig(
                total_scratch_capacity_factor=self.config.vmm_capacity_factor
            )
        )
        return dist_moe.Config(
            max_local_input_tokens=self.max_local_input_tokens,
            hidden_dim=module.hidden_dim,
            intermediate_dim=module.intermediate_dim,
            top_k=module.top_k,
            num_experts=module.num_experts,
            # Inert without saved activations, but the annex requires a
            # positive depth.
            max_moe_layers_per_activation_slot=len(self._modules),
            device_scratch_capacity_factor=self.config.scratch_capacity_factor,
            num_activation_slots=0,
            inference=True,
            vmm=vmm,
            num_sms=self.config.num_sms,
            bf16_grouped_gemm_preset=module.bf16_grouped_gemm_preset,
            block_scaled=module.block_scaled_config,
        )

    def _ep_group_max_tokens(self, num_tokens: int) -> int:
        """Largest local token count over the expert-parallel group.

        One CPU all-reduce on a private gloo group over the EP ranks, cached for
        the rest of the engine step. While a CUDA graph is captured the count is
        already the data-parallel-padded one and no host collective may run.
        """
        if torch.cuda.is_current_stream_capturing():
            return num_tokens
        step = _engine_step()
        cache = self._token_count_cache
        if (
            step is not None
            and cache is not None
            and cache[0] is step
            and cache[1] == num_tokens
        ):
            return cache[2]
        if self._token_count_group is None:
            # Collective: every rank of the default group reaches its first
            # forward together.
            self._token_count_group = dist.new_group(
                ranks=dist.get_process_group_ranks(self.ep_pg), backend="gloo"
            )
        count = torch.tensor([num_tokens], dtype=torch.int32)
        dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self._token_count_group)
        result = int(count.item())
        if step is not None:
            self._token_count_cache = (step, num_tokens, result)
        return result

    def route_padding_locally(
        self,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        padding_mask_T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route rows the caller already padded to this rank's own experts.

        vLLM pads a step's tokens (tensor-parallel rounding, CUDA-graph capture
        size, data-parallel equalization) before the model runs, so those rows
        went through the real router and carry real, often identical, expert
        IDs. Dist-MoE would dispatch them to whichever rank owns those experts.
        ``padding_mask_T`` is true for such rows: their scores become zero and
        their IDs local, so they cost no network traffic and no remote load.
        """
        return route_padding_to_local_experts(
            topk_scores_TK,
            topk_expert_ids_TK,
            padding_mask_T,
            first_local_expert=self.first_local_expert,
            num_local_experts=self.num_local_experts,
        )

    def equalize_inputs(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad this rank's expert input to the EP group's largest token count.

        Dist-MoE returns wrong results, or faults, when the ranks of its group
        pass different token counts. Generator ranks see different counts on
        every step vLLM does not pad across data-parallel ranks. Padding rows
        carry zero scores and are routed to this rank's own experts, so they add
        no network traffic and no load on remote experts. Callers slice the
        output back to the original token count.
        """
        target = self._ep_group_max_tokens(x_TD.shape[0])
        if target > self.max_local_input_tokens:
            raise ValueError(
                f"Dist-MoE EP group holds {target} local tokens, above the planned "
                f"{self.max_local_input_tokens}"
            )
        return pad_to_num_tokens(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            target,
            first_local_expert=self.first_local_expert,
            num_local_experts=self.num_local_experts,
        )

    def close(self) -> None:
        """Detach the expert modules and release the annex context."""
        if self._closed:
            return
        for module in self._modules:
            if module._runtime is self:
                module._runtime = None
        self.context.close()
        self._closed = True
