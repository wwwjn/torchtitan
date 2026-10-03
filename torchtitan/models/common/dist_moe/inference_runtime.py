# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-wide memory runtime for Dist-MoE experts in a generator.

``DistMoeRuntime`` plans for training: saved-activation slots, pipeline
liveness and weight gradients. A generator runs no backward and no pipeline, so
this runtime plans scratch memory only, sized by the scheduler's token bound.
It mirrors ``DistMoeRuntime``'s construction (same config field names, same
constructor arguments where they apply, ``context`` and ``close``).

Shape suffixes in this file use ``T`` for local input tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dist_moe
import torch
import torch.distributed as dist

from torchtitan.config import Configurable
from torchtitan.models.common.dist_moe.padding import LocalExpertPadding


if TYPE_CHECKING:
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.common.dist_moe.routed_experts import DistMoeRoutedExperts

__all__ = ["DistMoeInferenceRuntime"]


class DistMoeInferenceRuntime(Configurable):
    """Own one scratch-only annex context shared by a generator's Dist-MoE experts.

    Build it after the model is parallelized and before any CUDA-graph capture.
    Expert modules keep a non-owning reference and read ``context`` in forward;
    ``padding`` keeps padding rows off the network.

    Args:
        config: Scratch and VMM policy.
        model_parts: Final local model modules.
        parallelism_context: Final distributed mesh topology.
        device: CUDA device that owns the annex context and buffers.
        num_tokens_per_microbatch_per_dp_rank: Unsharded token count used to
            derive the local routing input bound (vLLM's
            ``max_num_batched_tokens`` for a generator).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configure rank-wide Dist-MoE scratch policy for inference.

        Args:
            scratch_capacity_factor: Routing imbalance that must fit in
                device-resident scratch. ``1.0`` covers balanced
                ``local_tokens * top_k`` routing; more skew is an illegal
                access, not an error.
            vmm_capacity_factor: Optional total device-plus-host scratch
                capacity. ``None`` disables VMM.
        """

        scratch_capacity_factor: float = 1.0
        vmm_capacity_factor: float | None = None

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

    def __init__(
        self,
        config: Config,
        *,
        model_parts: Sequence[torch.nn.Module],
        parallelism_context: ParallelismContext,
        device: torch.device,
        num_tokens_per_microbatch_per_dp_rank: int,
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
            raise ValueError("Dist-MoE runtime requires at least one expert module")
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
            raise ValueError("Dist-MoE requires an SM100-or-newer CUDA device")
        if parallelism_context.pp_enabled:
            raise ValueError("Dist-MoE inference does not support pipeline parallelism")

        ep_mesh = parallelism_context.get_optional_mesh(
            "ep", include_singleton_axes=True
        )
        if ep_mesh is None:
            raise RuntimeError("Dist-MoE requires an expert-parallel mesh")
        ep_pg = ep_mesh.get_group()

        num_token_shards = parallelism_context.cp * parallelism_context.tp
        if num_tokens_per_microbatch_per_dp_rank % num_token_shards:
            raise ValueError(
                "Dist-MoE input tokens must divide evenly across CP and TP"
            )
        self.max_local_input_tokens = (
            num_tokens_per_microbatch_per_dp_rank // num_token_shards
        )
        self.padding = LocalExpertPadding(
            ep_pg,
            num_local_experts=self._modules[0].num_experts // dist.get_world_size(ep_pg),
            max_local_input_tokens=self.max_local_input_tokens,
        )

        context_config = self._resolve_context_config(self._modules[0])
        for module in self._modules[1:]:
            if self._resolve_context_config(module) != context_config:
                raise ValueError(
                    "All local Dist-MoE layers must resolve one context configuration"
                )
        self.context = dist_moe.create_context(
            group=ep_pg, config=context_config, device=device
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
            # Inert without saved activations, but the annex wants a positive depth.
            max_moe_layers_per_activation_slot=len(self._modules),
            device_scratch_capacity_factor=self.config.scratch_capacity_factor,
            num_activation_slots=0,
            inference=True,
            vmm=vmm,
            bf16_grouped_gemm_preset=module.bf16_grouped_gemm_preset,
            block_scaled=module.block_scaled_config,
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
