# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel pieces shared by the linear-attention families."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.utils import _pytree as pytree

from torchtitan.distributed.context_parallel import get_token_fragments
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan.models.common.attn_gym import (
    _ContextParallelChunk,
    ContextParallelRouting,
    is_missing_kernel,
)

from .linear_attention import LinearAttentionMetadata

if not is_missing_kernel(_ContextParallelChunk):
    spmd.register_local_autograd_function(_ContextParallelChunk)


@dataclass(frozen=True, slots=True)
class ContextParallelLinearAttentionMetadata(LinearAttentionMetadata):
    """Linear-attention metadata with rank-local context-parallel routing."""

    cp_routing: ContextParallelRouting

    _ROUTING_SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.R,
        }
    )

    def annotate_spmd_types(self) -> None:
        """Annotate sequence offsets and rank-local CP routing tensors."""
        LinearAttentionMetadata.annotate_spmd_types(self)
        for value in vars(self.cp_routing).values():
            if isinstance(value, torch.Tensor):
                spmd.assert_type(value, self._ROUTING_SPMD_TYPE)


# CP metadata crosses CUDA-graph boundaries after the input sharding step.
# Expose every routing tensor so replay copies current values instead of
# treating either dataclass as an immutable non-tensor input. Guard the
# registration because test reloads can execute this module more than once.
if (
    not is_missing_kernel(ContextParallelRouting)
    and ContextParallelRouting not in pytree.SUPPORTED_NODES
):
    pytree.register_dataclass(ContextParallelRouting)
if ContextParallelLinearAttentionMetadata not in pytree.SUPPORTED_NODES:
    pytree.register_dataclass(ContextParallelLinearAttentionMetadata)


def prepare_linear_attention_cp_metadata(
    attention_metadata: LinearAttentionMetadata,
    *,
    permutation: torch.Tensor | None,
    family: str,
) -> ContextParallelLinearAttentionMetadata:
    """Build rank-local routing from global linear-attention sequence metadata.

    ``family`` names the attention (e.g. "KDA") in error messages.
    """
    if attention_metadata.varlen is None:
        if permutation is None:
            raise ValueError(
                f"{family} context parallelism requires global "
                "VarlenAttentionMetadata when using contiguous CP sharding."
            )
        cu_seqlens_global = [0, permutation.shape[1]]
        device = permutation.device
    else:
        cu_seqlens_global = attention_metadata.varlen.cu_seq_q.tolist()
        device = attention_metadata.varlen.cu_seq_q.device

    group = spmd_mesh_group(MeshAxisName.CP)
    if group is None:
        raise RuntimeError(
            f"{family} metadata preparation requires an active multi-rank CP "
            "mesh axis."
        )
    fragments = get_token_fragments(
        cu_seqlens_global[-1],
        cp_size=group.size(),
        permutation=permutation,
    )
    # Local document cuts vary even when the global offsets are padded.
    # Each additional fragment can split one document into another local
    # subsequence. Preserve that capacity at the CUDA-graph input boundary.
    max_subsequences = (
        len(cu_seqlens_global) - 1 + max(len(owned) for owned in fragments) - 1
    )
    routing = ContextParallelRouting.from_fragments(
        cu_seqlens_global=cu_seqlens_global,
        fragments=fragments,
        cp_rank=dist.get_rank(group),
        device=device,
        max_subsequences=max_subsequences,
        conv_history=attention_metadata.num_conv_history_tokens,
    )
    return ContextParallelLinearAttentionMetadata(
        varlen=attention_metadata.varlen,
        num_conv_history_tokens=attention_metadata.num_conv_history_tokens,
        cp_routing=routing,
    )
