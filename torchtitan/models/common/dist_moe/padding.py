# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Padding rows for Dist-MoE inference.

Dist-MoE requires every expert-parallel rank to pass exactly the planned
number of local tokens ``T``. A rank holding fewer real tokens must pad. Dist-MoE
cannot tell a padding row from a real one: it dispatches, multiplies and
combines every row, so padding costs network traffic, GEMM work and scratch
capacity on whichever rank owns the expert it is routed to.

Padding rows therefore get zero scores and are routed to experts owned by the
rank that sends them. Nothing crosses the network, and no remote expert gains
load. Rows are spread round-robin over the local experts so that no single
local expert becomes a hotspot.

Shape suffixes: ``T`` local tokens, ``K`` selected experts, ``D`` model dim.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def route_padding_to_local_experts(
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    padding_mask_T: torch.Tensor,
    *,
    first_local_expert: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the scores of padding rows and route those rows to local experts.

    ``padding_mask_T`` is TorchTitan's padding convention: a bool ``(T,)`` that
    is true for padding. Shapes stay static and the mask may be computed on
    the device from a stable buffer, so the operation is CUDA-graph capturable.
    The local experts of an EP rank are the contiguous range
    ``[first_local_expert, first_local_expert + num_local_experts)``, which is
    how Dist-MoE maps ``expert_id // num_local_experts`` to a rank.

    Args:
        topk_scores_TK: Routing weights.
        topk_expert_ids_TK: Selected global expert IDs.
        padding_mask_T: True for rows that are padding.
        first_local_expert: Global ID of this rank's first expert.
        num_local_experts: Number of experts this rank owns.

    Returns:
        Scores and expert IDs, with padding rows rewritten.
    """
    num_tokens, top_k = topk_expert_ids_TK.shape
    device = topk_expert_ids_TK.device
    row_TK = torch.arange(num_tokens, device=device)[:, None]
    slot_TK = torch.arange(top_k, device=device)[None, :]
    local_ids_TK = first_local_expert + (row_TK * top_k + slot_TK) % num_local_experts
    is_padding_T1 = padding_mask_T[:, None]
    expert_ids_TK = torch.where(
        is_padding_T1, local_ids_TK.to(topk_expert_ids_TK.dtype), topk_expert_ids_TK
    )
    scores_TK = torch.where(
        is_padding_T1, torch.zeros_like(topk_scores_TK), topk_scores_TK
    )
    return scores_TK, expert_ids_TK


def pad_to_num_tokens(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    num_tokens: int,
    *,
    first_local_expert: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append rows up to ``num_tokens``, routed to this rank's own experts."""
    num_valid_tokens = x_TD.shape[0]
    extra = num_tokens - num_valid_tokens
    if extra < 0:
        raise ValueError(f"cannot pad {num_valid_tokens} tokens down to {num_tokens}")
    if extra == 0:
        return x_TD, topk_scores_TK, topk_expert_ids_TK
    x_TD = F.pad(x_TD, (0, 0, 0, extra))
    topk_scores_TK = F.pad(topk_scores_TK, (0, 0, 0, extra))
    topk_expert_ids_TK = F.pad(topk_expert_ids_TK, (0, 0, 0, extra))
    padding_mask_T = torch.arange(num_tokens, device=x_TD.device) >= num_valid_tokens
    topk_scores_TK, topk_expert_ids_TK = route_padding_to_local_experts(
        topk_scores_TK,
        topk_expert_ids_TK,
        padding_mask_T,
        first_local_expert=first_local_expert,
        num_local_experts=num_local_experts,
    )
    return x_TD, topk_scores_TK, topk_expert_ids_TK


def _engine_step() -> object | None:
    """vLLM's per-step forward context, or None outside a vLLM forward."""
    try:
        from vllm.forward_context import get_forward_context

        return get_forward_context()
    except Exception:  # noqa: BLE001  (vLLM absent or not inside a forward)
        return None


class LocalExpertPadding:
    """Keep padding rows on the rank that sends them, for one EP group.

    Args:
        ep_pg: The expert-parallel process group.
        num_local_experts: Experts per EP rank (a contiguous block per rank).
        max_local_input_tokens: Planned local token count of the context.
    """

    def __init__(
        self,
        ep_pg: dist.ProcessGroup,
        *,
        num_local_experts: int,
        max_local_input_tokens: int,
    ) -> None:
        self.ep_pg = ep_pg
        self.num_local_experts = num_local_experts
        self.first_local_expert = dist.get_rank(ep_pg) * num_local_experts
        self.max_local_input_tokens = max_local_input_tokens
        self._count_group: dist.ProcessGroup | None = None
        self._count_cache: tuple[object, int, int] | None = None

    def route(
        self,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        padding_mask_T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route rows the caller already padded (vLLM) to this rank's experts."""
        return route_padding_to_local_experts(
            topk_scores_TK,
            topk_expert_ids_TK,
            padding_mask_T,
            first_local_expert=self.first_local_expert,
            num_local_experts=self.num_local_experts,
        )

    def equalize(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad this rank to the EP group's largest token count.

        Dist-MoE needs the same token count on every EP rank. vLLM only
        equalizes DP ranks in CUDA-graph-synced steps, so eager steps arrive
        unequal. Callers slice the output back to the original count.
        """
        target = self._group_max_tokens(x_TD.shape[0])
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

    def _group_max_tokens(self, num_tokens: int) -> int:
        """Largest local token count over the EP group, once per engine step.

        One CPU all-reduce on a private gloo group, cached on vLLM's forward
        context. Under CUDA-graph capture the count is already DP-padded and a
        host collective may not run.
        """
        if torch.cuda.is_current_stream_capturing():
            return num_tokens
        step = _engine_step()
        cache = self._count_cache
        if step is not None and cache and cache[0] is step and cache[1] == num_tokens:
            return cache[2]
        if self._count_group is None:
            # Collective: every rank reaches its first forward together.
            group = dist.new_group(
                ranks=dist.get_process_group_ranks(self.ep_pg), backend="gloo"
            )
            assert isinstance(group, dist.ProcessGroup)
            self._count_group = group
        count = torch.tensor([num_tokens], dtype=torch.int32)
        dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self._count_group)
        result = int(count.item())
        if step is not None:
            self._count_cache = (step, num_tokens, result)
        return result
