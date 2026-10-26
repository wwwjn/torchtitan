# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pinned CPU staging for the generator's model state dict loads."""

from collections.abc import Iterable, Mapping

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.utils import get_local_tensor

# Each staging tensor starts at a multiple of this within its buffer, so a view of any
# dtype is valid.
_ALIGNMENT_BYTES = 64


def top_level_module(fqn: str) -> str:
    """The top-level module a state-dict entry belongs to: ``layers.<i>`` for a decoder
    layer's entries, the top-level child (e.g. ``tok_embeddings``) otherwise, and ``""``
    for the root module's own entries."""
    parts = fqn.split(".", 2)
    if len(parts) == 1:
        return ""
    return ".".join(parts[:2]) if parts[0] == "layers" else parts[0]


def group_fqns_by_module(fqns: Iterable[str]) -> list[list[str]]:
    """Group state-dict FQNs by ``top_level_module``, in first-seen order."""
    groups: dict[str, list[str]] = {}
    for fqn in fqns:
        groups.setdefault(top_level_module(fqn), []).append(fqn)
    return list(groups.values())


def _aligned_nbytes(value: torch.Tensor) -> int:
    local = get_local_tensor(value)
    nbytes = local.numel() * local.element_size()
    return -(-nbytes // _ALIGNMENT_BYTES) * _ALIGNMENT_BYTES


class ChunkedStateDictStaging:
    """Pinned CPU buffers for a state dict, in chunks of whole top-level modules that take
    turns in two buffers:

        state dict:  tok_embeddings layers.0 | layers.1 layers.2 | layers.3 layers.4 | norm output
        chunk:       0                       | 1                 | 2                 | 3
        buffer:      A                       | B                 | A                 | B

    Chunk ``i`` lives in buffer ``i % 2``, so it can be staged once chunk ``i - 2`` has
    been copied out, while chunk ``i - 1`` is being copied. A chunk holds as many
    consecutive modules as fit in a buffer.

    Args:
        state_dict: The state dict to stage, read only for its keys (in load order),
            shapes, dtypes and DTensor layouts.
        max_pinned_bytes: Upper bound on the pinned memory of the two buffers together.
        pin_memory: Whether to pin the buffers.
    """

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        max_pinned_bytes: int,
        pin_memory: bool = True,
    ):
        # The CUDA host allocator rounds a pinned block up to a power of two, so a buffer
        # holds at most the largest power of two in half the budget.
        capacity = 1 << max((max_pinned_bytes // 2).bit_length() - 1, 0)
        self.chunks: list[list[str]] = []
        """Keys of each chunk, in state dict order."""
        chunk_nbytes: list[int] = []
        for fqns in group_fqns_by_module(state_dict):
            nbytes = sum(_aligned_nbytes(state_dict[fqn]) for fqn in fqns)
            if nbytes > capacity:
                raise ValueError(
                    f"{top_level_module(fqns[0]) or 'the root module'} needs "
                    f"{nbytes / 2**30:.2f} GiB of staging on this rank, but a staging "
                    f"buffer holds {capacity / 2**30:.2f} GiB (the largest power of two "
                    f"in half of the {max_pinned_bytes / 2**30:.2f} GiB budget); raise "
                    f"the budget to at least {2 * nbytes / 2**30:.2f} GiB"
                )
            if self.chunks and chunk_nbytes[-1] + nbytes <= capacity:
                self.chunks[-1].extend(fqns)
                chunk_nbytes[-1] += nbytes
            else:
                self.chunks.append(list(fqns))
                chunk_nbytes.append(nbytes)

        buffers = [
            torch.empty(
                max(chunk_nbytes[parity::2], default=0),
                dtype=torch.uint8,
                pin_memory=pin_memory,
            )
            for parity in (0, 1)
        ]
        self.state_dict: dict[str, torch.Tensor] = {}
        """Every key's staging tensor, a view of its chunk's buffer."""
        for index, fqns in enumerate(self.chunks):
            buffer = buffers[index % 2]
            offset = 0
            for fqn in fqns:
                value = state_dict[fqn]
                local = get_local_tensor(value)
                view = (
                    buffer[offset : offset + local.numel() * local.element_size()]
                    .view(local.dtype)
                    .view(local.shape)
                )
                offset += _aligned_nbytes(value)
                if isinstance(value, DTensor):
                    # Keep the mesh and placements around the CPU view.
                    staged = torch.empty_like(value, device="cpu")
                    staged._local_tensor = view
                    self.state_dict[fqn] = staged
                else:
                    self.state_dict[fqn] = view
