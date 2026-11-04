# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch's RDMA prompt image for the opt-in RDMA path (`Controller.Config.rdma_min_image_bytes`):
the rollout worker keeps an image file's bytes, and generator rank 0 reads them.
"""

from __future__ import annotations

import dataclasses
import functools

import torch
from monarch.rdma import RDMABuffer

from torchtitan.rl.actors.monarch.rdma_microbatch import (
    create_cuda_context_to_silence_monarch_log,
)
from torchtitan.rl.rollout.types import PromptImage


@dataclasses.dataclass(frozen=True, slots=True)
class MonarchRDMAPromptImage:
    """A `PromptImage` whose file bytes stay in its owner, the process that registered it.

    Example:
        rdma_image = register_rdma_prompt_image(image)  # owner
        image = await rdma_image.read()                  # generator rank 0
        await rdma_image.drop()                          # owner
    """

    image: PromptImage
    """The image, with empty ``data``; `read` fills it in."""

    buffer: RDMABuffer
    """The file bytes, in the owner."""

    async def read(self) -> PromptImage:
        data = torch.empty(self.buffer.size(), dtype=torch.uint8)
        await self.buffer.read_into(data)
        return dataclasses.replace(self.image, data=data.numpy().tobytes())

    async def drop(self) -> None:
        await self.buffer.drop()


def register_rdma_prompt_image(image: PromptImage) -> MonarchRDMAPromptImage:
    """Register ``image``'s file bytes as an RDMA buffer in this process.

    A module function, not a classmethod: a `GenerateFn` carries it to the rollout workers,
    and pickle finds a module function by name. Call it on one thread (the event loop's): the
    CUDA context it makes is current only there.
    """
    _create_cuda_context_once()
    # A copy: RDMABuffer takes a writable tensor, and bytes are read-only. The buffer keeps it
    # alive until drop().
    data = torch.frombuffer(bytearray(image.data), dtype=torch.uint8)
    return MonarchRDMAPromptImage(
        image=dataclasses.replace(image, data=b""), buffer=RDMABuffer(data)
    )


@functools.cache
def _create_cuda_context_once() -> None:
    # Without a current CUDA context, Monarch logs 2-3 lines per buffer (10.5k lines in a
    # 10-step SLAKE run), so every rollout worker that registers pays ~256 MiB of GPU 0, like
    # the batcher (see `create_cuda_context_to_silence_monarch_log` for the TODO).
    create_cuda_context_to_silence_monarch_log()
