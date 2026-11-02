# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in RDMA path for prompt image files (`Controller.Config.rdma_min_image_bytes`): the rollout
worker that registered a file keeps its bytes, and generator rank 0 reads them.

The actor system implements `RDMAPromptImage` (e.g. `actors/monarch/rdma_prompt_image.py`); the
controller's `GenerateFn` and the generator use only this interface.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from torchtitan.rl.rollout.types import PromptImage


@runtime_checkable
class RDMAPromptImage(Protocol):
    """A `PromptImage` whose file bytes stay in its owner, the process that registered it.

    Example:
        rdma_image = register_rdma_prompt_image(image)  # owner, before `generate`
        image = await rdma_image.read()                  # generator rank 0
        await rdma_image.drop()                          # owner, after `generate`
    """

    async def read(self) -> PromptImage:
        """Read the image, with its file bytes, from its owner."""
        ...

    async def drop(self) -> None:
        """Release the owner's buffer."""
        ...


RegisterRDMAPromptImage = Callable[[PromptImage], RDMAPromptImage]


async def read_prompt_images(
    images: Sequence[PromptImage | RDMAPromptImage],
) -> tuple[PromptImage, ...]:
    """Return ``images`` with every `RDMAPromptImage` read from its owner."""

    async def read(image: PromptImage | RDMAPromptImage) -> PromptImage:
        return await image.read() if isinstance(image, RDMAPromptImage) else image

    return tuple(await asyncio.gather(*(read(image) for image in images)))
