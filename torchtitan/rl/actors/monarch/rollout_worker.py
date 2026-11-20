# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_rank

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch.rdma_object import (
    create_cuda_context_to_silence_monarch_log,
    MonarchRDMAObject,
)
from torchtitan.rl.rollout.finished_group import (
    finish_rollout_group,
    FinishedRolloutGroup,
    RolloutGroupBuffers,
)
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.torch_transport_rdma import (
    make_rdma_probe,
    RDMAByteBuffers,
    RDMAProbe,
)


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        worker_kwargs: dict[str, Any],
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
    ) -> None:
        if structured_logger_init_fn is not None:
            structured_logger_init_fn(
                source="rl_rollout_worker", rank=current_rank().rank
            )
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        if worker_config.rdma_min_image_bytes is not None:
            # On the event loop's thread, where `run_group` registers the image files.
            create_cuda_context_to_silence_monarch_log()
            worker_kwargs = {
                **worker_kwargs,
                "register_rdma_object": MonarchRDMAObject.register,
            }
        self._worker = worker_config.build(**worker_kwargs)
        self._encoder: RolloutSampleEncoder
        # Large training groups wait here, packed, until the controller retires them.
        self._group_buffers: RolloutGroupBuffers | None = None
        self._rdma_byte_buffers: RDMAByteBuffers | None = None
        self._rdma_probe: RDMAProbe | None = None
        if worker_config.rdma_min_group_bytes is not None:
            self._rdma_byte_buffers = RDMAByteBuffers()
            self._group_buffers = RolloutGroupBuffers(
                min_bytes=worker_config.rdma_min_group_bytes,
                buffers=self._rdma_byte_buffers,
            )

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        rollout_recorder_config: RolloutSampleRecorder.Config,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        self._encoder = RolloutSampleEncoder(rollout_recorder_config)
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
        )

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        generate_fn: Any,
        release_session_fn: Any,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
        for_training: bool,
    ) -> tuple[FinishedRolloutGroup, float]:
        """Returns the finished group (see `finish_rollout_group`) and the seconds this handler
        took; the caller subtracts them from its wall time to get the hop's transfer time (see
        `compute_hop_metrics`)."""
        start = time.perf_counter()
        group = await self._worker.run_group(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )
        finished = finish_rollout_group(
            group, for_training=for_training, encoder=self._encoder
        )
        if for_training and self._group_buffers is not None:
            finished = self._group_buffers.pack(finished)
        return finished, time.perf_counter() - start

    @concurrent_endpoint
    async def release_groups(self, group_ids: list[int]) -> None:
        """Return the buffers of the retired groups packed here to the pool."""
        assert self._group_buffers is not None
        self._group_buffers.release(group_ids)

    @concurrent_endpoint
    async def make_rdma_probe(self) -> RDMAProbe:
        """A small buffer for every trainer DP replica leader to read at setup."""
        assert self._rdma_byte_buffers is not None and self._rdma_probe is None
        self._rdma_probe = make_rdma_probe(self._rdma_byte_buffers)
        return self._rdma_probe

    @concurrent_endpoint
    async def drop_rdma_probe(self) -> None:
        assert self._rdma_byte_buffers is not None and self._rdma_probe is not None
        self._rdma_byte_buffers.release(self._rdma_probe.buffer)
        self._rdma_probe = None

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
