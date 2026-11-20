# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How each trainer rank gets its microbatches: the DP replica's leader packs its column of the
step plan, and broadcasts each packed microbatch group to the replica's other ranks over gloo.

    controller --plan column + the rollouts it names--> leader (pipeline stage 0, CP and TP rank 0)
    leader, reader thread:  materialize -> pack group i -> broadcast -> preprocess_images
    other ranks, reader thread:                         receive   -> preprocess_images
    every rank, main thread: engine.forward_backward(reader), one group per pipeline step

Every rank of a replica trains on the same microbatches, so packing once per replica instead
of once per rank saves the pp * cp * tp - 1 other ranks' CPU, and they need neither the rollouts
nor the builder. Image files travel packed; every rank turns them into pixel values itself.
"""

from __future__ import annotations

import contextvars
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.distributed as dist

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.rollout.finished_group import join_rollout_group

if TYPE_CHECKING:
    from torchtitan.rl.rollout.finished_group import FinishedRolloutGroup
    from torchtitan.rl.trainer.batcher import MicrobatchPacker, TrainingMicrobatch
    from torchtitan.rl.trainer.training_sample_builder import (
        TrainingSample,
        TrainingSampleBuilder,
        TrainingSampleSpec,
    )

# Groups made ahead of the one the engine waits on. A rank holds at most 4 microbatch groups
# on the host: these 2, the one waited on, and the previous one, still in the engine's loop.
_NUM_MICROBATCH_GROUPS_READ_AHEAD = 2


@dataclass(frozen=True, slots=True)
class ReplicaColumn:
    """What a DP replica's leader packs: its column of the step plan and the rollouts it names.

    Example:
        # DP rank 1 of 2, one microbatch per group; its specs come from 2 rollouts of group 7
        ReplicaColumn(
            microbatch_groups=[[[spec_7_0, spec_7_2]], [[spec_7_0_turn_1]]],
            rollout_groups=[finished_group_7.select_rollouts({0, 2})],
        )
    """

    microbatch_groups: list[list[list[TrainingSampleSpec]]]
    """[num_groups][num_microbatches_per_group] -> the microbatch's specs, in packing order:
    column d of `StepPlan.microbatch_groups`."""

    rollout_groups: list[FinishedRolloutGroup]
    """Each group the column names, with only the rollouts it names."""


@dataclass(frozen=True, slots=True)
class _LeaderError:
    """What the leader broadcasts instead of a microbatch group when packing fails, so the
    other ranks raise instead of waiting for a group that never comes."""

    message: str


class MicrobatchGroupReader:
    """Yields this rank's microbatch groups for one step in order, made ahead on a background
    thread: packed and broadcast by the leader, received by the other ranks of the replica.

    Implements only what `TrainingEngine.forward_backward` uses: `len()` and in-order passes.
    Each pass packs and broadcasts again, on every rank of the replica.

    Example:
        reader = MicrobatchGroupReader(
            num_microbatch_groups=2,
            replica_group=replica_group,  # None when the replica is this rank alone
            column=column,  # on the leader; None on the other ranks
            training_sample_builder=builder,
            microbatch_packer=packer,
        )
        len(reader)  # 2
        for microbatch_group in reader:
            train(microbatch_group)

        leader thread:  pack 0, send 0 | pack 1, send 1 |
        other threads:  receive 0      | receive 1      |
        main threads:   wait           | train 0        | train 1
    """

    def __init__(
        self,
        *,
        num_microbatch_groups: int,
        replica_group: dist.ProcessGroup | None,
        column: ReplicaColumn | None,
        training_sample_builder: TrainingSampleBuilder,
        microbatch_packer: MicrobatchPacker,
    ) -> None:
        # The replica's rank 0 is its leader.
        assert column is not None or replica_group is not None
        self._num_microbatch_groups = num_microbatch_groups
        self._replica_group = replica_group
        self._column = column
        self._training_sample_builder = training_sample_builder
        self._microbatch_packer = microbatch_packer
        # The leader's materialized samples, in column order, and where each group's samples start.
        self._training_samples: list[TrainingSample] | None = None
        self._sample_starts: list[int] = []
        # Set once the leader fails to pack, on every rank: later groups raise it without a
        # broadcast, so no rank waits for a group the others never send.
        self._leader_error: str | None = None
        # This rank's total times, hidden or not; final once a pass ends.
        self.pack_s = 0.0
        self.broadcast_s = 0.0

    def __len__(self) -> int:
        return self._num_microbatch_groups

    def __iter__(self) -> Iterator[list[TrainingMicrobatch]]:
        def wait_for_microbatch_group(
            read: Future[list[TrainingMicrobatch]],
        ) -> list[TrainingMicrobatch]:
            # The span is the read time that training did not hide.
            with sl.log_trace_span("wait_for_microbatch"):
                return read.result()

        pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="microbatch_group_read"
        )
        reads: deque[Future[list[TrainingMicrobatch]]] = deque()
        try:
            for index in range(self._num_microbatch_groups):
                # Copy the actor's context: in a bare thread an actor framework can start a
                # second client (Monarch does).
                reads.append(
                    pool.submit(contextvars.copy_context().run, self._read, index)
                )
                if len(reads) > _NUM_MICROBATCH_GROUPS_READ_AHEAD:
                    yield wait_for_microbatch_group(reads.popleft())
            while reads:
                yield wait_for_microbatch_group(reads.popleft())
        finally:
            # The pass ended, or the generator was closed (e.g. a failed step). Cancel the queued
            # reads, and don't wait for the running one: a rank receiving from a leader that
            # failed would wait out the gloo timeout.
            pool.shutdown(wait=False, cancel_futures=True)

    def _read(self, index: int) -> list[TrainingMicrobatch]:
        if self._leader_error is not None:
            raise RuntimeError(
                f"The DP replica's leader failed to pack a microbatch group: {self._leader_error}"
            )
        if self._column is None:
            microbatch_group = self._receive()
        else:
            try:
                microbatch_group = self._pack(index, self._column)
            except Exception as error:
                self._leader_error = f"{type(error).__name__}: {error}"
                self._send(_LeaderError(self._leader_error))
                raise
            self._send(microbatch_group)
        # Here too, so training hides it like the packing.
        return [microbatch.preprocess_images() for microbatch in microbatch_group]

    def _pack(self, index: int, column: ReplicaColumn) -> list[TrainingMicrobatch]:
        start = time.perf_counter()
        if self._training_samples is None:
            specs = [
                spec
                for group in column.microbatch_groups
                for microbatch_specs in group
                for spec in microbatch_specs
            ]
            self._training_samples = self._training_sample_builder.materialize(
                training_sample_specs=specs,
                rollout_groups={
                    group.group_id: join_rollout_group(group.summary, group.arrays)
                    for group in column.rollout_groups
                },
            )
            num_samples = 0
            for group in column.microbatch_groups:
                self._sample_starts.append(num_samples)
                num_samples += sum(len(specs) for specs in group)
        microbatch_group = []
        sample_start = self._sample_starts[index]
        for microbatch_specs in column.microbatch_groups[index]:
            sample_end = sample_start + len(microbatch_specs)
            microbatch_group.append(
                self._microbatch_packer.pack(
                    self._training_samples[sample_start:sample_end]
                )
            )
            sample_start = sample_end
        self.pack_s += time.perf_counter() - start
        return microbatch_group

    def _send(self, value: list[TrainingMicrobatch] | _LeaderError) -> None:
        if self._replica_group is None:
            return
        start = time.perf_counter()
        dist.broadcast_object_list([value], group=self._replica_group, group_src=0)
        self.broadcast_s += time.perf_counter() - start

    def _receive(self) -> list[TrainingMicrobatch]:
        start = time.perf_counter()
        received: list[list[TrainingMicrobatch] | _LeaderError | None] = [None]
        dist.broadcast_object_list(received, group=self._replica_group, group_src=0)
        self.broadcast_s += time.perf_counter() - start
        value = received[0]
        if isinstance(value, _LeaderError):
            self._leader_error = value.message
            raise RuntimeError(
                f"The DP replica's leader failed to pack a microbatch group: {value.message}"
            )
        assert value is not None
        return value
