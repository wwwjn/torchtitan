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

Rollout groups packed in their rollout workers (`BaseRolloutWorker.Config.rdma_min_group_bytes`)
reach the leader as handles; before packing the first microbatch group, the leader reads the
rollouts its column names from every worker at once (`read_packed_rollout_groups`).
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
from torchtitan.rl.rollout.finished_group import (
    join_rollout_group,
    read_packed_rollout_groups,
)

if TYPE_CHECKING:
    from torchtitan.rl.rollout.finished_group import FinishedRolloutGroup
    from torchtitan.rl.rollout.types import RolloutTurnID
    from torchtitan.rl.torch_transport_rdma import RDMAByteBuffers
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
    """Each group the column names, with only the rollouts it names and no group metrics
    (`FinishedRolloutGroup.select_rollouts`): their arrays, or their handles when packed."""


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
        rdma_byte_buffers: RDMAByteBuffers | None = None,
    ) -> None:
        # The replica's rank 0 is its leader.
        assert column is not None or replica_group is not None
        self._num_microbatch_groups = num_microbatch_groups
        self._replica_group = replica_group
        # The leader's column. The reader keeps the rollout groups only until it materializes
        # them, so the caller must not hold `column` past this call.
        self._microbatch_groups = None if column is None else column.microbatch_groups
        self._rollout_groups = [] if column is None else list(column.rollout_groups)
        self._training_sample_builder = training_sample_builder
        self._microbatch_packer = microbatch_packer
        # Reads the column's packed groups; set on leaders when groups go by RDMA.
        self._rdma_byte_buffers = rdma_byte_buffers
        # The leader's materialized samples, in column order, and where each group's samples start.
        self._training_samples: list[TrainingSample] | None = None
        self._sample_starts: list[int] = []
        # Set once the leader fails to pack, on every rank: later groups raise it without a
        # broadcast, so no rank waits for a group the others never send.
        self._leader_error: str | None = None
        # This rank's total times, hidden or not; final once a pass ends.
        self.read_s = 0.0
        self.read_bytes = 0
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
        if self._microbatch_groups is None:
            microbatch_group = self._receive()
        else:
            try:
                microbatch_group = self._pack(index, self._microbatch_groups)
            except Exception as error:
                self._leader_error = f"{type(error).__name__}: {error}"
                self._send(_LeaderError(self._leader_error))
                raise
            self._send(microbatch_group)
        # Here too, so training hides it like the packing.
        return [microbatch.preprocess_images() for microbatch in microbatch_group]

    def _pack(
        self, index: int, microbatch_groups: list[list[list[TrainingSampleSpec]]]
    ) -> list[TrainingMicrobatch]:
        if self._training_samples is None:
            self._read_packed_rollout_groups()
        start = time.perf_counter()
        if self._training_samples is None:
            self._training_samples = self._materialize(microbatch_groups)
            num_samples = 0
            for group in microbatch_groups:
                self._sample_starts.append(num_samples)
                num_samples += sum(len(specs) for specs in group)
        microbatch_group = []
        sample_start = self._sample_starts[index]
        for microbatch_specs in microbatch_groups[index]:
            sample_end = sample_start + len(microbatch_specs)
            microbatch_group.append(
                self._microbatch_packer.pack(
                    self._training_samples[sample_start:sample_end]
                )
            )
            sample_start = sample_end
        self.pack_s += time.perf_counter() - start
        return microbatch_group

    def _read_packed_rollout_groups(self) -> None:
        """Read the column's packed groups from their rollout workers, all reads in flight
        together; groups sent by value pass through."""
        num_bytes = sum(group.num_packed_bytes for group in self._rollout_groups)
        if num_bytes == 0:
            return
        assert self._rdma_byte_buffers is not None
        start = time.perf_counter()
        self._rollout_groups = read_packed_rollout_groups(
            self._rollout_groups, self._rdma_byte_buffers
        )
        self.read_s += time.perf_counter() - start
        self.read_bytes += num_bytes

    def _materialize(
        self, microbatch_groups: list[list[list[TrainingSampleSpec]]]
    ) -> list[TrainingSample]:
        """The column's samples, in column order. One rollout group at a time, dropping its
        arrays as soon as its samples exist, so the column's arrays and its samples (copies)
        are never all alive together."""
        specs = [
            spec
            for group in microbatch_groups
            for microbatch_specs in group
            for spec in microbatch_specs
        ]
        specs_by_group_id: dict[int, list[TrainingSampleSpec]] = {}
        for spec in specs:
            specs_by_group_id.setdefault(spec.rollout_id.group_id, []).append(spec)
        samples_by_id: dict[RolloutTurnID, TrainingSample] = {}
        rollout_groups, self._rollout_groups = self._rollout_groups, []
        rollout_groups.reverse()
        while rollout_groups:
            rollout_group = rollout_groups.pop()
            group_specs = specs_by_group_id[rollout_group.group_id]
            assert rollout_group.arrays is not None
            samples = self._training_sample_builder.materialize(
                training_sample_specs=group_specs,
                rollout_groups={
                    rollout_group.group_id: join_rollout_group(
                        rollout_group.summary, rollout_group.arrays
                    )
                },
            )
            del rollout_group
            for spec, sample in zip(group_specs, samples, strict=True):
                samples_by_id[spec.rollout_id] = sample
        return [samples_by_id.pop(spec.rollout_id) for spec in specs]

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
