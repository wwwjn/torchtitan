# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker sends back for a finished rollout group: the group split into a summary
(what the batcher decides and plans from) and its arrays (what the trainer packs from).

With `BaseRolloutWorker.Config.rdma_min_group_bytes` set, a large training group's arrays stay in
the rollout worker, packed into one registered buffer (`RolloutGroupBuffers`), and the message
carries a handle instead (`PackedRolloutGroup`); each trainer DP replica leader reads the rollouts
it trains on (`read_packed_rollout_groups`).
"""

from __future__ import annotations

import functools
import itertools
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import numpy as np

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder
from torchtitan.rl.rollout.types import RolloutGroup

if TYPE_CHECKING:
    import torch

    from torchtitan.rl.torch_transport_rdma import RDMAByteBuffer, RDMAByteBuffers


@dataclass(frozen=True, slots=True)
class FinishedRolloutGroup:
    """A finished rollout group, as the rollout worker returns it: split into its summary and its
    arrays (`split_rollout_group`), so the batcher gets only the summary and the trainer's DP
    replica leaders get only the arrays of the rollouts they train on (`select_rollouts`).

    For training, the worker has already logged and recorded the full group, so the group keeps
    only what training needs (see `slim_rollout_group`). For validation, it is the full group and
    the controller records it (`join_rollout_group(summary, arrays)`).
    """

    summary: RolloutGroupSummary

    arrays: RolloutGroupArrays | None
    """None when `packed` holds them."""

    recorder_lines: str = ""
    """The group's `RolloutSampleRecorder` JSONL lines, encoded from the full group; "" for
    validation."""

    packed: PackedRolloutGroup | None = None
    """Where the arrays wait in the rollout worker, when they are not in the message."""

    @property
    def group_id(self) -> int:
        return self.summary.group.group_id

    @property
    def num_bytes(self) -> int:
        """The arrays' bytes it carries plus each image file once: the bulk of its pickle. 0 when
        packed."""
        if self.arrays is None:
            return 0
        return sum(
            array.nbytes
            for rollout_turn_arrays in self.arrays.turn_arrays
            for turn_arrays in rollout_turn_arrays
            for array in turn_arrays.values()
        ) + sum(len(data) for data in self.arrays.image_data.values())

    @property
    def num_packed_bytes(self) -> int:
        """The bytes a reader of its packed rollouts and images reads; 0 when not packed."""
        if self.packed is None:
            return 0
        return sum(region.num_bytes for region in self.packed.regions())

    def select_rollouts(self, rollout_ids: set[int]) -> FinishedRolloutGroup:
        """What a trainer DP replica leader gets of this group: only the rollouts in
        `rollout_ids` (by `Rollout.rollout_id`), their arrays and the image files they show.

        The rollouts are unchanged, so `rollout_to_training_samples` reads the same fields. The
        group's `metrics` are dropped: the batcher logs them from its copy of the summary, and
        they are many small objects (one per generator stat per turn) that the controller would
        pickle a second time. No recorder lines.

        Example:
            finished.select_rollouts({0, 3}).summary.group.rollouts  # [rollout 0, rollout 3]
        """
        kept = [
            index
            for index, rollout in enumerate(self.summary.group.rollouts)
            if rollout.rollout_id in rollout_ids
        ]
        group = self.summary.group
        rollouts = [group.rollouts[index] for index in kept]
        image_hashes = {
            image.hash_value
            for rollout in rollouts
            for turn in rollout.turns
            for image in turn.prompt_delta_images
        }
        return FinishedRolloutGroup(
            summary=RolloutGroupSummary(
                group=replace(group, rollouts=rollouts, metrics=[]),
                turn_lengths=tuple(self.summary.turn_lengths[index] for index in kept),
            ),
            arrays=(
                None
                if self.arrays is None
                else RolloutGroupArrays(
                    turn_arrays=tuple(self.arrays.turn_arrays[index] for index in kept),
                    image_data={
                        hash_value: data
                        for hash_value, data in self.arrays.image_data.items()
                        if hash_value in image_hashes
                    },
                )
            ),
            packed=(
                None
                if self.packed is None
                else self.packed.select(kept, image_hashes=image_hashes)
            ),
        )


def slim_rollout_group(group: RolloutGroup) -> RolloutGroup:
    """Return a copy of `group` without the fields only logging and the recorder read.

    Drops each rollout's `reward_breakdown` and `logs` and each turn's messages, `env_rewards`
    and `metrics`. It names only what it drops, so every other field survives, including
    per-token arrays that a feature adds to `RolloutTurn` (or a subclass of it).

    Example:
        group.rollouts[0].turns[0].prompt_messages   # [{"role": "user", "content": "sort..."}]
        slim_rollout_group(group).rollouts[0].turns[0].prompt_messages   # []
    """
    return replace(
        group,
        rollouts=[
            replace(
                rollout,
                reward_breakdown={},
                logs={},
                turns=[
                    replace(
                        turn,
                        prompt_messages=[],
                        completion_message=None,
                        env_messages=[],
                        env_rewards={},
                        metrics=[],
                    )
                    for turn in rollout.turns
                ],
            )
            for rollout in group.rollouts
        ],
    )


def finish_rollout_group(
    group: RolloutGroup, *, for_training: bool, encoder: RolloutSampleEncoder
) -> FinishedRolloutGroup:
    """Turn a scored group into what the rollout worker returns.

    For training: compute the group's rollout metrics and recorder lines from the full group,
    then slim it and split it. Validation aggregates metrics across groups and records full
    groups in the controller, so its full group is split as is.
    """
    if not for_training:
        return FinishedRolloutGroup(*split_rollout_group(group))
    group.metrics.extend(
        compute_rollout_metrics(prefix="rollout", rollouts=group.rollouts)
    )
    recorder_lines = encoder.encode(is_validation=False, rollout_groups=[group])
    summary, arrays = split_rollout_group(slim_rollout_group(group))
    return FinishedRolloutGroup(summary, arrays, recorder_lines)


@dataclass(frozen=True, slots=True)
class RolloutTurnLengths:
    """The lengths of a turn's arrays that deciding and planning read instead of the arrays."""

    num_prompt_delta_tokens: int
    """`len(prompt_delta_token_ids)`."""

    num_completion_tokens: int
    """`len(completion_token_ids)`."""

    completion_nonfinite_logprob_offsets: tuple[int, ...]
    """Positions in the completion whose generator logprob is not finite (no loss there);
    usually empty. Positions, not a count: an MTP depth's loss-token count skips a sample's
    first positions, so it depends on where they are."""


@dataclass(frozen=True, slots=True)
class RolloutGroupSummary:
    """A rollout group without its bytes: what the batcher decides and plans from.

    `group` keeps every field except the bytes: each `np.ndarray` field of each turn is cut to 0
    rows (same dtype and trailing shape, e.g. int32 [0, 32] for top-k ids) and each image's
    `data` is b"". The turns keep their class.
    """

    group: RolloutGroup

    turn_lengths: tuple[tuple[RolloutTurnLengths, ...], ...]
    """[rollout][turn], in `group.rollouts` and `Rollout.turns` order."""


@dataclass(frozen=True, slots=True)
class RolloutGroupArrays:
    """A rollout group's bytes: every `np.ndarray` field of every turn, and each image file once."""

    turn_arrays: tuple[tuple[dict[str, np.ndarray], ...], ...]
    """[rollout][turn] -> field name -> array."""

    image_data: dict[str, bytes]
    """`PromptImage.hash_value` -> the image file's bytes."""


@functools.cache
def _field_names(turn_type: type) -> tuple[str, ...]:
    return tuple(turn_field.name for turn_field in fields(turn_type))


def split_rollout_group(
    group: RolloutGroup,
) -> tuple[RolloutGroupSummary, RolloutGroupArrays]:
    """Split a group into its summary and its arrays; `join_rollout_group` reverses it.

    Every `np.ndarray`-valued field of a turn is an array, so per-token fields a feature adds
    (e.g. top-k logprobs) move with the arrays without code here; a `None` field stays in the
    summary. The envs read each image file once per rollout, so a group of G rollouts holds G
    equal copies; the arrays keep one per `hash_value`, so the group pickles each file once.

    Example:
        summary, arrays = split_rollout_group(group)
        summary.group.rollouts[0].turns[0].completion_token_ids  # array([], dtype=int32)
        summary.turn_lengths[0][0].num_completion_tokens          # 221
        arrays.turn_arrays[0][0]["completion_token_ids"]          # the 221 ids
        join_rollout_group(summary, arrays)                       # equal to group
    """
    summary_rollouts = []
    turn_lengths = []
    turn_arrays = []
    image_data: dict[str, bytes] = {}
    for rollout in group.rollouts:
        summary_turns = []
        rollout_turn_lengths = []
        rollout_turn_arrays = []
        for turn in rollout.turns:
            arrays = {
                name: value
                for name in _field_names(type(turn))
                if isinstance(value := getattr(turn, name), np.ndarray)
            }
            for image in turn.prompt_delta_images:
                image_data.setdefault(image.hash_value, image.data)
            finite = np.isfinite(turn.completion_logprobs)
            rollout_turn_lengths.append(
                RolloutTurnLengths(
                    num_prompt_delta_tokens=len(turn.prompt_delta_token_ids),
                    num_completion_tokens=len(turn.completion_token_ids),
                    completion_nonfinite_logprob_offsets=(
                        () if finite.all() else tuple(np.flatnonzero(~finite).tolist())
                    ),
                )
            )
            summary_turns.append(
                replace(
                    turn,
                    # A 0-row view; it pickles as an empty array.
                    **{name: array[:0] for name, array in arrays.items()},
                    prompt_delta_images=tuple(
                        replace(image, data=b"") for image in turn.prompt_delta_images
                    ),
                )
            )
            rollout_turn_arrays.append(arrays)
        summary_rollouts.append(replace(rollout, turns=summary_turns))
        turn_lengths.append(tuple(rollout_turn_lengths))
        turn_arrays.append(tuple(rollout_turn_arrays))
    summary = RolloutGroupSummary(
        group=replace(group, rollouts=summary_rollouts),
        turn_lengths=tuple(turn_lengths),
    )
    return summary, RolloutGroupArrays(
        turn_arrays=tuple(turn_arrays), image_data=image_data
    )


def join_rollout_group(
    summary: RolloutGroupSummary, arrays: RolloutGroupArrays
) -> RolloutGroup:
    """The group `split_rollout_group` split into `summary` and `arrays`."""
    return replace(
        summary.group,
        rollouts=[
            replace(
                rollout,
                turns=[
                    replace(
                        turn,
                        **turn_arrays,
                        prompt_delta_images=tuple(
                            replace(image, data=arrays.image_data[image.hash_value])
                            for image in turn.prompt_delta_images
                        ),
                    )
                    for turn, turn_arrays in zip(
                        rollout.turns, rollout_turn_arrays, strict=True
                    )
                ],
            )
            for rollout, rollout_turn_arrays in zip(
                summary.group.rollouts, arrays.turn_arrays, strict=True
            )
        ],
    )


# Every region, and every array in a region, starts at a multiple of this many bytes.
_ALIGNMENT = 64
# What each region starts and ends with: (group_id, generation).
_REGION_TAG = struct.Struct("<qq")


def _align(offset: int) -> int:
    return -(-offset // _ALIGNMENT) * _ALIGNMENT


@dataclass(frozen=True, slots=True)
class PackedArray:
    """Where one turn array sits in its rollout's region."""

    name: str
    """The `RolloutTurn` field, e.g. "completion_token_ids" or "generator_topk_logprobs"."""

    dtype: str
    """`np.dtype.str`, e.g. "<i4"."""

    shape: tuple[int, ...]

    offset: int
    """Bytes from the start of the region."""


@dataclass(frozen=True, slots=True)
class PackedRegion:
    """A byte range of a group's buffer: `(group_id, generation)`, the payload, then the same tag
    again, so a reader can tell the range still holds this group (`read_packed_rollout_groups`)."""

    offset: int
    """Bytes from the start of the buffer."""

    num_bytes: int
    """Both tags and the payload."""


@dataclass(frozen=True, slots=True)
class PackedRolloutGroup:
    """A rollout group's arrays, packed into one registered buffer in the rollout worker: one
    region per rollout and one per distinct image file, so a DP replica leader reads exactly the
    rollouts it trains on and the images they show.

        offset 0, every region 64-byte aligned
        | rollout 0 | rollout 1 | ... | rollout G-1 | image a | image b |
          region = [tag (group_id, generation) | payload | tag]
          rollout payload = each turn's arrays, in turn order (`turn_arrays`)
    """

    buffer: RDMAByteBuffer

    group_id: int

    generation: int
    """The worker's count of packed groups, so `(group_id, generation)` names one use of a
    pooled buffer."""

    rollout_regions: tuple[PackedRegion, ...]
    """In `RolloutGroupSummary.group.rollouts` order."""

    turn_arrays: tuple[tuple[tuple[PackedArray, ...], ...], ...]
    """[rollout][turn] -> where each of the turn's arrays sits in the rollout's region."""

    image_regions: dict[str, PackedRegion]
    """`PromptImage.hash_value` -> the region holding that image file."""

    def regions(self) -> list[PackedRegion]:
        return [*self.rollout_regions, *self.image_regions.values()]

    def select(
        self, rollout_indices: list[int], *, image_hashes: set[str]
    ) -> PackedRolloutGroup:
        """Only the rollouts at `rollout_indices` and the images in `image_hashes`: what a DP
        replica leader reads (see `FinishedRolloutGroup.select_rollouts`)."""
        return replace(
            self,
            rollout_regions=tuple(self.rollout_regions[i] for i in rollout_indices),
            turn_arrays=tuple(self.turn_arrays[i] for i in rollout_indices),
            image_regions={
                hash_value: region
                for hash_value, region in self.image_regions.items()
                if hash_value in image_hashes
            },
        )


def pack_rollout_group(
    arrays: RolloutGroupArrays,
    *,
    group_id: int,
    generation: int,
    allocate: Callable[[int], tuple[torch.Tensor, RDMAByteBuffer]],
) -> PackedRolloutGroup:
    """Copy `arrays` into one buffer from `allocate` (e.g. `RDMAByteBuffers.allocate`); the only
    copy. Every array field packs the same way whatever its shape (1-D token ids, `[N, k]` top-k
    rows), and a non-contiguous array is written contiguous.

    Example:
        packed = pack_rollout_group(arrays, group_id=7, generation=3, allocate=buffers.allocate)
        packed.rollout_regions[0]  # PackedRegion(offset=0, num_bytes=...)
    """
    position = 0
    rollout_regions = []
    turn_arrays = []
    for rollout_turn_arrays in arrays.turn_arrays:
        start = _align(position)
        position = start + _REGION_TAG.size
        rollout_layout = []
        for turn in rollout_turn_arrays:
            turn_layout = []
            for name, array in turn.items():
                position = _align(position)
                turn_layout.append(
                    PackedArray(
                        name=name,
                        dtype=array.dtype.str,
                        shape=array.shape,
                        offset=position - start,
                    )
                )
                position += array.nbytes
            rollout_layout.append(tuple(turn_layout))
        position += _REGION_TAG.size
        rollout_regions.append(PackedRegion(offset=start, num_bytes=position - start))
        turn_arrays.append(tuple(rollout_layout))
    image_regions = {}
    for hash_value, data in arrays.image_data.items():
        start = _align(position)
        position = start + 2 * _REGION_TAG.size + len(data)
        image_regions[hash_value] = PackedRegion(
            offset=start, num_bytes=position - start
        )

    tensor, handle = allocate(position)
    destination = tensor.numpy()
    tag = np.frombuffer(_REGION_TAG.pack(group_id, generation), dtype=np.uint8)

    def write_tags(region: PackedRegion) -> None:
        end = region.offset + region.num_bytes
        destination[region.offset : region.offset + _REGION_TAG.size] = tag
        destination[end - _REGION_TAG.size : end] = tag

    for region, rollout_turn_arrays, rollout_layout in zip(
        rollout_regions, arrays.turn_arrays, turn_arrays, strict=True
    ):
        for turn, turn_layout in zip(rollout_turn_arrays, rollout_layout, strict=True):
            for packed_array in turn_layout:
                array = turn[packed_array.name]
                start = region.offset + packed_array.offset
                np.copyto(
                    destination[start : start + array.nbytes]
                    .view(array.dtype)
                    .reshape(array.shape),
                    array,
                )
        write_tags(region)
    for hash_value, data in arrays.image_data.items():
        region = image_regions[hash_value]
        start = region.offset + _REGION_TAG.size
        destination[start : start + len(data)] = np.frombuffer(data, dtype=np.uint8)
        write_tags(region)
    return PackedRolloutGroup(
        buffer=handle,
        group_id=group_id,
        generation=generation,
        rollout_regions=tuple(rollout_regions),
        turn_arrays=tuple(turn_arrays),
        image_regions=image_regions,
    )


def read_packed_rollout_groups(
    groups: list[FinishedRolloutGroup], buffers: RDMAByteBuffers
) -> list[FinishedRolloutGroup]:
    """`groups` with each packed group's arrays read from its rollout worker: every region of
    every group in one `buffers.read` call, adjacent regions merged into one range. Groups sent by
    value pass through.

    Raises:
        RuntimeError: a region does not start and end with its group's `(group_id, generation)`,
            i.e. the buffer was released and reused before this read (a bug in the retire order).
    """
    packed_groups = [group for group in groups if group.packed is not None]
    requests = []
    for group in packed_groups:
        assert group.packed is not None
        requests.append((group.packed.buffer, _merged_ranges(group.packed.regions())))
    chunks_by_request = buffers.read(requests)
    read_groups = {}
    for group, (_, ranges), chunks in zip(
        packed_groups, requests, chunks_by_request, strict=True
    ):
        packed = group.packed
        assert packed is not None
        tag = _REGION_TAG.pack(packed.group_id, packed.generation)

        def region_bytes(region: PackedRegion) -> np.ndarray:
            # The merged range holding the region, then the region within it.
            for (range_start, range_length), chunk in zip(ranges, chunks, strict=True):
                if range_start <= region.offset < range_start + range_length:
                    start = region.offset - range_start
                    data = chunk[start : start + region.num_bytes]
                    head = data[: _REGION_TAG.size].tobytes()
                    tail = data[-_REGION_TAG.size :].tobytes()
                    if head != tag or tail != tag:
                        raise RuntimeError(
                            f"Rollout group {packed.group_id} (generation "
                            f"{packed.generation}): bytes {region.offset}.."
                            f"{region.offset + region.num_bytes} of its buffer hold "
                            f"{_REGION_TAG.unpack(head)} ... {_REGION_TAG.unpack(tail)}, so "
                            "the rollout worker reused the buffer before this read."
                        )
                    return data
            raise AssertionError(f"region {region} is in no read range")

        turn_arrays = []
        for region, rollout_layout in zip(
            packed.rollout_regions, packed.turn_arrays, strict=True
        ):
            data = region_bytes(region)
            turn_arrays.append(
                tuple(
                    {
                        packed_array.name: data[
                            packed_array.offset : packed_array.offset
                            + _num_bytes(packed_array)
                        ]
                        .view(np.dtype(packed_array.dtype))
                        .reshape(packed_array.shape)
                        for packed_array in turn_layout
                    }
                    for turn_layout in rollout_layout
                )
            )
        image_data = {
            hash_value: region_bytes(region)[
                _REGION_TAG.size : -_REGION_TAG.size
            ].tobytes()
            for hash_value, region in packed.image_regions.items()
        }
        read_groups[packed.group_id] = replace(
            group,
            arrays=RolloutGroupArrays(
                turn_arrays=tuple(turn_arrays), image_data=image_data
            ),
            packed=None,
        )
    return [read_groups.get(group.group_id, group) for group in groups]


def _num_bytes(packed_array: PackedArray) -> int:
    return (
        int(np.prod(packed_array.shape, dtype=np.int64))
        * np.dtype(packed_array.dtype).itemsize
    )


def _merged_ranges(regions: Iterable[PackedRegion]) -> list[tuple[int, int]]:
    """`(offset, num_bytes)` ranges covering `regions`, merging regions less than an alignment
    apart (only padding between them), so a leader that needs most of a group issues few reads."""
    ranges: list[list[int]] = []
    for region in sorted(regions, key=lambda packed_region: packed_region.offset):
        end = region.offset + region.num_bytes
        if ranges and region.offset - ranges[-1][1] < _ALIGNMENT:
            ranges[-1][1] = end
        else:
            ranges.append([region.offset, end])
    return [(start, end - start) for start, end in ranges]


class RolloutGroupBuffers:
    """A rollout worker's packed training groups: each in one registered buffer from `buffers`,
    from `pack` until the controller retires the group (`release`).

    Example:
        group_buffers = RolloutGroupBuffers(min_bytes=1 << 20, buffers=RDMAByteBuffers())
        finished = group_buffers.pack(finished)  # >= 1 MiB: arrays=None, packed=handle
        group_buffers.release([finished.group_id])  # after the step that trains it commits
    """

    def __init__(self, *, min_bytes: int, buffers: RDMAByteBuffers) -> None:
        self._min_bytes = min_bytes
        self._buffers = buffers
        self._generations = itertools.count(1)
        self._packed_by_group_id: dict[int, PackedRolloutGroup] = {}
        self._num_registrations_logged = 0

    def pack(self, finished: FinishedRolloutGroup) -> FinishedRolloutGroup:
        """`finished` with its arrays in a buffer here, if they are at least `min_bytes`;
        smaller groups go by value. Adds the worker's buffer metrics to the group's metrics."""
        if finished.num_bytes < self._min_bytes:
            return finished
        assert finished.arrays is not None
        start = time.perf_counter()
        packed = pack_rollout_group(
            finished.arrays,
            group_id=finished.group_id,
            generation=next(self._generations),
            allocate=self._buffers.allocate,
        )
        pack_s = time.perf_counter() - start
        self._packed_by_group_id[finished.group_id] = packed
        pool = self._buffers.pool
        new_registrations = pool.num_registrations - self._num_registrations_logged
        self._num_registrations_logged = pool.num_registrations
        finished.summary.group.metrics.extend(
            [
                m.Metric("timing/rollout_worker/pack_group", m.Mean(pack_s)),
                # 0 once the pool has a free buffer for every group size it sees.
                m.Metric("rollout_worker/pool_registrations", m.Sum(new_registrations)),
                m.Metric("rollout_worker/pool_bytes", m.Max(pool.registered_bytes)),
                m.Metric(
                    "rollout_worker/live_group_bytes",
                    m.Max(pool.registered_bytes - pool.free_bytes),
                ),
            ]
        )
        return replace(finished, arrays=None, packed=packed)

    def release(self, group_ids: Iterable[int]) -> None:
        """Return the buffers of the groups in `group_ids` that are here to the pool; ignore the
        others (the controller sends every worker the same ids)."""
        for group_id in group_ids:
            packed = self._packed_by_group_id.pop(group_id, None)
            if packed is not None:
                self._buffers.release(packed.buffer)
