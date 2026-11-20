# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.finished_group`."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, fields, replace

import numpy as np
import pytest
from renderers import PlaceholderRange

from torchtitan.rl.controller.metrics import (
    compute_hop_metrics,
    compute_rollout_metrics,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.finished_group import (
    finish_rollout_group,
    FinishedRolloutGroup,
    join_rollout_group,
    pack_rollout_group,
    read_packed_rollout_groups,
    RolloutGroupArrays,
    RolloutGroupBuffers,
    slim_rollout_group,
    split_rollout_group,
)
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from tests.rl.unit_tests.cpu.test_torch_transport_rdma import loopback_buffers

_IMAGE_BYTES = 1 << 20


@dataclass(kw_only=True, slots=True)
class _TurnWithTopkLogprobs(RolloutTurn):
    """A turn with one more per-token array, as score centering adds top-k logprobs."""

    completion_topk_logprobs: np.ndarray


def _group(
    *, group_size: int = 8, with_image: bool = True, image_per_rollout: bool = False
) -> RolloutGroup:
    """A scored group whose rollouts each read the same image file (or with
    `image_per_rollout`, their own file) into their own bytes."""
    rollouts = []
    for rollout_id in range(group_size):
        images = (
            (
                PromptImage(
                    hash_value=f"h{rollout_id}" if image_per_rollout else "h",
                    source_path="a.png",
                    grid_thw=(1, 2, 2),
                    # A copy per rollout, as each env reads the file.
                    data=bytes(bytearray(_IMAGE_BYTES)),
                ),
            )
            if with_image
            else ()
        )
        turn = _TurnWithTopkLogprobs(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=rollout_id, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.arange(4, dtype=np.int32),
            prompt_delta_images=images,
            prompt_delta_image_placeholders=(
                (PlaceholderRange(offset=1, length=1),) if with_image else ()
            ),
            completion_token_ids=np.array([5, 6], dtype=np.int32),
            completion_logprobs=np.array([-0.5, -1.5], dtype=np.float32),
            completion_topk_logprobs=np.zeros((2, 4), dtype=np.float32),
            min_policy_version=1,
            max_policy_version=1,
            prompt_messages=[{"role": "user", "content": "sort: b a"}],
            completion_message={"role": "assistant", "content": "a b"},
            env_messages=[{"role": "tool", "content": "ok"}],
            env_rewards={"format": 1.0},
            metrics=[m.Metric("generator/queue_time_ms", m.Mean(2.0))],
        )
        rollouts.append(
            Rollout(
                group_id=3,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                turns=[turn],
                reward=float(rollout_id),
                reward_breakdown={"correct": float(rollout_id)},
                advantage=float(rollout_id) - 3.5,
                logs={"finish": "stop"},
            )
        )
    return RolloutGroup(
        group_id=3,
        rollouts=rollouts,
        metrics=[m.Metric("worker/x", m.Mean(1.0))],
    )


def test_slim_rollout_group_drops_only_the_logging_fields() -> None:
    group = _group(group_size=2)
    slim = slim_rollout_group(group)

    for rollout, slim_rollout in zip(group.rollouts, slim.rollouts, strict=True):
        assert (slim_rollout.reward_breakdown, slim_rollout.logs) == ({}, {})
        assert (slim_rollout.reward, slim_rollout.advantage, slim_rollout.status) == (
            rollout.reward,
            rollout.advantage,
            rollout.status,
        )
        (turn,), (slim_turn,) = rollout.turns, slim_rollout.turns
        # Keeps the turn's class and every array, including one RolloutTurn does not declare.
        assert isinstance(slim_turn, _TurnWithTopkLogprobs)
        assert isinstance(turn, _TurnWithTopkLogprobs)
        assert slim_turn.completion_topk_logprobs is turn.completion_topk_logprobs
        assert slim_turn.completion_logprobs is turn.completion_logprobs
        assert slim_turn.prompt_delta_images == turn.prompt_delta_images
        assert slim_turn.rollout_id == turn.rollout_id
        assert (slim_turn.prompt_messages, slim_turn.completion_message) == ([], None)
        assert (slim_turn.env_messages, slim_turn.env_rewards, slim_turn.metrics) == (
            [],
            {},
            [],
        )
        # The input group is left as it was.
        assert turn.prompt_messages and rollout.reward_breakdown
    assert slim.metrics == group.metrics


def test_a_split_group_pickles_each_image_file_once() -> None:
    group = _group()
    assert len(pickle.dumps(group)) > 8 * _IMAGE_BYTES

    summary, arrays = split_rollout_group(group)

    assert list(arrays.image_data) == ["h"]
    assert {
        image.data
        for rollout in summary.group.rollouts
        for turn in rollout.turns
        for image in turn.prompt_delta_images
    } == {b""}
    assert len(pickle.dumps((summary, arrays))) < 2 * _IMAGE_BYTES


def test_finish_rollout_group_for_training_logs_records_then_slims() -> None:
    group = _group()
    expected_metrics = m.MetricsProcessor._aggregate_metrics(
        group.metrics
        + compute_rollout_metrics(prefix="rollout", rollouts=group.rollouts)
    )
    recorder_config = RolloutSampleRecorder.Config(log_logprobs=True)
    expected_lines = RolloutSampleEncoder(recorder_config).encode(
        is_validation=False, rollout_groups=[group]
    )

    finished = finish_rollout_group(
        group, for_training=True, encoder=RolloutSampleEncoder(recorder_config)
    )

    # Metrics and recorder lines come from the full group (messages, reward breakdown).
    assert (
        m.MetricsProcessor._aggregate_metrics(finished.summary.group.metrics)
        == expected_metrics
    )
    assert "rollout_reward/component/correct/mean" in expected_metrics
    assert finished.recorder_lines == expected_lines
    assert '"sort: b a"' in finished.recorder_lines
    # The group is slim and split: the summary has no bytes, the arrays every array and one
    # copy of the image file.
    for rollout in finished.summary.group.rollouts:
        assert rollout.reward_breakdown == {}
        assert rollout.turns[0].prompt_messages == []
        assert len(rollout.turns[0].completion_token_ids) == 0
    assert set(finished.arrays.turn_arrays[0][0]) == {
        "prompt_delta_token_ids",
        "completion_token_ids",
        "completion_logprobs",
        "completion_topk_logprobs",
    }
    assert finished.num_bytes == 8 * (16 + 8 + 8 + 32) + _IMAGE_BYTES
    assert len(pickle.dumps(finished)) < 2 * _IMAGE_BYTES


def test_finish_rollout_group_for_validation_returns_the_full_group() -> None:
    group = _group(group_size=2, with_image=False)
    finished = finish_rollout_group(
        group,
        for_training=False,
        encoder=RolloutSampleEncoder(RolloutSampleRecorder.Config()),
    )
    full_group = join_rollout_group(finished.summary, finished.arrays)
    assert finished.recorder_lines == ""
    assert [metric.key for metric in full_group.metrics] == ["worker/x"]
    for rollout, full_rollout in zip(group.rollouts, full_group.rollouts, strict=True):
        assert full_rollout.reward_breakdown == rollout.reward_breakdown
        (turn,), (full_turn,) = rollout.turns, full_rollout.turns
        assert full_turn.prompt_messages == turn.prompt_messages
        assert full_turn.completion_token_ids is turn.completion_token_ids


def test_select_rollouts_keeps_only_the_named_rollouts_and_their_image_files() -> None:
    finished = FinishedRolloutGroup(
        *split_rollout_group(_group(group_size=3, image_per_rollout=True)),
        recorder_lines="lines",
    )

    selected = finished.select_rollouts({0, 2})

    assert [rollout.rollout_id for rollout in selected.summary.group.rollouts] == [0, 2]
    assert selected.summary.turn_lengths == (
        finished.summary.turn_lengths[0],
        finished.summary.turn_lengths[2],
    )
    assert selected.arrays.turn_arrays == (
        finished.arrays.turn_arrays[0],
        finished.arrays.turn_arrays[2],
    )
    assert list(selected.arrays.image_data) == ["h0", "h2"]
    assert selected.recorder_lines == ""
    assert selected.group_id == 3
    assert finished.num_bytes - selected.num_bytes == (16 + 8 + 8 + 32) + _IMAGE_BYTES
    rollout_2 = join_rollout_group(selected.summary, selected.arrays).rollouts[1]
    assert rollout_2.rollout_id == 2
    assert rollout_2.turns[0].prompt_delta_images[0].data == bytes(_IMAGE_BYTES)


def test_a_leaders_rollouts_carry_no_group_metrics_and_build_the_same_samples() -> None:
    """What the controller sends a trainer DP replica leader (`select_rollouts` of a training
    group) has no group metrics, messages or logs, and the leader builds the same training
    samples from it as from the full group."""
    group = _group(group_size=3, with_image=False)
    builder = TrainingSampleBuilder(TrainingSampleBuilder.Config())
    expected = [
        sample
        for rollout in group.rollouts
        for sample in builder.rollout_to_training_samples(rollout)
    ]
    finished = finish_rollout_group(
        group,
        for_training=True,
        encoder=RolloutSampleEncoder(RolloutSampleRecorder.Config()),
    )
    # As the controller adds them after `run_group`.
    finished.summary.group.metrics.extend(
        compute_hop_metrics("run_group", wall_s=[1.0], handler_s=[0.9])
    )
    specs = builder.decide(
        summary=finished.summary, num_mtp_layers=0
    ).training_sample_specs

    selected = finished.select_rollouts({0, 1, 2})

    assert finished.summary.group.metrics
    leader_group = join_rollout_group(selected.summary, selected.arrays)
    assert leader_group.metrics == []
    for rollout in leader_group.rollouts:
        assert (rollout.logs, rollout.reward_breakdown) == ({}, {})
        for turn in rollout.turns:
            assert (turn.prompt_messages, turn.completion_message) == ([], None)
            assert (turn.env_messages, turn.metrics) == ([], [])
    samples = builder.materialize(
        training_sample_specs=specs, rollout_groups={3: leader_group}
    )
    assert len(samples) == len(expected) == 3
    for sample, expected_sample in zip(samples, expected, strict=True):
        for sample_field in fields(sample):
            value = getattr(sample, sample_field.name)
            expected_value = getattr(expected_sample, sample_field.name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected_value)
            else:
                assert value == expected_value


def _multiturn_group(*, empty_completion: bool = True) -> RolloutGroup:
    """3 rollouts: one with an image and two turns, one with an empty completion (unless
    `empty_completion` is False: the batcher drops such a group), and turns that carry `[N, k]`
    top-k rows (a non-contiguous `[:, 1:]` view, as vLLM returns them) and a
    `uint8 [rows, 48, 8]` field, as router replay would."""
    rollouts = []
    for rollout_id in range(3):
        turns = []
        for turn_id in range(2 if rollout_id == 0 else 1):
            num_completion = 0 if rollout_id == 1 and empty_completion else 3 + turn_id
            topk = np.arange(num_completion * 5, dtype=np.float32).reshape(-1, 5)[:, 1:]
            turns.append(
                _TurnWithTopkAndExperts(
                    rollout_id=RolloutTurnID(
                        group_id=9, rollout_id=rollout_id, turn_id=turn_id
                    ),
                    # Turn 1 continues turn 0's 4 prompt and 3 completion tokens.
                    prompt_prefix_len=7 * turn_id,
                    prompt_delta_token_ids=np.arange(4 + turn_id, dtype=np.int32),
                    prompt_delta_images=(
                        (
                            PromptImage(
                                hash_value="img",
                                source_path="a.png",
                                grid_thw=(1, 2, 2),
                                data=bytes(range(256)) * 3,
                            ),
                        )
                        if rollout_id == 0 and turn_id == 0
                        else ()
                    ),
                    prompt_delta_image_placeholders=(
                        (PlaceholderRange(offset=1, length=1),)
                        if rollout_id == 0 and turn_id == 0
                        else ()
                    ),
                    completion_token_ids=np.arange(num_completion, dtype=np.int32) + 10,
                    completion_logprobs=np.full(num_completion, -0.5, dtype=np.float32),
                    completion_topk_logprobs=topk,
                    routed_expert_ids=np.full(
                        (num_completion + 3, 48, 8), rollout_id, dtype=np.uint8
                    ),
                    min_policy_version=1,
                    max_policy_version=1,
                )
            )
        rollouts.append(
            Rollout(
                group_id=9,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                turns=turns,
                reward=float(rollout_id),
                advantage=float(rollout_id) - 1.0,
            )
        )
    return RolloutGroup(group_id=9, rollouts=rollouts)


@dataclass(kw_only=True, slots=True)
class _TurnWithTopkAndExperts(RolloutTurn):
    completion_topk_logprobs: np.ndarray
    routed_expert_ids: np.ndarray


def _assert_arrays_equal(
    actual: RolloutGroupArrays, expected: RolloutGroupArrays
) -> None:
    assert len(actual.turn_arrays) == len(expected.turn_arrays)
    for actual_rollout, expected_rollout in zip(
        actual.turn_arrays, expected.turn_arrays, strict=True
    ):
        for actual_turn, expected_turn in zip(
            actual_rollout, expected_rollout, strict=True
        ):
            assert list(actual_turn) == list(expected_turn)
            for name, expected_array in expected_turn.items():
                assert actual_turn[name].dtype == expected_array.dtype
                np.testing.assert_array_equal(actual_turn[name], expected_array)
    assert actual.image_data == expected.image_data


def _packed(
    finished: FinishedRolloutGroup, owner, *, generation: int = 1
) -> FinishedRolloutGroup:
    packed = pack_rollout_group(
        finished.arrays,
        group_id=finished.group_id,
        generation=generation,
        allocate=owner.allocate,
    )
    return replace(finished, arrays=None, packed=packed)


def test_a_packed_group_reads_back_equal_to_its_arrays() -> None:
    owner, reader = loopback_buffers()
    finished = FinishedRolloutGroup(*split_rollout_group(_multiturn_group()))

    packed = _packed(finished, owner)
    [read] = read_packed_rollout_groups([packed], reader)

    assert (packed.arrays, packed.num_bytes) == (None, 0)
    assert read.packed is None
    _assert_arrays_equal(read.arrays, finished.arrays)
    assert read.arrays.turn_arrays[0][0]["completion_topk_logprobs"].shape == (3, 4)
    regions = packed.packed.regions()
    assert len(regions) == 3 + 1  # one per rollout, one per distinct image
    assert all(region.offset % 64 == 0 for region in regions)
    assert packed.num_packed_bytes == sum(region.num_bytes for region in regions)


def test_a_leader_reads_only_its_rollouts_and_their_images() -> None:
    owner, reader = loopback_buffers()
    finished = FinishedRolloutGroup(*split_rollout_group(_multiturn_group()))
    packed = _packed(finished, owner)
    regions = packed.packed.rollout_regions

    without_image = packed.select_rollouts({1, 2})
    [read] = read_packed_rollout_groups([without_image], reader)

    _assert_arrays_equal(read.arrays, finished.select_rollouts({1, 2}).arrays)
    # Rollouts 1 and 2 are adjacent: one read covering both, nothing of rollout 0 or the image.
    assert reader._transport.read_ranges == [
        (
            regions[1].offset,
            regions[2].offset + regions[2].num_bytes - regions[1].offset,
        )
    ]

    reader._transport.read_ranges.clear()
    [read] = read_packed_rollout_groups([packed.select_rollouts({0})], reader)
    _assert_arrays_equal(read.arrays, finished.select_rollouts({0}).arrays)
    assert (
        len(reader._transport.read_ranges) == 2
    )  # rollout 0 and the image, not adjacent


def test_groups_sent_by_value_pass_through_the_read() -> None:
    owner, reader = loopback_buffers()
    by_value = FinishedRolloutGroup(*split_rollout_group(_group(group_size=2)))
    packed = _packed(
        FinishedRolloutGroup(*split_rollout_group(_multiturn_group())), owner
    )

    read = read_packed_rollout_groups([by_value, packed], reader)

    assert read[0] is by_value
    assert read[1].group_id == 9 and read[1].arrays is not None


def test_a_read_of_a_reused_buffer_raises_naming_both_tags() -> None:
    """A handle read after its buffer went to another group (a retire-order bug) raises instead
    of returning the other group's bytes."""
    owner, reader = loopback_buffers()
    group_buffers = RolloutGroupBuffers(min_bytes=1, buffers=owner)
    old = group_buffers.pack(
        FinishedRolloutGroup(*split_rollout_group(_group(group_size=2)))
    )
    group_buffers.release([old.group_id])
    new_group = replace(_group(group_size=2), group_id=4)
    new = group_buffers.pack(FinishedRolloutGroup(*split_rollout_group(new_group)))

    assert new.packed.buffer.buffer_index == old.packed.buffer.buffer_index
    with pytest.raises(RuntimeError, match=r"group 3 \(generation 1\).*\(4, 2\)"):
        read_packed_rollout_groups([old], reader)


def test_a_torn_region_raises() -> None:
    owner, reader = loopback_buffers()
    packed = _packed(
        FinishedRolloutGroup(*split_rollout_group(_multiturn_group())), owner
    )
    region = packed.packed.rollout_regions[1]
    tensor = owner.pool.buffer(packed.packed.buffer.buffer_index).tensor
    tensor[region.offset + region.num_bytes - 1] += 1  # the trailing tag

    with pytest.raises(RuntimeError, match="reused the buffer"):
        read_packed_rollout_groups([packed], reader)


def test_rollout_group_buffers_pack_large_groups_and_reuse_released_buffers() -> None:
    owner, reader = loopback_buffers()
    finished = FinishedRolloutGroup(*split_rollout_group(_group(group_size=2)))
    group_buffers = RolloutGroupBuffers(min_bytes=finished.num_bytes, buffers=owner)

    small = replace(_group(group_size=1), group_id=5)
    assert (
        group_buffers.pack(FinishedRolloutGroup(*split_rollout_group(small))).packed
        is None
    )
    packed = group_buffers.pack(finished)

    assert packed.arrays is None and packed.packed.group_id == 3
    metrics = {metric.key: metric.value for metric in packed.summary.group.metrics}
    assert metrics["rollout_worker/pool_registrations"].value == 1
    assert (
        metrics["rollout_worker/live_group_bytes"].value == owner.pool.registered_bytes
    )
    [read] = read_packed_rollout_groups([packed], reader)
    _assert_arrays_equal(read.arrays, finished.arrays)

    group_buffers.release([3, 77])  # 77 is on another worker
    group_buffers.release([3])  # already released: nothing to do
    assert owner.pool.free_bytes == owner.pool.registered_bytes
    again = group_buffers.pack(replace(finished, summary=replace(finished.summary)))
    assert again.packed.buffer.buffer_index == packed.packed.buffer.buffer_index
    assert owner._transport.num_registrations == 1
    assert owner._transport.num_unregistrations == 0


def test_a_packed_group_builds_the_same_samples_as_by_value() -> None:
    owner, reader = loopback_buffers()
    group = _multiturn_group(empty_completion=False)
    builder = TrainingSampleBuilder(TrainingSampleBuilder.Config())
    finished = FinishedRolloutGroup(*split_rollout_group(group))
    specs = builder.decide(
        summary=finished.summary, num_mtp_layers=0
    ).training_sample_specs

    [read] = read_packed_rollout_groups([_packed(finished, owner)], reader)

    expected = builder.materialize(
        training_sample_specs=specs, rollout_groups={9: group}
    )
    samples = builder.materialize(
        training_sample_specs=specs,
        rollout_groups={9: join_rollout_group(read.summary, read.arrays)},
    )
    assert len(samples) == len(expected) > 0
    for sample, expected_sample in zip(samples, expected, strict=True):
        for sample_field in fields(sample):
            value = getattr(sample, sample_field.name)
            expected_value = getattr(expected_sample, sample_field.name)
            if isinstance(value, np.ndarray):
                assert value.dtype == expected_value.dtype
                np.testing.assert_array_equal(value, expected_value)
            else:
                assert value == expected_value
