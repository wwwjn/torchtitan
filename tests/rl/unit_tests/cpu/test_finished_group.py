# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.rollout.finished_group`."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, fields

import numpy as np
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
    slim_rollout_group,
    split_rollout_group,
)
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

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
