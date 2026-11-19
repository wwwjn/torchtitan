# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The batcher plans from `TrainingSampleSpec`s, built from lengths; the samples
`materialize` builds from the arrays must agree with them."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from renderers import PlaceholderRange

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.finished_group import join_rollout_group, split_rollout_group
from torchtitan.rl.rollout.types import (
    Prompt,
    PromptImage,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
    split_prompt,
)
from torchtitan.rl.trainer.batcher import Batcher, MicrobatchPacker
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleBuilder,
)

_NAN = float("nan")


@dataclass(kw_only=True, slots=True)
class _TurnWithRoutedExperts(RolloutTurn):
    """A turn with one more per-token array, as router replay adds routed expert ids."""

    routed_expert_ids: np.ndarray  # [rows, num_layers, top_k] uint8


def _image(hash_value: str) -> PromptImage:
    return PromptImage(
        hash_value=hash_value,
        source_path=f"{hash_value}.png",
        grid_thw=(1, 4, 4),  # 4 placeholder tokens
        data=hash_value.encode() * 3,
    )


def _rollout(
    rollout_id: int,
    turns: list[tuple[list[int], list[float]]],
    *,
    group_id: int = 0,
    images: dict[int, tuple[tuple[PromptImage, int], ...]] | None = None,
    reward: float | None = None,
) -> Rollout:
    """A scored rollout from each turn's full prompt and completion logprobs (completion token
    ids are 200 + position). `images[turn]` lists the turn's full-prompt (image, offset)s."""
    previous = Prompt(token_ids=[])
    rollout_turns = []
    for turn_id, (prompt_token_ids, logprobs) in enumerate(turns):
        turn_images = (images or {}).get(turn_id, ())
        prompt = Prompt(
            token_ids=prompt_token_ids,
            images=tuple(image for image, _ in turn_images),
            image_placeholders=tuple(
                PlaceholderRange(offset=offset, length=4) for _, offset in turn_images
            ),
        )
        prefix_len, delta = split_prompt(prompt, previous=previous)
        num_prefix_images = sum(
            placeholder.offset < prefix_len for placeholder in prompt.image_placeholders
        )
        completion = np.arange(200, 200 + len(logprobs), dtype=np.int32)
        rollout_turns.append(
            RolloutTurn(
                rollout_id=RolloutTurnID(
                    group_id=group_id, rollout_id=rollout_id, turn_id=turn_id
                ),
                prompt_prefix_len=prefix_len,
                prompt_delta_token_ids=delta,
                prompt_delta_images=prompt.images[num_prefix_images:],
                prompt_delta_image_placeholders=prompt.image_placeholders[
                    num_prefix_images:
                ],
                completion_token_ids=completion,
                completion_logprobs=np.array(logprobs, dtype=np.float32),
                min_policy_version=turn_id,
                max_policy_version=turn_id + 1,
            )
        )
        previous = Prompt(
            token_ids=prompt_token_ids + completion.tolist(),
            images=prompt.images,
            image_placeholders=prompt.image_placeholders,
        )
    return Rollout(
        group_id=group_id,
        rollout_id=rollout_id,
        status=RolloutStatus.COMPLETED,
        turns=rollout_turns,
        reward=float(rollout_id) if reward is None else reward,
        advantage=float(rollout_id) - 1.0,
    )


def _trainable_group() -> RolloutGroup:
    image_a, image_b = _image("a"), _image("b")
    return RolloutGroup(
        group_id=0,
        rollouts=[
            # A 1-token prompt; NaN logprobs at the start and the middle of the completion.
            _rollout(0, [([1], [_NAN, -0.1, _NAN, -0.3, -0.4])]),
            # Two turns in one sample (the second prompt extends the first), NaN in the middle.
            _rollout(
                1,
                [
                    ([1, 2, 3], [-0.1, -0.2]),
                    ([1, 2, 3, 200, 201, 9], [-0.1, _NAN, -0.3]),
                ],
            ),
            # A history edit: two samples.
            _rollout(2, [([1, 2], [-0.1]), ([9, 9], [-0.1, -0.2])]),
            # Images: b joins in turn 1 (carried with a), then swapped in place in turn 2.
            _rollout(
                3,
                [
                    ([1, 7, 7, 7, 7, 2], [-0.1]),
                    ([1, 7, 7, 7, 7, 2, 200, 3, 7, 7, 7, 7], [-0.2, -0.3]),
                    ([1, 7, 7, 7, 7, 2, 200, 3, 7, 7, 7, 7, 200, 201], [-0.4]),
                ],
                images={
                    0: ((image_a, 1),),
                    1: ((image_a, 1), (image_b, 8)),
                    2: ((image_b, 1), (image_b, 8)),
                },
            ),
            # Only NaN logprobs: no loss token, so its sample is dropped.
            _rollout(4, [([1, 2], [_NAN, _NAN])]),
            # Later turn without a completion: the rollout is dropped.
            _rollout(5, [([1], [-0.1]), ([1, 200, 4], [])]),
        ],
        metrics=[m.Metric("rollout/x", m.Mean(1.0))],
    )


def _builder() -> TrainingSampleBuilder:
    return TrainingSampleBuilder.Config().build()


def _assert_samples_equal(actual: TrainingSample, expected: TrainingSample) -> None:
    for sample_field in dataclasses.fields(expected):
        actual_value = getattr(actual, sample_field.name)
        expected_value = getattr(expected, sample_field.name)
        if isinstance(expected_value, np.ndarray):
            assert actual_value.dtype == expected_value.dtype
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("num_mtp_layers", [0, 3])
@pytest.mark.parametrize("per_sample_pad_multiple", [1, 8])
def test_spec_token_counts_equal_the_packed_counts(
    num_mtp_layers: int, per_sample_pad_multiple: int
) -> None:
    group = _trainable_group()
    builder = _builder()
    summary, arrays = split_rollout_group(group)
    spec_group = builder.decide(summary=summary, num_mtp_layers=num_mtp_layers)
    specs = spec_group.training_sample_specs
    samples = builder.materialize(
        training_sample_specs=specs,
        rollout_groups={0: join_rollout_group(summary, arrays)},
    )
    packer = MicrobatchPacker(
        num_tokens_per_microbatch_per_dp_rank=64,
        max_context_length=64,
        pad_id=0,
        temperature=1.0,
        per_sample_pad_multiple=per_sample_pad_multiple,
        num_mtp_layers=num_mtp_layers,
        build_mrope_positions=False,
        vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    )

    # Rollouts 0 and 1 one sample each, rollouts 2 and 3 two each, rollout 4's sample dropped.
    assert [spec.rollout_id.rollout_id for spec in specs] == [0, 1, 2, 2, 3, 3]
    for spec, sample in zip(specs, samples, strict=True):
        assert spec.rollout_id == sample.rollout_id
        assert spec.num_tokens == len(sample.token_ids)
        assert (spec.images, spec.image_placeholders) == (
            sample.images,
            sample.image_placeholders,
        )
        assert spec.num_response_tokens == int(sample.loss_mask[1:].sum())
        microbatch = packer.pack([sample])
        assert list(spec.loss_token_counts) == microbatch.loss_token_counts.tolist()
        assert list(spec.routing_token_counts) == (
            microbatch.routing_token_counts.tolist()
        )
    # The samples are the ones rollout_to_training_samples builds from the full rollouts.
    expected = [
        sample
        for rollout in group.rollouts[:5]
        for sample in builder.rollout_to_training_samples(rollout)
        if sample.rollout_id.rollout_id != 4
    ]
    for sample, expected_sample in zip(samples, expected, strict=True):
        _assert_samples_equal(sample, expected_sample)


@pytest.mark.parametrize("dp_degree", [1, 2])
def test_plan_global_counts_equal_the_packed_microbatches(dp_degree: int) -> None:
    batcher = Batcher.Config(num_mtp_layers=3, per_sample_pad_multiple=8).build(
        num_tokens_per_microbatch_per_dp_rank=32,
        max_context_length=32,
        num_prompts_per_train_step=1,
        dp_degree=dp_degree,
        pad_id=0,
        temperature=1.0,
        vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
    )
    batch, _ = batcher.add_group(rollout_group=_trainable_group())

    assert batch is not None
    microbatches = [
        microbatch
        for group in batch.microbatch_groups
        for row in group
        for microbatch in row
    ]
    assert torch.equal(
        batch.global_loss_token_counts,
        torch.stack([microbatch.loss_token_counts for microbatch in microbatches]).sum(
            0
        ),
    )
    assert torch.equal(
        batch.global_routing_token_counts,
        torch.stack(
            [microbatch.routing_token_counts for microbatch in microbatches]
        ).sum(0),
    )
    assert batch.global_loss_token_counts.dtype == torch.int64


def test_decide_reads_lengths_and_keeps_the_filters_and_metrics() -> None:
    builder = _builder()
    summary, _ = split_rollout_group(_trainable_group())
    # The summary has no bytes: every array is empty and every image file is b"".
    for rollout in summary.group.rollouts:
        for turn in rollout.turns:
            assert len(turn.completion_token_ids) == len(turn.completion_logprobs) == 0
            assert all(image.data == b"" for image in turn.prompt_delta_images)

    spec_group = builder.decide(summary=summary, num_mtp_layers=0)

    assert m.MetricsProcessor._aggregate_metrics(spec_group.metrics) == {
        "rollout/x/mean": 1.0,
        "rollout_reward/group_zero_std_frac/mean": 0.0,
        "training_sample_builder/num_rollouts_dropped_empty_completion/sum": 1.0,
        "training_sample_builder/num_samples_dropped_no_valid_tokens/sum": 1.0,
        "training_sample_builder/num_training_samples/sum": 6.0,
        **m.MetricsProcessor._aggregate_metrics(
            [m.Metric("advantage", m.SummaryStats.from_list([-1.0, 0, 1, 2, 3, 4]))]
        ),
        "rollout/branches_per_rollout/mean": 7 / 5,
        "rollout/branches_per_rollout/max": 2.0,
        "rollout/min_policy_version/min": 0.0,
        "rollout/max_policy_version/max": 3.0,
    }


def test_decide_drops_failed_untrainable_and_zero_std_groups() -> None:
    builder = _builder()

    def decide(group: RolloutGroup) -> list[str]:
        spec_group = builder.decide(
            summary=split_rollout_group(group)[0], num_mtp_layers=0
        )
        assert spec_group.training_sample_specs == []
        return sorted(
            key
            for key in m.MetricsProcessor._aggregate_metrics(spec_group.metrics)
            if key.startswith("training_sample_builder/")
        )

    assert decide(RolloutGroup(group_id=1, rollouts=[])) == []
    # A first-turn prompt too long to generate from: the whole group is untrainable.
    untrainable = RolloutGroup(
        group_id=2, rollouts=[_rollout(0, [([1, 2], [-0.1])]), _rollout(1, [([1], [])])]
    )
    assert decide(untrainable) == [
        "training_sample_builder/num_groups_dropped_untrainable/sum"
    ]
    zero_std = RolloutGroup(
        group_id=3,
        rollouts=[
            _rollout(rollout_id, [([1], [-0.1])], reward=1.0) for rollout_id in (0, 1)
        ],
    )
    assert decide(zero_std) == [
        "training_sample_builder/num_groups_dropped_zero_std/sum"
    ]


def test_join_restores_the_split_group() -> None:
    group = _trainable_group()
    # A feature's extra per-token array rides with the arrays, by field name.
    turn = group.rollouts[1].turns[0]
    group.rollouts[1].turns[0] = _TurnWithRoutedExperts(
        **{
            turn_field.name: getattr(turn, turn_field.name)
            for turn_field in dataclasses.fields(RolloutTurn)
        },
        routed_expert_ids=np.ones((4, 48, 8), dtype=np.uint8),
    )

    summary, arrays = split_rollout_group(group)
    joined = join_rollout_group(summary, arrays)

    summary_turn = summary.group.rollouts[1].turns[0]
    assert isinstance(summary_turn, _TurnWithRoutedExperts)
    assert summary_turn.routed_expert_ids.shape == (0, 48, 8)
    assert summary_turn.routed_expert_ids.dtype == np.uint8
    assert summary.turn_lengths[0][0].completion_nonfinite_logprob_offsets == (0, 2)
    assert set(arrays.image_data) == {"a", "b"}
    for rollout, joined_rollout in zip(group.rollouts, joined.rollouts, strict=True):
        for turn, joined_turn in zip(rollout.turns, joined_rollout.turns, strict=True):
            assert type(joined_turn) is type(turn)
            for turn_field in dataclasses.fields(turn):
                value = getattr(turn, turn_field.name)
                joined_value = getattr(joined_turn, turn_field.name)
                if isinstance(value, np.ndarray):
                    assert joined_value is value
                else:
                    assert joined_value == value
            assert [image.data for image in joined_turn.prompt_delta_images] == [
                image.data for image in turn.prompt_delta_images
            ]
    assert joined.metrics == group.metrics


def test_materialize_raises_when_the_specs_disagree_with_the_samples() -> None:
    class _ShortSampleBuilder(TrainingSampleBuilder):
        """Overrides only the bytes half: its samples drop their last token."""

        def rollout_to_training_samples(self, rollout: Rollout) -> list[TrainingSample]:
            return [
                dataclasses.replace(
                    sample,
                    token_ids=sample.token_ids[:-1],
                    loss_mask=sample.loss_mask[:-1],
                    logprobs=sample.logprobs[:-1],
                    advantage=sample.advantage[:-1],
                )
                for sample in super().rollout_to_training_samples(rollout)
            ]

    group = RolloutGroup(group_id=0, rollouts=[_rollout(0, [([1, 2], [-0.1, -0.2])])])
    builder = _ShortSampleBuilder(TrainingSampleBuilder.Config())
    summary, arrays = split_rollout_group(group)
    spec_group = builder.decide(summary=summary, num_mtp_layers=0)

    with pytest.raises(ValueError, match="override rollout_to_training_samples"):
        builder.materialize(
            training_sample_specs=spec_group.training_sample_specs,
            rollout_groups={0: join_rollout_group(summary, arrays)},
        )
