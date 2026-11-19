# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker sends back for a finished rollout group, and the group's split into a
summary (what the batcher decides and plans from) and its arrays (what packing reads)."""

from __future__ import annotations

import functools
from dataclasses import dataclass, fields, replace

import numpy as np

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder
from torchtitan.rl.rollout.types import PromptImage, RolloutGroup


@dataclass(frozen=True, slots=True)
class FinishedRolloutGroup:
    """A finished rollout group, as the rollout worker returns it.

    For training, the worker has already logged and recorded the full group, so `group` keeps
    only what training needs (see `slim_rollout_group`). For validation, `group` is the full
    group and the controller records it.
    """

    group: RolloutGroup

    recorder_lines: str
    """The group's `RolloutSampleRecorder` JSONL lines, encoded from the full group; "" for
    validation."""


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


def share_image_data(group: RolloutGroup) -> None:
    """Point every image with the same `hash_value` at one `bytes` object, in place.

    The envs read each image file once per rollout, so a group of G rollouts holds G equal
    copies. Pickle writes an object it has seen before as a reference, so after this the group
    pickles each image file once.
    """
    data_by_hash: dict[str, bytes] = {}
    for rollout in group.rollouts:
        for turn in rollout.turns:
            if not turn.prompt_delta_images:
                continue
            images: list[PromptImage] = []
            for image in turn.prompt_delta_images:
                data = data_by_hash.setdefault(image.hash_value, image.data)
                images.append(
                    image if data is image.data else replace(image, data=data)
                )
            turn.prompt_delta_images = tuple(images)


def finish_rollout_group(
    group: RolloutGroup, *, for_training: bool, encoder: RolloutSampleEncoder
) -> FinishedRolloutGroup:
    """Turn a scored group into what the rollout worker returns.

    For training: compute the group's rollout metrics and recorder lines from the full group,
    then slim it and share its image files. Validation aggregates metrics across groups and
    records full groups in the controller, so its group is returned as is.
    """
    if not for_training:
        return FinishedRolloutGroup(group=group, recorder_lines="")
    group.metrics.extend(
        compute_rollout_metrics(prefix="rollout", rollouts=group.rollouts)
    )
    recorder_lines = encoder.encode(is_validation=False, rollout_groups=[group])
    slim_group = slim_rollout_group(group)
    share_image_data(slim_group)
    return FinishedRolloutGroup(group=slim_group, recorder_lines=recorder_lines)


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
    summary.

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
