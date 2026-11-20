# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker sends back for a finished rollout group: the group split into a summary
(what the batcher decides and plans from) and its arrays (what the trainer packs from)."""

from __future__ import annotations

import functools
from dataclasses import dataclass, fields, replace

import numpy as np

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder
from torchtitan.rl.rollout.types import RolloutGroup


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

    arrays: RolloutGroupArrays

    recorder_lines: str = ""
    """The group's `RolloutSampleRecorder` JSONL lines, encoded from the full group; "" for
    validation."""

    @property
    def group_id(self) -> int:
        return self.summary.group.group_id

    @property
    def num_bytes(self) -> int:
        """Its arrays' bytes plus each image file once: the bulk of its pickle."""
        return sum(
            array.nbytes
            for rollout_turn_arrays in self.arrays.turn_arrays
            for turn_arrays in rollout_turn_arrays
            for array in turn_arrays.values()
        ) + sum(len(data) for data in self.arrays.image_data.values())

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
            arrays=RolloutGroupArrays(
                turn_arrays=tuple(self.arrays.turn_arrays[index] for index in kept),
                image_data={
                    hash_value: data
                    for hash_value, data in self.arrays.image_data.items()
                    if hash_value in image_hashes
                },
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
