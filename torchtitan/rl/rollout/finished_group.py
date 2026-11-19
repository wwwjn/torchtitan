# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker sends back for a finished rollout group."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
