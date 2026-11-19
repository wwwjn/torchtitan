# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converts one generated rollout group into trainable training_samples, in two halves.

    decide:       RolloutGroupSummary -> group filters -> rollout_to_training_sample_specs
                  -> sample filters -> TrainingSampleSpecGroup                       (lengths only)
    materialize:  specs + their rollout groups -> rollout_to_training_samples
                  -> TrainingSamples, in spec order                                  (the arrays)
"""

import statistics
from dataclasses import dataclass, field

import numpy as np
from renderers import PlaceholderRange

from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup
from torchtitan.rl.rollout.finished_group import RolloutGroupSummary, RolloutTurnLengths
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID


@dataclass(kw_only=True, slots=True)
class TrainingSample:
    """A trainable token sequence from a rollout.

    Per-token fields are numpy arrays, like `RolloutTurn`'s:
    - not lists: a full GC walks every entry of a list, but doesn't track numpy arrays;
    - not tensors: the turns are already numpy, so a sample is built with no conversion; the batcher
      converts once, into the trainer's tensors.

    Example:
        # Turn 0: prompt P0 -> assistant A0 -> env reply E0
        # Turn 1: prompt [P0, A0, E0] -> assistant A1
        TrainingSample(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=1, turn_id=1),
            min_policy_version=7,
            max_policy_version=9,            # weights updated during rollout
            token_ids=np.array(P0 + A0 + E0 + A1),
            loss_mask=np.array([False]*len(P0) + [True]*len(A0) + [False]*len(E0) + [True]*len(A1)),
            logprobs=np.array([0.0]*len(P0) + logprobs_A0 + [0.0]*len(E0) + logprobs_A1, dtype=np.float32),
            advantage=np.array([0.0]*len(P0) + [adv]*len(A0) + [0.0]*len(E0) + [adv]*len(A1), dtype=np.float32),
        )
    """

    min_policy_version: int
    """Oldest policy version among this branch's trained turns."""
    max_policy_version: int
    """Newest policy version among this branch's trained turns."""
    rollout_id: RolloutTurnID
    """This sample identifier."""
    token_ids: np.ndarray  # int64
    """[L] packed prompt + completions + env replies."""
    loss_mask: np.ndarray  # bool
    """[L] True on assistant tokens to train."""
    logprobs: np.ndarray  # float32
    """[L] generator logprobs; 0.0 where loss_mask is False."""
    advantage: np.ndarray  # float32
    """[L] advantage on assistant tokens, 0.0 elsewhere."""
    images: tuple[PromptImage, ...] = ()
    """Every image in `token_ids`: the last turn's prompt images (images only appear in
    prompts, and that prompt is a prefix of `token_ids`)."""
    image_placeholders: tuple[PlaceholderRange, ...] = ()
    """Where each of `images` sits in `token_ids` (see `Prompt.image_placeholders`)."""


@dataclass(frozen=True, slots=True)
class TrainingSampleSpec:
    """What the batcher plans a `TrainingSample` with, computed from lengths without its arrays.

    `TrainingSampleBuilder.materialize` builds the sample a spec describes; their lengths
    agree, so the plan (bins, global token counts) made from specs holds for the samples.

    Example:
        # The single-turn rollout prompt [1, 2] -> completion [3, 4, 5], logprob of 4 NaN:
        TrainingSampleSpec(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=1, turn_id=0),
            last_turn_id=0,
            num_tokens=5,                 # len(token_ids)
            loss_token_counts=(2,),       # shifted completion positions with finite logprobs
            routing_token_counts=(4,),    # shifted positions
            num_response_tokens=3,        # shifted completion positions
            ...
        )
    """

    rollout_id: RolloutTurnID
    """The sample's `TrainingSample.rollout_id`: its rollout and the turn it begins at."""
    last_turn_id: int
    """The last turn in the sample (inclusive)."""
    num_tokens: int
    """`len(TrainingSample.token_ids)`."""
    loss_token_counts: tuple[int, ...]
    """[1 + num_mtp_layers] loss tokens at each MTP depth, as `get_mtp_token_counts` counts them
    for this sample in a packed microbatch."""
    routing_token_counts: tuple[int, ...]
    """[1 + num_mtp_layers] non-padding tokens at each MTP depth, as `get_mtp_token_counts`
    counts them."""
    num_response_tokens: int
    """Shifted completion positions (`TrainingSample.loss_mask[1:].sum()`), finite or not."""
    min_policy_version: int
    max_policy_version: int
    images: tuple[PromptImage, ...] = ()
    """`TrainingSample.images`, without their file bytes (`data` is b"")."""
    image_placeholders: tuple[PlaceholderRange, ...] = ()
    """`TrainingSample.image_placeholders`."""


@dataclass(frozen=True, slots=True)
class TrainingSampleSpecGroup:
    """The training sample specs + metrics decided for one rollout group.

    Example:
        TrainingSampleSpecGroup(group_id=3, training_sample_specs=[], metrics=[failure_metric])
        # -> failed / filtered / zero-std group; metrics still reach the trainer logger
    """

    group_id: int
    training_sample_specs: list[TrainingSampleSpec]
    metrics: list[m.Metric]


@dataclass(kw_only=True, slots=True)
class _SampleInProgress:
    """A training_sample being built: per field, a list of chunks that `to_training_sample` concatenates
    once, e.g. token_ids = [prompt0, completion0, prompt_delta1, completion1]."""

    rollout_id: RolloutTurnID
    min_policy_version: int
    max_policy_version: int
    token_ids: list[np.ndarray] = field(default_factory=list)
    loss_mask: list[np.ndarray] = field(default_factory=list)
    logprobs: list[np.ndarray] = field(default_factory=list)
    advantage: list[np.ndarray] = field(default_factory=list)
    images: tuple[PromptImage, ...] = ()
    image_placeholders: tuple[PlaceholderRange, ...] = ()

    def to_training_sample(self) -> TrainingSample:
        return TrainingSample(
            rollout_id=self.rollout_id,
            min_policy_version=self.min_policy_version,
            max_policy_version=self.max_policy_version,
            token_ids=np.concatenate(self.token_ids, dtype=np.int64),
            loss_mask=np.concatenate(self.loss_mask),
            logprobs=np.concatenate(self.logprobs, dtype=np.float32),
            advantage=np.concatenate(self.advantage),
            images=self.images,
            image_placeholders=self.image_placeholders,
        )


@dataclass(kw_only=True, slots=True)
class _SpecInProgress:
    """A training sample spec being built from turn lengths, as `_SampleInProgress` builds a
    sample from the turns' arrays."""

    rollout_id: RolloutTurnID
    min_policy_version: int
    max_policy_version: int
    last_turn_id: int
    num_tokens: int = 0
    # Per turn: where its completion starts in the sample's tokens, and the turn's lengths.
    completions: list[tuple[int, RolloutTurnLengths]] = field(default_factory=list)
    images: tuple[PromptImage, ...] = ()
    image_placeholders: tuple[PlaceholderRange, ...] = ()

    def to_training_sample_spec(self, *, num_mtp_layers: int) -> TrainingSampleSpec:
        # Labels are token_ids[1:], so token index i is shifted position i - 1. At MTP depth d,
        # `get_mtp_token_counts` counts a loss token at each shifted position j >= d whose label is
        # a completion token with a finite logprob (token index >= d + 1), and a routed token at
        # each shifted position j < n - d, for n = num_tokens - 1 shifted positions.
        loss_token_counts = []
        for depth in range(num_mtp_layers + 1):
            num_loss_tokens = 0
            for start, lengths in self.completions:
                first = max(start, depth + 1)
                end = start + lengths.num_completion_tokens
                num_loss_tokens += max(end - first, 0) - sum(
                    start + offset >= first
                    for offset in lengths.completion_nonfinite_logprob_offsets
                )
            loss_token_counts.append(num_loss_tokens)
        num_shifted_tokens = self.num_tokens - 1
        return TrainingSampleSpec(
            rollout_id=self.rollout_id,
            last_turn_id=self.last_turn_id,
            num_tokens=self.num_tokens,
            loss_token_counts=tuple(loss_token_counts),
            routing_token_counts=tuple(
                max(num_shifted_tokens - depth, 0)
                for depth in range(num_mtp_layers + 1)
            ),
            # A completion token at index 0 (an empty first prompt) is no label.
            num_response_tokens=sum(
                lengths.num_completion_tokens - (start == 0)
                for start, lengths in self.completions
            ),
            min_policy_version=self.min_policy_version,
            max_policy_version=self.max_policy_version,
            images=self.images,
            image_placeholders=self.image_placeholders,
        )


class TrainingSampleBuilder(Configurable):
    """Build trainable samples and metrics the trainer consumes from rollout groups, in two halves:
    `decide` filters a group and plans its samples from lengths (`TrainingSampleSpec`s), and
    `materialize` builds the samples of chosen specs from the groups' arrays.

    Example:
        builder = config.async_loop.batcher.training_sample_builder.build()
        summary, arrays = split_rollout_group(rollout_group)
        spec_group = builder.decide(summary=summary, num_mtp_layers=0)
        samples = builder.materialize(
            training_sample_specs=spec_group.training_sample_specs,
            rollout_groups={rollout_group.group_id: join_rollout_group(summary, arrays)},
        )

    To change how a rollout becomes samples, override both `rollout_to_training_samples` and
    `rollout_to_training_sample_specs`; `materialize` raises if their lengths disagree.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """TrainingSample builder filtering knobs."""

        drop_zero_std_reward_groups: bool = True
        """Drop zero-reward-variance groups;"""

    def __init__(self, config: Config) -> None:
        self.config = config

    def decide(
        self, *, summary: RolloutGroupSummary, num_mtp_layers: int
    ) -> TrainingSampleSpecGroup:
        """Apply data-validity filters and plan the surviving rollouts' training samples, from
        the group's lengths and rewards (no array is read).

        Example:
            # rollouts=[] from a failed generation
            decide(summary=split_rollout_group(failed_group)[0], num_mtp_layers=0)
            # -> TrainingSampleSpecGroup(group_id=..., training_sample_specs=[], metrics=[failure_metric])
        """
        rollout_group = summary.group
        metrics: list[m.Metric] = list(rollout_group.metrics)

        # TODO(async-rl): generalize these group-level filters into an ordered, user-pluggable filter
        # list (e.g. drop groups with all-0 / all-1 advantage), run in sequence on each RolloutGroup.
        # Failed generation: empty group carrying its failure metric.
        if not rollout_group.rollouts:
            return TrainingSampleSpecGroup(
                group_id=rollout_group.group_id,
                training_sample_specs=[],
                metrics=metrics,
            )

        # Untrainable: any sibling with no completion tokens -> drop the whole group
        if any(
            not any(lengths.num_completion_tokens > 0 for lengths in rollout_lengths)
            for rollout_lengths in summary.turn_lengths
        ):
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_untrainable", m.Sum(1.0)
                )
            )
            return TrainingSampleSpecGroup(
                group_id=rollout_group.group_id,
                training_sample_specs=[],
                metrics=metrics,
            )

        # Zero-std reward: no learning signal across siblings.
        rewards = [rollout.reward for rollout in rollout_group.rollouts]
        is_zero_std = len(rewards) > 1 and statistics.pstdev(rewards) == 0.0
        metrics.append(
            m.Metric(
                "rollout_reward/group_zero_std_frac",
                m.Mean(1.0 if is_zero_std else 0.0),
            )
        )
        if self.config.drop_zero_std_reward_groups and is_zero_std:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_zero_std", m.Sum(1.0)
                )
            )
            return TrainingSampleSpecGroup(
                group_id=rollout_group.group_id,
                training_sample_specs=[],
                metrics=metrics,
            )

        # One rollout may branch into multiple trainable training_samples.
        training_sample_specs: list[TrainingSampleSpec] = []
        unfiltered_training_sample_specs: list[TrainingSampleSpec] = []
        branches_per_rollout: list[float] = []
        num_samples_dropped_no_valid_tokens = 0
        num_rollouts_dropped_empty_completion = 0
        for rollout, rollout_lengths in zip(
            rollout_group.rollouts, summary.turn_lengths, strict=True
        ):
            # A later turn with no completion (e.g. a request aborted before its first token): drop the
            # rollout, so `rollout_to_training_samples` doesn't raise and stop the run.
            if any(
                lengths.num_completion_tokens == 0 for lengths in rollout_lengths[1:]
            ):
                num_rollouts_dropped_empty_completion += 1
                continue

            rollout_training_sample_specs = self.rollout_to_training_sample_specs(
                rollout, rollout_lengths, num_mtp_layers=num_mtp_layers
            )
            unfiltered_training_sample_specs.extend(rollout_training_sample_specs)
            # Keep samples with a token usable by the RL loss.
            kept = [
                spec
                for spec in rollout_training_sample_specs
                if spec.loss_token_counts[0] > 0
            ]
            num_samples_dropped_no_valid_tokens += len(
                rollout_training_sample_specs
            ) - len(kept)
            training_sample_specs.extend(kept)
            branches_per_rollout.append(float(len(rollout_training_sample_specs)))

        if num_rollouts_dropped_empty_completion:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_rollouts_dropped_empty_completion",
                    m.Sum(float(num_rollouts_dropped_empty_completion)),
                )
            )
        if num_samples_dropped_no_valid_tokens:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_samples_dropped_no_valid_tokens",
                    m.Sum(float(num_samples_dropped_no_valid_tokens)),
                )
            )

        min_policy_versions = [
            spec.min_policy_version for spec in unfiltered_training_sample_specs
        ]
        max_policy_versions = [
            spec.max_policy_version for spec in unfiltered_training_sample_specs
        ]
        advantages = [rollout.advantage for rollout in rollout_group.rollouts]
        metrics += [
            m.Metric(
                "training_sample_builder/num_training_samples",
                m.Sum(float(len(training_sample_specs))),
            ),
            m.Metric("advantage", m.SummaryStats.from_list(advantages)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/min_policy_version", m.Min.from_list(min_policy_versions)
            ),
            m.Metric(
                "rollout/max_policy_version", m.Max.from_list(max_policy_versions)
            ),
        ]
        return TrainingSampleSpecGroup(
            group_id=rollout_group.group_id,
            training_sample_specs=training_sample_specs,
            metrics=metrics,
        )

    def rollout_to_training_sample_specs(
        self,
        rollout: Rollout,
        turn_lengths: tuple[RolloutTurnLengths, ...],
        *,
        num_mtp_layers: int,
    ) -> list[TrainingSampleSpec]:
        """The specs of the samples `rollout_to_training_samples` builds from `rollout`, from its
        turns' lengths (`rollout` may be a summary rollout, whose arrays are empty).

        Same turn walk, branch rule and length checks as `rollout_to_training_samples`; the one
        check that reads token values (a prefix that could reuse one more token) runs when
        `materialize` builds the sample.

        Args:
            rollout: The scored rollout; reads its ids, advantage, policy versions, prompt prefix
                lengths and images, not its arrays.
            turn_lengths: Each turn's lengths (`RolloutGroupSummary.turn_lengths`).
            num_mtp_layers: MTP depth of the token counts.
        """
        if rollout.advantage is None:
            raise ValueError(
                f"rollout {rollout.group_id}/rollout={rollout.rollout_id} has no advantage; the Rollouter "
                "must fill it (via its advantage estimator) before training_samples are built."
            )
        training_sample_specs: list[TrainingSampleSpec] = []
        spec_in_progress: _SpecInProgress | None = None
        prev_len = 0
        prev_images: tuple[PromptImage, ...] = ()
        prev_image_placeholders: tuple[PlaceholderRange, ...] = ()
        for turn_idx, (rollout_turn, lengths) in enumerate(
            zip(rollout.turns, turn_lengths, strict=True)
        ):
            if lengths.num_completion_tokens == 0:
                if len(rollout.turns) == 1:
                    continue
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: "
                    f"non-final turn {turn_idx} has no completion"
                )

            prompt_prefix_len = rollout_turn.prompt_prefix_len
            num_prompt_tokens = prompt_prefix_len + lengths.num_prompt_delta_tokens
            if prompt_prefix_len > prev_len:
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn {turn_idx} "
                    f"reuses {prompt_prefix_len} tokens, but the previous turn has only "
                    f"{prev_len}; build it with `split_prompt`"
                )
            for placeholder in rollout_turn.prompt_delta_image_placeholders:
                if not (
                    prompt_prefix_len
                    <= placeholder.offset
                    <= num_prompt_tokens - placeholder.length
                ):
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn "
                        f"{turn_idx} has an image at tokens [{placeholder.offset}, "
                        f"{placeholder.offset + placeholder.length}), outside its prompt delta "
                        f"[{prompt_prefix_len}, {num_prompt_tokens}); image "
                        "placeholders are positions in the full prompt"
                    )

            num_prefix_images = sum(
                placeholder.offset < prompt_prefix_len
                for placeholder in prev_image_placeholders
            )
            images = prev_images[:num_prefix_images] + rollout_turn.prompt_delta_images
            image_placeholders = (
                prev_image_placeholders[:num_prefix_images]
                + rollout_turn.prompt_delta_image_placeholders
            )

            extends_prev = prompt_prefix_len == prev_len
            if spec_in_progress is not None and not extends_prev:
                training_sample_specs.append(
                    spec_in_progress.to_training_sample_spec(
                        num_mtp_layers=num_mtp_layers
                    )
                )
                spec_in_progress = None

            if spec_in_progress is None:
                spec_in_progress = _SpecInProgress(
                    rollout_id=RolloutTurnID(
                        group_id=rollout.group_id,
                        rollout_id=rollout.rollout_id,
                        turn_id=turn_idx,
                    ),
                    min_policy_version=rollout_turn.min_policy_version,
                    max_policy_version=rollout_turn.max_policy_version,
                    last_turn_id=turn_idx,
                )
                # A new sample starts with the full prompt.
                spec_in_progress.num_tokens += num_prompt_tokens
            else:
                spec_in_progress.num_tokens += lengths.num_prompt_delta_tokens

            spec_in_progress.completions.append((spec_in_progress.num_tokens, lengths))
            spec_in_progress.num_tokens += lengths.num_completion_tokens
            spec_in_progress.last_turn_id = turn_idx
            spec_in_progress.images = images
            spec_in_progress.image_placeholders = image_placeholders
            spec_in_progress.min_policy_version = min(
                spec_in_progress.min_policy_version, rollout_turn.min_policy_version
            )
            spec_in_progress.max_policy_version = max(
                spec_in_progress.max_policy_version, rollout_turn.max_policy_version
            )

            prev_len = num_prompt_tokens + lengths.num_completion_tokens
            prev_images, prev_image_placeholders = images, image_placeholders

        if spec_in_progress is not None:
            training_sample_specs.append(
                spec_in_progress.to_training_sample_spec(num_mtp_layers=num_mtp_layers)
            )
        return training_sample_specs

    def rollout_to_training_samples(self, rollout: Rollout) -> list[TrainingSample]:
        """Pack a scored `Rollout` into training training_samples — usually one, or several where its turns branch.

        Override this to change how a rollout becomes trainable training_samples (the main extension point of
        this class), together with `rollout_to_training_sample_specs`, which the batcher plans with.
        Turns that share a growing prefix (each prompt continues the previous prompt + completion) are
        packed into ONE training_sample: prompts and env replies are masked out, completions are trained.
        A new training_sample (branch) opens wherever that prefix breaks (history was edited).

        This branching has three common causes:
        1) The rollout loop purposely edits the history: For example, when compacting a long conversation;
        2) Thinking is removed from history: This is a flag in the `renderer` used in the `Rollouter`. Choose
            `preserve_all_thinking=True` to avoid stripping thinking (default=True);
        3) Some error or change from retokenization: Those can be avoided by doing token-in-token-out (TITO),
            i.e. you give tokens to the generators, and you append (bridge) the tokens received, leaving no room for changes.
            TITO is our default;

        Each packed training_sample records the min/max policy version across its turns: `min_policy_version`
        = the oldest version any of its turns depends on (the policy-age metrics and the max_offpolicy_steps
        check read it), `max_policy_version` = the newest version any of its turns reached.

        Example (5 turns; the env compacts history before turn 3, so the prefix breaks -> 2 training_samples).
        P = prompt; C = completion; E = env reply
            turn0 (v5): prompt=[P1]              completion=[C1]
            turn1 (v6): prompt=[P1,C1,E1]        completion=[C2]
            turn2 (v6): prompt=[P1,C1,E1,C2,E2]  completion=[C3]
            turn3 (v6): prompt=[P2]              completion=[C4]   # history compacted -> prefix breaks
            turn4 (v7): prompt=[P2,C4,E4]        completion=[C5]
            # -> training_sample 0: token_ids=[P1,C1,E1,C2,E2,C3], loss_mask=[0,1,0,1,0,1]
            #              rollout_id.turn_id=0, min_policy_version=5, max_policy_version=6
            #    training_sample 1: token_ids=[P2,C4,E4,C5],        loss_mask=[0,1,0,1]
            #              rollout_id.turn_id=3, min_policy_version=6, max_policy_version=7
        """
        rollout_advantage = rollout.advantage
        if rollout_advantage is None:
            raise ValueError(
                f"rollout {rollout.group_id}/rollout={rollout.rollout_id} has no advantage; the Rollouter "
                "must fill it (via its advantage estimator) before training_samples are built."
            )
        training_samples: list[TrainingSample] = []
        # The training_sample that later turns may still extend; None until the first trained turn
        sample_in_progress: _SampleInProgress | None = None
        # Length of the previous turn's prompt + completion
        prev_len = 0
        # The previous turn's prompt images and where they sit (images appear only in prompts)
        prev_images: tuple[PromptImage, ...] = ()
        prev_image_placeholders: tuple[PlaceholderRange, ...] = ()

        # Skip if no completion (nothing to train on). This happens when the prompt is too
        # long in the first turn, before any generation. We keep these rollouts for debugging.
        # TODO(async-rl): confirm this signal actually reaches the rollout_recorder.
        for turn_idx, rollout_turn in enumerate(rollout.turns):
            if len(rollout_turn.completion_token_ids) == 0:
                if len(rollout.turns) == 1:

                    continue

                # empty rollouts should only happen when initial prompt is too long
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: "
                    f"non-final turn {turn_idx} has no completion"
                )

            prompt_prefix_len = rollout_turn.prompt_prefix_len
            prompt_delta_token_ids = rollout_turn.prompt_delta_token_ids

            # Too many: e.g. prev = [0, 1, 2, 3] has 4 tokens; a turn can't reuse 6.
            if prompt_prefix_len > prev_len:
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn {turn_idx} "
                    f"reuses {prompt_prefix_len} tokens, but the previous turn has only "
                    f"{prev_len}; build it with `split_prompt`"
                )
            # Delta images sit in the delta, at full-prompt positions; an offset inside the
            # delta (e.g. 1 instead of prefix + 1) would put image embeddings on other tokens.
            for placeholder in rollout_turn.prompt_delta_image_placeholders:
                if not (
                    prompt_prefix_len
                    <= placeholder.offset
                    <= rollout_turn.num_prompt_tokens - placeholder.length
                ):
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn "
                        f"{turn_idx} has an image at tokens [{placeholder.offset}, "
                        f"{placeholder.offset + placeholder.length}), outside its prompt delta "
                        f"[{prompt_prefix_len}, {rollout_turn.num_prompt_tokens}); image "
                        "placeholders are positions in the full prompt"
                    )

            # The full prompt's images: the previous prompt's images in the shared prefix, then
            # the turn's own. `split_prompt` ends the prefix at the first changed image, so an
            # image swapped in place (equal placeholder tokens) shortens the prefix too.
            num_prefix_images = sum(
                placeholder.offset < prompt_prefix_len
                for placeholder in prev_image_placeholders
            )
            images = prev_images[:num_prefix_images] + rollout_turn.prompt_delta_images
            image_placeholders = (
                prev_image_placeholders[:num_prefix_images]
                + rollout_turn.prompt_delta_image_placeholders
            )

            # True when this prompt continues the previous one (prefix-preserving);
            # False when the env edited history -> open a new training_sample (branch).
            extends_prev = prompt_prefix_len == prev_len
            if sample_in_progress is not None and not extends_prev:
                training_samples.append(sample_in_progress.to_training_sample())
                sample_in_progress = None

            if sample_in_progress is None:
                # Each turn appended to the last training_sample, so its token_ids are exactly the
                # previous turn's prompt + completion.
                prev_prompt_and_completion = (
                    training_samples[-1].token_ids
                    if training_samples
                    else np.empty(0, dtype=np.int64)
                )

                # Too few: the first new token equals prev's token at the same position, so the turn
                # could reuse at least one more. E.g. prev = [0, 1, 2, 3], stored (0, [0, 1, 9, 3]):
                # 0 == prev[0]; split_prompt gives (2, [9, 3]). Catches a full prompt stored as the delta.
                # Not when a changed image starts the delta: `split_prompt` ends the prefix at
                # it, and its placeholder tokens equal the previous image's.
                could_reuse_one_more = (
                    prompt_prefix_len < prev_len
                    and len(prompt_delta_token_ids) > 0
                    and prompt_delta_token_ids[0]
                    == prev_prompt_and_completion[prompt_prefix_len]
                    and not any(
                        placeholder.offset == prompt_prefix_len
                        for placeholder in rollout_turn.prompt_delta_image_placeholders
                    )
                )
                if could_reuse_one_more:
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn {turn_idx} "
                        f"reuses {prompt_prefix_len} tokens but could reuse more; build it with "
                        "`split_prompt`"
                    )

                # Start a new training_sample; its RolloutTurnID marks the turn the segment begins at.
                sample_in_progress = _SampleInProgress(
                    min_policy_version=rollout_turn.min_policy_version,
                    max_policy_version=rollout_turn.max_policy_version,
                    rollout_id=RolloutTurnID(
                        group_id=rollout.group_id,
                        rollout_id=rollout.rollout_id,
                        turn_id=turn_idx,
                    ),
                )
                # A new training_sample needs the full prompt: the shared prefix + the delta.
                prompt_delta = np.concatenate(
                    [
                        prev_prompt_and_completion[:prompt_prefix_len],
                        prompt_delta_token_ids,
                    ]
                )
            else:
                prompt_delta = prompt_delta_token_ids

            # Append this turn's chunks: prompt delta (untrained) + completion (trained).
            num_delta = len(prompt_delta)
            num_completion = len(rollout_turn.completion_token_ids)
            sample_in_progress.token_ids += [
                prompt_delta,
                rollout_turn.completion_token_ids,
            ]
            sample_in_progress.loss_mask += [
                np.zeros(num_delta, dtype=bool),
                np.ones(num_completion, dtype=bool),
            ]
            sample_in_progress.logprobs += [
                np.zeros(num_delta, dtype=np.float32),
                rollout_turn.completion_logprobs,
            ]
            sample_in_progress.advantage += [
                np.zeros(num_delta, dtype=np.float32),
                np.full(num_completion, rollout_advantage, dtype=np.float32),
            ]
            sample_in_progress.images = images
            sample_in_progress.image_placeholders = image_placeholders
            sample_in_progress.min_policy_version = min(
                sample_in_progress.min_policy_version, rollout_turn.min_policy_version
            )
            sample_in_progress.max_policy_version = max(
                sample_in_progress.max_policy_version, rollout_turn.max_policy_version
            )

            prev_len = rollout_turn.num_prompt_tokens + num_completion
            prev_images, prev_image_placeholders = images, image_placeholders

        if sample_in_progress is not None:
            training_samples.append(sample_in_progress.to_training_sample())
        return training_samples

    def materialize(
        self,
        *,
        training_sample_specs: list[TrainingSampleSpec],
        rollout_groups: dict[int, RolloutGroup],
    ) -> list[TrainingSample]:
        """Build the training samples `training_sample_specs` describe, in their order.

        Calls `rollout_to_training_samples` once per distinct rollout and picks each spec's
        sample (same `rollout_id`).

        Args:
            training_sample_specs: Specs from `decide`, in the order to return their samples.
            rollout_groups: The full groups the specs came from, by group id.

        Raises:
            ValueError: A spec's sample is missing or has another length, i.e.
                `rollout_to_training_samples` and `rollout_to_training_sample_specs` disagree;
                override both.
        """
        samples_by_id: dict[RolloutTurnID, TrainingSample] = {}
        rollouts_by_id: dict[int, dict[int, Rollout]] = {}
        training_samples = []
        for spec in training_sample_specs:
            group_id, rollout_id = spec.rollout_id.group_id, spec.rollout_id.rollout_id
            if group_id not in rollouts_by_id:
                rollouts_by_id[group_id] = {
                    rollout.rollout_id: rollout
                    for rollout in rollout_groups[group_id].rollouts
                }
            rollout = rollouts_by_id[group_id].pop(rollout_id, None)
            if rollout is not None:
                for training_sample in self.rollout_to_training_samples(rollout):
                    samples_by_id[training_sample.rollout_id] = training_sample
            training_sample = samples_by_id.get(spec.rollout_id)
            num_tokens = (
                None if training_sample is None else len(training_sample.token_ids)
            )
            if training_sample is None or num_tokens != spec.num_tokens:
                raise ValueError(
                    f"Training sample {spec.rollout_id.to_string()} is missing or has "
                    f"{num_tokens} tokens, but its spec has {spec.num_tokens}; override "
                    "rollout_to_training_samples and rollout_to_training_sample_specs together."
                )
            training_samples.append(training_sample)
        return training_samples
