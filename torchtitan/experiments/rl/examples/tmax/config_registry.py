# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the tmax terminal-agent (host_loop) example.

``ConfigManager`` discovers these from the fully-qualified example module path::

    python -m torchtitan.experiments.rl.train \
        --module torchtitan.experiments.rl.examples.tmax \
        --config rl_grpo_qwen3_5_27b_tmax \
        --hf_assets_path <path/to/Qwen3.6-27B>

(The short ``--module tmax`` form additionally requires ``tmax`` in
``torchtitan/experiments/__init__.py::_supported_experiments``, a core file this
example deliberately does not modify. The MAST path uses ``--module mast_rl``.)

The tmax config clones the Qwen3.6-27B SWE-R2E recipe
(``rl_grpo_qwen3_5_27b_swe_r2e``) verbatim -- same model spec, FSDP/generator
split, memory setup (bf16 master + bf16 Adam + FullAC + chunked DAPO loss), and
async knobs -- but swaps the rollouter to ``TMaxRollouter`` + ``TMaxDataset``. The
tmax JSONL path comes from ``SWE_PROMPT_DATA`` (set by the launcher's
``PROMPT_DATA``), matching the swe_r2e convention.
"""

from __future__ import annotations

import dataclasses
import os

from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.examples.swe_r2e.config_registry import (
    rl_grpo_qwen3_5_27b_swe_r2e as _swe_27b,
    rl_grpo_qwen3_5_9b_swe_r2e as _swe_9b,
)
from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset
from torchtitan.experiments.rl.examples.tmax.rollouter import TMaxRollouter

# tmax JSONL path, supplied by the launcher (PROMPT_DATA -> SWE_PROMPT_DATA).
# Empty by default; TMaxDataset raises a clear error if it is not set.
_DEFAULT_DATA = os.environ.get("SWE_PROMPT_DATA", "")


def _tmax_rollouter() -> TMaxRollouter.Config:
    """Train/validation datasets for the tmax rollouter (rubric + env defaults live
    on the rollouter Config)."""
    return TMaxRollouter.Config(
        train_dataset=TMaxDataset.Config(data_path=_DEFAULT_DATA, seed=42),
        validation_dataset=TMaxDataset.Config(
            data_path=_DEFAULT_DATA, seed=99, shuffle=False
        ),
    )


def rl_grpo_qwen3_5_27b_tmax() -> Controller.Config:
    """Qwen3.6-27B (Gated DeltaNet hybrid) tmax terminal-agent on a single 8-GPU node.

    Same recipe as ``rl_grpo_qwen3_5_27b_swe_r2e`` (trainer FSDP-8 + vLLM-native GDN
    generator TP-4, bf16 master/Adam + FullAC + chunked DAPO loss, max_tokens=8192,
    off-policy 2, drop_zero_std False, 8 groups x group_size 8, host_loop agent)
    with the rollouter swapped to ``TMaxRollouter`` (runs the agent as root, grades
    with ``bash /tests/test.sh`` -> reward.txt in the agent's own sandbox).
    """
    config = _swe_27b()
    config.rollouter = _tmax_rollouter()
    return config


def rl_grpo_qwen3_5_9b_tmax() -> Controller.Config:
    """Qwen3.5-9B (Gated DeltaNet hybrid, text-only) AI2 tmax terminal-agent recipe.

    Base = ``rl_grpo_qwen3_5_9b_swe_r2e`` (9B GDN, gen TP-2), rollouter swapped to
    ``TMaxRollouter``. Applies the TMax DPPO recipe knobs from the paper's
    open-instruct run (``scripts/tmax/RL/qwen35_9b_8gpu_local.sh``): group_size 8
    and, critically, ``drop_zero_std_reward_groups=True``
    (``filter_zero_std_samples`` in open-instruct) -- terminal tasks are sparse
    binary, so keeping all-fail groups would zero out the gradient. off-policy 2,
    temperature 1.0, lr 1e-6 are inherited from the swe base and already match the
    paper. num_groups is kept at 8 (vs the paper's 4) since torchtitan's drop-only
    filter has no active resampling, so more groups per step raise the odds of a
    non-empty trained batch under sparse reward.
    """
    config = _swe_9b()
    config.rollouter = _tmax_rollouter()
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_groups_per_train_step=8,
        group_size=8,
        training_sample_builder=TrainingSampleBuilder.Config(
            drop_zero_std_reward_groups=True,
        ),
    )
    return config
