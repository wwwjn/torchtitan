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
from torchtitan.experiments.rl.controller import Controller, ValidationConfig
from torchtitan.experiments.rl.examples.swe_r2e.config_registry import (
    _set_max_seq_len,
    rl_grpo_qwen3_5_27b_swe_r2e as _swe_27b,
    rl_grpo_qwen3_5_9b_swe_r2e as _swe_9b,
)
from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset
from torchtitan.experiments.rl.examples.tmax.rollouter import TMaxRollouter

# tmax JSONL path, supplied by the launcher (PROMPT_DATA -> SWE_PROMPT_DATA).
# Empty by default; TMaxDataset raises a clear error if it is not set.
_DEFAULT_DATA = os.environ.get("SWE_PROMPT_DATA", "")

# Full TMax-9B recipe context (open-instruct qwen35_9b.sh: response_length 65536)
# and per-turn generation cap (per_turn_max_tokens 16384). The context is the
# generator's vLLM max_model_len AND the trainer batcher's packing width: both are
# raised together (the controller mirrors the batcher width into the trainer
# seq_len), or a full episode is truncated by vLLM / dropped during packing.
_TMAX_9B_CONTEXT = 65536
_TMAX_9B_PER_TURN_TOKENS = 16384
# Held-out prompts per periodic validation pass (greedy, n=1). Runs concurrently, so its
# wall time is ~one rollout regardless of count; 32 gives a stable enough solve-rate.
_TMAX_9B_VAL_SAMPLES = 32
# Reserve the last N rows of the JSONL as a held-out validation slice, disjoint from
# training, so periodic validation measures generalization (not training-set recall).
# Must be >= _TMAX_9B_VAL_SAMPLES so a validation pass can draw distinct held-out tasks.
_TMAX_9B_HOLDOUT_N = 64


def _tmax_rollouter() -> TMaxRollouter.Config:
    """Train/validation datasets for the tmax rollouter (rubric + env defaults live
    on the rollouter Config). Train and validation read the same JSONL but disjoint
    slices via holdout_n (last N rows = validation)."""
    return TMaxRollouter.Config(
        train_dataset=TMaxDataset.Config(
            data_path=_DEFAULT_DATA,
            seed=42,
            holdout_n=_TMAX_9B_HOLDOUT_N,
            split="train",
        ),
        validation_dataset=TMaxDataset.Config(
            data_path=_DEFAULT_DATA,
            seed=99,
            shuffle=False,
            holdout_n=_TMAX_9B_HOLDOUT_N,
            split="validation",
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

    Base = ``rl_grpo_qwen3_5_9b_swe_r2e`` (9B GDN, gen TP-4), rollouter swapped to
    ``TMaxRollouter``. Matches the paper's open-instruct run
    (``scripts/tmax/RL/qwen35_9b.sh``): ``group_size=32``
    (num_samples_per_prompt_rollout), off-policy 4 (async_steps), per-turn 16384,
    full 65536 context (response_length), and ``drop_zero_std_reward_groups=True``
    (``filter_zero_std_samples``) -- terminal tasks are sparse binary, so keeping
    all-fail groups would zero out the gradient. temperature 1.0, lr 1e-6, constant
    LR, GRPO/DAPO with beta 0 are inherited from the swe base and already match.
    num_groups is kept at 8 (vs the paper's 4) since torchtitan's drop-only filter
    has no active resampling, so more groups per step raise the odds of a non-empty
    trained batch under sparse reward.

    Two knobs must move together with the context: the batcher packing width
    (``seq_len``) and the model RoPE / vLLM max_model_len, both to 65536. The loss
    is re-chunked to 32 chunks (from 16) so the per-chunk fp32 logits stay in the
    validated ~1 GiB envelope at the 4x longer sequence. The active-rollout ceiling
    grows to (off+1) x num_groups x group_size = 5 x 8 x 32 = 1280 concurrent
    rollout slots; ``SWE_ROLLOUT_CONCURRENCY`` throttles the sandbox count below it.
    """
    config = _swe_9b()
    config.rollouter = _tmax_rollouter()
    _set_max_seq_len(config.model_spec, _TMAX_9B_CONTEXT)
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_groups_per_train_step=8,
        group_size=32,
        max_offpolicy_steps=4,
        training_sample_builder=TrainingSampleBuilder.Config(
            drop_zero_std_reward_groups=True,
        ),
        batcher=dataclasses.replace(
            config.async_loop.batcher,
            batch=dataclasses.replace(
                config.async_loop.batcher.batch, seq_len=_TMAX_9B_CONTEXT
            ),
        ),
        # Periodic held-out eval every 20 steps (+ start/end): the trained-batch reward is
        # locked near ~0.5 by drop_zero_std, so it is NOT a learning signal; a greedy
        # (temp=0, n=1) pass over held-out tmax tasks via the same Daytona rollout+grade path
        # gives the real solve-rate curve inline, with no separate eval job or ckpt download.
        # The swe_r2e base sets num_samples=0 (off); we turn it on here. To eval the real
        # terminal-bench@2.0 benchmark instead, point the rollouter's validation_dataset at
        # TB-2.0 tasks in the tmax task format (see examples/tmax/data.py schema).
        validation=ValidationConfig(num_samples=_TMAX_9B_VAL_SAMPLES, interval=20),
    )
    config.generator = dataclasses.replace(
        config.generator,
        sampling=dataclasses.replace(
            config.generator.sampling, max_tokens=_TMAX_9B_PER_TURN_TOKENS
        ),
    )
    # 32 chunks keeps per-chunk fp32 lm_head logits ~1.2 GiB at seq_len 65536
    # (16 chunks -> ~2.3 GiB, an OOM risk); 65536 % 32 == 0 for the chunk split.
    # Save a full training-state checkpoint every 20 steps (matches the paper's
    # save_freq 20) so the run is resumable after a crash and each snapshot is
    # eval-able; keep last_save_model_only so the final step-100 save is a clean
    # model-only export for serving. The swe base uses interval=10000 (final save
    # only), which risks losing the whole ~20h run to any mid-run crash.
    config.trainer = dataclasses.replace(
        config.trainer,
        loss=dataclasses.replace(config.trainer.loss, num_chunks=32),
        checkpoint=dataclasses.replace(config.trainer.checkpoint, interval=20),
    )
    return config
