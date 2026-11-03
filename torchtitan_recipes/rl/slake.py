# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO recipes for SLAKE medical VQA (image + question -> short answer).

Data: the English part of SLAKE (CC-BY-4.0, Hugging Face ``BoKelvin/SLAKE``).
Download ``train.json``, ``validation.json``, ``test.json`` and ``imgs.zip``
into ``./assets/datasets/slake`` and unzip ``imgs.zip`` there, so images live
under ``./assets/datasets/slake/imgs/``.

Run with ``--module torchtitan_recipes.rl.slake --config rl_grpo_*``.
"""

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_model_config
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.slake import RewardSlakeAnswer, SlakeDataset, SlakeEnv
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

SLAKE_DIR = "./assets/datasets/slake"


def _slake_rollouter_config(*, include_image: bool) -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRolloutDataLoader.Config(
            dataset=SlakeDataset.Config(data_dir=SLAKE_DIR, split="train", seed=42)
        ),
        validation_dataset=SlakeDataset.Config(
            data_dir=SLAKE_DIR, split="validation", seed=99
        ),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(reward_fns=[RewardSlakeAnswer.Config(weight=1.0)]),
            message_env=SlakeEnv.Config(include_image=include_image),
        ),
    )


def rl_grpo_qwen3_5_0_8b_slake_text() -> Controller.Config:
    """Qwen3.5-0.8B on SLAKE closed questions without the image (4 GPUs).

    The text-only baseline: how often the model answers from the question
    alone. Trainer FSDP=2 + 2 generators at TP=1.
    """
    # SLAKE images expand to 64-1024 vision tokens (median 256) at the default
    # pixel budget, plus the question and a short answer.
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(data_parallel_shard_degree=2)
    generator_parallelism = InferenceParallelismConfig(tensor_parallel_degree=1)
    trainer_model, generator_model = build_rl_model_configs(
        build_qwen3_5_model_config,
        "0.8B",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=seq_len,
        attn_backend="varlen",
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-0.8B",
        num_generators=2,
        async_loop=AsyncLoopConfig(
            num_training_steps=50,
            num_prompts_per_train_step=16,
            num_samples_per_prompt=8,
            validation=ValidationConfig(steps=10),
        ),
        rollouter=_slake_rollouter_config(include_image=False),
        renderer=from_renderers(Qwen35RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=4,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            # Linear-attention (GDN) CUDA graphs need capture sizes <= max_model_len.
            max_num_batched_tokens=seq_len,
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(temperature=1.0, max_tokens=128),
        ),
    )
