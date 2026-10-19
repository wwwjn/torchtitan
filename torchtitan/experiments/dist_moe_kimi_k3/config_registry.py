# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Production-shaped Kimi K3 recipes using upstream Dist-MoE integration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import spmd_types as spmd
import torch

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetBuildContext,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim.optimizer import DistMuon
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.spmd_types import spmd_sparse_mesh
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.attention.kda import InnerKDA
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan.models.common.moe import MoE, RoundRobinTokenChoiceTopKRouter
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.dist_moe import require_dist_moe
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    kimi_k3_debugmodel,
)


_KIMI_K3_MANIFOLD_HF_ASSETS_PATH = (
    "/mnt/torchtrain_datasets/tree/shuhuay/k3-oss/assets/hf/Kimi-K3"
)
_KIMI_K3_C4_DATA_FILES = (
    "/mnt/torchtrain_datasets/tree/shuhuay/c4/en/c4-train.*.json.gz"
)
_KIMI_K3_LONG_CONTEXT_DATA_ROOT = os.environ.get(
    "KIMI_K3_LONG_CONTEXT_DATA_ROOT",
    (
        "/mnt/torchtrain_datasets/tree/shuhuay/k3-oss/datasets/"
        "c4-kimi-k3-long-context-v1"
    ),
)
_KIMI_K3_LONG_CONTEXT_64K_DATA_ROOT = os.environ.get(
    "KIMI_K3_LONG_CONTEXT_64K_DATA_ROOT",
    (
        "/mnt/torchtrain_datasets/tree/shuhuay/k3-oss/datasets/"
        "c4-kimi-k3-long-context-64k-v1"
    ),
)


class _PretokenizedLongContextProcessor(SampleProcessor):
    """Build one next-token sequence from an exact-length tokenized row."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._max_context_length = context.max_context_length

    def __call__(
        self, sample: Mapping[str, Any], rng: np.random.Generator
    ) -> TextSequence:
        del rng
        token_ids = np.asarray(sample["input_ids"], dtype=np.int64)
        expected_num_tokens = self._max_context_length + 1
        if token_ids.ndim != 1 or len(token_ids) != expected_num_tokens:
            raise ValueError(
                "pretokenized long-context rows must contain exactly "
                f"{expected_num_tokens} token IDs, got shape {token_ids.shape}"
            )
        return TextSequence(
            input_ids=token_ids[:-1],
            labels=token_ids[1:],
        )


class _PhasedRoundRobinTokenChoiceTopKRouter(RoundRobinTokenChoiceTopKRouter):
    """Spread a partial final expert cycle uniformly across EP destinations."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoundRobinTokenChoiceTopKRouter.Config):
        pass

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        del expert_bias_E, router_kwargs
        expert_offset = 0
        if (sparse_mesh := spmd_sparse_mesh()) is not None:
            # Experts owned by one EP rank are contiguous. Advancing each source
            # by one ownership group balances receive rows even when T*K is not
            # divisible by the global expert count.
            with spmd.no_typecheck():
                ep_mesh = sparse_mesh["ep"]
                ep_size = ep_mesh.size()
                if self.num_experts % ep_size != 0:
                    raise ValueError(
                        f"num_experts={self.num_experts} must be divisible by "
                        f"EP size {ep_size}"
                    )
                expert_offset = ep_mesh.get_local_rank() * (self.num_experts // ep_size)
        return (
            torch.arange(
                scores_TE.shape[0] * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            )
            .add_(expert_offset)
            .reshape(scores_TE.shape[0], self.top_k)
            % self.num_experts
        )


def _use_phased_round_robin_routing(config: Trainer.Config) -> Trainer.Config:
    """Select deterministic routing that is balanced across EP ranks."""
    for _, moe, _, _ in config.model.traverse(MoE.Config):
        router = moe.router
        moe.router = _PhasedRoundRobinTokenChoiceTopKRouter.Config(
            num_experts=router.num_experts,
            gate=router.gate,
            score_func=router.score_func,
            top_k=router.top_k,
            route_norm=router.route_norm,
            route_norm_epsilon=router.route_norm_epsilon,
            route_scale=router.route_scale,
            aux_loss=router.aux_loss,
            param_init=router.param_init,
            sharding_config=router.sharding_config,
        )
    return config


def _set_debug_expert_topology(
    config: Trainer.Config,
    *,
    num_experts: int,
    top_k: int,
) -> None:
    """Set a non-divisible routing cycle before building the debug optimizer."""
    for _, moe, _, _ in config.model.traverse(MoE.Config):
        routed = moe.routed_experts
        moe.num_experts = num_experts
        moe.router = replace(
            moe.router,
            num_experts=num_experts,
            top_k=top_k,
            gate=replace(moe.router.gate, out_features=num_experts),
        )
        moe.routed_experts = replace(
            routed,
            w13=replace(routed.w13, group_size=num_experts),
            w2=replace(routed.w2, group_size=num_experts),
            token_dispatcher=replace(
                routed.token_dispatcher,
                num_experts=num_experts,
                top_k=top_k,
            ),
        )


def _full_recompute_runtime_config() -> DistMoeRuntime.Config:
    """Use the backend's device-only full-recompute memory contract."""
    return DistMoeRuntime.Config(
        activation_slot_capacity_factor=None,
        scratch_capacity_factor=1.0,
        vmm_capacity_factor=None,
    )


def _use_transient_dist_muon_buffers(config: Trainer.Config) -> Trainer.Config:
    """Release optimizer-only redistribution storage outside ``step()``."""
    optimizer = replace(
        config.optim.optimizer,
        optimizers=[
            replace(
                optimizer_config,
                persistent_redistribution_buffers=False,
            )
            if isinstance(optimizer_config, DistMuon.Config)
            else optimizer_config
            for optimizer_config in config.optim.optimizer.optimizers
        ],
    )
    return replace(config, optim=replace(config.optim, optimizer=optimizer))


def _kimi_k3_manifold_c4_dataloader() -> GrainDataLoader.Config:
    """Build the packed C4 stream used by the full-scale validation runs."""
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": _KIMI_K3_C4_DATA_FILES},
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=dataset),
        streaming_shuffle_buffer_size=128,
    )


def _kimi_k3_long_context_dataloader(
    seq_len: int,
    *,
    data_root: str = _KIMI_K3_LONG_CONTEXT_DATA_ROOT,
) -> GrainDataLoader.Config:
    """Build the exact-row dataset used by a long-context workload."""
    data_files = f"{data_root}/seq{seq_len}/part-*.json.gz"
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": data_files},
        ),
        processor=_PretokenizedLongContextProcessor.Config(),
    )
    return GrainDataLoader.Config(
        dataset=dataset,
        max_num_documents=1,
        streaming_shuffle_buffer_size=128,
    )


def kimi_k3_debugmodel_dist_moe_bf16_recompute_varlen_pp2_fsdp2_ep2() -> (
    Trainer.Config
):
    """Exercise the production memory policy and 64-microbatch 1F1B schedule."""
    require_dist_moe()
    seq_len = 256
    num_dp_shards = 2
    num_pp_microbatches = 64
    config = kimi_k3_debugmodel(seq_len=seq_len)
    parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=num_dp_shards,
        context_parallel_degree=1,
        tensor_parallel_degree=1,
        expert_parallel_degree=2,
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        num_pp_microbatches=num_pp_microbatches,
        fsdp_reshard_after_forward="always",
    )
    model = build_model_config(
        "debugmodel",
        attn_backend="varlen",
        seq_len=seq_len,
        parallelism=parallelism,
    )
    config.model = model
    _set_debug_expert_topology(config, num_experts=28, top_k=16)
    optimizer = _dist_muon_optimizer(
        model,
        muon_lr=8e-4,
        adamw_lr=8e-4,
        parallelism=parallelism,
    )
    training = replace(
        config.training,
        max_context_length=seq_len,
        num_tokens_per_microbatch_per_dp_rank=seq_len,
        num_tokens_per_train_step=(seq_len * num_pp_microbatches * num_dp_shards),
        steps=4,
        disable_cuda_graphs=True,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )
    config = replace(
        config,
        optim=replace(config.optim, optimizer=optimizer),
        parallelism=parallelism,
        training=training,
        activation_remat=ActivationRematConfig(save_regions=[]),
        checkpointer=None,
        dist_moe_runtime=_full_recompute_runtime_config(),
    )
    config = _use_phased_round_robin_routing(config)
    config = _use_transient_dist_muon_buffers(config)
    return apply_transforms(config, [DistMoeTransform()])


def _kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512(
    *,
    seq_len: int,
    num_tokens_per_microbatch: int,
) -> Trainer.Config:
    """Build one twelve-step, 512-GPU Kimi K3 Dist-MoE workload."""
    require_dist_moe()
    num_dp_shards = 64
    num_pp_microbatches = 64
    config = kimi_k3_debugmodel(seq_len=seq_len)
    parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=num_dp_shards,
        context_parallel_degree=1,
        tensor_parallel_degree=1,
        expert_parallel_degree=64,
        pipeline_parallel_degree=8,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_module_fqns_per_model_part=None,
        pipeline_parallel_layers_per_stage=None,
        num_pp_microbatches=num_pp_microbatches,
        pp_max_unsharded_active_stages=None,
        fsdp_reshard_after_forward="always",
    )
    model = build_model_config(
        "Kimi-K3",
        attn_backend="varlen",
        seq_len=seq_len,
        parallelism=parallelism,
    )
    layer_ranges = [
        range(first_layer, min(first_layer + 12, len(model.layers)))
        for first_layer in range(0, len(model.layers), 12)
    ]
    pipeline_parts = [
        [f"layers.{layer_id}" for layer_id in layer_range]
        for layer_range in layer_ranges
    ]
    pipeline_parts[0][:0] = ["vision_encoder", "tok_embeddings"]
    pipeline_parts[-1].extend(["norm", "lm_head", "output_res_proj", "output_res_norm"])
    parallelism = replace(
        parallelism, pipeline_parallel_module_fqns_per_model_part=pipeline_parts
    )
    optimizer = _dist_muon_optimizer(
        model,
        muon_lr=8e-4,
        adamw_lr=8e-4,
        parallelism=parallelism,
    )
    training = replace(
        config.training,
        max_context_length=seq_len,
        num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
        num_tokens_per_train_step=(
            num_tokens_per_microbatch * num_pp_microbatches * num_dp_shards
        ),
        steps=12,
        disable_cuda_graphs=True,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        # cuDNN SDPA allocates outside PyTorch's caching allocator. Release
        # inactive pages once per update, after all 64 PP actions complete.
        empty_device_cache_after_step=True,
    )
    config = replace(
        config,
        model=model,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            )
        ),
        hf_assets_path=_KIMI_K3_MANIFOLD_HF_ASSETS_PATH,
        dataloader=_kimi_k3_manifold_c4_dataloader(),
        metrics=replace(config.metrics, enable_wandb=True),
        optim=replace(config.optim, optimizer=optimizer),
        parallelism=parallelism,
        training=training,
        # ActivationRematConfig recomputes the surrounding Kimi block but honors the
        # non-recomputed Dist-MoE region, avoiding nested expert execution.
        activation_remat=ActivationRematConfig(save_regions=[]),
        checkpointer=None,
        comm=replace(
            config.comm,
            init_timeout_seconds=30 * 60,
            train_timeout_seconds=10 * 60,
        ),
        profiler=replace(config.profiler, enable_profiling=False),
        dist_moe_runtime=_full_recompute_runtime_config(),
    )
    config = _use_phased_round_robin_routing(config)
    config = _use_transient_dist_muon_buffers(config)
    return apply_transforms(config, [DistMoeTransform()])


def kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512() -> (Trainer.Config):
    """Run v1: two packed 2K rows in each 4096-token microbatch."""
    return _kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512(
        seq_len=2048,
        num_tokens_per_microbatch=4096,
    )


def kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512_v1_5() -> (
    Trainer.Config
):
    """Run v1.5: one 4K row in each 4096-token microbatch."""
    return _kimi_k3_full_bf16_pretrain_dist_moe_pp8_fsdp64_ep64_gb200_512(
        seq_len=4096,
        num_tokens_per_microbatch=4096,
    )


def _kimi_k3_full_long_context_v4_dist_moe_mxfp8_gb200_256(
    *,
    seq_len: int,
    num_pp_microbatches: int,
    data_root: str,
    resident_dtype: str,
) -> Trainer.Config:
    """Build the V4 PP4/VP2 topology with MXFP8 Dist-MoE experts."""
    dist_moe = require_dist_moe()
    num_dp_shards = 4

    # The debug flavor is only a container for shared trainer defaults; its
    # own context limit is lower than the released model's long-context workload.
    config = kimi_k3_debugmodel()
    parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=num_dp_shards,
        context_parallel_degree=16,
        context_parallel_load_balancer=None,
        tensor_parallel_degree=1,
        expert_parallel_degree=64,
        pipeline_parallel_degree=4,
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_module_fqns_per_model_part=None,
        pipeline_parallel_layers_per_stage=12,
        num_pp_microbatches=num_pp_microbatches,
        pp_max_unsharded_active_stages=1,
        fsdp_reshard_after_forward="always",
    )
    model = build_model_config(
        "Kimi-K3",
        attn_backend="varlen",
        seq_len=seq_len,
        parallelism=parallelism,
    )
    optimizer = _dist_muon_optimizer(
        model,
        muon_lr=8e-4,
        adamw_lr=8e-4,
        parallelism=parallelism,
    )
    training = replace(
        config.training,
        max_context_length=seq_len,
        num_tokens_per_microbatch_per_dp_rank=seq_len,
        num_tokens_per_train_step=(seq_len * num_pp_microbatches * num_dp_shards),
        steps=12,
        disable_cuda_graphs=True,
        dtype=resident_dtype,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="bfloat16",
        empty_device_cache_after_step=True,
    )
    # Set the complete topology before applying CP. Reconstructing the trainer
    # here would validate the unconverted attention configs too early.
    config.model = model
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model),
        )
    )
    config.hf_assets_path = _KIMI_K3_MANIFOLD_HF_ASSETS_PATH
    config.dataloader = _kimi_k3_long_context_dataloader(
        seq_len,
        data_root=data_root,
    )
    config.metrics = replace(config.metrics, enable_wandb=True)
    config.optim = replace(config.optim, optimizer=optimizer)
    config.parallelism = parallelism
    config.training = training
    # ActivationRematConfig retains the Dist-MoE boundary while recomputing the rest of the
    # Kimi block, avoiding a second expert forward during backward.
    config.activation_remat = ActivationRematConfig(save_regions=[])
    config.checkpointer = None
    config.comm = replace(
        config.comm,
        init_timeout_seconds=30 * 60,
        train_timeout_seconds=30 * 60,
    )
    config.profiler = replace(config.profiler, enable_profiling=False)
    config.dist_moe_runtime = _full_recompute_runtime_config()
    # Factor-one scratch requires exact aggregate balance. The phased source
    # offsets preserve that bound when CP and DP jointly form the EP64 mesh.
    config = _use_phased_round_robin_routing(config)
    config = apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    VarlenInnerAttention: UlyssesVarlenInnerAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            ),
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            ),
        ],
    )
    return _use_transient_dist_muon_buffers(config)


def kimi_k3_full_long_context_v4_dist_moe_mxfp8_gb200_256() -> Trainer.Config:
    """Mirror the V4 64K BF16-state topology with MXFP8 Dist-MoE experts."""
    return _kimi_k3_full_long_context_v4_dist_moe_mxfp8_gb200_256(
        seq_len=65_536,
        num_pp_microbatches=32,
        data_root=_KIMI_K3_LONG_CONTEXT_64K_DATA_ROOT,
        resident_dtype="bfloat16",
    )


def kimi_k3_full_long_context_v4_1_dist_moe_mxfp8_gb200_256() -> Trainer.Config:
    """Mirror the V4.1 32K BF16-state topology with MXFP8 Dist-MoE experts."""
    return _kimi_k3_full_long_context_v4_dist_moe_mxfp8_gb200_256(
        seq_len=32_768,
        num_pp_microbatches=64,
        data_root=_KIMI_K3_LONG_CONTEXT_DATA_ROOT,
        resident_dtype="bfloat16",
    )


def kimi_k3_full_long_context_v4_2_dist_moe_mxfp8_gb200_256() -> Trainer.Config:
    """Mirror the V4.2 32K FP32-state topology with MXFP8 Dist-MoE experts."""
    return _kimi_k3_full_long_context_v4_dist_moe_mxfp8_gb200_256(
        seq_len=32_768,
        num_pp_microbatches=64,
        data_root=_KIMI_K3_LONG_CONTEXT_DATA_ROOT,
        resident_dtype="float32",
    )
