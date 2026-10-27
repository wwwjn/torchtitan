# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed experts on FlashInfer's TRT-LLM-gen MoE kernels in the generator (EP = 1, SM100).

Native vLLM and SGLang run the routed experts of Qwen MoE models on Blackwell as
FlashInfer's ``trtllm_bf16_routed_moe``: a fused permute, two block-major bf16 GEMMs with
the SwiGLU in between, and a finalize that weights and sums each token's picks. Per
Qwen3.5-35B-A3B layer on GB300 it takes 60 / 101 / 145 us at 8 / 16 / 32 tokens, vs
110 / 168 / 234 us for TitanRL's fused dispatch, two ``torch._grouped_mm`` and the
score-sum combine (``tools/gb300/moe_gemm_gb300.py``): ~3 ms of a decode step.

The kernels read the expert weights in their own layout (gate / up rows reordered and
shuffled for the epilogue, K in 128-wide blocks). Weights still load and sync in
TorchTitan's layout; after each full load ``relayout`` rewrites every layer's w13 / w2
storage in place in the TRT-LLM layout, so CUDA graphs keep reading the same addresses
and no second copy is kept. The state dict is not meaningful in between: only loads
write it. TorchTitan's router still picks the experts and their scores.

Requires EP = 1, bf16 weights, no LoRA / quantized experts, and an intermediate size
divisible by 128. Not bitwise with the grouped-GEMM path. Inference only. Opt-in with
``TORCHTITAN_TRTLLM_MOE=1``.
"""

import os

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe.moe import RoutedExperts

ENABLED = os.environ.get("TORCHTITAN_TRTLLM_MOE") == "1"
_original_forward = RoutedExperts.forward


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _forward(
    self,
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,
    *,
    padding_mask_T: torch.Tensor | None = None,
) -> torch.Tensor:
    w13 = getattr(self, "_trtllm_w13", None)
    if w13 is None:
        return _original_forward(
            self,
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            padding_mask_T=padding_mask_T,
        )
    # After relayout the grouped-GEMM path would read the TRT-LLM layout.
    assert not torch.is_grad_enabled(), "TRT-LLM-gen experts are inference only"
    import flashinfer
    from flashinfer.fused_moe import WeightLayout

    num_experts = w13.shape[0]
    out_TD = flashinfer.fused_moe.trtllm_bf16_routed_moe(
        topk_ids=(topk_expert_ids_TK.to(torch.int32), topk_scores_TK),
        hidden_states=x_TD.bfloat16().contiguous(),
        gemm1_weights=w13,
        gemm2_weights=self._trtllm_w2,
        num_experts=num_experts,
        top_k=topk_expert_ids_TK.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=self._trtllm_intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # unused with precomputed routing
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
    )
    out_TD = out_TD[0] if isinstance(out_TD, list) else out_TD
    return out_TD.type_as(x_TD)


def maybe_patch_routed_experts() -> None:
    """Before parallelize, which binds each module's forward into its SPMD wrapper."""
    if ENABLED and RoutedExperts.forward is _original_forward:
        RoutedExperts.forward = _forward


def relayout(model: torch.nn.Module) -> None:
    """After a full weight load: rewrite every eligible layer's experts in the TRT-LLM layout."""
    if not ENABLED or is_in_batch_invariant_mode():
        return
    # Importing fused_moe first avoids a circular import inside vLLM.
    import vllm.model_executor.layers.fused_moe  # noqa: F401
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout,
    )

    permute_indices: dict[torch.Size, torch.Tensor] = {}
    # The loaded weights are inference tensors, writable in place only in inference mode.
    with torch.inference_mode():
        for module in model.modules():
            if not isinstance(module, RoutedExperts):
                continue
            dispatcher = module.token_dispatcher
            if getattr(dispatcher, "ep_mesh", None) is not None:
                continue
            if not (
                type(module.w13) is GroupedLinear and type(module.w2) is GroupedLinear
            ):
                continue
            w13, w2 = _local(module.w13.weight), _local(module.w2.weight)
            # w13 [E, 2 (gate, up), F, D]; w2 [E, D, F].
            if w13.dtype != torch.bfloat16 or w13.dim() != 4 or w13.shape[2] % 128:
                continue
            num_experts, _, intermediate, dim = w13.shape
            w13_t, w2_t = convert_moe_weights_to_flashinfer_trtllm_block_layout(
                permute_indices,
                w13.reshape(num_experts, 2 * intermediate, dim),
                w2,
                is_gated_act_gemm=True,
            )
            w13.view(-1).copy_(w13_t.view(-1))
            w2.view(-1).copy_(w2_t.view(-1))
            module._trtllm_w13 = w13.view(w13_t.shape)
            module._trtllm_w2 = w2.view(w2_t.shape)
            module._trtllm_intermediate_size = intermediate
