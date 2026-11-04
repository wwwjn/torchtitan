# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Engine-agnostic helpers for running a TorchTitan model inside an inference engine.

The vLLM and SGLang model wrappers both build the generator's TorchTitan model on the
meta device, parallelize it, keep FSDP's unsharded parameters as the compute
representation (the storage CUDA graphs read), and copy new weights in one FSDP group at a
time. This module holds the pieces that do not depend on the engine.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.distributed._composable.fsdp import FSDPModule

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.models.common.decoder import Decoder
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor


def sharded_param_storages(module: FSDPModule) -> Iterator[torch.Tensor]:
    """Yield the sharded parameter tensors of ``module``'s FSDP group."""
    for param_group in module._get_fsdp_state()._fsdp_param_groups:
        for param in param_group.fsdp_params:
            sharded_data = param._sharded_param_data
            if isinstance(sharded_data, _ShardedFSDPTensor):
                sharded_data = sharded_data._tensor
            yield sharded_data


def restore_sharded_params(module: FSDPModule) -> None:
    """Re-allocate the sharded buffers of `module`'s FSDP group so new weights can be copied in.

    before:  sharded: freed                     unsharded: old weights (what CUDA graphs read)
    after:   sharded: allocated, uninitialized  unsharded: unchanged

    FSDP's own load pre-hook may have resharded the group already (e.g.
    tok_embeddings), so ``is_sharded`` can be True while the storage is still
    freed; allocate unconditionally.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param import alloc_storage

    # TODO: replace this with FSDPModule._restore_sharded_params()
    for sharded_data in sharded_param_storages(module):
        alloc_storage(sharded_data)


def unshard_and_free_sharded_params(module: FSDPModule) -> None:
    """Unshard `module`'s FSDP group into its unsharded buffers, then free its sharded ones.

        before:  sharded: new weights   unsharded: old weights
        after:   sharded: freed         unsharded: new weights (what CUDA graphs read)

    The unshard is explicit because CUDA graph replays skip FSDP's forward pre-hook.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage

    with torch.inference_mode():
        module.unshard()
        # TODO: replace this with FSDPModule._free_sharded_params()
        for sharded_data in sharded_param_storages(module):
            free_storage(sharded_data)


def load_state_dict_by_fsdp_group(
    model: torch.nn.Module, state_dict: Mapping[str, Any]
) -> Any:
    """Copy new weights into ``model``, allocating about one FSDP group's sharded buffers at a time.

    Each group is allocated right before its first module loads and freed right after its last:

        load order:     tok_embeddings | layers.0 | ... | layers.N | norm     lm_head
        FSDP group:     [emb]          | [l0]     |     | [lN]     | [norm,   lm_head]
        allocate:       emb            | l0       |     | lN       | norm
        unshard, free:  emb            | l0       |     | lN       |          lm_head

    With tied embeddings the group is [tok_embeddings, norm, lm_head], so it stays
    allocated for the whole load.

    Why this shape:
      - Memory: allocating every sharded buffer first adds a second copy of the weights at
        FSDP degree 1: 64.6 GiB extra for bf16 Qwen3.5-35B-A3B at TP1, vs 1.6 GiB here.
      - Per group: modules from one `fully_shard` call share one FSDP state, so unsharding
        after `norm` would all-gather `lm_head` before it loads.
      - Hooks: a `load_state_dict` per module would also load its nested FSDP modules
        (the root holds every block), whose sharded buffers are still freed.

    Args:
        model: The parallelized TorchTitan model.
        state_dict: Keyed like ``model.state_dict()``, e.g. the generator's pinned CPU copy.
    """
    # One entry per FSDP group: fully_shard([norm, lm_head]) gives both modules one
    # state, and its sharded buffers are allocated and freed together.
    modules_by_fsdp_state = defaultdict(list)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            modules_by_fsdp_state[module._get_fsdp_state()].append(module)
    # Hooks run inside the one load_state_dict call, just before and after each group
    # loads; a call per module would also load its nested modules while they are freed.
    handles = []
    for modules in modules_by_fsdp_state.values():
        handles.append(
            modules[0].register_load_state_dict_pre_hook(
                lambda module, *args: restore_sharded_params(module)
            )
        )
        handles.append(
            modules[-1].register_load_state_dict_post_hook(
                lambda module, *args: unshard_and_free_sharded_params(module)
            )
        )
    incompatible_keys = model.load_state_dict(state_dict, strict=True)
    for handle in handles:
        handle.remove()
    return incompatible_keys


def load_initial_weights(
    model: torch.nn.Module,
    model_config: Decoder.Config,
    checkpointer_config: CheckpointManager.Config | None,
) -> None:
    """Load initial HF weights into ``model`` via CheckpointManager.

    A configured manager loads initial weights. ``None`` skips loading for the RL
    loop, where weights arrive via TorchStore.
    """
    if checkpointer_config is None:
        return

    adapter_cls = type(model).state_dict_adapter_cls
    sd_adapter = None
    if adapter_cls is not None:
        sd_adapter = adapter_cls(
            model_config=model_config,
            hf_assets_path=checkpointer_config.initial_load_path,
        )

    # Model-only CheckpointManager: initial_load_model_only=True (default)
    # ensures only MODEL state is loaded, so None optimizer/lr_scheduler
    # are never accessed.
    checkpointer = checkpointer_config.build(
        model_parts=[model],
        optimizers=None,
        lr_schedulers=None,
        ema=None,
        extra_states={},
        sd_adapter=sd_adapter,
    )
    checkpointer.load()
    # Free the large transient allocations the HF load/from_hf conversion left in the
    # caching allocator, so the later CUDA-graph capture (which needs its own private
    # pool) has room. Without this, large models (e.g. 235B) OOM capture even though
    # the live weights fit.
    torch.cuda.empty_cache()


def get_attention_dimensions(
    attention_config, model_dim: int
) -> tuple[int, int, int, int]:
    """Return query heads, KV heads, Q/K head dim, and value head dim."""
    num_heads = attention_config.n_heads
    num_kv_heads = getattr(attention_config, "n_kv_heads", None) or num_heads
    if hasattr(attention_config, "qk_nope_head_dim"):
        head_dim = attention_config.qk_nope_head_dim + attention_config.qk_rope_head_dim
        value_head_dim = attention_config.v_head_dim
    else:
        head_dim = getattr(attention_config, "head_dim", None)
        head_dim = head_dim if head_dim is not None else model_dim // num_heads
        value_head_dim = head_dim
    return num_heads, num_kv_heads, head_dim, value_head_dim
