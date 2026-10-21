# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import MODEL_FLAVORS
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

_LAYER_ZERO_ATTN_RES_NORM = (
    "language_model.model.layers.0.self_attention_res_norm.weight"
)
_LAYER_ZERO_ATTN_RES_PROJ = (
    "language_model.model.layers.0.self_attention_res_proj.weight"
)


def _stage_state_dict(
    state_dict: dict[str, torch.Tensor], layer_ids: set[int]
) -> dict[str, torch.Tensor]:
    """Keep the given decoder layers, as a pipeline stage's state dict does."""
    return {
        key: value
        for key, value in state_dict.items()
        if key.startswith("layers.") and int(key.split(".")[1]) in layer_ids
    }


def test_to_hf_adds_layer_zero_placeholders_only_on_its_pipeline_stage():
    build_config, _ = MODEL_FLAVORS["debugmodel"]
    config = build_config(
        attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
    )
    model = config.build()
    model.init_states()
    state_dict = model.state_dict()
    adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)

    # A later stage holds neither layer 0 nor layer 1.
    later_stage = adapter.to_hf(_stage_state_dict(state_dict, {2, 3}))
    assert _LAYER_ZERO_ATTN_RES_NORM not in later_stage
    assert _LAYER_ZERO_ATTN_RES_PROJ not in later_stage

    # The first stage may end at layer 0.
    first_stage = adapter.to_hf(_stage_state_dict(state_dict, {0}))
    mlp_res_proj = state_dict["layers.0.ffn_res_proj.weight"]
    torch.testing.assert_close(
        first_stage[_LAYER_ZERO_ATTN_RES_NORM],
        torch.ones_like(state_dict["layers.0.ffn_res_norm.weight"]),
    )
    torch.testing.assert_close(
        first_stage[_LAYER_ZERO_ATTN_RES_PROJ], torch.zeros_like(mlp_res_proj)
    )
    assert (
        adapter.from_hf(first_stage).keys() == _stage_state_dict(state_dict, {0}).keys()
    )
