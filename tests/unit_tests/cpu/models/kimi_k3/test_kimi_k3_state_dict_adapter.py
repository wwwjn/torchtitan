# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

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


def test_hf_load_writes_routed_experts_in_place():
    build_config, _ = MODEL_FLAVORS["debugmodel"]
    config = build_config(
        attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
    )
    model = config.build()
    model.init_states()
    state_dict = model.state_dict()
    adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
    expert_keys = [key for key in state_dict if ".routed_experts." in key]
    assert expert_keys

    # An HF load fills to_hf's tensors in place before from_hf runs.
    hf_state_dict = adapter.to_hf(state_dict)
    with torch.no_grad():
        for value in hf_state_dict.values():
            if value.is_floating_point():
                value.copy_(torch.randn_like(value))
    restored = adapter.from_hf(hf_state_dict)

    for key in expert_keys:
        # Routed experts come back as views of the parameters to_hf split,
        # which already hold the loaded values, not as a second copy.
        assert (
            restored[key].untyped_storage().data_ptr()
            == state_dict[key].untyped_storage().data_ptr()
        )
        torch.testing.assert_close(restored[key], state_dict[key], rtol=0, atol=0)
    model.load_state_dict(restored, strict=True)


def test_dt_bias_round_trips_an_uneven_head_shard():
    # The debug model's 4 KDA heads on 3 ranks shard unevenly, as K3's 96 heads
    # do on a 64-rank generator; the fake process group still runs DTensor's
    # sharding propagation, which rejected flattening such a shard.
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=3)
    try:
        mesh = init_device_mesh("cpu", (3,), mesh_dim_names=("dp",))
        build_config, _ = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
        )
        attention = config.layers[1].attention
        shape = (attention.n_heads, attention.head_dim)
        key = "layers.1.attention.dt_bias"
        dt_bias = DTensor.from_local(
            torch.zeros(2, attention.head_dim),
            mesh,
            (Shard(0),),
            run_check=False,
            shape=torch.Size(shape),
            stride=(attention.head_dim, 1),
        )
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)

        hf_state_dict = adapter.to_hf({key: dt_bias})
        (hf_value,) = hf_state_dict.values()
        assert hf_value.shape == (attention.n_heads * attention.head_dim,)
        assert hf_value.placements == (Replicate(),)

        restored = adapter.from_hf(hf_state_dict)[key]
        assert restored.shape == shape
        assert restored.placements == (Shard(0),)
    finally:
        dist.destroy_process_group()
