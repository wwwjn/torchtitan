# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import importlib
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    InnerAttention,
    LinearAttention,
    MLAAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_config
from torchtitan.models.kimi_k2_7 import build_model_config as build_kimi_k2_7_config
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.vllm import model_wrapper
from torchtitan.rl.generator.vllm.attention import (
    _VLLMKVBProj,
    _VLLMNativeMLAAttention,
    VLLMCompressedMLAAttention,
    VLLMInnerAttention,
    VLLMMLAAttention,
    VLLMMLAInnerAttention,
)
from torchtitan.rl.generator.vllm.model_wrapper import (
    _replace_vllm_layer_configs,
    VLLMModelWrapper,
)


@pytest.mark.parametrize(
    "model_name, expected_scale",
    [
        ("gpt_oss", 0.22665755425493697),
        ("qwen3", None),
    ],
)
def test_vllm_replacement_preserves_attention_scale(model_name, expected_scale):
    """Preserve YaRN scaling as well as the backend's default scale."""
    model_config = importlib.import_module(
        f"torchtitan.models.{model_name}"
    ).build_model_config("debugmodel", seq_len=128, parallelism=ParallelismConfig())
    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        attention_config = model_layer.attention
        scale = vllm_layer.attention.inner_attention.scale
        if expected_scale is None:
            assert scale is None
        else:
            assert scale == pytest.approx(expected_scale)
            with torch.device("meta"):
                attention = attention_config.build()
            assert scale == attention.softmax_scale


@pytest.mark.parametrize(
    "build_config",
    [build_qwen3_config, build_model_config],
    ids=["qwen3", "qwen3_5"],
)
def test_vllm_replacement_preserves_full_attention_selection(build_config) -> None:
    model_config = build_config("debugmodel", parallelism=ParallelismConfig())
    full_attention_index = next(
        i
        for i, layer in enumerate(model_config.layers)
        if layer.attention is model_config.first_full_attention
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    assert (
        vllm_config.first_full_attention
        is vllm_config.layers[full_attention_index].attention
    )
    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        inner_attention = vllm_layer.attention.inner_attention
        assert isinstance(inner_attention, InnerAttention.Config)
        assert inner_attention._owner is not None
        assert issubclass(inner_attention._owner, InnerAttention)
        assert (
            inner_attention.is_full_attention
            == model_layer.attention.inner_attention.is_full_attention
        )
        assert inner_attention.build_attention_metadata(torch.arange(8)) is None


@pytest.mark.parametrize("sliding_window_size", [None, 128])
def test_vllm_attention_classifies_sliding_window(sliding_window_size) -> None:
    config = VLLMInnerAttention.Config(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sliding_window_size=sliding_window_size,
    )
    assert config.is_full_attention is (sliding_window_size is None)


@pytest.mark.parametrize(
    "build_config", [build_model_config, build_kimi_k3_config], ids=["gdn", "kda"]
)
def test_vllm_replacement_preserves_attention_sharding(build_config) -> None:
    """Pass the trainer attention's sharding config through unchanged."""
    model_config = build_config(
        "debugmodel",
        attn_backend="flex",
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2, enable_sequence_parallel=True
        ),
    )
    # Exercise KDA independently of the unsupported MLA layers in Kimi K3.
    model_config = dataclasses.replace(
        model_config,
        layers=[
            layer
            for layer in model_config.layers
            if not isinstance(layer.attention.inner_attention, MLAAttention.Config)
        ],
    )
    assert model_config.layers
    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        assert model_layer.attention is not None
        assert vllm_layer.attention is not None
        model_sharding = model_layer.attention.inner_attention.sharding_config
        vllm_sharding = vllm_layer.attention.inner_attention.sharding_config
        assert model_sharding is not None
        assert vllm_sharding is not None
        assert vllm_sharding.in_src_shardings is model_sharding.in_src_shardings
        assert vllm_sharding.in_dst_shardings is model_sharding.in_dst_shardings
        assert vllm_sharding.out_src_shardings is model_sharding.out_src_shardings
        assert vllm_sharding.out_dst_shardings is model_sharding.out_dst_shardings
        assert vllm_sharding.local_spmd is model_sharding.local_spmd
        for name, layout in model_sharding.state_shardings.items():
            assert vllm_sharding.state_shardings[name] is layout


class _Wrapper:
    """Stand-in `self` for VLLMModelWrapper's real model state dict load methods (no vLLM)."""

    load_state_dict = VLLMModelWrapper.load_state_dict

    def __init__(self, model: nn.Module):
        self.model = model


def test_load_state_dict_allocates_one_fsdp_group_at_a_time(
    tmp_path, monkeypatch
) -> None:
    """Load new weights with at most two FSDP groups' sharded buffers allocated at once.

    Guards that each group's sharded buffers are allocated before its copy, even when
    FSDP's own load pre-hook already resharded the group (tok_embeddings here).
    """
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        # The generator's layout: a tied [tok_embeddings, norm, lm_head] group, one per block.
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(32, 16)
        model.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)) for _ in range(3)
        )
        model.norm = nn.LayerNorm(16)
        model.lm_head = nn.Linear(16, 32, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        mesh = init_device_mesh("cpu", (1,))
        fully_shard([model.tok_embeddings, model.norm, model.lm_head], mesh=mesh)
        for block in model.layers:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model.set_keep_unsharded_storage(True)
        new_state_dict = {k: v.detach() + 1.0 for k, v in model.state_dict().items()}
        groups = [
            group
            for module in (model.tok_embeddings, *model.layers)
            for group in module._get_fsdp_state()._fsdp_param_groups
        ]

        def is_allocated(group) -> bool:
            return all(
                p._sharded_param_data.untyped_storage().size() > 0
                for p in group.fsdp_params
            )

        wrapper = _Wrapper(model)
        max_allocated_groups = 0

        restore_sharded_params = model_wrapper._restore_sharded_params

        def checked_restore_sharded_params(module) -> None:
            nonlocal max_allocated_groups
            restore_sharded_params(module)
            # The load copies into these buffers next, so freed ones would be overrun.
            assert all(map(is_allocated, module._get_fsdp_state()._fsdp_param_groups))
            max_allocated_groups = max(
                max_allocated_groups, sum(map(is_allocated, groups))
            )

        monkeypatch.setattr(
            model_wrapper, "_restore_sharded_params", checked_restore_sharded_params
        )
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)
        wrapper.load_state_dict(new_state_dict)

        assert max_allocated_groups <= 2
        assert all(
            p._sharded_param_data.untyped_storage().size() == 0
            for group in groups
            for p in group.fsdp_params
        )
        for name, param in model.named_parameters(remove_duplicate=False):
            assert param.equal(new_state_dict[name].to_local()), name
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("build_config", "attn_backend", "expected_adapter"),
    [
        pytest.param(
            build_deepseek_config,
            "flex",
            VLLMCompressedMLAAttention,
            id="deepseek_v3-flex",
        ),
        pytest.param(
            build_deepseek_config,
            "varlen",
            VLLMCompressedMLAAttention,
            id="deepseek_v3-varlen",
        ),
        pytest.param(
            build_kimi_k2_7_config,
            "flex",
            VLLMCompressedMLAAttention,
            id="kimi_k2_7-flex",
        ),
        pytest.param(build_kimi_k3_config, "flex", VLLMMLAAttention, id="kimi_k3-flex"),
        pytest.param(
            build_kimi_k3_config, "varlen", VLLMMLAAttention, id="kimi_k3-varlen"
        ),
    ],
)
def test_vllm_replaces_mla(build_config, attn_backend, expected_adapter) -> None:
    """Debug models with a vLLM-supported latent size use the compressed cache."""
    model_config = build_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        attn_backend=attn_backend,
        seq_len=128,
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    num_mla_layers = 0
    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        if not isinstance(model_layer.attention.inner_attention, MLAAttention.Config):
            continue
        num_mla_layers += 1
        vllm_mla_config = vllm_layer.attention.inner_attention
        assert type(vllm_mla_config) is expected_adapter.Config
        assert vllm_mla_config.wkv_b is model_layer.attention.inner_attention.wkv_b
        assert vllm_mla_config.inner_attention.is_full_attention
    assert num_mla_layers > 0


def test_vllm_mla_adapter_builds_wkv_b_from_config():
    config = VLLMMLAAttention.Config(
        wkv_b=Linear.Config(in_features=5, out_features=28),
        packed_kv_head_dim=7,
        inner_attention=VLLMInnerAttention.Config(
            hidden_size=24,
            num_heads=4,
            num_kv_heads=4,
            head_dim=6,
            value_head_dim=3,
        ),
    )

    def initialize_module(attention, config):
        torch.nn.Module.__init__(attention)

    with mock.patch.object(
        VLLMInnerAttention,
        "__init__",
        autospec=True,
        side_effect=initialize_module,
    ):
        attention = VLLMMLAAttention(config)

    assert attention.wkv_b.weight.shape == (28, 5)
    assert attention.packed_kv_head_dim == 7


def test_vllm_mla_adapter_materializes_kv():
    q_THK = torch.randn(8, 4, 6)
    kv_c_normed_TL = torch.randn(8, 5)
    kv_THP = torch.randn(8, 4, 7)
    k_shared_TR = torch.randn(8, 2)
    wkv_b = mock.Mock(return_value=kv_THP.flatten(-2))
    expected_k_THK = torch.cat(
        (kv_THP[..., :4], k_shared_TR.unsqueeze(1).expand(-1, 4, -1)), dim=-1
    )
    expected_v_THV = kv_THP[..., 4:]
    out_THV = torch.randn(8, 4, 3)
    attention = VLLMMLAAttention.__new__(VLLMMLAAttention)
    torch.nn.Module.__init__(attention)
    object.__setattr__(attention, "wkv_b", wkv_b)
    attention.packed_kv_head_dim = 7
    qkv_attention = VLLMInnerAttention.__new__(VLLMInnerAttention)
    torch.nn.Module.__init__(qkv_attention)
    attention.inner_attention = qkv_attention

    with mock.patch.object(
        VLLMInnerAttention,
        "forward",
        autospec=True,
        return_value=out_THV,
    ) as vllm_forward:
        result = attention.forward(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR,
        )

    assert result is out_THV
    wkv_b.assert_called_once_with(kv_c_normed_TL)
    args = vllm_forward.call_args.args
    assert args[0] is qkv_attention
    assert args[1] is q_THK
    torch.testing.assert_close(args[2], expected_k_THK)
    torch.testing.assert_close(args[3], expected_v_THV)


def _kimi_k3_config_with_latent_rank(kv_lora_rank: int):
    config = build_kimi_k3_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        attn_backend="varlen",
        seq_len=128,
    )
    for layer in config.layers:
        if isinstance(layer.attention.inner_attention, MLAAttention.Config):
            layer.attention = dataclasses.replace(
                layer.attention, kv_lora_rank=kv_lora_rank
            )
    return config


def test_vllm_replaces_mla_with_compressed_cache_adapter():
    model_config = _kimi_k3_config_with_latent_rank(512)

    vllm_config = _replace_vllm_layer_configs(model_config)

    num_mla_layers = 0
    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        if isinstance(vllm_layer.attention, LinearAttention.Config):
            continue
        num_mla_layers += 1
        attention = model_layer.attention
        mla_config = vllm_layer.attention.inner_attention
        assert isinstance(mla_config, VLLMCompressedMLAAttention.Config)
        assert mla_config.wkv_b is attention.inner_attention.wkv_b
        inner_config = mla_config.inner_attention
        assert isinstance(inner_config, VLLMMLAInnerAttention.Config)
        assert inner_config.is_full_attention
        assert inner_config.num_heads == attention.n_heads
        assert inner_config.kv_lora_rank == 512
        assert inner_config.qk_nope_head_dim == attention.qk_nope_head_dim
        assert inner_config.qk_rope_head_dim == attention.qk_rope_head_dim
        assert inner_config.v_head_dim == attention.v_head_dim
        assert inner_config.scale == attention.softmax_scale
    assert num_mla_layers > 0


def test_vllm_batch_invariant_mla_keeps_materialized_adapter(monkeypatch):
    model_config = _kimi_k3_config_with_latent_rank(512)
    monkeypatch.setattr(
        model_wrapper, "uses_vllm_compressed_mla", lambda model_config: False
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        if not isinstance(vllm_layer.attention, LinearAttention.Config):
            mla_config = vllm_layer.attention.inner_attention
            assert isinstance(mla_config, VLLMMLAAttention.Config)
            assert (
                mla_config.inner_attention.scale
                == model_layer.attention.softmax_scale
            )


def test_vllm_kv_b_proj_reads_live_weight():
    linear = nn.Linear(5, 6, bias=False)
    kv_b_proj = _VLLMKVBProj(linear)
    x = torch.randn(3, 5)

    out, bias = kv_b_proj(x)

    assert bias is None
    torch.testing.assert_close(out, linear(x))
    linear.weight = nn.Parameter(torch.randn(6, 5))
    assert kv_b_proj.weight is linear.weight
    assert kv_b_proj.params_dtype == torch.float32
    torch.testing.assert_close(kv_b_proj(x)[0], linear(x))


def test_vllm_compressed_mla_owns_wkv_b_and_forwards_compact_inputs():
    config = VLLMCompressedMLAAttention.Config(
        wkv_b=Linear.Config(in_features=5, out_features=28),
        packed_kv_head_dim=7,
        inner_attention=VLLMMLAInnerAttention.Config(
            num_heads=4,
            kv_lora_rank=5,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=3,
            scale=0.5,
        ),
    )
    kv_b_projs = []

    def initialize_module(attention, config, *, kv_b_proj):
        torch.nn.Module.__init__(attention)
        kv_b_projs.append(kv_b_proj)

    with mock.patch.object(
        VLLMMLAInnerAttention,
        "__init__",
        autospec=True,
        side_effect=initialize_module,
    ):
        attention = VLLMCompressedMLAAttention(config)

    assert kv_b_projs == [attention.wkv_b]
    assert "wkv_b.weight" in attention.state_dict()
    q_THK = torch.randn(8, 4, 6)
    kv_c_normed_TL = torch.randn(8, 5)
    k_shared_TR = torch.randn(8, 2)
    out_THV = torch.randn(8, 4, 3)
    with mock.patch.object(
        VLLMMLAInnerAttention, "forward", autospec=True, return_value=out_THV
    ) as inner_forward:
        result = attention(q_THK, kv_c_normed_TL, k_shared_TR, scale=0.5)

    assert result is out_THV
    inner_forward.assert_called_once_with(
        attention.inner_attention, q_THK, kv_c_normed_TL, k_shared_TR, scale=0.5
    )


def _vllm_mla_inner_attention(num_heads: int, v_head_dim: int, scale: float):
    attention = VLLMMLAInnerAttention.__new__(VLLMMLAInnerAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = num_heads
    attention.v_head_dim = v_head_dim
    vllm_mla = mock.Mock()
    vllm_mla.scale = scale
    object.__setattr__(attention, "vllm_mla", vllm_mla)
    return attention, vllm_mla


def test_vllm_mla_inner_attention_passes_headless_shared_key():
    attention, vllm_mla = _vllm_mla_inner_attention(4, 3, 0.5)
    q_THK = torch.randn(8, 4, 6)
    kv_c_normed_TL = torch.randn(8, 5)
    k_shared_TR = torch.randn(8, 2)
    vllm_mla.return_value = torch.randn(8, 12)

    out_THV = attention(q_THK, kv_c_normed_TL, k_shared_TR, scale=0.5)

    args, kwargs = vllm_mla.call_args
    assert args[0] is q_THK
    assert args[1] is kv_c_normed_TL
    torch.testing.assert_close(args[2], k_shared_TR.unsqueeze(1))
    assert kwargs == {"output_shape": (8, 12)}
    torch.testing.assert_close(out_THV, vllm_mla.return_value.view(8, 4, 3))


def test_vllm_mla_inner_attention_rejects_mismatched_scale():
    attention, _ = _vllm_mla_inner_attention(4, 3, 0.5)

    with pytest.raises(ValueError, match="differs from the vLLM MLA scale"):
        attention(
            torch.randn(8, 4, 6), torch.randn(8, 5), torch.randn(8, 2), scale=0.25
        )


def _vllm_native_mla_attention(wkv_b: nn.Linear) -> _VLLMNativeMLAAttention:
    attention = _VLLMNativeMLAAttention.__new__(_VLLMNativeMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.impl = mock.Mock()
    attention.is_amx_bmm_enabled = False
    attention.is_aiter_triton_fp4_bmm_enabled = False
    attention.is_aiter_triton_fp8_bmm_enabled = False
    attention.dcp_q_replicate = False
    attention.quant_config = None
    attention.kv_lora_rank = 5
    attention.num_heads = 2
    attention.qk_nope_head_dim = 3
    attention.v_head_dim = 4
    attention.kv_b_proj = _VLLMKVBProj(wkv_b)
    for name in VLLMInnerAttention.vllm_attn_scale_buffer_names:
        attention.register_buffer(name, torch.tensor(0.0))
    return attention


def _absorbed_weights(wkv_b: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    weight = wkv_b.weight.detach().to(torch.bfloat16).T.view(5, 2, 7)
    w_uk, w_uv = weight.split([3, 4], dim=-1)
    return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)


def test_vllm_native_mla_refreshes_absorbed_weights_in_place():
    wkv_b = nn.Linear(5, 2 * (3 + 4), bias=False)
    attention = _vllm_native_mla_attention(wkv_b)
    model = nn.Sequential(attention)

    attention.process_weights_after_loading(torch.bfloat16)

    assert not any(
        name in attention.state_dict() for name in ("W_UK_T", "W_UV")
    )
    assert {"W_UK_T", "W_UV"} <= {name for name, _ in attention.named_buffers()}
    expected_w_uk_t, expected_w_uv = _absorbed_weights(wkv_b)
    torch.testing.assert_close(attention.W_UK_T, expected_w_uk_t)
    torch.testing.assert_close(attention.W_UV, expected_w_uv)

    data_ptrs = (attention.W_UK_T.data_ptr(), attention.W_UV.data_ptr())
    with torch.no_grad():
        wkv_b.weight.mul_(3.0)
    model_wrapper._refresh_absorbed_mla_weights(model)

    expected_w_uk_t, expected_w_uv = _absorbed_weights(wkv_b)
    torch.testing.assert_close(attention.W_UK_T, expected_w_uk_t)
    torch.testing.assert_close(attention.W_UV, expected_w_uv)
    assert (attention.W_UK_T.data_ptr(), attention.W_UV.data_ptr()) == data_ptrs
