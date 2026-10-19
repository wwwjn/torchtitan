# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_config
from torchtitan.rl.generator.sglang.registry import (
    model_config_to_sglang_hf_config,
    SGLANG_MODEL_NAME,
)


def test_hf_config_comes_from_model_config():
    config = build_qwen3_config("0.6B", seq_len=1024, parallelism=ParallelismConfig())
    overrides = model_config_to_sglang_hf_config(config)
    attention = config.first_full_attention
    assert overrides["architectures"] == [SGLANG_MODEL_NAME]
    assert overrides["num_hidden_layers"] == len(config.layers)
    assert overrides["num_attention_heads"] == attention.n_heads
    assert overrides["num_key_value_heads"] == attention.n_kv_heads
    assert overrides["head_dim"] == attention.head_dim
    assert overrides["max_position_embeddings"] == config.max_context_length
    assert overrides["vocab_size"] == config.vocab_size


def test_replacement_keeps_layer_index_and_scale():
    from torchtitan.rl.generator.sglang.attention import SGLangInnerAttention
    from torchtitan.rl.generator.sglang.model_wrapper import (
        _replace_sglang_layer_configs,
    )

    config = build_qwen3_config("debugmodel", parallelism=ParallelismConfig())
    replaced = _replace_sglang_layer_configs(config)
    for layer_idx, (layer, new_layer) in enumerate(
        zip(config.layers, replaced.layers, strict=True)
    ):
        inner = new_layer.attention.inner_attention
        assert isinstance(inner, SGLangInnerAttention.Config)
        assert inner.layer_id == layer_idx
        assert inner.scale == layer.attention.softmax_scale
        assert inner.sharding_config == layer.attention.inner_attention.sharding_config


def test_replacement_maps_gdn_layers():
    from torchtitan.models.common.attention.gdn import GDN
    from torchtitan.rl.generator.sglang.gdn import SGLangInnerGDN
    from torchtitan.rl.generator.sglang.model_wrapper import (
        _replace_sglang_layer_configs,
    )

    config = build_qwen3_5_config("debugmodel", parallelism=ParallelismConfig())
    replaced = _replace_sglang_layer_configs(config)
    gdn_layers = 0
    for layer_idx, (layer, new_layer) in enumerate(
        zip(config.layers, replaced.layers, strict=True)
    ):
        if not isinstance(layer.attention, GDN.Config):
            continue
        gdn_layers += 1
        inner = new_layer.attention.inner_attention
        assert isinstance(inner, SGLangInnerGDN.Config)
        assert inner.layer_idx == layer_idx
        assert inner.head_k_dim == layer.attention.key_head_dim
        assert inner.conv_kernel_size == layer.attention.conv_kernel_size
    assert gdn_layers > 0


def test_replacement_rejects_kda():
    from torchtitan.models.kimi_k3 import build_model_config as build_kimi_config
    from torchtitan.rl.generator.sglang.model_wrapper import (
        _replace_sglang_layer_configs,
    )

    config = build_kimi_config("debugmodel", parallelism=ParallelismConfig())
    with pytest.raises(NotImplementedError, match="GDN linear attention only"):
        _replace_sglang_layer_configs(config)


def test_hf_config_lists_gdn_layers():
    from torchtitan.models.common.attention.gdn import GDN

    config = build_qwen3_5_config("debugmodel", parallelism=ParallelismConfig())
    hf_config = model_config_to_sglang_hf_config(config)
    assert hf_config["model_type"] == "torchtitan_hybrid"
    assert hf_config["layer_types"] == [
        "linear_attention"
        if isinstance(layer.attention, GDN.Config)
        else "full_attention"
        for layer in config.layers
    ]
    gdn = next(
        layer.attention
        for layer in config.layers
        if isinstance(layer.attention, GDN.Config)
    )
    assert hf_config["linear_key_head_dim"] == gdn.key_head_dim
    assert hf_config["linear_conv_kernel_dim"] == gdn.conv_kernel_size


def _sampler():
    """A TorchTitanSampler without SGLang's process groups (its __init__ reads them)."""
    from sglang.srt.layers.sampler import OutputLogprobProcessor

    from torchtitan.rl.generator.sglang.sampler import TorchTitanSampler

    sampler = TorchTitanSampler.__new__(TorchTitanSampler)
    torch.nn.Module.__init__(sampler)
    sampler._generators = {}
    sampler._batch_reqs = None
    sampler.output_logprob_processor = OutputLogprobProcessor()
    return sampler


def test_sampler_matches_vllm_sampling():
    """Tokens and logprobs match vLLM's v1 sampling ops over several steps."""
    from vllm.v1.sample.ops.topk_topp_sampler import random_sample
    from vllm.v1.sample.sampler import Sampler as VLLMSampler

    torch.manual_seed(0)
    vocab_size = 1000
    # (temperature, seed): greedy, seeded at T=1, seeded at T=0.7.
    rows = [(0.0, None), (1.0, 11), (0.7, 12)]
    reqs = [
        SimpleNamespace(
            rid=f"r{i}",
            sampling_params=SimpleNamespace(sampling_seed=seed),
            inflight_middle_chunks=0,
        )
        for i, (_, seed) in enumerate(rows)
    ]
    # SGLang stores a greedy request as temperature 1.0 and top_k 1.
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([1, 1 << 30, 1 << 30]),
        temperatures=torch.tensor([[1.0], [1.0], [0.7]]),
        is_all_greedy=False,
        has_custom_logit_processor=False,
    )
    sampler = _sampler()
    vllm_generators = {
        i: torch.Generator().manual_seed(seed)
        for i, (_, seed) in enumerate(rows)
        if seed is not None
    }
    for _ in range(3):
        logits = torch.randn(len(rows), vocab_size, dtype=torch.bfloat16)
        logits_output = SimpleNamespace(next_token_logits=logits.clone())
        sampler.set_batch_reqs(reqs)
        token_ids = sampler(
            logits_output,
            sampling_info,
            True,
            [0] * len(rows),
            [None] * len(rows),
            torch.zeros(len(rows), dtype=torch.int64),
        )

        # vLLM: fp32 logits, greedy argmax, temperature, processed logprobs, and
        # exponential-race sampling with per-request generators.
        expected_logits = logits.to(torch.float32)
        greedy = expected_logits.argmax(dim=-1)
        temperatures = torch.tensor([1.0 if t == 0.0 else t for t, _ in rows])
        expected_logits.div_(temperatures.unsqueeze(1))
        expected_logprobs = VLLMSampler.compute_logprobs(expected_logits)
        probs = expected_logits.softmax(dim=-1, dtype=torch.float32)
        random = random_sample(probs, vllm_generators)
        expected = torch.where(
            torch.tensor([t == 0.0 for t, _ in rows]), greedy, random
        )

        assert torch.equal(token_ids, expected)
        expected_token_logprobs = expected_logprobs.gather(
            -1, expected.unsqueeze(-1)
        ).squeeze(-1)
        assert torch.equal(
            torch.as_tensor(logits_output.next_token_logprobs), expected_token_logprobs
        )


def test_sampler_partial_prefill_row_draws_no_noise():
    """A partially prefilled request does not advance its generator."""
    sampler = _sampler()
    req = SimpleNamespace(
        rid="r0",
        sampling_params=SimpleNamespace(sampling_seed=5),
        inflight_middle_chunks=1,
    )
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([1 << 30]),
        temperatures=torch.tensor([[1.0]]),
        is_all_greedy=False,
        has_custom_logit_processor=False,
    )
    sampler.set_batch_reqs([req])
    sampler(
        SimpleNamespace(next_token_logits=torch.randn(1, 100)),
        sampling_info,
        False,
        [0],
        [None],
        torch.zeros(1, dtype=torch.int64),
    )
    assert "r0" not in sampler._generators


def test_sampler_samples_padding_rows_greedily():
    """Rows after the step's requests (data-parallel attention padding) take argmax."""
    sampler = _sampler()
    req = SimpleNamespace(
        rid="r0",
        sampling_params=SimpleNamespace(sampling_seed=3),
        inflight_middle_chunks=0,
    )
    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([1 << 30]),
        temperatures=torch.tensor([[1.0]]),
        is_all_greedy=False,
        has_custom_logit_processor=False,
    )
    logits = torch.randn(3, 50)
    sampler.set_batch_reqs([req])
    token_ids = sampler(
        SimpleNamespace(next_token_logits=logits.clone()),
        sampling_info,
        False,
        [0],
        [None],
        torch.zeros(3, dtype=torch.int64),
    )
    assert token_ids.shape == (3,)
    assert torch.equal(token_ids[1:], logits[1:].argmax(dim=-1))


def test_release_sessions_closes_this_ranks_sessions():
    """A release closes the named sessions and every session of the named groups,
    skipping sessions this rank never tagged, in sorted order."""
    from torchtitan.rl.generator.sglang.engine import SGLangEngine
    from torchtitan.rl.generator.types import EngineRequest, SamplingConfig

    closed = []
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.config = SGLangEngine.Config(hold_session_kv=True)
    engine._reset_kv_cache_on_model_state_dict_load = True
    engine._max_context_length = 64
    engine._vocab_size = 100
    engine._session_groups = {}
    engine._group_sessions = {}
    engine._scheduler = SimpleNamespace(
        process_input_requests=lambda requests: None,
        waiting_queue=[],
        close_session=lambda request: closed.append(request.session_id),
    )
    requests = []
    for group_id, session_id in [
        (1, "g1/s0"),
        (1, "g1/s1"),
        (2, "g2/s1"),
        (2, "g2/s0"),
    ]:
        request = EngineRequest(
            request_id=session_id,
            prompt_token_ids=[1, 2, 3],
            sampling=SamplingConfig(max_tokens=4),
            group_id=group_id,
            routing_session_id=session_id,
            arrival_time=0.0,
        )
        request.min_policy_version = 0
        requests.append(request)
    assert engine.add_requests(requests) == []

    engine.release_sessions(["g1/s1", "on-another-rank"], [])
    assert closed == ["g1/s1"]
    engine.release_sessions([], [2])
    assert closed == ["g1/s1", "g2/s0", "g2/s1"]
    engine.release_sessions(["g2/s0"], [1])
    assert closed == ["g1/s1", "g2/s0", "g2/s1", "g1/s0"]
