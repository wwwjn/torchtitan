# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchtitan.experiments.rl.models.gdn_projection_ablation import (
    _adjusted_offset_weight,
    _clear_merged_weight_cache,
    _clear_offset_weight_cache,
    _merged_gdn_ba,
    _merged_gdn_qkvz,
    _merged_qwen35_qkv_gate,
    _qwen35_text_rope_cache,
)


def test_adjusted_offset_weight_matches_qwen35_semantics() -> None:
    weight = torch.tensor([-0.5, 0.25, 1.0], dtype=torch.bfloat16)

    actual = _adjusted_offset_weight(weight, torch.bfloat16)
    expected = (weight.float() + 1.0).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected)


def test_adjusted_offset_weight_cache_can_be_refreshed() -> None:
    weight = torch.zeros(3, dtype=torch.bfloat16)
    _adjusted_offset_weight(weight, torch.bfloat16)

    with torch.no_grad():
        weight.add_(1)
    _clear_offset_weight_cache()

    expected = (weight.float() + 1.0).to(torch.bfloat16)
    actual = _adjusted_offset_weight(weight, torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_qwen35_text_rope_cache_matches_vllm_layout() -> None:
    cos = torch.arange(8, dtype=torch.float32).view(2, 4)
    sin = cos + 10
    torchtitan_cache = torch.cat((cos, cos, sin, sin), dim=-1)

    actual = _qwen35_text_rope_cache(torchtitan_cache, rotary_dim=8)
    expected = torch.cat((cos, sin), dim=-1)

    torch.testing.assert_close(actual, expected)


def test_merged_gdn_projections_match_separate_linears() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, 13)

    qkvz_weights = tuple(torch.randn(size, 13) for size in (5, 5, 11, 11))
    expected_qkvz = torch.cat([F.linear(x, weight) for weight in qkvz_weights], dim=-1)
    actual_qkvz = _merged_gdn_qkvz(x, *qkvz_weights)
    torch.testing.assert_close(actual_qkvz, expected_qkvz)

    ba_weights = tuple(torch.randn(size, 13) for size in (3, 3))
    expected_ba = torch.cat([F.linear(x, weight) for weight in ba_weights], dim=-1)
    actual_ba = _merged_gdn_ba(x, *ba_weights)
    torch.testing.assert_close(actual_ba, expected_ba)

    attention_weights = tuple(torch.randn(size, 13) for size in (14, 5, 5))
    expected_attention = torch.cat(
        [F.linear(x, weight) for weight in attention_weights], dim=-1
    )
    actual_attention = _merged_qwen35_qkv_gate(x, *attention_weights)
    torch.testing.assert_close(actual_attention, expected_attention)


def test_merged_gdn_projection_cache_can_be_refreshed() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    weights = tuple(torch.randn(3, 4) for _ in range(2))
    _merged_gdn_ba(x, *weights)

    with torch.no_grad():
        weights[0].add_(1)
    _clear_merged_weight_cache()

    expected = torch.cat([F.linear(x, weight) for weight in weights], dim=-1)
    actual = _merged_gdn_ba(x, *weights)
    torch.testing.assert_close(actual, expected)
