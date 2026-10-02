# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)
from torchtitan.models.common.dist_moe.inference_runtime import DistMoeInferenceRuntime
from torchtitan.models.common.dist_moe.padding import (
    pad_to_num_tokens,
    route_padding_to_local_experts,
)


NUM_EXPERTS = 16
NUM_LOCAL = 4
FIRST_LOCAL = 8  # EP rank 2 of 4


def _routing(num_tokens: int, top_k: int = 4):
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(num_tokens, top_k, generator=generator)
    ids = torch.randint(0, NUM_EXPERTS, (num_tokens, top_k), generator=generator).to(
        torch.int32
    )
    return scores, ids


def _assert_padding_is_local(scores, ids, num_valid):
    assert torch.all(scores[num_valid:] == 0)
    padding_ids = ids[num_valid:]
    assert torch.all(padding_ids >= FIRST_LOCAL)
    assert torch.all(padding_ids < FIRST_LOCAL + NUM_LOCAL)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_route_padding_rewrites_only_rows_past_num_valid(as_tensor):
    scores, ids = _routing(10)
    num_valid = torch.tensor(6) if as_tensor else 6
    new_scores, new_ids = route_padding_to_local_experts(
        scores,
        ids,
        num_valid,
        first_local_expert=FIRST_LOCAL,
        num_local_experts=NUM_LOCAL,
    )
    torch.testing.assert_close(new_scores[:6], scores[:6])
    torch.testing.assert_close(new_ids[:6], ids[:6])
    _assert_padding_is_local(new_scores, new_ids, 6)
    assert new_ids.dtype == ids.dtype


def test_route_padding_spreads_rows_over_all_local_experts():
    scores, ids = _routing(64)
    _, new_ids = route_padding_to_local_experts(
        scores, ids, 0, first_local_expert=FIRST_LOCAL, num_local_experts=NUM_LOCAL
    )
    counts = torch.bincount(new_ids.flatten().long(), minlength=NUM_EXPERTS)
    local = counts[FIRST_LOCAL : FIRST_LOCAL + NUM_LOCAL]
    assert counts.sum() == local.sum()
    assert local.max() == local.min()


def test_pad_to_num_tokens_pads_and_routes_locally():
    x = torch.randn(5, 8)
    scores, ids = _routing(5)
    new_x, new_scores, new_ids = pad_to_num_tokens(
        x, scores, ids, 9, first_local_expert=FIRST_LOCAL, num_local_experts=NUM_LOCAL
    )
    assert new_x.shape == (9, 8) and new_scores.shape == (9, 4)
    torch.testing.assert_close(new_x[:5], x)
    assert torch.all(new_x[5:] == 0)
    torch.testing.assert_close(new_ids[:5], ids)
    _assert_padding_is_local(new_scores, new_ids, 5)


def test_pad_to_num_tokens_is_identity_when_equal_and_rejects_shrinking():
    x = torch.randn(5, 8)
    scores, ids = _routing(5)
    out = pad_to_num_tokens(
        x, scores, ids, 5, first_local_expert=FIRST_LOCAL, num_local_experts=NUM_LOCAL
    )
    assert out[0] is x and out[1] is scores and out[2] is ids
    with pytest.raises(ValueError, match="cannot pad"):
        pad_to_num_tokens(
            x,
            scores,
            ids,
            4,
            first_local_expert=FIRST_LOCAL,
            num_local_experts=NUM_LOCAL,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scratch_capacity_factor": 0.0},
        {"scratch_capacity_factor": float("inf")},
        {"vmm_capacity_factor": 0.0},
        {"num_sms": 0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DistMoeInferenceRuntime.Config(**kwargs)


def _runtime(max_local_input_tokens: int, group_max: int) -> DistMoeInferenceRuntime:
    runtime = object.__new__(DistMoeInferenceRuntime)
    runtime.max_local_input_tokens = max_local_input_tokens
    runtime.first_local_expert = FIRST_LOCAL
    runtime.num_local_experts = NUM_LOCAL
    runtime._ep_group_max_tokens = lambda num_tokens: group_max  # type: ignore[method-assign]
    return runtime


def test_equalize_inputs_pads_to_group_max():
    runtime = _runtime(max_local_input_tokens=32, group_max=12)
    x = torch.randn(7, 8)
    scores, ids = _routing(7)
    new_x, new_scores, new_ids = runtime.equalize_inputs(x, scores, ids)
    assert new_x.shape[0] == new_scores.shape[0] == new_ids.shape[0] == 12
    _assert_padding_is_local(new_scores, new_ids, 7)


def test_equalize_inputs_rejects_group_above_planned_tokens():
    runtime = _runtime(max_local_input_tokens=8, group_max=12)
    x = torch.randn(7, 8)
    scores, ids = _routing(7)
    with pytest.raises(ValueError, match="above the planned"):
        runtime.equalize_inputs(x, scores, ids)


def test_runtime_rejects_pipeline_parallelism_and_missing_experts():
    context = SimpleNamespace(pp_enabled=True, cp=1, tp=1)
    with pytest.raises(ValueError, match="at least one expert module"):
        DistMoeInferenceRuntime(
            DistMoeInferenceRuntime.Config(),
            model_parts=[torch.nn.Linear(2, 2)],
            parallelism_context=context,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            max_num_batched_tokens=8,
        )
