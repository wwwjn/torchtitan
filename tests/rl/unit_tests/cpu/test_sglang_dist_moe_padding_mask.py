# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Dist-MoE padding mask and per-call size of the SGLang engine, without a GPU."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("sglang")

from torchtitan.rl.generator.sglang.model_wrapper import SGLangTransformerBody


class _RecordingDecoder:
    """Records the rows and kwargs the body passes to the TorchTitan model."""

    def __init__(self) -> None:
        self.seen: list[tuple[torch.Tensor, dict]] = []

    def __call__(self, input_ids, attention_metadata=None, positions=None, **kwargs):
        self.seen.append((input_ids, kwargs))
        return input_ids.float()[:, None].expand(-1, 4).contiguous()


class _RecordingRuntime:
    def __init__(self) -> None:
        self.num_tokens_per_call: list[int] = []

    def set_num_local_input_tokens_per_call(self, num_tokens: int) -> None:
        self.num_tokens_per_call.append(num_tokens)


@pytest.fixture(autouse=True)
def _no_sglang_forward_batch_context():
    with patch(
        "torchtitan.rl.generator.sglang.model_wrapper.set_forward_batch",
        lambda forward_batch: nullcontext(),
    ):
        yield


def _body(*, tp: int = 1, dp: int = 1, dist_moe: bool = True) -> Any:
    body = cast(Any, object.__new__(SGLangTransformerBody))
    torch.nn.Module.__init__(body)

    @contextmanager
    def activate_spmd():
        yield

    body.decoder = _RecordingDecoder()
    body.parallelism_context = SimpleNamespace(
        activate_spmd=activate_spmd, tp=tp, dp_shard=dp, ep_enabled=tp * dp > 1
    )
    body._dist_moe_runtime = _RecordingRuntime() if dist_moe else None
    body._max_prefill_graph_tokens = 64
    return body


def _forward_batch(out_cache_loc, *, extend=False, global_num_tokens=None):
    return SimpleNamespace(
        out_cache_loc=torch.tensor(out_cache_loc),
        forward_mode=SimpleNamespace(is_extend=lambda: extend),
        global_num_tokens_cpu=global_num_tokens,
    )


def test_graph_padding_rows_point_at_kv_slot_zero():
    # A CUDA graph replay pads its bucket with rows whose KV slot is 0.
    body = _body()
    body(torch.arange(4), torch.arange(4), _forward_batch([17, 18, 0, 0]))
    ((_, kwargs),) = body.decoder.seen
    assert kwargs["padding_mask"].tolist() == [False, False, True, True]
    assert body._dist_moe_runtime.num_tokens_per_call == [4]


def test_rows_appended_for_expert_sequence_parallelism_are_padding():
    # An eager step has KV slots for its real tokens only; the body pads 3 tokens
    # to a multiple of TP.
    body = _body(tp=2)
    body(torch.arange(3), torch.arange(3), _forward_batch([17, 18, 19]))
    ((input_ids, kwargs),) = body.decoder.seen
    assert input_ids.shape[0] == 4
    assert kwargs["padding_mask"].tolist() == [False, False, False, True]
    assert body._dist_moe_runtime.num_tokens_per_call == [2]


def test_idle_step_rows_are_all_padding():
    body = _body(tp=2)
    body(torch.arange(0), torch.arange(0), _forward_batch([]))
    ((_, kwargs),) = body.decoder.seen
    assert kwargs["padding_mask"].tolist() == [True, True]
    assert body._dist_moe_runtime.num_tokens_per_call == [1]


def test_eager_dp_step_uses_the_largest_dp_rank():
    # This DP rank has 3 tokens, another 9: with TP=2 both call Dist-MoE with
    # ceil(9 / 2) = 5 rows.
    body = _body(tp=2, dp=2)
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        body(
            torch.arange(3),
            torch.arange(3),
            _forward_batch([17, 18, 19], global_num_tokens=[3, 9]),
        )
    assert body._dist_moe_runtime.num_tokens_per_call == [5]


@pytest.mark.parametrize("extend,expected", [(False, 4), (True, 32)])
def test_dp_graph_capture_uses_a_size_every_dp_rank_replays(extend, expected):
    # Every DP rank replays the same decode bucket, but its own prefill bucket.
    body = _body(tp=2, dp=2)
    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        body(
            torch.arange(8),
            torch.arange(8),
            _forward_batch([0] * 8, extend=extend, global_num_tokens=[8, 8]),
        )
    assert body._dist_moe_runtime.num_tokens_per_call == [expected]


def test_without_dist_moe_passes_no_padding_mask():
    body = _body(dist_moe=False)
    body(torch.arange(2), torch.arange(2), _forward_batch([17, 0]))
    ((_, kwargs),) = body.decoder.seen
    assert kwargs == {}
