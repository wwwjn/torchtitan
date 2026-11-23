# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`paged_varlen_attention_out` against plain attention over the gathered KV pages."""

import pytest
import torch
from torch.nn.attention import current_flash_attention_impl

from torchtitan.rl.generator.attention.attention import paged_varlen_attention_out

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# FA2 paged KV requires pages that are a multiple of 256 tokens.
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 8, 2, 64, 256


def _reference(query, key_cache, value_cache, query_lens, kv_lens, block_table):
    """fp32 causal attention per sequence; a sequence's query tokens are its last KV positions."""
    outputs = []
    query_start = 0
    group = NUM_HEADS // NUM_KV_HEADS
    for row, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        pages = block_table[row, : (kv_len + PAGE_SIZE - 1) // PAGE_SIZE]
        key = key_cache[pages].flatten(0, 1)[:kv_len].float()
        value = value_cache[pages].flatten(0, 1)[:kv_len].float()
        q = query[query_start : query_start + query_len].float()
        query_start += query_len
        scores = torch.einsum(
            "qhd,khd->hqk", q, key.repeat_interleave(group, dim=1)
        ) * (HEAD_DIM**-0.5)
        query_positions = torch.arange(kv_len - query_len, kv_len, device=q.device)
        mask = torch.arange(kv_len, device=q.device)[None, :] > query_positions[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
        outputs.append(
            torch.einsum(
                "hqk,khd->qhd",
                scores.softmax(dim=-1),
                value.repeat_interleave(group, dim=1),
            )
        )
    return torch.cat(outputs)


@pytest.mark.parametrize(
    "query_lens",
    [
        pytest.param([1, 1, 1, 1, 1], id="decode_only"),
        pytest.param([1, 1, 7, 33], id="mixed"),
    ],
)
def test_paged_varlen_attention_matches_reference(query_lens):
    torch.manual_seed(0)
    device = torch.device("cuda")
    kv_lens = [300, 1, 520, 700, 256][: len(query_lens)]
    kv_lens = [max(kv_len, query_len) for kv_len, query_len in zip(kv_lens, query_lens)]
    num_pages_per_seq = max((kv_len + PAGE_SIZE - 1) // PAGE_SIZE for kv_len in kv_lens)
    num_seqs = len(query_lens)
    num_pages = num_seqs * num_pages_per_seq
    block_table = (
        torch.randperm(num_pages, device=device, dtype=torch.int32)
        .view(num_seqs, num_pages_per_seq)
        .contiguous()
    )
    key_cache = torch.randn(
        num_pages,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    num_tokens = sum(query_lens)
    query = torch.randn(
        num_tokens, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cu_seqlens_q = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(query_lens, device=device).cumsum(0)
    seqused_k = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    # NaN catches a kernel path that leaves `out` unwritten.
    out = torch.full_like(query, float("nan"))
    result = paged_varlen_attention_out(
        out,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        seqused_k=seqused_k,
        block_table=block_table,
        scale=HEAD_DIM**-0.5,
        window_size=(-1, 0),
        enable_gqa=True,
        return_lse=False,
    )

    assert result.data_ptr() == out.data_ptr()
    expected = _reference(
        query, key_cache, value_cache, query_lens, kv_lens, block_table
    )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_flash_attention_impl() not in (None, "FA2"),
    reason="FA2 decode packing",
)
def test_fa2_decode_packing_matches_unpacked_call_bitwise():
    """The packed FA2 decode call computes the same bits as the (-1, 0) call it replaces."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_seqs, num_pages_per_seq = 6, 4
    kv_lens = torch.randint(1, num_pages_per_seq * PAGE_SIZE, (num_seqs,))
    block_table = (
        torch.randperm(num_seqs * num_pages_per_seq, device=device, dtype=torch.int32)
        .view(num_seqs, num_pages_per_seq)
        .contiguous()
    )
    key_cache = torch.randn(
        num_seqs * num_pages_per_seq,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        num_seqs, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kwargs = dict(
        cu_seqlens_q=torch.arange(num_seqs + 1, device=device, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=int(kv_lens.max()),
        seqused_k=kv_lens.to(device=device, dtype=torch.int32),
        block_table=block_table,
        scale=HEAD_DIM**-0.5,
        window_size=(-1, 0),
        return_lse=False,
    )

    packed = paged_varlen_attention_out(
        torch.full_like(query, float("nan")),
        query,
        key_cache,
        value_cache,
        enable_gqa=True,
        **kwargs,
    )
    # Repeating the KV heads skips the packing (no GQA) and runs the (-1, 0) call.
    unpacked = paged_varlen_attention_out(
        torch.full_like(query, float("nan")),
        query,
        key_cache.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=2),
        value_cache.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=2),
        enable_gqa=False,
        **kwargs,
    )

    assert torch.equal(packed, unpacked)
