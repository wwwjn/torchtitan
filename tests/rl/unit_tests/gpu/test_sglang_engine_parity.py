# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise parity between the SGLang and vLLM generator engines on one TorchTitan model.

Each case launches ``engine_parity`` under torchrun once per engine and compares the
tokens and logprobs of every request bitwise. The vLLM reference runs once per
parallel layout; the SGLang runs vary admission order, chunked prefill, CUDA graphs,
KV memory pressure (SGLang retracts requests) and a weight reload mid-generation.
The multi-turn cases check that holding session KV keeps later turns' histories cached
while other sessions churn the KV pool, without changing any output.

Requires SGLang, two or more GPUs, and the Qwen3-0.6B HF checkpoint
(``ENGINE_PARITY_MODEL_PATH``, default: the recipe's ``./assets/hf/Qwen3-0.6B``).
The GDN cases also need the Qwen3.5-0.8B checkpoint (``ENGINE_PARITY_QWEN3_5_PATH``,
default: ``./assets/hf/Qwen3.5-0.8B``).
"""

import functools
import json
import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("sglang")

pytestmark = pytest.mark.multi_gpu

_MODEL_PATH = os.environ.get(
    "ENGINE_PARITY_MODEL_PATH", os.path.abspath("./assets/hf/Qwen3-0.6B")
)


def _run(
    tmp_dir: str,
    name: str,
    nproc: int,
    args: list[str],
    model_path: str = _MODEL_PATH,
) -> str:
    out = os.path.join(tmp_dir, f"{name}.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            "-m",
            "tests.rl.unit_tests.gpu.engine_parity",
            "--model-path",
            model_path,
            "--out",
            out,
            *args,
        ],
        check=True,
    )
    return out


@functools.cache
def _vllm_reference(tmp_dir: str, tp: int) -> str:
    return _run(tmp_dir, f"vllm_tp{tp}", tp, ["--engine", "vllm", "--tp", str(tp)])


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("engine_parity"))


@pytest.mark.skipif(
    not os.path.isdir(_MODEL_PATH) or torch.cuda.device_count() < 2,
    reason="needs the Qwen3-0.6B checkpoint and 2 GPUs",
)
@pytest.mark.parametrize(
    "name, tp, sglang_args",
    [
        ("eager", 1, []),
        ("cuda_graph", 1, ["--cuda-graph"]),
        (
            "staggered_chunked",
            1,
            ["--admission", "staggered", "--max-num-batched-tokens", "512"],
        ),
        (
            "kv_pressure_retract",
            1,
            [
                "--cuda-graph",
                "--sglang-extra-server-args",
                '{"max_total_tokens": 4096}',
            ],
        ),
        ("weight_reload", 1, ["--weight-update-at-step", "20"]),
        ("tp2_eager", 2, []),
        (
            "tp2_cuda_graph_staggered_chunked_reload",
            2,
            [
                "--cuda-graph",
                "--admission",
                "staggered",
                "--max-num-batched-tokens",
                "512",
                "--weight-update-at-step",
                "40",
            ],
        ),
    ],
)
def test_sglang_matches_vllm(tmp_dir, name, tp, sglang_args):
    from tests.rl.unit_tests.gpu.engine_parity import compare

    reference = _vllm_reference(tmp_dir, tp)
    sglang = _run(
        tmp_dir,
        f"sglang_{name}",
        tp,
        ["--engine", "sglang", "--tp", str(tp), *sglang_args],
    )
    assert compare(reference, sglang) == []


# Three-turn sessions on a 64-page KV pool. Between turns, 60 single-turn sessions churn
# the pool, enough to evict every waiting session's history unless it is held.
_SESSION_ARGS = [
    "--turns",
    "3",
    "--filler-prompts",
    "60",
    "--max-num-seqs",
    "8",
    "--stop-token-ids",
    "151645",
    "--cuda-graph",
]


@functools.cache
def _multi_turn_reference(tmp_dir: str, tp: int) -> str:
    # vLLM's default KV pool keeps every history: the hits holding can reach.
    return _run(
        tmp_dir,
        f"vllm_multi_turn_tp{tp}",
        tp,
        ["--engine", "vllm", "--tp", str(tp), *_SESSION_ARGS],
    )


def _cached_tokens(path: str, turn: int) -> int:
    with open(path) as f:
        results = json.load(f)["results"]
    return sum(
        result["num_cached_tokens"]
        for request_id, result in results.items()
        if request_id.endswith(f"/turn={turn}")
    )


@pytest.mark.skipif(
    not os.path.isdir(_MODEL_PATH) or torch.cuda.device_count() < 2,
    reason="needs the Qwen3-0.6B checkpoint and 2 GPUs",
)
@pytest.mark.parametrize(
    "name, tp, sglang_args, held_turns",
    [
        ("no_hold", 1, [], ()),
        ("hold", 1, ["--hold-session-kv"], (2, 3)),
        # Without a KV reset, a reload keeps the KV and its session tags.
        (
            "hold_reload",
            1,
            ["--hold-session-kv", "--no-kv-reset", "--weight-update-during-turn", "2"],
            (2, 3),
        ),
        # The reset flushes turn 1's histories; turn 2's requests, retracted or
        # waiting at the reload, still tag theirs for turn 3.
        (
            "hold_reload_kv_reset",
            1,
            ["--hold-session-kv", "--weight-update-during-turn", "2"],
            (3,),
        ),
        ("tp2_hold", 2, ["--hold-session-kv"], (2, 3)),
    ],
)
def test_sglang_session_kv(tmp_dir, name, tp, sglang_args, held_turns):
    """Holding session KV keeps the histories of later turns cached through the churn,
    and changes no output."""
    from tests.rl.unit_tests.gpu.engine_parity import compare

    reference = _multi_turn_reference(tmp_dir, tp)
    sglang = _run(
        tmp_dir,
        f"sglang_session_{name}",
        tp,
        [
            "--engine",
            "sglang",
            "--tp",
            str(tp),
            *_SESSION_ARGS,
            "--sglang-extra-server-args",
            '{"max_total_tokens": 16384}',
            *sglang_args,
        ],
    )
    assert compare(reference, sglang) == []
    for turn in (2, 3):
        cached, reachable = _cached_tokens(sglang, turn), _cached_tokens(
            reference, turn
        )
        if turn in held_turns:
            assert cached >= 0.8 * reachable, (turn, cached, reachable)
        else:
            assert cached <= 0.1 * reachable, (turn, cached, reachable)


@functools.cache
def _debug_moe_reference(tmp_dir: str, model_path: str, layout: tuple[int, ...]) -> str:
    dp, tp, ep = layout
    return _run(
        tmp_dir,
        f"vllm_moe_dp{dp}_tp{tp}_ep{ep}",
        dp * tp,
        ["--engine", "vllm", *_MOE_ARGS, *_layout_args(layout)],
        model_path=model_path,
    )


def _layout_args(layout: tuple[int, ...]) -> list[str]:
    dp, tp, ep = layout
    return ["--dp", str(dp), "--tp", str(tp), "--ep", str(ep)]


_MOE_ARGS = ["--model", "qwen3_moe_debug", "--stop-token-ids", "7"]


@pytest.fixture(scope="module")
def debug_moe_path(tmp_dir) -> str:
    from tests.rl.unit_tests.gpu.engine_parity import save_debug_moe_checkpoint

    path = os.path.join(tmp_dir, "qwen3_moe_debug")
    save_debug_moe_checkpoint(path)
    return path


@pytest.mark.parametrize(
    "name, layout, sglang_args",
    [
        ("eager", (1, 1, 1), []),
        (
            "staggered_chunked",
            (1, 1, 1),
            ["--admission", "staggered", "--max-num-batched-tokens", "512"],
        ),
        (
            "reload_kv_pressure",
            (1, 1, 1),
            [
                "--weight-update-at-step",
                "20",
                "--sglang-extra-server-args",
                '{"max_total_tokens": 4096}',
            ],
        ),
        # Expert parallelism over TP: tokens are padded to a TP multiple.
        ("tp2_ep2", (1, 2, 2), []),
        # Generator DP supplies EP ranks: SGLang's data-parallel attention.
        ("dp2_tp1_ep2", (2, 1, 2), []),
        # DP rank 1 runs empty idle steps through the MoE layers.
        ("dp2_tp1_ep2_imbalanced", (2, 1, 2), ["--dp-imbalanced"]),
        (
            "dp2_tp2_ep4_staggered_chunked_reload",
            (2, 2, 4),
            [
                "--admission",
                "staggered",
                "--max-num-batched-tokens",
                "512",
                "--weight-update-at-step",
                "30",
            ],
        ),
    ],
)
def test_sglang_matches_vllm_moe(tmp_dir, debug_moe_path, name, layout, sglang_args):
    """A random-weight Qwen3 MoE across DP x TP = EP layouts."""
    from tests.rl.unit_tests.gpu.engine_parity import compare

    dp, tp, _ = layout
    if torch.cuda.device_count() < dp * tp:
        pytest.skip(f"needs {dp * tp} GPUs")
    reference = _debug_moe_reference(tmp_dir, debug_moe_path, layout)
    sglang = _run(
        tmp_dir,
        f"sglang_moe_{name}",
        dp * tp,
        ["--engine", "sglang", *_MOE_ARGS, *_layout_args(layout), *sglang_args],
        model_path=debug_moe_path,
    )
    assert compare(reference, sglang) == []


_QWEN3_5_PATH = os.environ.get(
    "ENGINE_PARITY_QWEN3_5_PATH", os.path.abspath("./assets/hf/Qwen3.5-0.8B")
)
_QWEN3_5_ARGS = ["--model", "qwen3_5_0_8b", "--stop-token-ids", "248046,13"]


@functools.cache
def _qwen3_5_reference(tmp_dir: str, tp: int) -> str:
    return _run(
        tmp_dir,
        f"vllm_qwen3_5_tp{tp}",
        tp,
        ["--engine", "vllm", *_QWEN3_5_ARGS, "--tp", str(tp)],
        model_path=_QWEN3_5_PATH,
    )


@pytest.mark.skipif(
    not os.path.isdir(_QWEN3_5_PATH) or torch.cuda.device_count() < 2,
    reason="needs the Qwen3.5-0.8B checkpoint and 2 GPUs",
)
@pytest.mark.parametrize(
    "name, tp, sglang_args",
    [
        ("eager", 1, []),
        (
            "cuda_graph_staggered_chunked_reload",
            1,
            [
                "--cuda-graph",
                "--admission",
                "staggered",
                "--max-num-batched-tokens",
                "512",
                "--weight-update-at-step",
                "20",
            ],
        ),
        # Fewer state slots than requests: finished and retracted requests must
        # return their slots.
        (
            "slot_reuse_kv_pressure_reload",
            1,
            [
                "--max-num-seqs",
                "8",
                "--weight-update-at-step",
                "20",
                "--sglang-extra-server-args",
                '{"max_total_tokens": 4096}',
            ],
        ),
        (
            "mixed_chunks",
            1,
            [
                "--admission",
                "staggered",
                "--max-num-batched-tokens",
                "512",
                "--sglang-extra-server-args",
                '{"enable_mixed_chunk": true}',
            ],
        ),
        (
            "tp2_cuda_graph_staggered_chunked",
            2,
            [
                "--cuda-graph",
                "--admission",
                "staggered",
                "--max-num-batched-tokens",
                "512",
            ],
        ),
    ],
)
def test_sglang_matches_vllm_gdn(tmp_dir, name, tp, sglang_args):
    """Qwen3.5: GDN linear attention with a full-attention layer every fourth layer."""
    from tests.rl.unit_tests.gpu.engine_parity import compare

    reference = _qwen3_5_reference(tmp_dir, tp)
    sglang = _run(
        tmp_dir,
        f"sglang_qwen3_5_{name}",
        tp,
        ["--engine", "sglang", *_QWEN3_5_ARGS, "--tp", str(tp), *sglang_args],
        model_path=_QWEN3_5_PATH,
    )
    assert compare(reference, sglang) == []
