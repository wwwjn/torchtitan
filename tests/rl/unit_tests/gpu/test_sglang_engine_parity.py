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

Requires SGLang, two or more GPUs, and the Qwen3-0.6B HF checkpoint
(``ENGINE_PARITY_MODEL_PATH``, default: the recipe's ``./assets/hf/Qwen3-0.6B``).
"""

import functools
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
