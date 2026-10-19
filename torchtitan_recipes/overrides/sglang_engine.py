# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an RL recipe's generator with SGLang instead of vLLM.

``sglang_engine`` replaces the generator's ``VLLMEngine.Config`` with an
``SGLangEngine.Config``; the TorchTitan model, parallelism and sampling stay the
recipe's:

    python -m torchtitan.rl.train --module torchtitan_recipes.rl.alphabet_sort \\
        --config rl_grpo_qwen3_0_6b_varlen_batch_invariant \\
        --override torchtitan_recipes.overrides.sglang_engine.sglang_engine
"""

from torchtitan.config import override
from torchtitan.rl.generator.sglang.engine import SGLangEngine
from torchtitan.rl.generator.vllm.engine import VLLMEngine

__all__ = ["sglang_engine"]


@override(
    target=VLLMEngine.Config,
    description="Run the RL generator's TorchTitan model in SGLang",
)
def sglang_engine(
    cfg: VLLMEngine.Config,
    *,
    enable_cuda_graph: bool | None = None,
    hold_session_kv: bool | None = None,
) -> SGLangEngine.Config:
    """Return the SGLang engine config for a vLLM engine config.

    Args:
        cfg: The recipe's vLLM engine config.
        enable_cuda_graph: Capture SGLang's decode CUDA graphs. None: capture them
            unless the recipe's vLLM CUDA graph mode is ``"NONE"``.
        hold_session_kv: Keep multi-turn sessions' KV cached between turns (see
            ``SGLangEngine.Config.hold_session_kv``). None: keep it when the recipe
            sets ``session_kv_holder``.
    """
    if cfg.extra_engine_args:
        raise ValueError(
            "extra_engine_args are vLLM EngineArgs; set SGLangEngine.Config."
            "extra_server_args instead"
        )
    if enable_cuda_graph is None:
        enable_cuda_graph = cfg.cuda_graph.mode != "NONE"
    if hold_session_kv is None:
        hold_session_kv = cfg.session_kv_holder is not None
    return SGLangEngine.Config(
        enable_cuda_graph=enable_cuda_graph, hold_session_kv=hold_session_kv
    )
