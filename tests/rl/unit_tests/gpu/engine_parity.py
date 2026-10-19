# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate through a ``GenerationEngine`` and dump each request's tokens and logprobs.

Each engine runs in its own torchrun launch (one process per generator rank); a
comparison of the dumps checks that two engines running the same TorchTitan model
produce bitwise-identical tokens and logprobs in batch-invariant mode:

    torchrun --nproc_per_node=1 -m tests.rl.unit_tests.gpu.engine_parity \\
        --engine vllm --out /tmp/vllm.json
    torchrun --nproc_per_node=1 -m tests.rl.unit_tests.gpu.engine_parity \\
        --engine sglang --out /tmp/sglang.json
    python -m tests.rl.unit_tests.gpu.engine_parity --compare /tmp/vllm.json /tmp/sglang.json

The requests mix greedy and seeded temperature sampling, and GRPO-style groups that
share a prompt (prefix-cache hits). ``--admission`` changes the batches each engine
sees, and ``--weight-update-at-step`` reloads the current weights through
``load_prefetched_model_state_dict`` while requests are in flight; neither may change
the outputs.
"""

from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()

import argparse
import importlib
import json
import os
import random
import struct
import sys

import torch
import torch.distributed as dist

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig

# name -> (model builder module, flavor, default HF checkpoint)
MODELS = {
    "qwen3_0_6b": ("torchtitan.models.qwen3", "0.6B", "./assets/hf/Qwen3-0.6B"),
    "qwen3_30b_a3b": (
        "torchtitan.models.qwen3",
        "30B-A3B",
        "./assets/hf/Qwen3-30B-A3B",
    ),
    # Hybrid: GDN linear attention with a full-attention layer every fourth layer.
    "qwen3_5_0_8b": ("torchtitan.models.qwen3_5", "0.8B", "./assets/hf/Qwen3.5-0.8B"),
    # Random weights saved in HF format; no tokenizer, so prompts are synthetic.
    "qwen3_moe_debug": (
        "torchtitan.models.qwen3",
        "debugmodel_moe",
        "./assets/hf/Qwen3-MoE-debug",
    ),
}


_WORDS = (
    "alpha beta gamma delta epsilon sort these names alphabetically and explain why "
    "the river bends near the old mill while the scientist measures entropy of a gas "
    "in a sealed container under pressure"
).split()


def _build_prompts(
    model_path: str, num_prompts: int, vocab_size: int
) -> list[list[int]]:
    """Chat prompts from the checkpoint's tokenizer, or random token ids without one."""
    rng = random.Random(1234)
    # Lengths from a few tokens to more than one 256-token KV page.
    lengths = [8, 40, 120, 300, 16, 600, 64, 200]
    if not os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        return [
            [rng.randrange(vocab_size) for _ in range(2 * lengths[i % 8] + i)]
            for i in range(num_prompts)
        ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = []
    for i in range(num_prompts):
        num_words = lengths[i % 8] + rng.randint(0, 7)
        text = " ".join(rng.choice(_WORDS) for _ in range(num_words))
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(list(tokenizer.encode(rendered, add_special_tokens=False)))
    return prompts


def _build_requests(args, prompts: list[list[int]]) -> list[EngineRequest]:
    stop_token_ids = [int(t) for t in args.stop_token_ids.split(",")]
    requests = []
    # Per prompt: one greedy request, then a group of seeded samples sharing the prompt.
    for i, prompt in enumerate(prompts):
        samples = [(0.0, None)] + [
            (1.0, 1000 + 100 * i + j) for j in range(args.group_size)
        ]
        for j, (temperature, seed) in enumerate(samples):
            request = EngineRequest(
                request_id=f"req-{i}-{j}",
                prompt_token_ids=prompt,
                sampling=SamplingConfig(
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                    seed=seed,
                    stop_token_ids=stop_token_ids,
                ),
                group_id=i,
                routing_session_id=f"session-{i}-{j}",
                arrival_time=float(len(requests)),
            )
            request.min_policy_version = 0
            requests.append(request)
    return requests


def save_debug_moe_checkpoint(path: str) -> None:
    """Save Qwen3's ``debugmodel_moe`` with seeded random weights as an HF checkpoint."""
    from safetensors.torch import save_file

    from torchtitan.config import ParallelismConfig
    from torchtitan.models.qwen3 import build_model_config
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

    torch.manual_seed(1234)
    config = build_model_config(
        "debugmodel_moe",
        seq_len=2048,
        attn_backend="varlen",
        parallelism=ParallelismConfig(),
    )
    with torch.device("cpu"):
        model = config.build()
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    state_dict = {
        k: v.detach().to(torch.bfloat16).contiguous()
        for k, v in model.state_dict().items()
    }
    hf_state_dict = Qwen3StateDictAdapter(config, hf_assets_path=None).to_hf(state_dict)
    os.makedirs(path, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in hf_state_dict.items()},
        os.path.join(path, "model.safetensors"),
    )
    attention, moe = config.first_full_attention, config.first_moe
    hf_config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": config.dim,
        "num_hidden_layers": len(config.layers),
        "vocab_size": config.vocab_size,
        "num_attention_heads": attention.n_heads,
        "num_key_value_heads": attention.n_kv_heads,
        "head_dim": attention.head_dim,
        "num_experts": moe.router.num_experts,
        "num_experts_per_tok": moe.router.top_k,
        "moe_intermediate_size": moe.routed_experts.w2.in_features,
        "intermediate_size": 4 * config.dim,
        "max_position_embeddings": config.max_context_length,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(hf_config, f)


@torch.no_grad()
def _copy_live_weights_into_prefetched(engine) -> None:
    """Fill the engine's prefetched model state dict with the weights it runs now.

    The prefetched buffers hold each rank's FSDP shard (generator DP shards dim 0);
    the live parameters are the unsharded ones the forward reads.
    """
    from torch.distributed.tensor import DTensor

    # Both engines expose the TorchTitan model wrapper they run through _get_model().
    model = engine._get_model().model
    live = dict(model.named_parameters())
    live.update(dict(model.named_buffers()))
    dp_rank, dp_size = engine.data_parallel_rank, engine.data_parallel_size
    for key, dst in engine.prefetched_model_state_dict.items():
        src = live[key]
        src = src.to_local() if isinstance(src, DTensor) else src
        dst = dst.to_local() if isinstance(dst, DTensor) else dst
        if dst.shape != src.shape:
            # FSDP's dim-0 shard: torch.chunk rows of this DP rank.
            rows_per_rank = -(-src.shape[0] // dp_size)
            src = src[dp_rank * rows_per_rank :][: dst.shape[0]]
        dst.copy_(src)


def _build_engine(args):
    from torchtitan.config import DebugConfig, ParallelismConfig
    from torchtitan_recipes.rl.model_configs import build_rl_model_configs

    module_name, flavor, default_model_path = MODELS[args.model]
    parallelism = InferenceParallelismConfig(
        data_parallel_degree=args.dp,
        tensor_parallel_degree=args.tp,
        expert_parallel_degree=args.ep,
    )
    # The generator config of an RL recipe, sharded for this layout.
    _, model_config = build_rl_model_configs(
        importlib.import_module(module_name).build_model_config,
        flavor,
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=parallelism,
        seq_len=args.seq_len,
        attn_backend="varlen",
    )
    # Local compile regions do not support batch-invariant mode.
    model_config.local_compile_regions = []
    debug = DebugConfig(batch_invariant=True, deterministic=True, seed=0)
    # What Generator.__init__ does before it builds the engine.
    set_batch_invariance(debug.batch_invariant)
    Generator._set_determinism(debug)

    if args.engine == "vllm":
        from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
        from torchtitan.rl.generator.vllm.engine import VLLMEngine

        engine_config = VLLMEngine.Config(
            cuda_graph=VLLMCudaGraphConfig(mode="FULL" if args.cuda_graph else "NONE"),
            extra_engine_args=json.loads(args.vllm_extra_engine_args),
        )
    else:
        from torchtitan.rl.generator.sglang.engine import SGLangEngine

        engine_config = SGLangEngine.Config(
            enable_cuda_graph=args.cuda_graph,
            extra_server_args=json.loads(args.sglang_extra_server_args),
        )
    model_path = args.model_path or os.path.abspath(default_model_path)
    engine = engine_config.build(
        model=model_config,
        parallelism=parallelism,
        model_path=model_path,
        max_num_seqs=args.max_num_seqs,
        model_dtype="bfloat16",
        gpu_memory_limit=args.gpu_memory_limit,
        max_num_batched_tokens=args.max_num_batched_tokens,
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True, initial_load_path=model_path
        ),
        dist_moe_runtime=None,
        debug=debug,
        reset_kv_cache_on_model_state_dict_load=True,
        rank=int(os.environ.get("RANK", "0")),
        generator_name="engine_parity",
        output_dir="/tmp/engine_parity",
    )
    return engine, model_path, model_config.vocab_size


def generate(args) -> None:
    engine, model_path, vocab_size = _build_engine(args)
    requests = _build_requests(
        args, _build_prompts(model_path, args.num_prompts, vocab_size)
    )
    results: dict[str, dict] = {}
    num_steps = 0

    def step() -> None:
        nonlocal num_steps
        if num_steps == args.weight_update_at_step:
            _copy_live_weights_into_prefetched(engine)
            engine.load_prefetched_model_state_dict()
        for finished in engine.step():
            results[finished.request_id] = {
                "token_ids": finished.token_ids,
                "token_logprobs": finished.token_logprobs,
                "finish_reason": finished.finish_reason,
            }
        num_steps += 1

    # As the generator does, each DP rank admits its own share of the requests;
    # every loop below runs the same iterations on every rank, because each
    # engine call is a collective across ranks.
    dp_rank, dp_size = engine.data_parallel_rank, engine.data_parallel_size
    if args.dp_imbalanced:
        # DP rank r > 0 takes only request r - 1, so it runs idle (empty) steps
        # while rank 0 works through the rest.
        requests = (
            requests[dp_size - 1 :] if dp_rank == 0 else requests[dp_rank - 1 : dp_rank]
        )
    else:
        requests = requests[dp_rank::dp_size]

    def any_rank(flag: bool) -> bool:
        if not dist.is_initialized():
            return flag
        flag_tensor = torch.tensor([int(flag)], device="cuda")
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
        return bool(flag_tensor.item())

    if args.admission == "all":
        assert not engine.add_requests(requests)
        while engine.has_unfinished_requests():
            step()
    elif args.admission == "sequential":
        pending = list(requests)
        while any_rank(bool(pending)):
            if pending:
                assert not engine.add_requests([pending.pop(0)])
            while engine.has_unfinished_requests():
                step()
    else:  # staggered: admit one request per step
        pending = list(requests)
        while any_rank(bool(pending)) or engine.has_unfinished_requests():
            if pending:
                assert not engine.add_requests([pending.pop(0)])
            if engine.has_unfinished_requests():
                step()

    all_results = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
    if dist.is_initialized():
        dist.all_gather_object(all_results, results)
    else:
        all_results = [results]
    if int(os.environ.get("RANK", "0")) == 0:
        merged = {k: v for rank_results in all_results for k, v in rank_results.items()}
        with open(args.out, "w") as f:
            json.dump({"engine": args.engine, "results": merged}, f)
    engine.shutdown()
    if dist.is_initialized():
        dist.barrier()


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def compare(path_a: str, path_b: str) -> list[str]:
    """Return the ids of requests whose tokens or logprobs differ bitwise."""
    with open(path_a) as f:
        results_a = json.load(f)["results"]
    with open(path_b) as f:
        results_b = json.load(f)["results"]
    assert results_a.keys() == results_b.keys(), (sorted(results_a), sorted(results_b))
    mismatches = []
    for request_id, a in results_a.items():
        b = results_b[request_id]
        if a["token_ids"] != b["token_ids"] or [
            _bits(x) for x in a["token_logprobs"]
        ] != [_bits(x) for x in b["token_logprobs"]]:
            mismatches.append(request_id)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument(
        "--save-debug-moe-checkpoint", metavar="DIR", help="write it and exit"
    )
    parser.add_argument("--engine", choices=["vllm", "sglang"])
    parser.add_argument("--out")
    parser.add_argument("--model", default="qwen3_0_6b", choices=sorted(MODELS))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--model-path", default=None, help="HF checkpoint (default: MODELS)"
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--dp-imbalanced", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-limit", type=float, default=0.6)
    parser.add_argument("--stop-token-ids", default="151645,13")
    parser.add_argument(
        "--admission", choices=["all", "sequential", "staggered"], default="all"
    )
    parser.add_argument("--weight-update-at-step", type=int, default=-1)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--sglang-extra-server-args", default="{}")
    parser.add_argument("--vllm-extra-engine-args", default="{}")
    args = parser.parse_args()
    if args.save_debug_moe_checkpoint:
        save_debug_moe_checkpoint(args.save_debug_moe_checkpoint)
        return
    if args.compare:
        mismatches = compare(*args.compare)
        print(f"{len(mismatches)} mismatching requests: {sorted(mismatches)}")
        sys.exit(1 if mismatches else 0)
    generate(args)


if __name__ == "__main__":
    main()
