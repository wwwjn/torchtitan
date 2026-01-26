#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness check script to compare model outputs between vLLM Native and vLLM TorchTitan.

This script directly runs the forward pass on both models with the same input
and compares the output logits to verify correctness.

Usage:
    # Single GPU (TP=1)
    python scripts/check_inference_correctness.py --checkpoint /path/to/checkpoint

    # Multi-GPU (TP=2)
    python scripts/check_inference_correctness.py --checkpoint /path/to/checkpoint --tp 2
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torchtitan.experiments.rl.unified 

# Must set spawn method before any CUDA operations or vLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Enable vLLM V1 engine with multiprocessing disabled for direct model access
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


@dataclass
class CorrectnessConfig:
    """Configuration for correctness checking."""

    model_path: str = "Qwen/Qwen3-1.7B"
    torchtitan_checkpoint_path: str = ""
    tp: int = 1
    seq_len: int = 32  # Length of fake input sequence
    rtol: float = 1e-3  # Relative tolerance for comparison
    atol: float = 1e-3  # Absolute tolerance for comparison
    swap_attention: bool = False  # Swap TorchTitan attention with vLLM native
    compare_weights: bool = False  # Compare model weights between native and TorchTitan
    test_native: bool = False  # Test native TorchTitan model (not through vLLM)


def create_fake_input(seq_len: int, vocab_size: int = 32000, device: str = "cuda"):
    """Create a fake tokenized input sequence.

    vLLM expects flattened 1D tensors of shape (total_tokens,) rather than 2D (batch, seq_len).
    """
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    # vLLM uses flattened input_ids (seq_len,) not (1, seq_len)
    input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long, device=device)
    return input_ids


def get_worker_from_engine(engine):
    """Extract the driver worker from vLLM engine."""
    # In V1 engine, model_executor is exposed directly on the engine
    return engine.model_executor.driver_worker


def get_model_from_worker(worker):
    """Extract the underlying model from vLLM worker."""
    return worker.model_runner.model


def create_dummy_forward_context(worker, seq_len: int):
    """Create a dummy forward context for direct model forward pass."""
    from vllm.forward_context import ForwardContext, override_forward_context

    vllm_config = worker.vllm_config
    model_runner = worker.model_runner

    # Get the static forward context which contains attention layer references
    no_compile_layers = vllm_config.compilation_config.static_forward_context

    # Create a dict mapping layer names to None for attention metadata
    # This is a minimal setup - attention ops will receive None metadata
    attn_metadata = {layer_name: None for layer_name in no_compile_layers.keys()}

    # Create a minimal ForwardContext
    forward_context = ForwardContext(
        no_compile_layers=no_compile_layers,
        attn_metadata=attn_metadata,
        virtual_engine=0,
        dp_metadata=None,
    )

    return override_forward_context(forward_context)


def compare_model_weights(native_model, torchtitan_model):
    """
    Compare weights between vLLM native and TorchTitan models.

    Weight mappings:
    - Attention: qkv_proj = concat(wq, wk, wv), o_proj = wo
    - FeedForward: gate_up_proj = concat(w1, w3), down_proj = w2

    Args:
        native_model: vLLM native Qwen3 model
        torchtitan_model: TorchTitan model (TorchTitanVLLMModelWrapper)
    """
    from torch.distributed._tensor import DTensor

    print("\n" + "=" * 60)
    print("WEIGHT COMPARISON")
    print("=" * 60)

    def get_local_tensor(param):
        """Extract local tensor from DTensor if needed."""
        if isinstance(param, DTensor):
            return param.to_local()
        return param

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol=1e-5, atol=1e-5):
        """Compare two tensors and print results."""
        t1 = get_local_tensor(t1).float()
        t2 = get_local_tensor(t2).float()

        if t1.shape != t2.shape:
            print(f"  {name}: SHAPE MISMATCH - native {t1.shape} vs torchtitan {t2.shape}")
            return False

        close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
        max_diff = (t1 - t2).abs().max().item()
        mean_diff = (t1 - t2).abs().mean().item()

        status = "MATCH" if close else "DIFFER"
        print(f"  {name}: {status} (shape={list(t1.shape)}, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return close

    # Get inner models
    # vLLM native: Qwen3ForCausalLM -> .model -> .layers[i] -> .self_attn, .mlp
    # TorchTitan: TorchTitanVLLMModelWrapper -> .model -> .layers['i'] -> .attention, .feed_forward

    native_inner = native_model.model if hasattr(native_model, 'model') else native_model
    tt_wrapper = torchtitan_model
    tt_inner = tt_wrapper.model if hasattr(tt_wrapper, 'model') else tt_wrapper

    all_match = True

    # Compare embeddings
    print("\n[Embeddings]")
    if hasattr(native_inner, 'embed_tokens') and hasattr(tt_inner, 'tok_embeddings'):
        match = compare_tensors(
            "tok_embeddings",
            native_inner.embed_tokens.weight,
            tt_inner.tok_embeddings.weight
        )
        all_match = all_match and match

    # Compare layers
    native_layers = native_inner.layers if hasattr(native_inner, 'layers') else []
    tt_layers = tt_inner.layers if hasattr(tt_inner, 'layers') else {}

    # Determine number of layers
    num_layers = min(len(native_layers), len(tt_layers))
    print(f"\nComparing {num_layers} layers...")

    for i in range(num_layers):
        print(f"\n[Layer {i}]")
        native_layer = native_layers[i]
        tt_layer = tt_layers[str(i)]

        # Attention weights
        native_attn = native_layer.self_attn
        tt_attn = tt_layer.attention

        # Get TorchTitan attention weights
        wq = get_local_tensor(tt_attn.wq.weight)
        wk = get_local_tensor(tt_attn.wk.weight)
        wv = get_local_tensor(tt_attn.wv.weight)
        wo = get_local_tensor(tt_attn.wo.weight)

        # vLLM qkv_proj = concat(Q, K, V)
        native_qkv = get_local_tensor(native_attn.qkv_proj.weight)
        tt_qkv = torch.cat([wq, wk, wv], dim=0)
        match = compare_tensors("attention.qkv_proj (vs concat(wq,wk,wv))", native_qkv, tt_qkv)
        all_match = all_match and match

        # o_proj = wo
        native_o = get_local_tensor(native_attn.o_proj.weight)
        match = compare_tensors("attention.o_proj (vs wo)", native_o, wo)
        all_match = all_match and match

        # q_norm, k_norm
        if hasattr(native_attn, 'q_norm') and hasattr(tt_attn, 'q_norm'):
            native_q_norm = get_local_tensor(native_attn.q_norm.weight)
            tt_q_norm = get_local_tensor(tt_attn.q_norm.weight)
            match = compare_tensors("attention.q_norm", native_q_norm, tt_q_norm)
            all_match = all_match and match

            native_k_norm = get_local_tensor(native_attn.k_norm.weight)
            tt_k_norm = get_local_tensor(tt_attn.k_norm.weight)
            match = compare_tensors("attention.k_norm", native_k_norm, tt_k_norm)
            all_match = all_match and match

        # FeedForward/MLP weights
        native_mlp = native_layer.mlp
        tt_ff = tt_layer.feed_forward

        # Get TorchTitan FeedForward weights
        w1 = get_local_tensor(tt_ff.w1.weight)  # gate_proj
        w2 = get_local_tensor(tt_ff.w2.weight)  # down_proj
        w3 = get_local_tensor(tt_ff.w3.weight)  # up_proj

        # vLLM gate_up_proj = concat(gate=w1, up=w3)
        native_gate_up = get_local_tensor(native_mlp.gate_up_proj.weight)
        tt_gate_up = torch.cat([w1, w3], dim=0)
        match = compare_tensors("mlp.gate_up_proj (vs concat(w1,w3))", native_gate_up, tt_gate_up)
        all_match = all_match and match

        # down_proj = w2
        native_down = get_local_tensor(native_mlp.down_proj.weight)
        match = compare_tensors("mlp.down_proj (vs w2)", native_down, w2)
        all_match = all_match and match

        # Layer norms
        native_input_ln = get_local_tensor(native_layer.input_layernorm.weight)
        tt_attn_norm = get_local_tensor(tt_layer.attention_norm.weight)
        match = compare_tensors("input_layernorm (vs attention_norm)", native_input_ln, tt_attn_norm)
        all_match = all_match and match

        native_post_ln = get_local_tensor(native_layer.post_attention_layernorm.weight)
        tt_ffn_norm = get_local_tensor(tt_layer.ffn_norm.weight)
        match = compare_tensors("post_attention_layernorm (vs ffn_norm)", native_post_ln, tt_ffn_norm)
        all_match = all_match and match

    # Compare final norm
    print("\n[Final Norm]")
    if hasattr(native_inner, 'norm') and hasattr(tt_inner, 'norm'):
        native_norm = get_local_tensor(native_inner.norm.weight)
        tt_norm = get_local_tensor(tt_inner.norm.weight)
        match = compare_tensors("final_norm", native_norm, tt_norm)
        all_match = all_match and match

    # Compare output/lm_head
    print("\n[Output/LM Head]")
    if hasattr(native_model, 'lm_head') and hasattr(tt_inner, 'output'):
        native_lm_head = get_local_tensor(native_model.lm_head.weight)
        tt_output = get_local_tensor(tt_inner.output.weight)
        match = compare_tensors("lm_head (vs output)", native_lm_head, tt_output)
        all_match = all_match and match

    print("\n" + "=" * 60)
    if all_match:
        print("WEIGHT COMPARISON: ALL WEIGHTS MATCH")
    else:
        print("WEIGHT COMPARISON: SOME WEIGHTS DIFFER")
    print("=" * 60)

    return all_match


def swap_torchtitan_attention_with_vllm_native(model, vllm_config):
    """
    Replace TorchTitan attention blocks with vLLM native Qwen3Attention.

    This allows us to isolate whether differences in logits come from the
    attention implementation or other parts of the model.

    Args:
        model: TorchTitan model (the inner model from TorchTitanVLLMModelWrapper)
        vllm_config: VllmConfig for cache config access
    """
    from vllm.model_executor.models.qwen3 import Qwen3Attention
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.config import set_current_vllm_config
    from torch.distributed._tensor import DTensor

    print("=" * 60)
    print("Swapping TorchTitan attention with vLLM native Qwen3Attention...")
    print("=" * 60)

    # Get model config
    model_args = model.config if hasattr(model, 'config') else None
    if model_args is None:
        raise ValueError("Model does not have config attribute")

    cache_config = vllm_config.cache_config if hasattr(vllm_config, 'cache_config') else None

    # Get the inner model (TorchTitanVLLMModelWrapper.model)
    inner_model = model.model if hasattr(model, 'model') else model

    swapped_count = 0
    for layer_name, layer in inner_model.layers.items():
        if not hasattr(layer, 'attention'):
            continue

        old_attn = layer.attention
        print(f"Swapping attention in layer {layer_name}")

        # Create vLLM native Qwen3Attention
        # Note: rope_parameters format for vLLM
        rope_parameters = {
            "rope_type": "default",
            "base": model_args.rope_theta,
        }

        # Must use set_current_vllm_config context for vLLM layers to initialize properly
        with set_current_vllm_config(vllm_config):
            new_attn = Qwen3Attention(
                hidden_size=model_args.dim,
                num_heads=model_args.n_heads,
                num_kv_heads=model_args.n_kv_heads if hasattr(model_args, 'n_kv_heads') else model_args.n_heads,
                rope_parameters=rope_parameters,
                max_position=model_args.max_seq_len if hasattr(model_args, 'max_seq_len') else 4096 * 32,
                head_dim=model_args.head_dim,
                rms_norm_eps=model_args.norm_eps,
                qkv_bias=False,
                cache_config=cache_config,
                quant_config=None,
                prefix=f"model.layers.{layer_name}.self_attn",
            )

        # Move to same device as old attention
        device = next(old_attn.parameters()).device
        dtype = next(old_attn.parameters()).dtype
        new_attn = new_attn.to(device=device, dtype=dtype)

        # Map weights from TorchTitan format to vLLM format
        # TorchTitan: wq, wk, wv (separate) -> vLLM: qkv_proj (merged)
        # TorchTitan: wo -> vLLM: o_proj

        with torch.no_grad():
            # Get weights, handling DTensor if needed
            def get_local_weight(param):
                if isinstance(param, DTensor):
                    return param.to_local()
                return param

            wq_weight = get_local_weight(old_attn.wq.weight)
            wk_weight = get_local_weight(old_attn.wk.weight)
            wv_weight = get_local_weight(old_attn.wv.weight)
            wo_weight = get_local_weight(old_attn.wo.weight)

            # QKV projection: stack [Q, K, V] weights
            # vLLM's QKVParallelLinear expects interleaved format for TP
            # For TP=1, we can just concatenate
            tp_size = get_tensor_model_parallel_world_size()

            if tp_size == 1:
                # Simple case: just concatenate
                qkv_weight = torch.cat([wq_weight, wk_weight, wv_weight], dim=0)
                new_attn.qkv_proj.weight.copy_(qkv_weight)
            else:
                # For TP > 1, weights are already sharded, need careful mapping
                # This is complex - for now, just concatenate and hope for the best
                print(f"  Warning: TP={tp_size} weight mapping may not be correct")
                qkv_weight = torch.cat([wq_weight, wk_weight, wv_weight], dim=0)
                new_attn.qkv_proj.weight.copy_(qkv_weight)

            # Output projection
            new_attn.o_proj.weight.copy_(wo_weight)

            # Q/K norms
            q_norm_weight = get_local_weight(old_attn.q_norm.weight)
            k_norm_weight = get_local_weight(old_attn.k_norm.weight)
            new_attn.q_norm.weight.copy_(q_norm_weight)
            new_attn.k_norm.weight.copy_(k_norm_weight)

        # Create a wrapper that adapts the interface
        # TorchTitan calls: attention(x, rope_cache, attention_masks, positions)
        # vLLM Qwen3Attention expects: forward(positions, hidden_states)
        class VLLMNativeAttentionWrapper(torch.nn.Module):
            def __init__(self, vllm_attn):
                super().__init__()
                self.vllm_attn = vllm_attn

            def forward(self, x, rope_cache=None, attention_masks=None, positions=None):
                # x: [batch, seq_len, hidden_size]
                # positions: [batch, seq_len] or [seq_len]
                batch_size, seq_len, hidden_size = x.shape

                # Flatten for vLLM: [batch * seq_len, hidden_size]
                x_flat = x.view(batch_size * seq_len, hidden_size)

                # Flatten positions if needed: [batch * seq_len]
                if positions is not None:
                    if positions.dim() == 2:
                        positions_flat = positions.view(-1)
                    else:
                        positions_flat = positions
                else:
                    positions_flat = torch.arange(seq_len, device=x.device).repeat(batch_size)

                # Call vLLM attention
                output_flat = self.vllm_attn(positions_flat, x_flat)

                # Reshape back: [batch, seq_len, hidden_size]
                output = output_flat.view(batch_size, seq_len, hidden_size)

                return output

        # Replace the attention module
        layer.attention = VLLMNativeAttentionWrapper(new_attn)
        swapped_count += 1
        print(f"  Successfully swapped layer {layer_name}")

    print(f"Swapped {swapped_count} attention layers")
    return model


def load_vllm_native_engine(config: CorrectnessConfig):
    """Load vLLM engine with native HuggingFace model."""
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.engine.arg_utils import EngineArgs

    print("=" * 60)
    print("Loading vLLM with native HuggingFace model...")
    print(f"Model: {config.model_path}")
    print(f"Tensor Parallel Size: {config.tp}")
    print("=" * 60)

    engine_args = EngineArgs(
        model=config.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        tensor_parallel_size=config.tp,
    )

    engine = LLMEngine.from_engine_args(engine_args)
    print(f"Engine type: {type(engine)}")

    return engine


def run_vllm_native_forward(config: CorrectnessConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass on vLLM with native HuggingFace model."""
    # Load vocab_size from checkpoint config to ensure consistent input
    config_path = os.path.join(config.torchtitan_checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        hf_config = json.load(f)
    vocab_size = hf_config["vocab_size"]

    engine = load_vllm_native_engine(config)

    # Get the worker and model
    worker = get_worker_from_engine(engine)
    model = get_model_from_worker(worker)
    print(f"Model type: {type(model)}")

    # Create fake input (use vocab_size from config for consistency)
    input_ids = create_fake_input(config.seq_len, vocab_size=vocab_size)
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens (first 10): {input_ids[:10].tolist()}")

    # Run forward pass with dummy forward context
    with torch.no_grad():
        # vLLM expects 1D positions tensor (seq_len,)
        positions = torch.arange(config.seq_len, device="cuda")
        with create_dummy_forward_context(worker, config.seq_len):
            output = model(input_ids, positions=positions)

    # Handle different output types
    if hasattr(output, 'logits'):
        logits = output.logits
    elif isinstance(output, torch.Tensor):
        logits = output
    else:
        logits = output[0] if isinstance(output, tuple) else output

    print(f"Output logits shape: {logits.shape}")

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    # Add batch dimension for comparison (vLLM outputs 2D, compare expects 3D)
    return input_ids.unsqueeze(0).cpu(), logits.unsqueeze(0).cpu()


def load_vllm_torchtitan_engine(config: CorrectnessConfig):
    """Load vLLM engine with TorchTitan model."""
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.engine.arg_utils import EngineArgs

    print("=" * 60)
    print("Loading vLLM with TorchTitan model...")
    print(f"Checkpoint: {config.torchtitan_checkpoint_path}")
    print(f"Tensor Parallel Size: {config.tp}")
    print("=" * 60)

    engine_args = EngineArgs(
        model=config.torchtitan_checkpoint_path,
        hf_overrides={
            "architectures": ["Qwen3TorchTitanForCausalLM"],
        },
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        tensor_parallel_size=config.tp,
    )

    engine = LLMEngine.from_engine_args(engine_args)
    print(f"Engine type: {type(engine)}")

    return engine


def run_vllm_torchtitan_forward(config: CorrectnessConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass on vLLM with TorchTitan model."""
    # Load vocab_size from checkpoint config to match native TorchTitan input
    config_path = os.path.join(config.torchtitan_checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        hf_config = json.load(f)
    vocab_size = hf_config["vocab_size"]

    engine = load_vllm_torchtitan_engine(config)

    # Get the worker and model
    worker = get_worker_from_engine(engine)
    model = get_model_from_worker(worker)
    print(f"Model type: {type(model)}")

    # Optionally swap TorchTitan attention with vLLM native attention
    if config.swap_attention:
        swap_torchtitan_attention_with_vllm_native(model, engine.vllm_config)

    # Create fake input (same as native - use same vocab_size)
    input_ids = create_fake_input(config.seq_len, vocab_size=vocab_size)
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens (first 10): {input_ids[:10].tolist()}")

    # Run forward pass with dummy forward context
    with torch.no_grad():
        # vLLM expects 1D positions tensor (seq_len,)
        positions = torch.arange(config.seq_len, device="cuda")
        with create_dummy_forward_context(worker, config.seq_len):
            output = model(input_ids, positions=positions)

    # Handle different output types
    if hasattr(output, 'logits'):
        logits = output.logits
    elif isinstance(output, torch.Tensor):
        logits = output
    else:
        logits = output[0] if isinstance(output, tuple) else output

    print(f"Output logits shape: {logits.shape}")

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    # Add batch dimension for comparison (vLLM outputs 2D, compare expects 3D)
    return input_ids.unsqueeze(0).cpu(), logits.unsqueeze(0).cpu()


def run_torchtitan_native_forward(config: CorrectnessConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run forward pass on native TorchTitan model (not through vLLM).

    This directly instantiates the TorchTitan Qwen3Model, loads weights from HF checkpoint,
    and runs forward pass with debug output enabled.
    """
    from safetensors import safe_open
    from torchtitan.models.qwen3.model.model import Qwen3Model
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
    from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter

    print("=" * 60)
    print("Running Native TorchTitan Qwen3 Model Forward Pass...")
    print(f"Checkpoint: {config.torchtitan_checkpoint_path}")
    print("=" * 60)

    # Load HuggingFace config
    config_path = os.path.join(config.torchtitan_checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        hf_config = json.load(f)

    print(f"Model type: {hf_config.get('model_type', 'unknown')}")
    print(f"Hidden size: {hf_config['hidden_size']}")
    print(f"Num layers: {hf_config['num_hidden_layers']}")
    print(f"Num heads: {hf_config['num_attention_heads']}")

    # Create model args
    model_args = Qwen3ModelArgs(
        dim=hf_config["hidden_size"],
        n_layers=hf_config["num_hidden_layers"],
        n_heads=hf_config["num_attention_heads"],
        n_kv_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        vocab_size=hf_config["vocab_size"],
        hidden_dim=hf_config["intermediate_size"],
        max_seq_len=2048,
        rope_theta=hf_config.get("rope_theta", 1000000.0),
        norm_eps=hf_config.get("rms_norm_eps", 1e-6),
        head_dim=hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"]),
        qk_norm=hf_config.get("qk_norm", True),
        attn_type="sdpa",  # Use SDPA for native inference
    )
    print(f"Model args: dim={model_args.dim}, n_layers={model_args.n_layers}, n_heads={model_args.n_heads}")

    # Create model
    print("Creating Qwen3Model...")
    model = Qwen3Model(model_args)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load weights from HF checkpoint
    print("Loading HuggingFace weights...")
    hf_weights = {}
    safetensor_files = [f for f in os.listdir(config.torchtitan_checkpoint_path) if f.endswith(".safetensors")]
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {config.torchtitan_checkpoint_path}")

    for filename in sorted(safetensor_files):
        filepath = os.path.join(config.torchtitan_checkpoint_path, filename)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key)
    print(f"Loaded {len(hf_weights)} weight tensors")

    # Convert using state dict adapter
    print("Converting weights using state dict adapter...")
    adapter = Qwen3StateDictAdapter(
        model_args=model_args,
        hf_assets_path=None,
    )
    torchtitan_weights = adapter.from_hf(hf_weights)

    # Move weights to device and dtype
    for name in torchtitan_weights:
        torchtitan_weights[name] = torchtitan_weights[name].to(device="cuda", dtype=torch.bfloat16)

    # Load state dict
    missing, unexpected = model.load_state_dict(torchtitan_weights, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    print("Weights loaded successfully!")

    # Enable debug output
    print("Enabling debug output...")
    model.enable_debug(True)

    # Create fake input (same as other tests)
    # Note: create_fake_input returns 1D tensor, we need 2D for native model
    input_ids_1d = create_fake_input(config.seq_len, vocab_size=hf_config["vocab_size"])
    input_ids = input_ids_1d.unsqueeze(0)  # [1, seq_len]
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens (first 10): {input_ids[0, :10].tolist()}")

    # Create positions tensor
    positions = torch.arange(config.seq_len, device="cuda").unsqueeze(0)  # [1, seq_len]
    print(f"Positions shape: {positions.shape}")

    # Run forward pass
    print("\n" + "=" * 60)
    print("Running forward pass with debug output...")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_masks=None, positions=positions)

    print("\n" + "=" * 60)
    print("Native TorchTitan Forward Pass Complete")
    print("=" * 60)
    print(f"Output logits shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    print(f"Output min: {logits.min().item():.6f}")
    print(f"Output max: {logits.max().item():.6f}")
    print(f"Output mean: {logits.float().mean().item():.6f}")
    print(f"Output std: {logits.float().std().item():.6f}")
    print(f"Has NaN: {logits.isnan().any().item()}")
    print(f"Has Inf: {logits.isinf().any().item()}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return input_ids.cpu(), logits.cpu()


def compare_native_vs_vllm_torchtitan_weights(config: CorrectnessConfig) -> bool:
    """
    Compare weights between native TorchTitan model and vLLM+TorchTitan model.

    Both models use the same TorchTitan Qwen3Model architecture with identical weight names.
    This function verifies that weights are loaded identically in both approaches.

    Returns:
        True if all weights match, False otherwise.
    """
    from safetensors import safe_open
    from torch.distributed._tensor import DTensor
    from torchtitan.models.qwen3.model.model import Qwen3Model
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
    from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.engine.arg_utils import EngineArgs

    print("\n" + "=" * 60)
    print("WEIGHT COMPARISON: Native TorchTitan vs vLLM+TorchTitan")
    print("=" * 60)

    def get_local_tensor(param):
        """Extract local tensor from DTensor if needed."""
        if isinstance(param, DTensor):
            return param.to_local()
        return param

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol=1e-5, atol=1e-5):
        """Compare two tensors and print results."""
        t1 = get_local_tensor(t1).float().cpu()
        t2 = get_local_tensor(t2).float().cpu()

        if t1.shape != t2.shape:
            print(f"  {name}: SHAPE MISMATCH - native {t1.shape} vs vllm {t2.shape}")
            return False

        close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
        max_diff = (t1 - t2).abs().max().item()
        mean_diff = (t1 - t2).abs().mean().item()

        status = "MATCH" if close else "DIFFER"
        print(f"  {name}: {status} (shape={list(t1.shape)}, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return close

    # Step 1: Load native TorchTitan model
    print("\n[1/3] Loading native TorchTitan model...")
    config_path = os.path.join(config.torchtitan_checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        hf_config = json.load(f)

    model_args = Qwen3ModelArgs(
        dim=hf_config["hidden_size"],
        n_layers=hf_config["num_hidden_layers"],
        n_heads=hf_config["num_attention_heads"],
        n_kv_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        vocab_size=hf_config["vocab_size"],
        hidden_dim=hf_config["intermediate_size"],
        max_seq_len=2048,
        rope_theta=hf_config.get("rope_theta", 1000000.0),
        norm_eps=hf_config.get("rms_norm_eps", 1e-6),
        head_dim=hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"]),
        qk_norm=hf_config.get("qk_norm", True),
        attn_type="sdpa",
    )

    native_model = Qwen3Model(model_args)
    native_model = native_model.to(device="cuda", dtype=torch.bfloat16)

    # Load weights
    hf_weights = {}
    safetensor_files = [f for f in os.listdir(config.torchtitan_checkpoint_path) if f.endswith(".safetensors")]
    for filename in sorted(safetensor_files):
        filepath = os.path.join(config.torchtitan_checkpoint_path, filename)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key)

    adapter = Qwen3StateDictAdapter(model_args=model_args, hf_assets_path=None)
    torchtitan_weights = adapter.from_hf(hf_weights)
    for name in torchtitan_weights:
        torchtitan_weights[name] = torchtitan_weights[name].to(device="cuda", dtype=torch.bfloat16)
    native_model.load_state_dict(torchtitan_weights, strict=False)
    print(f"Native TorchTitan model loaded with {sum(p.numel() for p in native_model.parameters()):,} parameters")

    # Step 2: Load vLLM+TorchTitan model
    print("\n[2/3] Loading vLLM+TorchTitan model...")
    tt_engine_args = EngineArgs(
        model=config.torchtitan_checkpoint_path,
        hf_overrides={
            "architectures": ["Qwen3TorchTitanForCausalLM"],
        },
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        tensor_parallel_size=config.tp,
    )
    tt_engine = LLMEngine.from_engine_args(tt_engine_args)
    tt_worker = get_worker_from_engine(tt_engine)
    tt_wrapper = get_model_from_worker(tt_worker)
    vllm_model = tt_wrapper.model if hasattr(tt_wrapper, 'model') else tt_wrapper
    print(f"vLLM+TorchTitan model type: {type(tt_wrapper)}")

    # Step 3: Compare weights
    print("\n[3/3] Comparing weights...")
    all_match = True

    # Compare embeddings
    print("\n[Embeddings]")
    match = compare_tensors("tok_embeddings", native_model.tok_embeddings.weight, vllm_model.tok_embeddings.weight)
    all_match = all_match and match

    # Compare layers
    native_layers = native_model.layers
    vllm_layers = vllm_model.layers
    num_layers = min(len(native_layers), len(vllm_layers))
    print(f"\nComparing {num_layers} layers...")

    for i in range(num_layers):
        print(f"\n[Layer {i}]")
        native_layer = native_layers[str(i)]
        vllm_layer = vllm_layers[str(i)]

        # Attention weights
        native_attn = native_layer.attention
        vllm_attn = vllm_layer.attention

        match = compare_tensors("attention.wq", native_attn.wq.weight, vllm_attn.wq.weight)
        all_match = all_match and match

        match = compare_tensors("attention.wk", native_attn.wk.weight, vllm_attn.wk.weight)
        all_match = all_match and match

        match = compare_tensors("attention.wv", native_attn.wv.weight, vllm_attn.wv.weight)
        all_match = all_match and match

        match = compare_tensors("attention.wo", native_attn.wo.weight, vllm_attn.wo.weight)
        all_match = all_match and match

        # Q/K norms
        if hasattr(native_attn, 'q_norm') and hasattr(vllm_attn, 'q_norm'):
            match = compare_tensors("attention.q_norm", native_attn.q_norm.weight, vllm_attn.q_norm.weight)
            all_match = all_match and match

            match = compare_tensors("attention.k_norm", native_attn.k_norm.weight, vllm_attn.k_norm.weight)
            all_match = all_match and match

        # FeedForward weights
        native_ff = native_layer.feed_forward
        vllm_ff = vllm_layer.feed_forward

        match = compare_tensors("feed_forward.w1", native_ff.w1.weight, vllm_ff.w1.weight)
        all_match = all_match and match

        match = compare_tensors("feed_forward.w2", native_ff.w2.weight, vllm_ff.w2.weight)
        all_match = all_match and match

        match = compare_tensors("feed_forward.w3", native_ff.w3.weight, vllm_ff.w3.weight)
        all_match = all_match and match

        # Layer norms
        match = compare_tensors("attention_norm", native_layer.attention_norm.weight, vllm_layer.attention_norm.weight)
        all_match = all_match and match

        match = compare_tensors("ffn_norm", native_layer.ffn_norm.weight, vllm_layer.ffn_norm.weight)
        all_match = all_match and match

    # Compare final norm
    print("\n[Final Norm]")
    match = compare_tensors("norm", native_model.norm.weight, vllm_model.norm.weight)
    all_match = all_match and match

    # Compare output
    print("\n[Output/LM Head]")
    match = compare_tensors("output", native_model.output.weight, vllm_model.output.weight)
    all_match = all_match and match

    # Cleanup
    del native_model
    del tt_engine
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if all_match:
        print("WEIGHT COMPARISON: ALL WEIGHTS MATCH")
    else:
        print("WEIGHT COMPARISON: SOME WEIGHTS DIFFER")
    print("=" * 60)

    return all_match


def run_weight_comparison(config: CorrectnessConfig) -> bool:
    """
    Run weight comparison between vLLM native and TorchTitan models.

    This loads both models sequentially (to manage GPU memory) and compares weights.

    Returns:
        True if all weights match, False otherwise.
    """
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.engine.arg_utils import EngineArgs

    print("\n" + "=" * 60)
    print("WEIGHT COMPARISON MODE")
    print("=" * 60)

    # Load native model first
    print("\n[1/3] Loading vLLM Native model...")
    native_engine_args = EngineArgs(
        model=config.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=config.tp,
    )
    native_engine = LLMEngine.from_engine_args(native_engine_args)
    native_worker = get_worker_from_engine(native_engine)
    native_model = get_model_from_worker(native_worker)
    print(f"Native model type: {type(native_model)}")

    # Extract weights from native model before cleanup
    # Store as CPU tensors to free GPU memory
    native_weights = {}

    def get_local_tensor(param):
        from torch.distributed._tensor import DTensor
        if isinstance(param, DTensor):
            return param.to_local()
        return param

    # Get native model inner structure
    native_inner = native_model.model if hasattr(native_model, 'model') else native_model

    # Store embeddings
    if hasattr(native_inner, 'embed_tokens'):
        native_weights['embed_tokens'] = get_local_tensor(native_inner.embed_tokens.weight).cpu().clone()

    # Store layer weights
    native_layers = native_inner.layers if hasattr(native_inner, 'layers') else []
    for i, layer in enumerate(native_layers):
        prefix = f'layer_{i}'
        native_attn = layer.self_attn
        native_mlp = layer.mlp

        native_weights[f'{prefix}.qkv_proj'] = get_local_tensor(native_attn.qkv_proj.weight).cpu().clone()
        native_weights[f'{prefix}.o_proj'] = get_local_tensor(native_attn.o_proj.weight).cpu().clone()
        if hasattr(native_attn, 'q_norm'):
            native_weights[f'{prefix}.q_norm'] = get_local_tensor(native_attn.q_norm.weight).cpu().clone()
            native_weights[f'{prefix}.k_norm'] = get_local_tensor(native_attn.k_norm.weight).cpu().clone()

        native_weights[f'{prefix}.gate_up_proj'] = get_local_tensor(native_mlp.gate_up_proj.weight).cpu().clone()
        native_weights[f'{prefix}.down_proj'] = get_local_tensor(native_mlp.down_proj.weight).cpu().clone()

        native_weights[f'{prefix}.input_layernorm'] = get_local_tensor(layer.input_layernorm.weight).cpu().clone()
        native_weights[f'{prefix}.post_attention_layernorm'] = get_local_tensor(layer.post_attention_layernorm.weight).cpu().clone()

    # Store final norm and lm_head
    if hasattr(native_inner, 'norm'):
        native_weights['final_norm'] = get_local_tensor(native_inner.norm.weight).cpu().clone()
    if hasattr(native_model, 'lm_head'):
        native_weights['lm_head'] = get_local_tensor(native_model.lm_head.weight).cpu().clone()

    num_native_layers = len(native_layers)
    print(f"Extracted weights from {num_native_layers} native layers")

    # Cleanup native model
    del native_engine
    torch.cuda.empty_cache()

    # Load TorchTitan model
    print("\n[2/3] Loading vLLM TorchTitan model...")
    tt_engine_args = EngineArgs(
        model=config.torchtitan_checkpoint_path,
        hf_overrides={
            "architectures": ["Qwen3TorchTitanForCausalLM"],
        },
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=config.tp,
    )
    tt_engine = LLMEngine.from_engine_args(tt_engine_args)
    tt_worker = get_worker_from_engine(tt_engine)
    tt_model = get_model_from_worker(tt_worker)
    print(f"TorchTitan model type: {type(tt_model)}")

    # Compare weights
    print("\n[3/3] Comparing weights...")
    tt_inner = tt_model.model if hasattr(tt_model, 'model') else tt_model

    all_match = True

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol=1e-5, atol=1e-5):
        t1 = t1.float()
        t2 = get_local_tensor(t2).float().cpu()

        if t1.shape != t2.shape:
            print(f"  {name}: SHAPE MISMATCH - native {t1.shape} vs torchtitan {t2.shape}")
            return False

        close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
        max_diff = (t1 - t2).abs().max().item()
        mean_diff = (t1 - t2).abs().mean().item()

        status = "MATCH" if close else "DIFFER"
        print(f"  {name}: {status} (shape={list(t1.shape)}, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return close

    # Compare embeddings
    print("\n[Embeddings]")
    if 'embed_tokens' in native_weights and hasattr(tt_inner, 'tok_embeddings'):
        match = compare_tensors("tok_embeddings", native_weights['embed_tokens'], tt_inner.tok_embeddings.weight)
        all_match = all_match and match

    # Compare layers
    tt_layers = tt_inner.layers if hasattr(tt_inner, 'layers') else {}
    num_layers = min(num_native_layers, len(tt_layers))
    print(f"\nComparing {num_layers} layers...")

    for i in range(num_layers):
        print(f"\n[Layer {i}]")
        prefix = f'layer_{i}'
        tt_layer = tt_layers[str(i)]
        tt_attn = tt_layer.attention
        tt_ff = tt_layer.feed_forward

        # Attention: qkv_proj = concat(wq, wk, wv)
        wq = get_local_tensor(tt_attn.wq.weight)
        wk = get_local_tensor(tt_attn.wk.weight)
        wv = get_local_tensor(tt_attn.wv.weight)
        tt_qkv = torch.cat([wq, wk, wv], dim=0).cpu()
        match = compare_tensors("attention.qkv_proj (vs concat(wq,wk,wv))", native_weights[f'{prefix}.qkv_proj'], tt_qkv)
        all_match = all_match and match

        wo = get_local_tensor(tt_attn.wo.weight)
        match = compare_tensors("attention.o_proj (vs wo)", native_weights[f'{prefix}.o_proj'], wo)
        all_match = all_match and match

        if f'{prefix}.q_norm' in native_weights:
            match = compare_tensors("attention.q_norm", native_weights[f'{prefix}.q_norm'], tt_attn.q_norm.weight)
            all_match = all_match and match
            match = compare_tensors("attention.k_norm", native_weights[f'{prefix}.k_norm'], tt_attn.k_norm.weight)
            all_match = all_match and match

        # FeedForward: gate_up_proj = concat(w1, w3)
        w1 = get_local_tensor(tt_ff.w1.weight)
        w3 = get_local_tensor(tt_ff.w3.weight)
        tt_gate_up = torch.cat([w1, w3], dim=0).cpu()
        match = compare_tensors("mlp.gate_up_proj (vs concat(w1,w3))", native_weights[f'{prefix}.gate_up_proj'], tt_gate_up)
        all_match = all_match and match

        w2 = get_local_tensor(tt_ff.w2.weight)
        match = compare_tensors("mlp.down_proj (vs w2)", native_weights[f'{prefix}.down_proj'], w2)
        all_match = all_match and match

        # Layer norms
        match = compare_tensors("input_layernorm (vs attention_norm)", native_weights[f'{prefix}.input_layernorm'], tt_layer.attention_norm.weight)
        all_match = all_match and match
        match = compare_tensors("post_attention_layernorm (vs ffn_norm)", native_weights[f'{prefix}.post_attention_layernorm'], tt_layer.ffn_norm.weight)
        all_match = all_match and match

    # Compare final norm
    print("\n[Final Norm]")
    if 'final_norm' in native_weights and hasattr(tt_inner, 'norm'):
        match = compare_tensors("final_norm", native_weights['final_norm'], tt_inner.norm.weight)
        all_match = all_match and match

    # Compare output/lm_head
    print("\n[Output/LM Head]")
    if 'lm_head' in native_weights and hasattr(tt_inner, 'output'):
        match = compare_tensors("lm_head (vs output)", native_weights['lm_head'], tt_inner.output.weight)
        all_match = all_match and match

    # Cleanup
    del tt_engine
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if all_match:
        print("WEIGHT COMPARISON: ALL WEIGHTS MATCH")
    else:
        print("WEIGHT COMPARISON: SOME WEIGHTS DIFFER")
    print("=" * 60)

    return all_match


def compare_outputs(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    name1: str,
    name2: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Tuple[bool, str]:
    """Compare model outputs and generate detailed report."""
    report_lines = []
    report_lines.append("\n" + "=" * 60)
    report_lines.append("COMPARISON RESULTS")
    report_lines.append("=" * 60)

    # Check shapes
    report_lines.append(f"\n{name1} output shape: {logits1.shape}")
    report_lines.append(f"{name2} output shape: {logits2.shape}")

    if logits1.shape != logits2.shape:
        report_lines.append(f"\n[FAIL] Output shapes differ!")
        return False, "\n".join(report_lines)

    report_lines.append("[PASS] Output shapes match")

    # Compare values
    logits1_float = logits1.float()
    logits2_float = logits2.float()

    # Element-wise comparison
    close_mask = torch.isclose(logits1_float, logits2_float, rtol=rtol, atol=atol)
    num_close = close_mask.sum().item()
    num_total = close_mask.numel()
    pct_close = 100.0 * num_close / num_total

    report_lines.append(f"\nElement-wise comparison (rtol={rtol}, atol={atol}):")
    report_lines.append(f"  Matching elements: {num_close:,} / {num_total:,} ({pct_close:.2f}%)")

    # Compute differences
    diff = (logits1_float - logits2_float).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()

    report_lines.append(f"\nDifference statistics:")
    report_lines.append(f"  Max diff:    {max_diff:.6e}")
    report_lines.append(f"  Mean diff:   {mean_diff:.6e}")
    report_lines.append(f"  Median diff: {median_diff:.6e}")

    # Find positions with largest differences
    flat_diff = diff.flatten()
    top_k = min(10, flat_diff.numel())
    top_diffs, top_indices = torch.topk(flat_diff, top_k)

    report_lines.append(f"\nTop {top_k} largest differences:")
    for i, (d, idx) in enumerate(zip(top_diffs, top_indices)):
        # Convert flat index to multi-dimensional index
        idx = idx.item()
        shape = logits1.shape
        coords = []
        for dim_size in reversed(shape):
            coords.append(idx % dim_size)
            idx //= dim_size
        coords = tuple(reversed(coords))

        val1 = logits1_float.flatten()[top_indices[i]].item()
        val2 = logits2_float.flatten()[top_indices[i]].item()
        report_lines.append(f"  {i+1}. Position {coords}: {name1}={val1:.6f}, {name2}={val2:.6f}, diff={d:.6e}")

    # Compare log probabilities (softmax)
    report_lines.append(f"\nLog probability comparison (last position):")
    logprobs1 = torch.log_softmax(logits1_float[0, -1, :], dim=-1)
    logprobs2 = torch.log_softmax(logits2_float[0, -1, :], dim=-1)

    logprob_diff = (logprobs1 - logprobs2).abs()
    report_lines.append(f"  Max logprob diff:  {logprob_diff.max().item():.6e}")
    report_lines.append(f"  Mean logprob diff: {logprob_diff.mean().item():.6e}")

    # Check if top predicted tokens match
    top_k_tokens = 5
    top1_tokens1 = torch.topk(logits1_float[0, -1, :], top_k_tokens).indices.tolist()
    top1_tokens2 = torch.topk(logits2_float[0, -1, :], top_k_tokens).indices.tolist()

    report_lines.append(f"\nTop {top_k_tokens} predicted tokens (last position):")
    report_lines.append(f"  {name1}: {top1_tokens1}")
    report_lines.append(f"  {name2}: {top1_tokens2}")

    if top1_tokens1 == top1_tokens2:
        report_lines.append(f"  [PASS] Top {top_k_tokens} tokens match!")
    else:
        report_lines.append(f"  [WARN] Top {top_k_tokens} tokens differ")

    # Overall verdict
    all_match = pct_close >= 99.0 and max_diff < 0.1

    report_lines.append("\n" + "=" * 60)
    if all_match:
        report_lines.append(f"RESULT: PASS - Outputs are sufficiently close ({pct_close:.2f}% match)")
    else:
        report_lines.append(f"RESULT: FAIL - Outputs differ significantly ({pct_close:.2f}% match)")
    report_lines.append("=" * 60)

    return all_match, "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Check inference correctness between vLLM Native and TorchTitan")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path to HuggingFace model (for vLLM native)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TorchTitan checkpoint",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (1 or 2)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Length of fake input sequence",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for comparison",
    )
    parser.add_argument(
        "--swap-attention",
        action="store_true",
        help="Swap TorchTitan attention with vLLM native Qwen3Attention for comparison",
    )
    parser.add_argument(
        "--compare-weights",
        action="store_true",
        help="Compare model weights between vLLM native and TorchTitan models",
    )
    parser.add_argument(
        "--test-native",
        action="store_true",
        help="Test native TorchTitan model forward pass (not through vLLM) with debug output",
    )

    args = parser.parse_args()

    config = CorrectnessConfig(
        model_path=args.model_path,
        torchtitan_checkpoint_path=args.checkpoint,
        tp=args.tp,
        seq_len=args.seq_len,
        rtol=args.rtol,
        atol=args.atol,
        swap_attention=args.swap_attention,
        compare_weights=args.compare_weights,
        test_native=args.test_native,
    )

    print("\n" + "=" * 60)
    print("INFERENCE CORRECTNESS CHECK")
    print("Direct Forward Pass Comparison")
    print("=" * 60)
    print(f"Native model: {config.model_path}")
    print(f"TorchTitan checkpoint: {config.torchtitan_checkpoint_path}")
    print(f"TP: {config.tp}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Tolerance: rtol={config.rtol}, atol={config.atol}")
    print(f"Swap attention: {config.swap_attention}")
    print(f"Compare weights: {config.compare_weights}")
    print(f"Test native: {config.test_native}")
    print("=" * 60)

    # If test-native is requested, run both native TorchTitan and vLLM+TorchTitan tests
    if config.test_native:
        print("\n[TEST NATIVE MODE] Comparing native TorchTitan vs vLLM+TorchTitan models...")

        # Step 1: Run native TorchTitan forward pass
        print("\n" + "=" * 60)
        print("[1/3] Running Native TorchTitan forward pass...")
        print("=" * 60)
        native_input_ids, native_logits = run_torchtitan_native_forward(config)
        print(f"Native TorchTitan output shape: {native_logits.shape}")

        # Step 2: Run vLLM+TorchTitan forward pass
        print("\n" + "=" * 60)
        print("[2/3] Running vLLM+TorchTitan forward pass...")
        print("=" * 60)
        vllm_input_ids, vllm_logits = run_vllm_torchtitan_forward(config)
        print(f"vLLM+TorchTitan output shape: {vllm_logits.shape}")

        # Step 3: Optionally compare weights between native TorchTitan and vLLM+TorchTitan
        weights_match = True
        if config.compare_weights:
            print("\n" + "=" * 60)
            print("[3/3] Comparing weights between Native TorchTitan and vLLM+TorchTitan...")
            print("=" * 60)
            # We need to load both models again for weight comparison
            # Since we already ran forward passes, we need a new comparison function
            weights_match = compare_native_vs_vllm_torchtitan_weights(config)
        else:
            print("\n[3/3] Skipping weight comparison (use --compare-weights to enable)")

        # Compare outputs
        # Native TorchTitan outputs [batch, seq_len, vocab_size] (3D)
        # vLLM+TorchTitan outputs [batch, seq_len, vocab_size] (3D after unsqueeze in run_vllm_torchtitan_forward)
        print("\n" + "=" * 60)
        print("Comparing outputs between Native TorchTitan and vLLM+TorchTitan...")
        print("=" * 60)

        outputs_match, report = compare_outputs(
            native_logits, vllm_logits,
            "NativeTorchTitan", "VLLMTorchTitan",
            rtol=config.rtol, atol=config.atol,
        )
        print(report)

        # Overall verdict
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Weights match: {weights_match}")
        print(f"Outputs match: {outputs_match}")
        all_match = weights_match and outputs_match
        if all_match:
            print("RESULT: PASS - Native TorchTitan and vLLM+TorchTitan are consistent")
        else:
            print("RESULT: FAIL - Differences detected between Native and vLLM+TorchTitan")
        print("=" * 60)

        return 0 if all_match else 1

    # Run weight comparison if requested
    weights_match = True
    if config.compare_weights:
        weights_match = run_weight_comparison(config)

    # Run vLLM Native forward pass
    print("\n[1/2] Running vLLM Native forward pass...")
    input_ids1, logits1 = run_vllm_native_forward(config)
    print("vLLM Native completed")

    # Run vLLM TorchTitan forward pass
    print("\n[2/2] Running vLLM TorchTitan forward pass...")
    input_ids2, logits2 = run_vllm_torchtitan_forward(config)
    print("vLLM TorchTitan completed")

    # Verify inputs are the same
    assert torch.equal(input_ids1, input_ids2), "Input IDs should be identical!"

    # Compare outputs
    outputs_match, report = compare_outputs(
        logits1, logits2,
        "VLLMNative", "VLLMTorchTitan",
        rtol=config.rtol, atol=config.atol,
    )

    print(report)

    # Return success only if both weights and outputs match (if weight comparison was done)
    all_match = outputs_match and weights_match
    return 0 if all_match else 1


if __name__ == "__main__":
    exit(main())
