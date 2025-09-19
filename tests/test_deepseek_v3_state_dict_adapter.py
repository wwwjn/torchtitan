import os
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed.checkpoint as dist_cp


def test_state_dict_adapter():
    """Test the DeepSeek V3 state dict adapter with real HF checkpoint."""
    print("Starting DeepSeek V3 state dict adapter test...")

    from torch.distributed.checkpoint.quantized_hf_storage import (
        QuantizedHuggingFaceStorageReader,
    )

    # Import the necessary components
    from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
    from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model
    from torchtitan.models.deepseek_v3.model.state_dict_adapter import (
        DeepSeekV3StateDictAdapter,
    )
    from torchtitan.models.moe import MoEArgs

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Path to the actual DeepSeek-V3.1-Base checkpoint
    CHECKPOINT_DIR = "/home/jianiw/tmp/mffuse/deepseek-v3/DeepSeek-V3.1-Base"

    # Verify checkpoint directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"✗ Checkpoint directory not found: {CHECKPOINT_DIR}")
        return False

    print(f"Using checkpoint directory: {CHECKPOINT_DIR}")

    # Create model args based on the actual DeepSeek-V3.1-Base configuration
    # Using smaller values for testing to avoid memory issues
    moe_args = MoEArgs(
        num_experts=256,  # Real model has 256 experts
        top_k=8,  # Real model uses top_k=8
        router_aux_loss_coef=0.001,
        use_grouped_mm=False,
    )

    model_args = DeepSeekV3ModelArgs(
        vocab_size=129280,  # From config.json
        dim=7168,  # hidden_size from config.json
        n_layers=4,  # Reduced from 61 for testing
        n_heads=128,  # num_attention_heads from config.json
        n_dense_layers=3,  # first_k_dense_replace from config.json
        inter_dim=18432,  # intermediate_size from config.json
        moe_inter_dim=2048,  # moe_intermediate_size from config.json
        q_lora_rank=1536,  # From config.json
        kv_lora_rank=512,  # From config.json
        qk_nope_head_dim=128,  # From config.json
        qk_rope_head_dim=64,  # From config.json
        v_head_dim=128,  # From config.json
        max_seq_len=163840,  # max_position_embeddings from config.json
        norm_eps=1e-6,  # rms_norm_eps from config.json
        rope_theta=10000,  # From config.json
        rope_factor=40,  # From config.json
        beta_fast=32,  # From config.json
        beta_slow=1,  # From config.json
        mscale=1.0,  # From config.json
        moe_args=moe_args,
        hf_weight_quantized=True,  # Checkpoint is quantized
    )

    print(
        f"Created model args: vocab_size={model_args.vocab_size}, dim={model_args.dim}, "
        f"n_layers={model_args.n_layers}, n_heads={model_args.n_heads}"
    )

    # Create TT model
    print("Creating TorchTitan model...")
    tt_model = DeepSeekV3Model(model_args)
    tt_model.eval()
    print("Created TT model successfully")

    # Initialize state dict adapter
    adapter = DeepSeekV3StateDictAdapter(model_args, CHECKPOINT_DIR)

    # Load HF state dict using QuantizedHuggingFaceStorageReader
    print("Loading HF state dict from real checkpoint...")

    # Create a minimal state dict template for loading only a subset of weights
    # to avoid memory issues
    hf_state_dict = {}

    # Create template state dict with the expected keys and shapes
    # We'll load only embedding, output, and a few layers for testing
    keys_to_load = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]

    # Add weights for first few layers only
    layers_to_load = min(4, model_args.n_layers)
    for layer_id in range(layers_to_load):
        layer_prefix = f"model.layers.{layer_id}"

        # Layer norms
        keys_to_load.extend(
            [
                f"{layer_prefix}.input_layernorm.weight",
                f"{layer_prefix}.post_attention_layernorm.weight",
            ]
        )

        # Attention weights
        keys_to_load.extend(
            [
                f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{layer_prefix}.self_attn.kv_a_layernorm.weight",
                f"{layer_prefix}.self_attn.kv_b_proj.weight",
                f"{layer_prefix}.self_attn.o_proj.weight",
            ]
        )

        # Query projection
        if model_args.q_lora_rank > 0:
            keys_to_load.extend(
                [
                    f"{layer_prefix}.self_attn.q_a_proj.weight",
                    f"{layer_prefix}.self_attn.q_a_layernorm.weight",
                    f"{layer_prefix}.self_attn.q_b_proj.weight",
                ]
            )
        else:
            keys_to_load.append(f"{layer_prefix}.self_attn.q_proj.weight")

        # MLP/MoE weights
        if layer_id >= model_args.n_dense_layers:
            # MoE layer - load only a few experts for testing
            keys_to_load.append(f"{layer_prefix}.mlp.gate.weight")

            # Load only first 2 experts to reduce memory usage
            experts_to_load = min(2, model_args.moe_args.num_experts)
            for expert_id in range(experts_to_load):
                keys_to_load.extend(
                    [
                        f"{layer_prefix}.mlp.experts.{expert_id}.gate_proj.weight",
                        f"{layer_prefix}.mlp.experts.{expert_id}.up_proj.weight",
                        f"{layer_prefix}.mlp.experts.{expert_id}.down_proj.weight",
                    ]
                )

            # Shared experts
            keys_to_load.extend(
                [
                    f"{layer_prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{layer_prefix}.mlp.shared_experts.up_proj.weight",
                    f"{layer_prefix}.mlp.shared_experts.down_proj.weight",
                ]
            )
        else:
            # Dense layer
            keys_to_load.extend(
                [
                    f"{layer_prefix}.mlp.gate_proj.weight",
                    f"{layer_prefix}.mlp.up_proj.weight",
                    f"{layer_prefix}.mlp.down_proj.weight",
                ]
            )

    # Initialize state dict with zero tensors to avoid loading the full model
    print(f"Preparing to load {len(keys_to_load)} keys from checkpoint...")
    for key in keys_to_load:
        # Create placeholder tensors - these will be overwritten by the loader
        if "embed_tokens.weight" in key or "lm_head.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.vocab_size, model_args.dim)
        elif "norm.weight" in key and "layernorm" not in key:
            hf_state_dict[key] = torch.ones(model_args.dim)
        elif "layernorm.weight" in key:
            hf_state_dict[key] = torch.ones(model_args.dim)
        elif "kv_a_proj_with_mqa.weight" in key:
            hf_state_dict[key] = torch.zeros(
                model_args.kv_lora_rank + model_args.qk_rope_head_dim, model_args.dim
            )
        elif "kv_a_layernorm.weight" in key:
            hf_state_dict[key] = torch.ones(model_args.kv_lora_rank)
        elif "kv_b_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(
                model_args.n_heads
                * (model_args.qk_nope_head_dim + model_args.v_head_dim),
                model_args.kv_lora_rank,
            )
        elif "o_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(
                model_args.dim, model_args.n_heads * model_args.v_head_dim
            )
        elif "q_a_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.q_lora_rank, model_args.dim)
        elif "q_a_layernorm.weight" in key:
            hf_state_dict[key] = torch.ones(model_args.q_lora_rank)
        elif "q_b_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(
                model_args.n_heads
                * (model_args.qk_nope_head_dim + model_args.qk_rope_head_dim),
                model_args.q_lora_rank,
            )
        elif "q_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(
                model_args.n_heads
                * (model_args.qk_nope_head_dim + model_args.qk_rope_head_dim),
                model_args.dim,
            )
        elif "mlp.gate.weight" in key:
            hf_state_dict[key] = torch.zeros(
                model_args.moe_args.num_experts, model_args.dim
            )
        elif "experts." in key and "gate_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.moe_inter_dim, model_args.dim)
        elif "experts." in key and "up_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.moe_inter_dim, model_args.dim)
        elif "experts." in key and "down_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.dim, model_args.moe_inter_dim)
        elif "shared_experts.gate_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.moe_inter_dim, model_args.dim)
        elif "shared_experts.up_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.moe_inter_dim, model_args.dim)
        elif "shared_experts.down_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.dim, model_args.moe_inter_dim)
        elif "gate_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.inter_dim, model_args.dim)
        elif "up_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.inter_dim, model_args.dim)
        elif "down_proj.weight" in key:
            hf_state_dict[key] = torch.zeros(model_args.dim, model_args.inter_dim)
        else:
            print(f"⚠ Unknown key pattern: {key}")
            hf_state_dict[key] = torch.zeros(1)  # fallback

    try:
        # Load from the checkpoint using QuantizedHuggingFaceStorageReader
        block_size = 128
        print(f"Loading weights with block_size={block_size}...")

        dist_cp.load(
            state_dict=hf_state_dict,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=CHECKPOINT_DIR,
                target_dtype=torch.float32,
                block_size=block_size,
                thread_count=2,
            ),
        )
        print(f"✓ Successfully loaded HF state dict with {len(hf_state_dict)} keys")

    except Exception as e:
        print(f"✗ Failed to load HF checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test from_hf conversion
    print("Testing from_hf conversion...")
    tt_state_dict = adapter.from_hf(hf_state_dict)
    print(f"Converted to TT state dict with {len(tt_state_dict)} keys")

    # Test to_hf conversion (roundtrip)
    print("Testing to_hf conversion (roundtrip)...")
    hf_state_dict_again = adapter.to_hf(tt_state_dict)
    print(f"Converted back to HF state dict with {len(hf_state_dict_again)} keys")

    # Verify roundtrip consistency
    def compare_state_dicts(sd1, sd2, name1="sd1", name2="sd2"):
        """Compare two state dicts and report differences."""
        keys1 = set(sd1.keys())
        keys2 = set(sd2.keys())

        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            extra_in_2 = keys2 - keys1
            print(f"Key mismatch between {name1} and {name2}:")
            if missing_in_2:
                print(f"  Missing in {name2}: {sorted(missing_in_2)}")
            if extra_in_2:
                print(f"  Extra in {name2}: {sorted(extra_in_2)}")
            return False

        for key in keys1:
            if sd1[key].shape != sd2[key].shape:
                print(
                    f"Shape mismatch for key '{key}': {sd1[key].shape} vs {sd2[key].shape}"
                )
                return False

            if not torch.allclose(sd1[key], sd2[key], rtol=1e-6, atol=1e-8):
                diff = (sd1[key] - sd2[key]).abs().max().item()
                print(f"Value mismatch for key '{key}': max_diff={diff}")
                return False

        return True

    print("Comparing original and roundtrip HF state dicts...")
    if compare_state_dicts(
        hf_state_dict, hf_state_dict_again, "original_hf", "roundtrip_hf"
    ):
        print("✓ Roundtrip test passed!")
    else:
        print("✗ Roundtrip test failed!")
        return False

    # Test loading into TT model
    print("Testing loading into TT model...")
    tt_model_state_dict = tt_model.state_dict()
    missing_keys = []
    unexpected_keys = []

    for key in tt_model_state_dict.keys():
        if key == "freqs_cis":
            continue  # Skip freqs_cis as it's computed
        if key not in tt_state_dict:
            missing_keys.append(key)

    for key in tt_state_dict.keys():
        if key not in tt_model_state_dict:
            unexpected_keys.append(key)

    if missing_keys:
        print(f"Missing keys in converted state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in converted state dict: {unexpected_keys}")

    # Load the state dict (allowing missing freqs_cis)
    try:
        missing, unexpected = tt_model.load_state_dict(tt_state_dict, strict=False)
        # Filter out freqs_cis from missing keys
        missing = [k for k in missing if k != "freqs_cis"]

        if not missing and not unexpected:
            print("✓ State dict loaded successfully into TT model!")
        else:
            print(f"State dict loaded with issues:")
            if missing:
                print(f"  Missing: {missing}")
            if unexpected:
                print(f"  Unexpected: {unexpected}")
    except Exception as e:
        print(f"✗ Failed to load state dict: {e}")
        return False

    # Test forward pass to ensure model is functional
    print("Testing forward pass...")
    try:
        batch_size = 2
        seq_len = 10
        tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = tt_model(tokens)
            print(f"Forward pass successful! Output shape: {output.shape}")
            expected_shape = (batch_size, seq_len, model_args.vocab_size)
            if output.shape == expected_shape:
                print("✓ Output shape is correct!")
            else:
                print(
                    f"✗ Output shape mismatch: expected {expected_shape}, got {output.shape}"
                )
                return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        success = test_state_dict_adapter()
        if success:
            print("\n🎉 DeepSeek V3 state dict adapter test completed successfully!")
        else:
            print("\n💥 DeepSeek V3 state dict adapter test failed!")
            sys.exit(1)

    except Exception as e:
        print(f"💥 Error occurred: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
