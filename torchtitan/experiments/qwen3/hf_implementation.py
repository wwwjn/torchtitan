#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hugging Face implementation for Qwen3 dense model inference.
"""

import argparse
import gc
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def print_gpu_memory_usage(message=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"GPU Memory ({message}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )


def run_huggingface_implementation(args, _):
    """Run the Qwen3 dense model using Hugging Face Transformers."""

    print("\n" + "=" * 50)
    print("Running Hugging Face Implementation")
    print("=" * 50)

    # Set precision
    dtype = torch.bfloat16
    print(f"Using precision: {dtype}")

    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print_gpu_memory_usage("Before loading model")

    # Disable FP8 quantization and other features that can cause issues
    os.environ["TRANSFORMERS_DISABLE_FP8"] = "1"
    os.environ["TRANSFORMERS_DISABLE_FLASH_ATTN_2"] = "1"
    os.environ["TRANSFORMERS_DISABLE_CUSTOM_KERNELS"] = "1"
    os.environ["TRANSFORMERS_DISABLE_TRITON"] = "1"

    # We're not using the tokenizer anymore, using fake inputs instead
    print("Using fake inputs instead of tokenizer")

    start_time = time.time()

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.num_hidden_layers = args.num_layers
    print(f"Modified config to use only {args.num_layers} layers")


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        config=config,
        attn_implementation="eager",  # Force eager attention to support output_attentions=True
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")

    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} billion"
    )

    # Print head dimension information
    print("\nAttention Head Dimensions:")
    if hasattr(config, "head_dim"):
        print(f"Explicit head_dim in config: {config.head_dim}")
    else:
        print("No explicit head_dim in config")

    print(f"hidden_size: {config.hidden_size}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"Calculated head_dim: {config.hidden_size // config.num_attention_heads}")
    print(f"Calculated kv_head_dim: {config.hidden_size // config.num_key_value_heads}")

    # Try to access the actual head dimension from model weights
    try:
        # Get the query projection weight from the first layer
        query_weight = model.model.layers[0].self_attn.q_proj.weight
        key_weight = model.model.layers[0].self_attn.k_proj.weight

        # Calculate dimensions
        q_out_features, q_in_features = query_weight.shape
        k_out_features, k_in_features = key_weight.shape

        print(f"\nFrom model weights:")
        print(f"Query projection: {q_out_features}x{q_in_features}")
        print(f"Key projection: {k_out_features}x{k_in_features}")
        print(
            f"Query head_dim (out_features/num_heads): {q_out_features // config.num_attention_heads}"
        )
        print(
            f"Key head_dim (out_features/num_kv_heads): {k_out_features // config.num_key_value_heads}"
        )
    except Exception as e:
        print(f"Could not access attention weights: {e}")

    # Get the device where the model is loaded
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # Create fake input directly on the correct device
    print("\nCreating fake input with the same shape as tokenized input")

    # Define sequence length for fake input
    seq_length = 4096  # You can adjust this based on your needs

    # Create fake input_ids directly on the device - using random integers between 0 and 50000 (typical vocab size)
    torch.manual_seed(23)  # Set seed for reproducibility
    input_ids = torch.randint(
        0, 50000, (1, seq_length), dtype=torch.long, device=device
    )

    # Create fake attention_mask directly on the device - all 1s for full attention
    attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)

    # Create inputs dictionary similar to what tokenizer would produce
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Print input information
    print(f"Fake input token IDs: {inputs['input_ids'][0][:10].cpu().numpy()}...")
    print(f"Fake input shape: {inputs['input_ids'].shape}")
    print(f"Input tensors device: {inputs['input_ids'].device}")

    # Run a single forward pass
    print("\nRunning single forward pass...")
    start_time = time.time()

    # For Qwen3, check the MLP weights structure
    try:
        # Qwen3 uses a different structure for attention layers
        layer_0_mlp_weights = model.model.layers[0].mlp.gate_proj.weight
        print(f"Layer 0 MLP gate weights shape: {layer_0_mlp_weights.shape}")
    except Exception as e:
        print(f"Could not access MLP weights: {e}")

    with torch.no_grad():
        # Forward pass through the model with output_hidden_states=True and output_attentions=True
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    forward_time = time.time() - start_time

    # Get the logits from the output
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    # Print attention outputs
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        print("\nAttention Outputs:")
        attentions = outputs.attentions

        # Check if attentions is None or empty
        if attentions is None or len(attentions) == 0:
            print(
                "No attention weights available. The model might be using an attention implementation that doesn't support returning attention weights."
            )
        else:
            print(f"Number of attention layers: {len(attentions)}")

            for i, attn in enumerate(attentions):
                # Check if this specific attention layer is None
                if attn is None:
                    print(f"Layer {i} attention weights are None")
                    continue

                # Print shape and statistics for each attention layer
                print(
                    f"Layer {i} attention weights - Shape: {attn.shape}, "
                    f"Mean: {attn.mean().item():.6f}, "
                    f"Std: {attn.std().item():.6f}, "
                    f"Min: {attn.min().item():.6f}, "
                    f"Max: {attn.max().item():.6f}"
                )

                # Print a small sample of attention weights from the first head
                if i == 0:  # Just for the first layer as an example
                    print(f"Sample attention weights (first head, first 5x5):")
                    # Convert to float32 before converting to numpy to avoid BFloat16 error
                    sample = attn[0, 0, :5, :5].to(torch.float32).cpu().numpy()
                    for row in sample:
                        print(" ".join([f"{val:.4f}" for val in row]))
    else:
        print("\nAttention Outputs:")
        print(
            "No attention weights available. The model might be using an attention implementation that doesn't support returning attention weights."
        )

    # Print hidden states information if available
    if hasattr(outputs, "hidden_states"):
        hidden_states = outputs.hidden_states
        for i, hidden_state in enumerate(hidden_states):
            # Calculate statistics.
            layer_mean = hidden_state.mean().item()
            layer_std = hidden_state.std().item()
            layer_min = hidden_state.min().item()
            layer_max = hidden_state.max().item()

            if i == 0:
                print(
                    f"Input Embeddings - Mean: {layer_mean:.6f}, Std: {layer_std:.6f}, Min: {layer_min:.6f}, Max: {layer_max:.6f}"
                )
                # Print embedding weights statistics
                if i == 0:
                    embedding_weights = model.model.embed_tokens.weight
                    emb_mean = embedding_weights.mean().item()
                    emb_std = embedding_weights.std().item()
                    emb_min = embedding_weights.min().item()
                    emb_max = embedding_weights.max().item()

                    print(
                        f"Embedding Weights - Mean: {emb_mean:.6f}, Std: {emb_std:.6f}, Min: {emb_min:.6f}, Max: {emb_max:.6f} shape {embedding_weights.shape}"
                    )
            else:
                # dtype = bfloat16
                print(
                    f"Layer {i-1} - Mean: {layer_mean:.6f}, Std: {layer_std:.6f}, Min: {layer_min:.6f}, Max: {layer_max:.6f},  shape {hidden_state.shape} dtype {hidden_state.dtype}"
                )
    else:
        print(
            "\nNo hidden states found in model outputs. Try setting return_dict=True and output_hidden_states=True in the model configuration."
        )

    # Get the predictions for the next token (highest probability)
    next_token_logits = logits[:, -1, :]
    print(f"\nNext token logits : {next_token_logits}")
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    print(f"\nNext token probabilities: {next_token_probs}")
    top_k_values, top_k_indices = torch.topk(next_token_probs, 5, dim=-1)

    print("\nForward Pass Results:")
    print(f"- Output logits shape: {logits.shape}")
    print(f"- Sequence length: {logits.shape[1]}")
    print(f"- Vocabulary size: {logits.shape[2]}")

    print(
        "\nTop 5 predicted next tokens (showing IDs only since we're not using tokenizer):"
    )
    for i, (value, index) in enumerate(zip(top_k_values[0], top_k_indices[0])):
        print(f"  {i+1}. Token ID: {index} - Probability: {value.item():.4f}")

    print(f"\nForward pass stats:")
    print(f"- Time: {forward_time:.4f} seconds")
    print(f"- Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"- Tokens per second: {inputs['input_ids'].shape[1] / forward_time:.2f}")
    print_gpu_memory_usage("After forward pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3 model using Hugging Face")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=28,
        help="Number of layers to use in the model (default: 28)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (default: cuda:0)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/data/users/jianiw/model/qwen3-0.6b",
        help="Path to the model weights (default: /data/users/jianiw/model/qwen3-0.6b)",
    )

    args = parser.parse_args()

    # Print arguments for debugging
    print("Script arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    run_huggingface_implementation(args, None)
