#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hugging Face implementation for DeepSeek-V3 model inference.
"""

import gc
import os
import time

import torch


def print_gpu_memory_usage(message=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"GPU Memory ({message}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )


def run_huggingface_implementation(args, _):
    """Run the DeepSeek-V3 model using Hugging Face Transformers."""
    from transformers import AutoConfig, AutoModelForCausalLM

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

    # Use local path for model weights if specified, otherwise use model_name
    model_path = "/data/users/jianiw/dsv3-weights"
    print(f"Loading model from local path: {model_path}")
    start_time = time.time()

    # ============= Change config to only use a few layers  =============
    config = None
    if args.num_layers > 0:
        # Try to load config from local path first, fall back to model_name if needed
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Could not load config from local path: {e}")
            print(f"Falling back to loading config from {args.model_name}")
            config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        config.num_hidden_layers = args.num_layers
        print(f"Modified config to use only {args.num_layers} layers")

        config.n_group = 1  # make n_groups = a huge group
        config.topk_group = 1  # make topk_group = a huge group

    # Define quantization config
    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    print(f"Using quantization config: {quantization_config}")

    # Load the model from local path
    try:
        # First try with specific device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=args.device,  # Try with specific device first
            config=config,
            trust_remote_code=True,
            # Disable features that can cause issues with device mapping
            attn_implementation="eager",  # Use standard attention instead of flash attention
            quantization_config=quantization_config,
        )
    except Exception as e:
        print(f"Error loading model from local path with device_map={args.device}: {e}")
        print("Trying with device_map='cpu' first, then moving to GPU...")

        try:
            # Load on CPU first, then move to GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="cpu",  # Load on CPU first
                config=config,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
            # Move model to GPU after loading
            device = torch.device(args.device if args.device != "auto" else "cuda:0")
            model = model.to(device)
            print(f"Successfully loaded model on CPU and moved to {device}")
        except Exception as e:
            print(f"Error loading model on CPU: {e}")
            print(f"Falling back to loading model from {args.model_path}...")

            # Last resort: try loading from HF hub with auto device mapping
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=dtype,
                device_map="auto",
                config=config,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
            model = model.to(torch.float32)

    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")

    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} billion"
    )

    # Get the device where the model is loaded
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # Create fake input directly on the correct device
    print("\nCreating fake input with the same shape as tokenized input")

    # Define sequence length for fake input
    seq_length = 2048  # You can adjust this based on your needs

    # Create fake input_ids directly on the device - using random integers between 0 and 50000 (typical vocab size)
    torch.manual_seed(42)  # Set seed for reproducibility
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

    # # dense layer
    # layer_0_mlp_weights = model.model.layers[0].self_attn.q_a_proj.weight

    with torch.no_grad():
        # Forward pass through the model with output_hidden_states=True and output_attentions=True
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=False)

    forward_time = time.time() - start_time

    # Get the logits from the output
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    # Print attention outputs
    if hasattr(outputs, "attentions"):
        print("\nAttention Outputs:")
        attentions = outputs.attentions
        print(f"Number of attention layers: {len(attentions)}")

        for i, attn in enumerate(attentions):
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
