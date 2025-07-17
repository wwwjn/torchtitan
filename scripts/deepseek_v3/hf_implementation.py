#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hugging Face implementation for DeepSeek-V3 model inference.
"""

import time
import torch
import os
import gc


def print_gpu_memory_usage(message=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU Memory ({message}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def run_huggingface_implementation(args, hf_tokenizer):
    """Run the DeepSeek-V3 model using Hugging Face Transformers."""
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print("\n" + "="*50)
    print("Running Hugging Face Implementation")
    print("="*50)
    
    # Set precision
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Using precision: {dtype}")
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print_gpu_memory_usage("Before loading model")
    
    # Disable FP8 quantization which can cause issues
    os.environ["TRANSFORMERS_DISABLE_FP8"] = "1"
    
    # Use the provided HF tokenizer instead of creating a new torchtitan tokenizer
    # This ensures compatibility with the model
    tokenizer = hf_tokenizer
    print("Using provided Hugging Face tokenizer")
    
    # Load model configuration
    print(f"Loading model: {args.model_name}")
    start_time = time.time()
    
    # Change config to only use a few layers for testing if specified
    config = None
    if args.num_layers > 0:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        config.num_hidden_layers = args.num_layers
        print(f"Modified config to use only {args.num_layers} layers")
    
    # Load the model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map=args.device,
            config=config,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model with device_map={args.device}: {e}")
        print("Trying with device_map='auto'...")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
            config=config,
            trust_remote_code=True,
        )
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")
    
    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} billion")
    
    # Tokenize input and ensure it's on the correct device
    print(f"\nPrompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    # Print input information
    print(f"Input token IDs: {inputs['input_ids'][0][:10]}...")
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Explicitly move all input tensors to the same device as the model
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run a single forward pass
    print("\nRunning single forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**inputs)
    
    forward_time = time.time() - start_time
    
    # Get the logits from the output
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    
    # Get the predictions for the next token (highest probability)
    next_token_logits = logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    top_k_values, top_k_indices = torch.topk(next_token_probs, 5, dim=-1)
    
    print("\nForward Pass Results:")
    print(f"- Output logits shape: {logits.shape}")
    print(f"- Sequence length: {logits.shape[1]}")
    print(f"- Vocabulary size: {logits.shape[2]}")
    
    print("\nTop 5 predicted next tokens:")
    for i, (value, index) in enumerate(zip(top_k_values[0], top_k_indices[0])):
        token = tokenizer.decode([index])
        print(f"  {i+1}. Token: '{token}' (ID: {index}) - Probability: {value.item():.4f}")
    
    print(f"\nForward pass stats:")
    print(f"- Time: {forward_time:.4f} seconds")
    print(f"- Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"- Tokens per second: {inputs['input_ids'].shape[1] / forward_time:.2f}")
    print_gpu_memory_usage("After forward pass")
