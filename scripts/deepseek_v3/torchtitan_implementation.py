#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan implementation for DeepSeek-V3 model inference.
"""

import time
import torch
import os
import gc
from pathlib import Path


def print_gpu_memory_usage(message=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU Memory ({message}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def run_torchtitan_implementation(args):
    """Run the DeepSeek-V3 model using TorchTitan."""
    from torchtitan.config_manager import ConfigManager
    from torchtitan.tools.logging import init_logger, logger
    from torchtitan.protocols.train_spec import get_train_spec
    from torchtitan.distributed import ParallelDims, utils as dist_utils
    
    print("\n" + "="*50)
    print("Running TorchTitan Implementation")
    print("="*50)
    
    # Initialize logger
    init_logger()
    
    # Set environment variables for distributed setup (single GPU mode)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_path)
    
    # Override checkpoint path if provided
    if args.checkpoint_path:
        config.checkpoint.initial_load_path = args.checkpoint_path
    
    # Set precision
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Using precision: {dtype}")
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print_gpu_memory_usage("Before loading model")
    
    # Initialize distributed environment
    dist_utils.init_distributed(config)
    
    # Get the train spec for DeepSeek-V3
    train_spec = get_train_spec("deepseek_v3")
    
    # Build tokenizer
    print("Loading tokenizer...")
    start_time = time.time()
    tokenizer = train_spec.build_tokenizer_fn(config)
    print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    
    # Initialize model
    print(f"Loading model from config: {config.model.name} {config.model.flavor}")
    start_time = time.time()
    
    # Get model arguments
    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config, tokenizer)
    
    # Modify number of layers if specified
    if args.num_layers > 0:
        print(f"Modified config to use only {args.num_layers} layers")
        model_args.num_hidden_layers = args.num_layers
    
    # Build model in meta device first
    with torch.device("meta"):
        model = train_spec.model_cls(model_args)
    
    # Create parallel dimensions
    parallel_dims = ParallelDims(
        dp_shard=config.parallelism.data_parallel_shard_degree,
        dp_replicate=config.parallelism.data_parallel_replicate_degree,
        cp=config.parallelism.context_parallel_degree,
        tp=config.parallelism.tensor_parallel_degree,
        pp=config.parallelism.pipeline_parallel_degree,
        ep=config.parallelism.expert_parallel_degree,
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    
    # Apply parallelism
    model = train_spec.parallelize_fn(model, parallel_dims, config)
    
    # Move model to device and initialize weights
    device = torch.device(args.device)
    model.to_empty(device=args.device)
    
    # Initialize weights
    print("Initializing model weights...")
    with torch.no_grad():
        model.init_weights()
    
    # Set model to eval mode
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {config.checkpoint.initial_load_path}")
    checkpoint_path = Path(config.checkpoint.initial_load_path)
    
    # Use torch.distributed.checkpoint to load the model
    from torch.distributed.checkpoint import load_checkpoint
    load_checkpoint(checkpoint_path, model)
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")
    
    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} billion")
    
    # Tokenize input
    print(f"\nPrompt: {args.prompt}")
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"Input token IDs: {input_ids[0][:10]}...")
    print(f"Input shape: {input_ids.shape}")
    
    # Run a single forward pass
    print("\nRunning single forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(input_ids)
    
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
    print(f"- Input tokens: {input_ids.shape[1]}")
    print(f"- Tokens per second: {input_ids.shape[1] / forward_time:.2f}")
    print_gpu_memory_usage("After forward pass")
