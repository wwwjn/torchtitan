#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple script to load and run inference with DeepSeek-V3 model from Hugging Face.
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os


def print_gpu_memory_usage(message=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU Memory ({message}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Load and test DeepSeek-V3 model")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-V3-0324", 
                        help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Explain the concept of attention in deep learning in simple terms.", 
                        help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--bf16", action="store_true", 
                        help="Use bfloat16 precision")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--load_in_8bit", action="store_true", 
                        help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Load model in 4-bit quantization")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    print(f"Device: {args.device}")
    
    # Set precision
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Using precision: {dtype}")
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print_gpu_memory_usage("Before loading model")
    
    # Load tokenizer
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    
    # Load model with appropriate quantization settings
    start_time = time.time()
    
    # Prepare quantization config if needed
    quantization_config = None
    # if args.load_in_8bit or args.load_in_4bit:
    #     from transformers import BitsAndBytesConfig
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_8bit=args.load_in_8bit,
    #         load_in_4bit=args.load_in_4bit,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    #     print(f"Using quantization: {'8-bit' if args.load_in_8bit else '4-bit'}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        # quantization_config=quantization_config
    )
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")
    
    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} billion")
    
    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print(f"\nPrompt: {args.prompt}")
    
    # Run a single forward pass
    print("\nRunning single forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**inputs)
    
    forward_time = time.time() - start_time
    
    # Get the logits from the output
    logits = outputs.logits
    
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
    print_gpu_memory_usage("After generation")


if __name__ == "__main__":
    main()
