#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to load and run inference with DeepSeek-V3 model using either:
1. Hugging Face Transformers library
2. TorchTitan native implementation
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add torchtitan to path if needed
torchtitan_path = Path("/data/users/jianiw/torchtitan")
if str(torchtitan_path) not in sys.path:
    sys.path.insert(0, str(torchtitan_path))

# Import implementations
from hf_implementation import run_huggingface_implementation, print_gpu_memory_usage
from torchtitan_implementation import run_torchtitan_implementation


def main():
    parser = argparse.ArgumentParser(description="Load and test DeepSeek-V3 model")
    parser.add_argument("--implementation", type=str, choices=["hf", "torchtitan", "both"], default="hf",
                        help="Which implementation to use: 'hf' for Hugging Face, 'torchtitan' for TorchTitan, or 'both'")
    
    # Common arguments
    parser.add_argument("--prompt", type=str, 
                        default="Explain the concept of attention in deep learning in simple terms.",
                        help="Prompt for inference")
    parser.add_argument("--bf16", action="store_true", 
                        help="Use bfloat16 precision")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num_layers", type=int, default=0,
                        help="Number of layers to use (0 for all layers)")
    
    # Hugging Face specific arguments
    parser.add_argument("--model_name", type=str, 
                        default="deepseek-ai/DeepSeek-V3-0324",
                        help="Hugging Face model name or path")
    parser.add_argument("--load_in_8bit", action="store_true", 
                        help="Load model in 8-bit quantization (HF only)")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Load model in 4-bit quantization (HF only)")
    
    # TorchTitan specific arguments
    parser.add_argument("--config_path", type=str, 
                        default="/data/users/jianiw/torchtitan/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b_test.toml",
                        help="Path to the TorchTitan config file")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="/data/users/jianiw/torchtitan/outputs/checkpoint-dsv3/step-0",
                        help="Path to the checkpoint directory")
    
    args = parser.parse_args()
    
    # Run the selected implementation(s)
    if args.implementation in ["hf", "both"]:
        run_huggingface_implementation(args)
    
    if args.implementation in ["torchtitan", "both"]:
        run_torchtitan_implementation(args)


if __name__ == "__main__":
    main()
