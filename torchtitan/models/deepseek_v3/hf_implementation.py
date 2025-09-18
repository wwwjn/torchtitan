#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hugging Face implementation for DeepSeek-V3 model inference.
"""

import argparse
import gc
import json
import os
import time

import numpy as np

import torch

# Global dictionary to store intermediate results
saved_tensors = {}

def dequantize_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype=torch.bfloat16,
    BLOCK_SIZE: int = 128,
) -> torch.Tensor:
    """
    Dequantize FP8 quantized weights using block-wise scaling.
    
    Args:
        weight: Quantized FP8 weight tensor
        scale_inv: Inverse scaling factors for each block
        dtype: Target dtype for dequantized weights
        BLOCK_SIZE: Size of quantization blocks
    
    Returns:
        Dequantized weight tensor
    """
    orig_shape = weight.shape
    float_weight = weight.float()
    
    block_rows = (orig_shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_cols = (orig_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    expected_scale_shape = torch.Size((block_rows, block_cols))
    
    if scale_inv.shape != expected_scale_shape:
        raise ValueError(
            f"scale_inv shape {scale_inv.shape} doesn't match expected shape {expected_scale_shape}"
        )
    
    # NOTE: When processing large models on-the-fly, misalignment between block boundaries
    # and DTensor local shape partitioning can lead to silent numerical inaccuracies.
    dequantized = float_weight.detach().clone().to(dtype=dtype)
    
    # Apply scaling factors to each block
    for i in range(block_rows):
        row_start = i * BLOCK_SIZE
        row_end = min(row_start + BLOCK_SIZE, orig_shape[0])
        
        for j in range(block_cols):
            col_start = j * BLOCK_SIZE
            col_end = min(col_start + BLOCK_SIZE, orig_shape[1])
            
            block = dequantized[row_start:row_end, col_start:col_end]
            scale = scale_inv[i, j]
            block = block * scale
            
            # Explicitly convert block to dtype
            block_converted = block.to(dtype=torch.float32)
            # Store the dequantized block
            dequantized[row_start:row_end, col_start:col_end] = block_converted
    
    return dequantized

def tensor_to_json_compatible(tensor, state_dict=None):
    """
    Convert tensor to JSON-compatible format.
    If the tensor is quantized (has a corresponding _scale_inv tensor), dequantize it first.
    
    Args:
        tensor: The tensor to convert
        state_dict: Optional state_dict to look for quantization scale tensors
    
    Returns:
        Dictionary with tensor metadata and statistics
    """
    if isinstance(tensor, torch.Tensor):
        try:
            # Detach tensor from computation graph and move to CPU
            tensor_cpu = tensor.detach().cpu()
            
            # Check if this tensor is quantized by looking for a corresponding scale_inv tensor
            is_quantized = False
            dequantized_tensor = tensor_cpu
            quantization_info = {}
            
            if state_dict is not None:
                # We need the tensor name to check for quantization, but we don't have it here
                # This will be handled in the calling function
                pass
            
            # Only save first 10 values to avoid memory issues
            values = []
            if dequantized_tensor.numel() > 0:
                values = dequantized_tensor.flatten().numpy().tolist()
            
            result = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "mean": float(dequantized_tensor.mean().item()),
                "std": float(dequantized_tensor.std().item()),
                "min": float(dequantized_tensor.min().item()),
                "max": float(dequantized_tensor.max().item()),
                "values": values,
                "is_quantized": is_quantized,
            }
            
            # Add quantization info if available
            if quantization_info:
                result.update(quantization_info)
                
            return result
            
        except Exception as e:
            # Fallback for tensors that can't be processed
            return {
                "shape": list(tensor.shape) if hasattr(tensor, 'shape') else "unknown",
                "dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else "unknown",
                "device": str(tensor.device) if hasattr(tensor, 'device') else "unknown",
                "error": f"Could not process tensor: {str(e)}",
                "values": [],
                "is_quantized": False,
            }
    return tensor

def tensor_to_json_compatible_with_dequant(tensor_name, tensor, state_dict):
    """
    Convert tensor to JSON-compatible format with dequantization support.
    
    Args:
        tensor_name: Name/key of the tensor in state_dict
        tensor: The tensor to convert
        state_dict: Full state_dict to look for quantization scale tensors
    
    Returns:
        Dictionary with tensor metadata and statistics
    """
    if isinstance(tensor, torch.Tensor):
        try:
            # Detach tensor from computation graph and move to CPU
            tensor_cpu = tensor.detach().cpu()
            
            # Check if this tensor is quantized by looking for a corresponding scale_inv tensor
            is_quantized = False
            dequantized_tensor = tensor_cpu
            quantization_info = {}
            
            scale_inv_key = tensor_name + "_scale_inv"
            if scale_inv_key in state_dict:
                print(f"Found quantized tensor: {tensor_name}")
                is_quantized = True
                scale_inv = state_dict[scale_inv_key].detach().cpu()
                
                try:
                    # Dequantize the tensor
                    dequantized_tensor = dequantize_from_fp8(
                        tensor_cpu, scale_inv, dtype=torch.float32
                    )
                      
                    # For Float8 tensors, we need to convert to a supported dtype first to compute statistics
                    try:
                        # Convert to float32 for statistics computation
                        tensor_for_stats = tensor_cpu.float()
                        quantization_info = {
                            "quantization_blocks": list(scale_inv.shape),
                            "original_mean": float(tensor_for_stats.mean().item()),
                            "original_std": float(tensor_for_stats.std().item()),
                            "original_min": float(tensor_for_stats.min().item()),
                            "original_max": float(tensor_for_stats.max().item()),
                        }
                    except Exception as stats_error:
                        print(f"Could not compute original tensor statistics for {tensor_name}: {stats_error}")
                        quantization_info = {
                            "quantization_blocks": list(scale_inv.shape),
                            "original_stats_error": str(stats_error),
                        }
                      
                    print(f"Successfully dequantized {tensor_name}")
                except Exception as dequant_error:
                    print(f"Failed to dequantize {tensor_name}: {dequant_error}")
                    # Fall back to using original quantized tensor
                    dequantized_tensor = tensor_cpu
                    quantization_info["dequantization_error"] = str(dequant_error)
            
            # Only save first 10 values to avoid memory issues
            values = []
            if dequantized_tensor.numel() > 0:
                values = dequantized_tensor.flatten().numpy().tolist()
            
            result = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "mean": float(dequantized_tensor.mean().item()),
                "std": float(dequantized_tensor.std().item()),
                "min": float(dequantized_tensor.min().item()),
                "max": float(dequantized_tensor.max().item()),
                "values": values,
                "is_quantized": is_quantized,
            }
            
            # Add quantization info if available
            if quantization_info:
                result.update(quantization_info)
                
            return result
            
        except Exception as e:
            # Fallback for tensors that can't be processed
            return {
                "shape": list(tensor.shape) if hasattr(tensor, 'shape') else "unknown",
                "dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else "unknown",
                "device": str(tensor.device) if hasattr(tensor, 'device') else "unknown",
                "error": f"Could not process tensor: {str(e)}",
                "value": [],
                "is_quantized": False,
            }
    return tensor


def save_model_state_dict_subset(model, save_path):
    """Save subset of model state_dict to JSON file with dequantization support."""
    state_dict = model.state_dict()
    filtered_dict = {}
    
    print(f"Processing model state_dict with {len(state_dict)} total keys")

    # Filter keys that start with "model.layers.0.mlp"
    mlp_keys = [key for key in state_dict.keys() if key.startswith("model.layers.0.mlp")]
    print(f"Found {len(mlp_keys)} MLP keys in layer 0:")
    
    for key in mlp_keys:
        print(f"  Processing: {key}")
        # Skip scale_inv keys as they are processed with their corresponding weight tensors
        if key.endswith("_scale_inv"):
            print(f"    Skipping scale_inv key: {key}")
            continue
        
        value = state_dict[key]
        # Use the dequantization-aware conversion function
        filtered_dict[key] = tensor_to_json_compatible_with_dequant(key, value, state_dict)

    # Add intermediate results from global dictionary
    if saved_tensors:
        filtered_dict["intermediate_results"] = saved_tensors

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(filtered_dict, f, indent=2)

    print(f"Saved tensor data to {save_path}")
    print(f"Saved {len(filtered_dict)} tensor entries")
    
    # Print summary of quantized vs non-quantized tensors
    quantized_count = sum(1 for v in filtered_dict.values() 
                         if isinstance(v, dict) and v.get("is_quantized", False))
    non_quantized_count = len(filtered_dict) - quantized_count - (1 if "intermediate_results" in filtered_dict else 0)
    print(f"Summary: {quantized_count} quantized tensors, {non_quantized_count} non-quantized tensors")


def hook_intermediate_results(model):
    """Add hooks to capture intermediate results."""

    def capture_layer_outputs(module, input, output):
        """Hook to capture layer outputs at different stages"""
        if hasattr(output, "__len__") and len(output) > 0:
            hidden_states = output[0]  # First element is usually hidden_states
            # This captures the final output of the decoder layer (after x+mlp)
            saved_tensors["hidden_states_after_x_plus_mlp"] = tensor_to_json_compatible(
                hidden_states
            )

    def capture_self_attention_output(module, input, output):
        """Hook to capture self-attention output"""
        if hasattr(output, "__len__") and len(output) > 0:
            attn_output = output[0]  # Attention output
            saved_tensors["self_attention_output"] = tensor_to_json_compatible(
                attn_output
            )

    def capture_mlp_output(module, input, output):
        """Hook to capture MLP output"""
        if isinstance(output, torch.Tensor):
            saved_tensors["mlp_output"] = tensor_to_json_compatible(output)

    # Register hooks on the first decoder layer (layer 0)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and len(model.model.layers) > 0
    ):
        first_layer = model.model.layers[0]

        # Hook the entire decoder layer to capture final output (after x+mlp)
        first_layer.register_forward_hook(capture_layer_outputs)

        # Hook the self-attention module to capture attention output
        if hasattr(first_layer, "self_attn"):
            first_layer.self_attn.register_forward_hook(capture_self_attention_output)

        # Hook the MLP module to capture MLP output
        if hasattr(first_layer, "mlp"):
            first_layer.mlp.register_forward_hook(capture_mlp_output)

    # # Also try to monkey patch the modeling_deepseek print_tensor_stats function to capture data
    # try:
    #     import os

    #     # Import the modeling module to patch it
    #     import sys

    #     model_path = "/data/users/jianiw/model/DeepSeek-V3.1-Base"
    #     if model_path not in sys.path:
    #         sys.path.insert(0, model_path)

    #     # Try to import and patch the print_tensor_stats function
    #     try:
    #         import modeling_deepseek

    #         original_print_tensor_stats = modeling_deepseek.print_tensor_stats

    #         def patched_print_tensor_stats(name, tensor):
    #             # Call original function
    #             original_print_tensor_stats(name, tensor)

    #             # Save specific tensors we're interested in
    #             if "hidden_states after x + self attention" in name:
    #                 saved_tensors["hidden_states_after_x_plus_self_attention"] = (
    #                     tensor_to_json_compatible(tensor)
    #                 )
    #             elif "hidden_states after x+mlp" in name:
    #                 saved_tensors["hidden_states_after_x_plus_mlp"] = (
    #                     tensor_to_json_compatible(tensor)
    #                 )

    #         # Replace the function
    #         modeling_deepseek.print_tensor_stats = patched_print_tensor_stats
    #         print("Successfully patched print_tensor_stats function")

    #     except ImportError:
    #         print("Could not import modeling_deepseek module for patching")

    # except Exception as e:
    #     print(f"Error setting up patching: {e}")


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
    # Disable Hugging Face cache
    from transformers import AutoConfig, AutoModelForCausalLM

    # We're not using the tokenizer anymore, using fake inputs instead
    # Use local path for model weights if specified, otherwise use model_name
    model_path = args.model_path
    print(f"Loading model from local path: {model_path}")
    start_time = time.time()

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",  # Updated from fp8 to fbgemm_fp8
        "weight_block_size": [128, 128],
    }
    print(f"Using quantization config: {quantization_config}")

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

        config.n_group = 1  # make n_groups = a huge group
        config.topk_group = 1  # make topk_group = a huge group
        # tailer the first several layers
        config.num_hidden_layers = args.num_layers
        # Explicitly set rope_interleaved to True to use the interleaved rope implementation
        config.rope_interleaved = True
        print(f"Modified config to use only {args.num_layers} layers")
        print(f"Config of Deepseek: {config}")

    # Load the model from local path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # Try with specific device first
        config=config,
        trust_remote_code=True,
        # Disable features that can cause issues with device mapping
        attn_implementation="eager",  # Use standard attention instead of flash attention
        quantization_config=None,
        local_files_only=True,  # Only use local files, don't fetch from cache
        use_auth_token=False,  # Don't try to authenticate with HF
    )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")

    # Get the device where the model is loaded
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # Set up hooks to capture intermediate results if JSON saving is enabled
    if args.save_tensors_json:
        print("Setting up hooks to capture intermediate results...")
        hook_intermediate_results(model)

    # Create fake input directly on the correct device
    print("\nCreating fake input with the same shape as tokenized input")

    # Define sequence length for fake input
    seq_length = 2048  # You can adjust this based on your needs
    vocab_size = 50000

    with torch.no_grad():
        # Create fake input_ids directly on the device - using random integers between 0 and 50000 (typical vocab size)
        torch.manual_seed(42)
        tokens = torch.randint(
            0, vocab_size, (1, seq_length), dtype=torch.long, device="cuda"
        )

        # Create fake attention_mask directly on the device - all 1s for full attention
        attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)

        # Create inputs dictionary similar to what tokenizer would produce
        inputs = {"input_ids": tokens}

        # Print input information
        print(f"Fake input token IDs: {inputs['input_ids'][0][:10].cpu().numpy()}...")
        print(f"Fake input shape: {inputs['input_ids'].shape}")
        print(f"Input tensors device: {inputs['input_ids'].device}")

    # Run a single forward pass
    print("\nRunning single forward pass...")
    start_time = time.time()

    with torch.no_grad():
        # Forward pass through the model with output_hidden_states=True and output_attentions=True
        outputs = model(
            **inputs, output_hidden_states=True, output_attentions=True, use_cache=False
        )

    forward_time = time.time() - start_time

    # Get the logits from the output
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

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

    # Save tensors to JSON if requested
    if args.save_tensors_json:
        print(f"\nSaving tensors to JSON file: {args.save_tensors_json}")
        save_model_state_dict_subset(model, args.save_tensors_json)


def main():
    parser = argparse.ArgumentParser(description="Load and test DeepSeek-V3 model")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,  # tailered to 1 layers for 671B model
        help="Number of layers to use (0 for all layers)",
    )

    # Hugging Face specific arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/users/jianiw/model/DeepSeek-V3.1-Base",
        help="Hugging Face model name or path",
    )

    # JSON saving option
    parser.add_argument(
        "--save_tensors_json",
        type=str,
        default=None,
        help="Path to save intermediate tensors to JSON file",
    )

    args = parser.parse_args()
    run_huggingface_implementation(args, None)


if __name__ == "__main__":
    main()
