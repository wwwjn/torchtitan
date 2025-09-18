#!/usr/bin/env python3
"""
Test script to compare HuggingFace quantized checkpoint loading approaches.
Only applies FSDP to the model and compares:
1. Original HF weight + manually dequantized result
2. dcp.load(QuantizedHFReader)'s result

To run this script: torchrun --nproc_per_node=2 tests/fsdp_dequantized_load.py
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh
import json
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import QuantizedHuggingFaceStorageReader, HuggingFaceStorageReader
from safetensors import safe_open



class SimpleMLP(nn.Module):
    """A simple MLP with configurable hidden dimensions."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def setup_distributed(backend="nccl"):
    """Initialize the distributed environment."""
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Initialize the process group
    dist.init_process_group(backend=backend)
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


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


def load_hf_weights_manually(checkpoint_path, target_keys):
    """
    Load HuggingFace weights manually by directly reading safetensor files.
    Extract quantized weights and scale_inv, then perform manual dequantization.
    
    Args:
        checkpoint_path: Path to the HuggingFace checkpoint
        target_keys: List of keys to load
        
    Returns:
        Dictionary of manually dequantized weights
    """
    manually_dequantized = {}
    
    # Read the safetensors index file to find which files contain our target keys
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map", {})
    
    # Group keys by their safetensor file
    file_to_keys = {}
    for key in target_keys:
        if key in weight_map:
            safetensor_file = weight_map[key]
            if safetensor_file not in file_to_keys:
                file_to_keys[safetensor_file] = []
            file_to_keys[safetensor_file].append(key)
        else:
            print(f"Warning: Key {key} not found in weight_map")
    
    # Load weights from each safetensor file
    for safetensor_file, keys_in_file in file_to_keys.items():
        safetensor_path = os.path.join(checkpoint_path, safetensor_file)
        
        if not os.path.exists(safetensor_path):
            print(f"Warning: Safetensor file not found: {safetensor_path}")
            continue
        
        print(f"Loading {len(keys_in_file)} keys from {safetensor_file}")
        
        # Open the safetensor file and read the target keys
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            available_keys = f.keys()
            print(f"Available keys in {safetensor_file}: {list(available_keys)[:10]}...")  # Show first 10 keys
            
            for key in keys_in_file:
                try:
                    # Load the quantized weight tensor
                    if key in available_keys:
                        weight_tensor = f.get_tensor(key)
                        print(f"Loaded {key}: shape={weight_tensor.shape}, dtype={weight_tensor.dtype}")
                        
                        # Look for corresponding scale_inv tensor
                        scale_key = key.replace('.weight', '.weight_scale_inv')
                        assert scale_key in available_keys
                        scale_inv_tensor = f.get_tensor(scale_key)
                        print(f"Loaded {scale_key}: shape={scale_inv_tensor.shape}, dtype={scale_inv_tensor.dtype}")
                        
                        # Perform manual dequantization using FP8 method
                        try:
                            dequantized = dequantize_from_fp8(
                                weight=weight_tensor,
                                scale_inv=scale_inv_tensor,
                                dtype=torch.bfloat16,
                                BLOCK_SIZE=128
                            )
                            manually_dequantized[key] = dequantized
                            print(f"Successfully dequantized {key} to shape={dequantized.shape}, dtype={dequantized.dtype}")
                            
                        except Exception as e:
                            print(f"Failed to dequantize {key} using FP8 method: {e}")
                            # Fallback to simple element-wise multiplication
                            raise e
                    
                    else:
                        print(f"Key {key} not found in {safetensor_file}")
                        
                except Exception as e:
                    print(f"Error loading {key} from {safetensor_file}: {e}")
    
    print(f"Successfully loaded {len(manually_dequantized)} weights manually")
    return manually_dequantized


def load_hf_weights_with_quantized_reader(checkpoint_path, target_keys, args, model):
    """
    Load HuggingFace weights using QuantizedHuggingFaceStorageReader.
    
    Args:
        checkpoint_path: Path to the HuggingFace checkpoint
        target_keys: List of keys to load
        
    Returns:
        Dictionary of weights loaded via QuantizedHuggingFaceStorageReader
    """
    quantized_reader_weights = {}
    
    # Use QuantizedHuggingFaceStorageReader
    storage_reader = QuantizedHuggingFaceStorageReader(checkpoint_path)
    
    # Create a state dict with the target keys
    # Get the model's state dict and apply key mapping
    original_state_dict = model.state_dict()
    state_dict = {}
    key_mapping = {
        f"model.layers.{args.layer_id}.mlp.gate_proj.weight": f"w1.weight",
        f"model.layers.{args.layer_id}.mlp.up_proj.weight": f"w3.weight", 
        f"model.layers.{args.layer_id}.mlp.down_proj.weight": f"w2.weight"
    }
    for target_key in target_keys:
        mapped_key = key_mapping.get(target_key, target_key)
        if mapped_key in original_state_dict:
            state_dict[target_key] = original_state_dict[mapped_key]

    # Check all the state_dict values are DTensor, and print the placements of DTensor
    from torch.distributed.tensor import DTensor
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            print(f"[load_hf_weights_with_quantized_reader]Before dcp.load() {key}: DTensor with placements={tensor.placements}, shape={tensor.shape}")
    
    # Load using the quantized reader
    dcp.load(state_dict, storage_reader=storage_reader)

    # After dcp.load(), the DTensor placement and the shape should be the same
    for key, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            print(f"[load_hf_weights_with_quantized_reader]After dcp.load() {key}: DTensor with placements={tensor.placements}, shape={tensor.shape}")
    
    return state_dict


def tensor_to_local(tensor):
    """
    Convert tensor to local/regular tensor, handling DTensors properly.
    
    Args:
        tensor: Input tensor (regular tensor or DTensor)
        
    Returns:
        Regular tensor
    """
    from torch.distributed.tensor import DTensor
    
    if isinstance(tensor, DTensor):
        # For DTensor, get the full tensor
        return tensor.full_tensor().detach().cpu().float()
    else:
        # For regular tensor, just move to CPU
        return tensor.detach().cpu().float()


def get_tensor_stats(tensor):
    """
    Get tensor statistics, handling DTensors properly.
    
    Args:
        tensor: Input tensor (regular tensor or DTensor)
        
    Returns:
        Dictionary with min, max, mean statistics
    """
    from torch.distributed.tensor import DTensor
    
    if isinstance(tensor, DTensor):
        # For DTensor, get the full tensor first
        full_tensor = tensor.full_tensor().detach().cpu().float()
    else:
        # For regular tensor
        full_tensor = tensor.detach().cpu().float()
    
    return {
        "min": full_tensor.min().item(),
        "max": full_tensor.max().item(), 
        "mean": full_tensor.mean().item()
    }, full_tensor


def normalize_for_kl(tensor1, tensor2, epsilon=1e-8):
    """
    Normalize tensors to probability distributions for KL divergence calculation.
    
    Args:
        tensor1, tensor2: Input tensors
        epsilon: Small value to avoid log(0)
    
    Returns:
        Normalized probability distributions
    """
    # Flatten tensors
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    # Shift to make all values positive (for probability distribution)
    min_val = min(flat1.min(), flat2.min())
    if min_val < 0:
        flat1 = flat1 - min_val + epsilon
        flat2 = flat2 - min_val + epsilon
    
    # Add epsilon to avoid zeros
    flat1 = flat1 + epsilon
    flat2 = flat2 + epsilon
    
    # Normalize to probability distributions
    prob1 = flat1 / flat1.sum()
    prob2 = flat2 / flat2.sum()
    
    return prob1.numpy(), prob2.numpy()


def compare_tensors(tensor1, tensor2, name1="Method 1", name2="Method 2"):
    """
    Compare two tensors and compute various similarity metrics including KL divergence.
    Handles both regular tensors and DTensors.
    
    Args:
        tensor1, tensor2: Tensors to compare (can be regular tensors or DTensors)
        name1, name2: Names for the comparison methods
        
    Returns:
        Dictionary of comparison metrics
    """
    # Convert to local tensors for comparison
    t1 = tensor_to_local(tensor1)
    t2 = tensor_to_local(tensor2)
    
    # Basic checks
    if t1.shape != t2.shape:
        return {"error": f"Shape mismatch: {t1.shape} vs {t2.shape}"}
    
    # Compute basic metrics
    mse = torch.mean((t1 - t2) ** 2).item()
    mae = torch.mean(torch.abs(t1 - t2)).item()
    max_abs_diff = torch.max(torch.abs(t1 - t2)).item()
    
    # Cosine similarity
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
    
    # Relative error
    rel_error = torch.mean(torch.abs(t1 - t2) / (torch.abs(t1) + 1e-8)).item()
    
    # KL divergence calculation
    try:
        # Convert to probability distributions
        prob1, prob2 = normalize_for_kl(t1, t2)
        
        # Compute KL divergences (both directions)
        kl_1_to_2 = entropy(prob1, prob2)  # KL(P||Q)
        kl_2_to_1 = entropy(prob2, prob1)  # KL(Q||P)
        
        # Jensen-Shannon divergence (symmetric)
        js_divergence = jensenshannon(prob1, prob2) ** 2
        
    except Exception as e:
        print(f"Warning: KL divergence calculation failed: {e}")
        kl_1_to_2 = float('nan')
        kl_2_to_1 = float('nan')
        js_divergence = float('nan')
    
    return {
        "mse": mse,
        "mae": mae,
        "max_abs_diff": max_abs_diff,
        "cosine_similarity": cos_sim,
        "relative_error": rel_error,
        "kl_divergence_1_to_2": kl_1_to_2,
        "kl_divergence_2_to_1": kl_2_to_1,
        "js_divergence": js_divergence,
        "shape": t1.shape,
        "t1_stats": {"min": t1.min().item(), "max": t1.max().item(), "mean": t1.mean().item()},
        "t2_stats": {"min": t2.min().item(), "max": t2.max().item(), "mean": t2.mean().item()},
    }


def apply_fsdp_only(model, use_mixed_precision=True):
    """
    Apply FSDP to the model without TP.
    
    Args:
        model: The model to wrap with FSDP
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        The FSDP-wrapped model
    """
    if dist.is_initialized():
        world_size = dist.get_world_size()
        device_mesh = DeviceMesh("cuda", list(range(world_size)))
        
        # Define mixed precision policy
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) if use_mixed_precision else None
        
        # FSDP configuration
        fsdp_config = {"mesh": device_mesh}
        if mp_policy:
            fsdp_config["mp_policy"] = mp_policy
        
        # Wrap the model with FSDP
        fully_shard(model, **fsdp_config)
    else:
        # Single GPU case, no FSDP needed
        pass


def main():
    parser = argparse.ArgumentParser(description="Compare HuggingFace quantized loading approaches")
    parser.add_argument("--checkpoint-path", type=str, 
                        default="/data/users/jianiw/model/DeepSeek-V3.1-Base",
                        help="Path to HuggingFace checkpoint directory")
    parser.add_argument("--fsdp-size", type=int, default=1, help="FSDP parallel size")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--layer-id", type=int, default=0, help="Layer ID to test (default: 0)")
    args = parser.parse_args()

    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"Testing HuggingFace quantized loading approaches")
        print(f"Checkpoint path: {args.checkpoint_path}")
        print(f"Layer ID: {args.layer_id}")
    
    # Define the target keys for the MLP weights we want to compare
    target_keys = [
        f"model.layers.{args.layer_id}.mlp.gate_proj.weight",
        f"model.layers.{args.layer_id}.mlp.up_proj.weight", 
        f"model.layers.{args.layer_id}.mlp.down_proj.weight",
    ]
    
    if rank == 0:
        print(f"Target keys: {target_keys}")
    
    # Create a simple model for FSDP testing (this is just for demonstration)
    # In practice, you would use your actual model architecture
    model = SimpleMLP(dim=7168, hidden_dim=18432)
    model = model.cuda()
    
    # Apply FSDP only (no TP)
    apply_fsdp_only(model, use_mixed_precision=not args.no_mixed_precision)
    
    if rank == 0:
        print("Applied FSDP to model")
    
    try:
        # Method 1: Load original HF weights + manual dequantization
        if rank == 0:
            print("\n" + "="*60)
            print("METHOD 1: Original HF weights + Manual dequantization")
            print("="*60)
        
        manually_dequantized_weights = load_hf_weights_manually(args.checkpoint_path, target_keys)
    
        print("Successfully loaded weights with manual dequantization")
        for key in target_keys:
            if key in manually_dequantized_weights:
                tensor = manually_dequantized_weights[key]
                stats, _ = get_tensor_stats(tensor)
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Mean: {stats['mean']:.6f}")
    
    except Exception as e:
        if rank == 0:
            print(f"Method 1 failed: {e}")
        manually_dequantized_weights = {}
    
    try:
        # Method 2: Load with QuantizedHuggingFaceStorageReader
        if rank == 0:
            print("\n" + "="*60)
            print("METHOD 2: dcp.load(QuantizedHuggingFaceStorageReader)")
            print("="*60)
        
        quantized_reader_weights = load_hf_weights_with_quantized_reader(args.checkpoint_path, target_keys, args, model)
        
        
        print("Successfully loaded weights with QuantizedHFStorageReader, performing all-gather to get full_tensor")
        for key in target_keys:
            if key in quantized_reader_weights:
                tensor = quantized_reader_weights[key]
                stats, full_tensor = get_tensor_stats(tensor)
                # store full tensor instead of DTensor on each rank
                quantized_reader_weights[key] = full_tensor
                print(f"  {key}: shape={full_tensor.shape}, dtype={full_tensor.dtype}")
                print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Mean: {stats['mean']:.6f}")
    
    except Exception as e:
        if rank == 0:
            print(f"Method 2 failed: {e}")
        quantized_reader_weights = {}
    
    # Compare the results
    if rank == 0 and manually_dequantized_weights and quantized_reader_weights:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        comparison_results = {}
        
        for key in target_keys:
            if key in manually_dequantized_weights and key in quantized_reader_weights:
                print(f"\nComparing {key}:")
                
                tensor1 = manually_dequantized_weights[key]
                tensor2 = quantized_reader_weights[key]
                
                metrics = compare_tensors(tensor1, tensor2, "Manual Dequant", "QuantizedHFReader")
                comparison_results[key] = metrics
                
                if "error" in metrics:
                    print(f"  ERROR: {metrics['error']}")
                else:
                    print(f"  Shape: {metrics['shape']}")
                    print(f"  MSE: {metrics['mse']:.2e}")
                    print(f"  MAE: {metrics['mae']:.2e}")
                    print(f"  Max Abs Diff: {metrics['max_abs_diff']:.2e}")
                    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
                    print(f"  Relative Error: {metrics['relative_error']:.2e}")
                    print(f"  KL Divergence (Manual->QuantizedHF): {metrics['kl_divergence_1_to_2']:.6f}")
                    print(f"  KL Divergence (QuantizedHF->Manual): {metrics['kl_divergence_2_to_1']:.6f}")
                    print(f"  JS Divergence: {metrics['js_divergence']:.6f}")
                    print(f"  Manual Dequant Stats: min={metrics['t1_stats']['min']:.6f}, max={metrics['t1_stats']['max']:.6f}, mean={metrics['t1_stats']['mean']:.6f}")
                    print(f"  QuantizedHuggingFaceReader Stats: min={metrics['t2_stats']['min']:.6f}, max={metrics['t2_stats']['max']:.6f}, mean={metrics['t2_stats']['mean']:.6f}")
        
        # Save detailed comparison results
        output_file = "/data/users/jianiw/torchtitan/hf_quantized_comparison_results.json"
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"\nDetailed comparison results saved to: {output_file}")
    
    elif rank == 0:
        print("\nCannot perform comparison - one or both methods failed to load weights")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
