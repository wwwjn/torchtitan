#!/usr/bin/env python3
"""
Simple script to scan a folder for safetensor files and check if a specific key exists in them.
If the key exists, it prints information about the tensor including its hash.
"""

import argparse
import hashlib
import os
import sys
from typing import List


try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Install with 'pip install numpy'")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors library is required. Install with 'pip install safetensors'")
    sys.exit(1)

# Try to import PyTorch for bfloat16 support
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch is available. Will use it for bfloat16 support.")
except ImportError:
    print("Warning: PyTorch is not available. bfloat16 tensors may not be handled correctly.")


def calculate_hash(tensor_data: np.ndarray) -> str:
    """Calculate SHA-256 hash of a tensor."""
    tensor_bytes = tensor_data.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def scan_folder(folder_path: str, key: str) -> List[str]:
    """
    Scan a folder for safetensor files and check if they contain the specified key.
    
    Args:
        folder_path: Path to the folder containing safetensor files
        key: The tensor key to look for
        
    Returns:
        List of files that contain the key
    """
    matching_files = []
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return matching_files
    
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.safetensors')]
    if not files:
        print(f"No safetensor files found in '{folder_path}'")
        return matching_files
    
    print(f"Found {len(files)} safetensor files in '{folder_path}'")
    
    # Check each file for the key
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            # Try with NumPy as fallback
            with safe_open(file_path, framework="pt") as f:
                if key in f.keys():
                    matching_files.append(file_path)
                    print(f"Found key '{key}' in file: {filename}")
                    # Print tensor value
                    try:
                        tensor = f.get_tensor(key)
                        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
                        tensor_np = tensor.to(torch.float32).detach().cpu().numpy()
                        print(f"Hash: {calculate_hash(tensor_np)}")
                    except Exception as e:
                        print(f"Error getting tensor details: {e}")
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
    
    return matching_files


def print_tensor_info(file_path: str, key: str) -> None:
    """
    Print information about a tensor in a safetensor file.
    
    Args:
        file_path: Path to the safetensor file
        key: The tensor key to extract
    """
    try:
        # Try using PyTorch first if available (handles bfloat16)
        if TORCH_AVAILABLE:
            try:
                from safetensors.torch import load_file
                tensors = load_file(file_path, device="cpu")
                if key in tensors:
                    tensor = tensors[key]
                    
                    # Convert to numpy for consistent handling
                    tensor_np = tensor.detach().cpu().numpy()
                    
                    # Calculate hash
                    # tensor_hash = calculate_hash(tensor_np)
                    
                    # Print tensor information
                    print(f"\nFile: {os.path.basename(file_path)}")
                    print(f"Tensor key: {key}")
                    print(f"Shape: {tensor.shape}")
                    print(f"Dtype: {tensor.dtype} (PyTorch)")
                    # print(f"SHA-256 Hash: {tensor_hash}")
                    print(f"First 10 values: {tensor.flatten()[:10].tolist()}")
                    print("-" * 50)
                    return
            except Exception as e:
                print(f"Warning: PyTorch loading failed, falling back to NumPy: {e}")
        
        # Fall back to NumPy if PyTorch is not available or failed
        with safe_open(file_path, framework="numpy") as f:
            tensor = f.get_tensor(key)
            
            # Calculate hash
            tensor_hash = calculate_hash(tensor)
            
            # Print tensor information
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"Tensor key: {key}")
            print(f"Shape: {tensor.shape}")
            print(f"Dtype: {tensor.dtype}")
            print(f"SHA-256 Hash: {tensor_hash}")
            print(f"First 10 values: {tensor.flatten()[:10]}")
            print("-" * 50)
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Scan a folder for safetensor files and check for a specific key'
    )
    parser.add_argument('folder_path', type=str, help='Path to the folder containing safetensor files')
    parser.add_argument('--key', type=str, default='model.layers.0.self_attn.q_proj.weight',
                        help='Tensor key to look for (default: model.layers.0.self_attn.q_proj.weight)')
    parser.add_argument('--use-torch', action='store_true', default=True,
                        help='Force using PyTorch for loading tensors (better bfloat16 support)')
    args = parser.parse_args()
    
    # If user requested PyTorch but it's not available
    if args.use_torch and not TORCH_AVAILABLE:
        print("Error: --use-torch option specified but PyTorch is not available.")
        print("Install PyTorch with 'pip install torch' and try again.")
        return 1
    
    # Scan the folder for files containing the key
    matching_files = scan_folder(args.folder_path, args.key)
    
    if not matching_files:
        print(f"No files found containing the key '{args.key}'")
        return 1
    
    print(f"\nFound {len(matching_files)} files containing the key '{args.key}'")
    
    # Print information about the tensor in each matching file
    for file_path in matching_files:
        print_tensor_info(file_path, args.key)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
