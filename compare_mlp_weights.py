#!/usr/bin/env python3
"""
Script to compare corresponding tensor pairs from two JSON weight files
and compute KL divergence between them.
"""

import json
import numpy as np
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import sys
import time

def load_json_weights(file_path):
    """Load weights from JSON file with progress indication."""
    print(f"Loading weights from {file_path}...")
    start_time = time.time()
    
    try:
        with open(file_path, 'r') as f:
            weights = json.load(f)
        
        load_time = time.time() - start_time
        print(f"Successfully loaded {len(weights)} weight tensors in {load_time:.2f} seconds")
        return weights
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def json_to_tensor(tensor_data):
    """Convert JSON tensor data back to numpy array."""
    if isinstance(tensor_data, dict):
        if 'values' in tensor_data:
            # Handle format with 'values' field
            data = np.array(tensor_data['values'])
            if 'shape' in tensor_data:
                data = data.reshape(tensor_data['shape'])
            return data
        elif 'data' in tensor_data:
            # Handle the format from our checkpoint saver
            data = np.array(tensor_data['data'])
            if 'shape' in tensor_data:
                data = data.reshape(tensor_data['shape'])
            return data
        else:
            print(f"Dict format found but no 'values' or 'data' field. Keys: {list(tensor_data.keys())}")
            return None
    elif isinstance(tensor_data, list):
        # Handle direct list format
        return np.array(tensor_data)
    else:
        print(f"Unknown tensor format: {type(tensor_data)}")
        return None

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
    
    return prob1, prob2

def compute_kl_divergence(tensor1, tensor2):
    """
    Compute KL divergence between two tensors.
    
    Returns:
        dict: Contains KL divergences and other similarity metrics
    """
    try:
        # Convert to probability distributions
        prob1, prob2 = normalize_for_kl(tensor1, tensor2)
        
        # Compute KL divergences (both directions)
        kl_1_to_2 = entropy(prob1, prob2)  # KL(P||Q)
        kl_2_to_1 = entropy(prob2, prob1)  # KL(Q||P)
        
        # Symmetric KL divergence (Jensen-Shannon divergence is better for this)
        js_divergence = jensenshannon(prob1, prob2) ** 2
        
        # Additional metrics
        mse = np.mean((tensor1 - tensor2) ** 2)
        max_abs_diff = np.max(np.abs(tensor1 - tensor2))
        cosine_sim = np.dot(tensor1.flatten(), tensor2.flatten()) / (
            np.linalg.norm(tensor1.flatten()) * np.linalg.norm(tensor2.flatten())
        )
        
        return {
            'kl_divergence_1_to_2': kl_1_to_2,
            'kl_divergence_2_to_1': kl_2_to_1,
            'js_divergence': js_divergence,
            'mse': mse,
            'max_abs_diff': max_abs_diff,
            'cosine_similarity': cosine_sim
        }
    except Exception as e:
        print(f"Error computing divergence: {e}")
        return None

def apply_key_mapping(key, mapping_dict, reverse=False):
    """Apply key mapping using the provided mapping dictionary."""
    if not mapping_dict:
        return key
    
    for pattern, replacement in mapping_dict.items():
        if reverse:
            # Swap pattern and replacement for reverse mapping
            pattern, replacement = replacement, pattern
        
        # Handle layer number extraction and replacement
        if "{}" in pattern and "{}" in replacement:
            # Extract layer number from the key using the pattern
            import re
            # Convert pattern to regex by escaping special chars and replacing {} with capture group
            pattern_regex = re.escape(pattern).replace("\\{\\}", r"(\d+)")
            match = re.match(pattern_regex, key)
            if match:
                layer_num = match.group(1)
                return replacement.format(layer_num)
        elif pattern == key:
            return replacement
    
    return key

def find_matching_keys(weights1, weights2, key_mapping=None):
    """Find keys that exist in both weight dictionaries, optionally using key mapping."""
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    
    if key_mapping:
        print("Using key mapping to find corresponding tensors...")
        # Create mapping pairs
        mapped_pairs = []
        
        for key1 in keys1:
            # Try to map key1 to see if it exists in keys2
            mapped_key = apply_key_mapping(key1, key_mapping)
            if mapped_key in keys2:
                mapped_pairs.append((key1, mapped_key))
            else:
                # Try reverse mapping
                for key2 in keys2:
                    reverse_mapped = apply_key_mapping(key2, key_mapping, reverse=True)
                    if reverse_mapped == key1:
                        mapped_pairs.append((key1, key2))
                        break
        
        print(f"Found {len(mapped_pairs)} mapped tensor pairs")
        
        # Show unmapped keys
        mapped_keys1 = set(pair[0] for pair in mapped_pairs)
        mapped_keys2 = set(pair[1] for pair in mapped_pairs)
        
        unmapped_1 = keys1 - mapped_keys1
        unmapped_2 = keys2 - mapped_keys2
        
        if unmapped_1:
            print(f"Unmapped keys in file 1: {list(unmapped_1)[:5]}")
        if unmapped_2:
            print(f"Unmapped keys in file 2: {list(unmapped_2)[:5]}")
        
        return mapped_pairs
    
    else:
        # Original exact key matching
        common_keys = keys1.intersection(keys2)
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        print(f"Common keys: {len(common_keys)}")
        print(f"Keys only in file 1: {len(only_in_1)}")
        print(f"Keys only in file 2: {len(only_in_2)}")
        
        if only_in_1:
            print("Keys only in file 1:", list(only_in_1)[:5])  # Show first 5
        if only_in_2:
            print("Keys only in file 2:", list(only_in_2)[:5])  # Show first 5
        
        # Return as pairs for consistency
        return [(key, key) for key in sorted(common_keys)]

def main():
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Compare tensor weights between two JSON files')
    parser.add_argument('--file1', default="/data/users/jianiw/model/mlp_weights_check1.json", 
                        help='Path to first JSON file')
    parser.add_argument('--file2', default="/data/users/jianiw/torchtitan/titan_online_conversion_layer1.json",
                        help='Path to second JSON file') 
    parser.add_argument('--no-mapping', action='store_true', 
                        help='Disable key mapping and use exact key matching')
    
    args = parser.parse_args()
    
    file1 = args.file1
    file2 = args.file2
    
    # Optional key mapping between file1 and file2
    # Set to None to disable mapping and use exact key matching
    if args.no_mapping:
        key_mapping = None
        print("Key mapping disabled - using exact key matching")
    else:
        key_mapping = {
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight", 
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        }
        print("Using key mapping between different naming conventions")
    
    print(f"Comparing files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    # Load both weight files
    weights1 = load_json_weights(file1)
    if weights1 is None:
        return 1
        
    weights2 = load_json_weights(file2)
    if weights2 is None:
        return 1
    
    # Find matching keys (returns list of tuples: (key1, key2))
    matching_pairs = find_matching_keys(weights1, weights2, key_mapping)
    
    if not matching_pairs:
        print("No matching tensor pairs found between the two files!")
        return 1
    
    print(f"\nComparing {len(matching_pairs)} tensor pairs...")
    print("=" * 80)
    
    results = []
    
    for i, (key1, key2) in enumerate(matching_pairs):
        if key_mapping:
            print(f"\nProcessing {i+1}/{len(matching_pairs)}: {key1} -> {key2}")
        else:
            print(f"\nProcessing {i+1}/{len(matching_pairs)}: {key1}")
        
        # Convert JSON data to tensors
        tensor1 = json_to_tensor(weights1[key1])
        tensor2 = json_to_tensor(weights2[key2])
        
        if tensor1 is None or tensor2 is None:
            print(f"  Skipping {key1} -> {key2}: failed to convert to tensor")
            continue
        
        # Check shapes match
        if tensor1.shape != tensor2.shape:
            print(f"  Skipping {key1} -> {key2}: shape mismatch {tensor1.shape} vs {tensor2.shape}")
            continue
        
        print(f"  Shape: {tensor1.shape}")
        print(f"  Min/Max values - File1: [{tensor1.min():.6f}, {tensor1.max():.6f}]")
        print(f"  Min/Max values - File2: [{tensor2.min():.6f}, {tensor2.max():.6f}]")
        
        # Compute metrics
        metrics = compute_kl_divergence(tensor1, tensor2)
        if metrics is None:
            print(f"  Failed to compute metrics for {key1} -> {key2}")
            continue
        
        print(f"  KL Divergence (1->2): {metrics['kl_divergence_1_to_2']:.6f}")
        print(f"  KL Divergence (2->1): {metrics['kl_divergence_2_to_1']:.6f}")
        print(f"  JS Divergence: {metrics['js_divergence']:.6f}")
        print(f"  MSE: {metrics['mse']:.6e}")
        print(f"  Max Abs Diff: {metrics['max_abs_diff']:.6e}")
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        
        results.append({
            'key1': key1,
            'key2': key2,
            'shape': tensor1.shape,
            **metrics
        })
    
    # Summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        kl_1_to_2_values = [r['kl_divergence_1_to_2'] for r in results]
        kl_2_to_1_values = [r['kl_divergence_2_to_1'] for r in results]
        js_values = [r['js_divergence'] for r in results]
        mse_values = [r['mse'] for r in results]
        cosine_values = [r['cosine_similarity'] for r in results]
        
        print(f"KL Divergence (1->2) - Mean: {np.mean(kl_1_to_2_values):.6f}, Std: {np.std(kl_1_to_2_values):.6f}")
        print(f"KL Divergence (2->1) - Mean: {np.mean(kl_2_to_1_values):.6f}, Std: {np.std(kl_2_to_1_values):.6f}")
        print(f"JS Divergence - Mean: {np.mean(js_values):.6f}, Std: {np.std(js_values):.6f}")
        print(f"MSE - Mean: {np.mean(mse_values):.6e}, Std: {np.std(mse_values):.6e}")
        print(f"Cosine Similarity - Mean: {np.mean(cosine_values):.6f}, Std: {np.std(cosine_values):.6f}")
        
        # Find tensors with highest and lowest divergences
        max_kl_idx = np.argmax(kl_1_to_2_values)
        min_kl_idx = np.argmin(kl_1_to_2_values)
        
        print(f"\nHighest KL Divergence: {results[max_kl_idx]['key1']} -> {results[max_kl_idx]['key2']} ({kl_1_to_2_values[max_kl_idx]:.6f})")
        print(f"Lowest KL Divergence: {results[min_kl_idx]['key1']} -> {results[min_kl_idx]['key2']} ({kl_1_to_2_values[min_kl_idx]:.6f})")
        
        # Save detailed results
        output_file = "/data/users/jianiw/torchtitan/tensor_comparison_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
