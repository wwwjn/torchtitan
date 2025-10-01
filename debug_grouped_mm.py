#!/usr/bin/env python3

import torch
import os

def load_saved_data(data_file, grad_file):
    """Load the saved forward inputs and backward gradients from PyTorch files."""
    
    print(f"[DEBUG] Loading data from {data_file}")
    print(f"[DEBUG] Loading gradients from {grad_file}")
    
    # Check file sizes first
    data_size = os.path.getsize(data_file)
    grad_size = os.path.getsize(grad_file)
    print(f"[DEBUG] Data file size: {data_size} bytes")
    print(f"[DEBUG] Grad file size: {grad_size} bytes")
    
    # Load forward inputs
    forward_data = torch.load(data_file, map_location='cpu')
    
    # Load backward gradients  
    backward_data = torch.load(grad_file, map_location='cpu')
    
    return forward_data, backward_data

def reproduce_grouped_mm(forward_data, backward_data):
    """Reproduce the grouped_mm operation with saved data."""
    
    # Extract data from saved tensors
    h = forward_data["h"]
    w2 = forward_data["w2"] 
    offsets = forward_data["offsets"]
    target_grad = backward_data["grad"]
    layer_id = forward_data["layer_id"]
    
    print(f"[REPRO] Layer {layer_id} - Input shapes:")
    print(f"  h: {h.shape}, dtype: {h.dtype}")
    print(f"  w2: {w2.shape}, dtype: {w2.dtype}")
    print(f"  offsets: {offsets.shape if offsets is not None else None}")
    print(f"  target_grad: {target_grad.shape}, dtype: {target_grad.dtype}")
    
    # Convert to the right dtypes and enable gradients
    h = h.to(torch.bfloat16).requires_grad_(True)
    w2 = w2.to(torch.bfloat16).requires_grad_(True) 
    offsets_tensor = offsets.to(torch.int32) if offsets is not None else None
    target_grad = target_grad.to(torch.bfloat16)
    
    # Forward pass
    print(f"\n[REPRO] Layer {layer_id} - Running forward pass...")
    out = torch._grouped_mm(h, w2, offs=offsets_tensor)
    print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"  Output has_nan: {torch.isnan(out).any()}")
    print(f"  Output has_inf: {torch.isinf(out).any()}")
    print(f"  Output norm: {out.norm().item():.6f}")
    
    # Backward pass with the stored gradient
    print(f"\n[REPRO] Layer {layer_id} - Running backward pass...")
    try:
        out.backward(target_grad)
        print(f"  Backward completed successfully!")
        
        # Check gradients
        if h.grad is not None:
            print(f"  h.grad: shape={h.grad.shape}, has_nan={torch.isnan(h.grad).any()}, norm={h.grad.norm().item():.6f}")
        else:
            print(f"  h.grad is None")
            
        if w2.grad is not None:
            print(f"  w2.grad: shape={w2.grad.shape}, has_nan={torch.isnan(w2.grad).any()}, norm={w2.grad.norm().item():.6f}")
        else:
            print(f"  w2.grad is None")
            
    except Exception as e:
        print(f"  ERROR during backward: {e}")
        print(f"  Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the reproduction."""
    
    debug_dir = "/tmp/grouped_mm_debug"
    
    # Find the saved files
    data_files = []
    grad_files = []
    
    if os.path.exists(debug_dir):
        for filename in os.listdir(debug_dir):
            if filename.startswith("grouped_mm_data_layer_") and filename.endswith(".pt"):
                layer_id = filename.split("_")[-1].split(".")[0]
                data_file = os.path.join(debug_dir, filename)
                grad_file = os.path.join(debug_dir, f"grouped_mm_gradient_layer_{layer_id}.pt")
                
                if os.path.exists(grad_file):
                    data_files.append(data_file)
                    grad_files.append(grad_file)
                    print(f"Found data pair for layer {layer_id}")
    
    if not data_files:
        print("No saved data files found! Make sure to run the model training first to generate the debug files.")
        return
    
    # Process each saved layer
    for data_file, grad_file in zip(data_files, grad_files):
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(data_file)}")
        print(f"{'='*60}")
        
        try:
            forward_data, backward_data = load_saved_data(data_file, grad_file)
            reproduce_grouped_mm(forward_data, backward_data)
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
