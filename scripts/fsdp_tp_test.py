#!/usr/bin/env python3
"""
Simple test script to create a model with one MLP layer and apply both FSDP and TP.
Demonstrates how to create a 2D device mesh with TP=2 and FSDP=2.

Test loading [Shard(0), Shard(0)] tensor to the fc1 & fc2 weight

To run this script: torchrun --nproc_per_node=4 scripts/fsdp_tp_test.py --tp-size=2 --fsdp-size=2
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard
import hashlib
import numpy as np

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)


def calculate_hash(tensor_data: np.ndarray) -> str:
    """Calculate SHA-256 hash of a tensor."""
    tensor_bytes = tensor_data.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

class SimpleMLP(nn.Module):
    """A simple MLP with configurable hidden dimensions."""
    
    def __init__(self, input_dim=1024, hidden_dim=4096, output_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


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


def create_2d_mesh(fsdp_size, tp_size):
    """
    Create a 2D device mesh with FSDP and TP dimensions.
    
    Args:
        fsdp_size: Size of the FSDP dimension
        tp_size: Size of the TP dimension
        
    Returns:
        DeviceMesh: A 2D device mesh with shape [FSDP_SIZE, TP_SIZE]
    """
    world_size = dist.get_world_size()
    if world_size != fsdp_size * tp_size:
        raise ValueError(f"World size ({world_size}) must equal FSDP size ({fsdp_size}) * TP size ({tp_size})")
    
    # Get all ranks
    ranks = list(range(world_size))
    
    # Reshape into 2D mesh: [FSDP_SIZE, TP_SIZE]
    # This creates a mesh where the first dimension is for FSDP and the second for TP
    mesh_2d = torch.arange(world_size).reshape(fsdp_size, tp_size)
    
    # Create the device mesh with dimension names
    device_mesh = DeviceMesh("cuda", mesh_2d, mesh_dim_names=["dp", "tp"])
    
    rank = dist.get_rank()
    if rank == 0:
        print(f"Created 2D device mesh with shape {device_mesh.shape}:")
        print(mesh_2d)
    
    return device_mesh


def apply_tensor_parallel(model, device_mesh):
    """
    Apply Tensor Parallelism to the model using a device mesh.
    
    Args:
        model: The model to parallelize
        device_mesh: The device mesh to use for parallelization
        
    Returns:
        The parallelized model
    """
    # For 2D mesh, we use the second dimension (dim=1) for TP
    # Create a submesh for TP by selecting the appropriate dimension
    # In older PyTorch versions, we need to use the mesh directly
    
    # Define parallel styles for each layer
    parallel_styles = {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    
    # Parallelize the model using the device mesh
    # Pass the full mesh and specify which dimension to use for TP
    tp_model = parallelize_module(model, device_mesh, parallel_styles)
    
    return tp_model


def apply_fsdp(model, device_mesh, use_mixed_precision=True):
    """
    Apply FSDP to the model using a device mesh.
    
    Args:
        model: The model to wrap with FSDP
        device_mesh: The device mesh to use for FSDP
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        The FSDP-wrapped model
    """
    
    # Define mixed precision policy
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    
    # Use the mesh directly with the mesh_dim parameter
    fsdp_config = {
        "mesh": device_mesh, 
        "mp_policy": mp_policy,
    }
    
    # Wrap the model with FSDP using the device mesh
    fully_shard(model, **fsdp_config)
    


def main():
    parser = argparse.ArgumentParser(description="Test FSDP and TP on a simple MLP")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int, default=1024, help="Output dimension")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--fsdp-size", type=int, default=2, help="FSDP parallel size")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    # Check if TP size * FSDP size equals world size
    if world_size != args.tp_size * args.fsdp_size:
        if rank == 0:
            print(f"Error: TP size ({args.tp_size}) * FSDP size ({args.fsdp_size}) must equal world size ({world_size})")
            print(f"For example, with 4 GPUs, you can use TP=2 and FSDP=2")
        dist.destroy_process_group()
        return
    
    # Create model
    model = SimpleMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    
    # Move model to GPU
    model = model.cuda()
    
    if rank == 0:
        print(f"Created model with input_dim={args.input_dim}, hidden_dim={args.hidden_dim}, output_dim={args.output_dim}")
    
    # Create 2D device mesh
    device_mesh = create_2d_mesh(args.fsdp_size, args.tp_size)
    print(f"rank {rank} device_mesh: {device_mesh}")
    
    dp_mesh = device_mesh["dp"]
    tp_mesh = device_mesh["tp"]

    # Apply TP first
    model = apply_tensor_parallel(model, tp_mesh)
    
    if rank == 0:
        print("Applied Tensor Parallelism")
    
    # Apply FSDP
    apply_fsdp(model, dp_mesh, use_mixed_precision=not args.no_mixed_precision)
    
    if rank == 0:
        print("Applied FSDP")

    # Create dummy input and target
    torch.manual_seed(42)
    # Create the same full tensor on all ranks
    fc1_weight_tensor = torch.randn(args.hidden_dim, args.input_dim , device=torch.cuda.current_device(), dtype=torch.float32)
    fc1_dist_tensor = torch.distributed.tensor.distribute_tensor(
        fc1_weight_tensor, device_mesh, placements=[Shard(0), Shard(0)]
    )
    print(f"rank {rank} dtype {fc1_weight_tensor.dtype}, dtensor placement {fc1_dist_tensor.placements}, fc1_weight_tensor hash: {calculate_hash(fc1_weight_tensor.detach().cpu().numpy())}")

    fc2_weight_tensor = torch.randn(args.output_dim, args.hidden_dim, device=torch.cuda.current_device(), dtype=torch.float32)
    fc2_dist_tensor = torch.distributed.tensor.distribute_tensor(
        fc2_weight_tensor, device_mesh, placements=[Shard(0), Shard(0)]
    )
    print(f"rank {rank} dtype {fc2_weight_tensor.dtype}, dtensor placement {fc2_dist_tensor.placements}, fc2_weight_tensor hash: {calculate_hash(fc2_weight_tensor.detach().cpu().numpy())}")

    # Now, try to load the (Shard(0), Shard(0)) tensor to the fc1 weight
    state_dict = {
        "fc1.weight": fc1_dist_tensor,
        "fc2.weight": fc2_dist_tensor,
    }

    def load_state_dict(model, state_dict):
        import functools
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        func(model)

    def get_state_dict(model):
        state_dict = {k: v for k, v in get_model_state_dict(model).items()}
        return state_dict


    load_state_dict(model, state_dict)
    
    # Check After loading
    loaded_weights = get_state_dict(model)

    fc1_dist_tensor = loaded_weights["fc1.weight"]
    fc2_dist_tensor = loaded_weights["fc2.weight"]
    fc1_full_tensor = fc1_dist_tensor.full_tensor()
    fc2_full_tensor = fc2_dist_tensor.full_tensor()
    print(f"rank {rank} loaded fc1_full_tensor dtype {fc1_full_tensor.dtype}, dtensor placement {fc1_dist_tensor.placements}, hash: {calculate_hash(fc1_full_tensor.detach().cpu().numpy())}")
    print(f"rank {rank} loaded fc2_full_tensor dtype {fc2_full_tensor.dtype}, dtensor placement {fc2_dist_tensor.placements} hash: {calculate_hash(fc2_full_tensor.detach().cpu().numpy())}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
