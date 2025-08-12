import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
)
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed as dist
import os
from torch.distributed.fsdp import fully_shard

from torchtitan.models.llama3.model.model import Transformer


class MLP(nn.Module):
    def __init__(self, dim = None, hidden_dim=None):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class FakeModel(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()  # Fixed: removed extra underscore
        # Only 1 layer
        self.layers = nn.ModuleDict(
            {
                "0": DecoderLayer(dim, hidden_dim)
            }
        )
    def forward(self, x: torch.Tensor):
        for layer in self.layers.values():
            x = layer(x, self.freqs_cis)
        return x

def apply_fsdp(model: nn.Module, dp_mesh: DeviceMesh):
    fsdp_config = {"mesh": dp_mesh} 
    for layer_id, layer in model.layers.items():
        fully_shard(
            layer,
            **fsdp_config,
        )


def init_distributed():
    """Initialize the distributed environment."""
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Get local rank and world size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

def main():

    # Load the weights from HF safetensor format
    hf_checkpoint_path = "/data/users/jianiw/fake-weights"
    # (pytorch-3.12) [jianiw@devvm7508]/data/users/jianiw/fake-weights% ll -a
    # total 5111472
    # drwxr-xr-x 1 jianiw users        122 Aug 11 17:43 .
    # drwxr-xr-x 1 jianiw users        676 Aug 11 17:42 ..
    # -rw-r--r-- 1 jianiw users 5234139343 Aug 11 17:42 model-00001-of-000163.safetensors
    # -rw-r--r-- 1 jianiw users        584 Aug 11 17:43 model.safetensors.index.json
    
    # Intructions:  
    # model-00001-of-000163.safetensors can be downloaded here: https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/model-00001-of-000163.safetensors
    # model.safetensors.index.json can be find in the same folder as this script
    
    local_rank, world_size = init_distributed()
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use the actual world_size for the mesh
    mesh_devices = list(range(world_size))
    mesh = DeviceMesh(device, mesh_devices)
    
    model = FakeModel(dim=7168, hidden_dim=18432)
    apply_fsdp(model=model, dp_mesh=mesh)
    print(model)

    state_dict = {"model": model.state_dict()}
    # TODO: Need to support dequantize, and then load into state dict
    # Now it doesn't perform dequantize, it will only load the wegiths without scaled by weight_scale_inv
    dcp.load(state_dict, storage_reader=HuggingFaceStorageReader(path=hf_checkpoint_path))
    if local_rank == 0:
        print(f"Loaded state_dict: {state_dict}")
    model.load_state_dict(state_dict["model"])

if __name__ == "__main__":
    main()
