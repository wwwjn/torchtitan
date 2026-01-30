# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for vLLM integration with TorchTitan models.

This script tests vLLM running TorchTitan model vs vLLM running native Qwen3 model.
Both models use vLLM's attention and forward context, and we compare their outputs.
"""

import os
import tempfile
from dataclasses import dataclass

import pytest
import torch

# Import vLLM (skip tests if not available)
vllm = pytest.importorskip("vllm")

from torch.distributed._tensor import DTensor
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.config import set_current_vllm_config
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backend import CommonAttentionMetadata

# Import debug control for vLLM Qwen3 model
from vllm.model_executor.models.qwen3 import enable_qwen3_debug, disable_qwen3_debug

# TorchTitan imports
from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
from torchtitan.experiments.rl.unified.models.vllm_wrapper import TorchTitanVLLMModelWrapper
from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter

BLOCK_SIZE = 32

# Track if distributed environment has been initialized
_distributed_initialized = False


def get_distributed_info():
    """Get distributed environment info from environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, rank, local_rank


def ensure_distributed_initialized():
    """Initialize distributed environment if not already done."""
    global _distributed_initialized

    # Check if already initialized
    if model_parallel_is_initialized() or _distributed_initialized:
        return

    world_size, rank, local_rank = get_distributed_info()

    temp_file = tempfile.mkstemp()[1]
    os.environ["DIST_INIT_METHOD"] = f"file://{temp_file}"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=os.environ["DIST_INIT_METHOD"],
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(world_size, 1)  # TP=world_size, PP=1
    _distributed_initialized = True

    if rank == 0:
        print(f"Initialized distributed environment: TP={world_size}")


@dataclass
class TestContext:
    """Context object holding setup results for tests."""

    vllm_native_model: torch.nn.Module
    torchtitan_vllm_model: torch.nn.Module
    vllm_config: object
    tokens: torch.Tensor
    positions: torch.Tensor
    attn_metadata: FlashAttentionMetadata
    model_dim: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int
    hidden_dim: int
    seq_len: int
    q_size: int
    kv_size: int


def setup_models_and_inputs(
    model_name: str = "Qwen/Qwen3-0.6B",
    seq_len: int = 8,
) -> TestContext:
    """
    Set up vLLM native and TorchTitan models with shared weights and inputs.

    This function:
    1. Initializes distributed environment
    2. Creates vLLM config and loads native Qwen3 model
    3. Creates TorchTitan model with vLLM wrapper
    4. Copies weights from vLLM native to TorchTitan
    5. Creates input tokens, positions, and attention metadata
    6. Sets up KV caches

    Returns:
        TestContext containing both models, inputs, and configuration.
    """
    # Setup distributed environment
    ensure_distributed_initialized()

    # Get world size for tensor parallel
    world_size, rank, local_rank = get_distributed_info()
    device = f"cuda:{local_rank}"

    # Create vLLM EngineArgs for native Qwen3 model
    engine_args = EngineArgs(
        model=model_name,
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        tensor_parallel_size=world_size,
        dtype="bfloat16",
    )

    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)

    # Extract model dimensions from vLLM model config
    hf_config = vllm_config.model_config.hf_config
    model_dim = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    num_layers = hf_config.num_hidden_layers
    head_dim = getattr(hf_config, "head_dim", model_dim // num_heads)
    hidden_dim = hf_config.intermediate_size

    if rank == 0:
        print(f"\n=== Model dimensions from {model_name} ===")
        print(f"model_dim={model_dim}, vocab_size={vocab_size}, num_heads={num_heads}, "
              f"num_kv_heads={num_kv_heads}, num_layers={num_layers}, hidden_dim={hidden_dim}, "
              f"head_dim={head_dim}")

    # Create TorchTitan model args matching vLLM model config
    model_args = Qwen3ModelArgs(
        dim=model_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=seq_len * 2,
        qk_norm=getattr(hf_config, "qk_norm", True),
        depth_init=False,
        attn_type="sdpa",
        attn_mask_type="causal",
    )

    # Load vLLM native Qwen3 model
    if rank == 0:
        print("\n=== Loading vLLM native Qwen3 model ===")
    loader = get_model_loader(vllm_config.load_config)
    vllm_native_model = loader.load_model(vllm_config, vllm_config.model_config)
    if rank == 0:
        print(f"vLLM native model: {type(vllm_native_model)}")

    # Create TorchTitan model with vLLM wrapper
    if rank == 0:
        print("\n=== Creating TorchTitan model with vLLM wrapper ===")
    with set_current_vllm_config(vllm_config):
        torchtitan_vllm_model = TorchTitanVLLMModelWrapper(
            model_cls=Qwen3Model,
            model_args=model_args,
            state_dict_adapter=Qwen3StateDictAdapter,
            parallelize_fn=parallelize_qwen3,
            vllm_config=vllm_config,
        )
    torchtitan_vllm_model.to(device=device, dtype=torch.bfloat16)
    torchtitan_vllm_model.eval()

    # Replace TorchTitan's RMSNorm with vLLM's RMSNorm for numerical equivalence
    replace_rmsnorm_with_vllm(torchtitan_vllm_model.model, vllm_config, rank)

    # Copy weights from vLLM native model to TorchTitan model
    if rank == 0:
        print("\n=== Copying weights from vLLM native to TorchTitan ===")

    # Calculate sizes for splitting qkv_proj
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Convert vLLM native model state dict to HF format
    hf_state_dict = {}
    param_names = [name for name, _ in vllm_native_model.named_parameters()]
    if rank == 0:
        print(f"Sample vLLM native model parameter names: {param_names[:5]}")

    has_lm_head = any("lm_head" in name for name in param_names)
    if rank == 0:
        print(f"Has lm_head in vLLM native model: {has_lm_head}")

    for name, param in vllm_native_model.named_parameters():
        tensor = param.data.clone()

        def ensure_model_prefix(n):
            if n.startswith("model."):
                return n
            return f"model.{n}"

        if "qkv_proj" in name:
            expected_size = q_size + kv_size + kv_size
            actual_size = tensor.shape[0]
            if rank == 0:
                print(f"  {name}: shape={tuple(tensor.shape)}, expected_size={expected_size}, actual_size={actual_size}")
            if actual_size != expected_size:
                raise ValueError(
                    f"QKV tensor size mismatch for {name}: expected {expected_size} "
                    f"(q={q_size} + k={kv_size} + v={kv_size}), got {actual_size}. "
                    f"Check if head_dim is correct (current: {head_dim})"
                )
            q, k, v = tensor.split([q_size, kv_size, kv_size], dim=0)
            q_name = name.replace("qkv_proj", "q_proj")
            k_name = name.replace("qkv_proj", "k_proj")
            v_name = name.replace("qkv_proj", "v_proj")
            hf_state_dict[ensure_model_prefix(q_name)] = q
            hf_state_dict[ensure_model_prefix(k_name)] = k
            hf_state_dict[ensure_model_prefix(v_name)] = v

        elif "gate_up_proj" in name:
            gate, up = tensor.chunk(2, dim=0)
            gate_name = name.replace("gate_up_proj", "gate_proj")
            up_name = name.replace("gate_up_proj", "up_proj")
            hf_state_dict[ensure_model_prefix(gate_name)] = gate
            hf_state_dict[ensure_model_prefix(up_name)] = up

        elif name.startswith("lm_head"):
            hf_state_dict[name] = tensor

        else:
            hf_state_dict[ensure_model_prefix(name)] = tensor

    if rank == 0:
        print(f"\nConverted {len(hf_state_dict)} weight keys to HF format")

    # Handle weight tying: Because vLLM's native model might not explicitly have lm_head.weight in its parameters (it's tied)
    if "lm_head.weight" not in hf_state_dict and "model.embed_tokens.weight" in hf_state_dict:
        hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"].clone()
        if rank == 0:
            print("Applied weight tying: copied model.embed_tokens.weight -> lm_head.weight")

    # Use the adapter to convert HF weights to TorchTitan format
    adapter = Qwen3StateDictAdapter(
        model_args=model_args,
        hf_assets_path=None,
    )
    torchtitan_state_dict = adapter.from_hf(hf_state_dict)

    # Load weights into TorchTitan model
    torchtitan_params = dict(torchtitan_vllm_model.model.named_parameters())

    model_keys = set(torchtitan_params.keys())
    converted_keys = set(torchtitan_state_dict.keys())
    missing_in_converted = model_keys - converted_keys
    extra_in_converted = converted_keys - model_keys

    if rank == 0:
        print(f"\n=== Weight key analysis ===")
        print(f"TorchTitan model has {len(model_keys)} parameters")
        print(f"Converted state dict has {len(converted_keys)} keys")
        if missing_in_converted:
            print(f"Missing keys (in model but not in converted): {sorted(missing_in_converted)}")
        if extra_in_converted:
            print(f"Extra keys (in converted but not in model): {sorted(extra_in_converted)}")

    copied_count = 0
    for name, param in torchtitan_state_dict.items():
        if name in torchtitan_params:
            torchtitan_params[name].data.copy_(param.to(torchtitan_params[name].device))
            copied_count += 1
        else:
            if rank == 0:
                print(f"Warning: Key '{name}' in converted state dict not found in model")

    if rank == 0:
        print(f"Successfully copied {copied_count}/{len(torchtitan_state_dict)} weights")

    # Create input tokens
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (seq_len,), device=device)
    positions = torch.arange(seq_len, device=device)
    if rank == 0:
        print(f"\nInput tokens shape: {tokens.shape}")

    # Build attention metadata
    attn_metadata = build_attn_metadata(seq_len, local_rank)

    # Bind KV caches for all attention layers in static_forward_context
    num_blocks = 30
    static_ctx = vllm_config.compilation_config.static_forward_context
    if rank == 0:
        print(f"\n=== Setting up KV caches ===")

    for layer_name, layer in static_ctx.items():
        if hasattr(layer, 'kv_cache'):
            kv_cache = torch.zeros(
                (2, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            layer.kv_cache = [kv_cache]

    if rank == 0:
        print("=== Setup complete ===\n")

    return TestContext(
        vllm_native_model=vllm_native_model,
        torchtitan_vllm_model=torchtitan_vllm_model,
        vllm_config=vllm_config,
        tokens=tokens,
        positions=positions,
        attn_metadata=attn_metadata,
        model_dim=model_dim,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        q_size=q_size,
        kv_size=kv_size,
    )


def print_tensor_stats(name: str, tensor: torch.Tensor, prefix: str = "") -> None:
    """Print tensor statistics in a consistent format."""
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    t = tensor.float()  # Convert to float for stats
    flat = t.flatten()  # Use float tensor for first_10 to avoid bfloat16 printing issues
    first_10 = flat[:10].tolist()
    print(f"{prefix}[STATS] {name}:")
    print(f"{prefix}  shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"{prefix}  min={t.min().item():.6f}, max={t.max().item():.6f}, "
          f"mean={t.mean().item():.6f}, std={t.std().item():.6f}")
    print(f"{prefix}  first_10={[f'{x:.6f}' for x in first_10]}")


def replace_rmsnorm_with_vllm(model: torch.nn.Module, vllm_config, rank: int = 0) -> None:
    """
    Replace all torch.nn.RMSNorm in the model with vLLM's RMSNorm.

    This ensures numerical equivalence between TorchTitan and vLLM models.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm
    from vllm.config import set_current_vllm_config

    replaced_count = 0

    def replace_in_module(parent_module, parent_name=""):
        nonlocal replaced_count
        for name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Check if this is a torch.nn.RMSNorm
            if isinstance(child, torch.nn.RMSNorm):
                # Get the original parameters
                normalized_shape = child.normalized_shape
                eps = child.eps
                weight = child.weight.data.clone()

                # Handle tuple normalized_shape
                if isinstance(normalized_shape, tuple):
                    hidden_size = normalized_shape[0]
                else:
                    hidden_size = normalized_shape

                # Create vLLM RMSNorm within config context
                with set_current_vllm_config(vllm_config):
                    vllm_norm = VLLMRMSNorm(hidden_size=hidden_size, eps=eps)
                vllm_norm.weight.data.copy_(weight)
                vllm_norm = vllm_norm.to(device=child.weight.device, dtype=child.weight.dtype)

                # Replace the module
                setattr(parent_module, name, vllm_norm)
                replaced_count += 1
                if rank == 0:
                    print(f"  Replaced {full_name}: torch.nn.RMSNorm -> VLLMRMSNorm")
            else:
                # Recurse into child modules
                replace_in_module(child, full_name)

    replace_in_module(model)


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor,
                    label1: str = "tensor1", label2: str = "tensor2") -> float:
    """Compare two tensors and print detailed stats."""
    if isinstance(t1, DTensor):
        t1 = t1.full_tensor()
    if isinstance(t2, DTensor):
        t2 = t2.full_tensor()

    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n=== Comparing {name} ===")
    print_tensor_stats(f"{label1}", t1)
    print_tensor_stats(f"{label2}", t2)
    print(f"  DIFF: max={max_diff:.6e}, mean={mean_diff:.6e}")

    # Find where max diff occurs
    if max_diff > 1e-5:
        max_idx = diff.argmax().item()
        print(f"  Max diff at flat index {max_idx}: {label1}={t1.flatten()[max_idx].item():.6f}, "
              f"{label2}={t2.flatten()[max_idx].item():.6f}")

    return max_diff


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""

    seq_lens: list[int]
    query_lens: list[int]

    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def __post_init__(self):
        assert len(self.seq_lens) == len(self.query_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    max_block_idx: int = 1000,
    arange_block_indices: bool = False,
) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata from a BatchSpec."""
    # Create query start locations
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()

    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    max_seq_len = int(seq_lens.max().item())

    # Create block table and slot mapping
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    if arange_block_indices:
        num_blocks = batch_spec.batch_size * max_blocks
        block_table_tensor = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(batch_spec.batch_size, max_blocks)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device).view(
            num_tokens
        )
    else:
        block_table_tensor = torch.randint(
            0,
            max_block_idx,
            (batch_spec.batch_size, max_blocks),
            dtype=torch.int32,
            device=device,
        )
        slot_mapping = torch.randint(
            0, max_block_idx, (num_tokens,), dtype=torch.int64, device=device
        )

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor + 1,
        slot_mapping=slot_mapping + BLOCK_SIZE,
        causal=True,
    )


def build_attn_metadata(num_tokens: int, local_rank: int = 0) -> FlashAttentionMetadata:
    """Build FlashAttentionMetadata for a given number of tokens."""
    batch_spec = BatchSpec(seq_lens=[num_tokens], query_lens=[num_tokens])
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=BLOCK_SIZE,
        device=torch.device(f"cuda:{local_rank}"),
        arange_block_indices=True,
    )
    return FlashAttentionMetadata(
        causal=common_attn_metadata.causal,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=common_attn_metadata.query_start_loc,
        max_seq_len=num_tokens,
        seq_lens=common_attn_metadata.seq_lens,
        block_table=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def create_debug_model_args(
    model_dim: int = 64,
    vocab_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    num_layers: int = 2,
    max_seq_len: int = 32,
) -> Qwen3ModelArgs:
    """Create a small debug model args for testing."""
    head_dim = model_dim // num_heads
    hidden_dim = model_dim * 4  # Standard 4x multiplier for FFN

    return Qwen3ModelArgs(
        dim=model_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=max_seq_len,
        qk_norm=True,
        depth_init=False,
        attn_type="sdpa",
        attn_mask_type="causal",
    )


class TestVLLMTorchTitan:
    """Test class for vLLM integration with TorchTitan models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Set default dtype to bfloat16 for vLLM compatibility
        torch.set_default_dtype(torch.bfloat16)

    @pytest.fixture(scope="class")
    def ctx(self):
        """Shared test context fixture that sets up models once per class."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.set_default_dtype(torch.bfloat16)
        return setup_models_and_inputs()

    @torch.inference_mode()
    def test_forward_end_to_end(self, ctx: TestContext):
        """
        Test end-to-end forward pass comparing vLLM native vs TorchTitan.

        This runs the full model forward pass on both models and compares
        the final logits output.
        """
        print("\n" + "=" * 70)
        print("=== TEST: End-to-End Forward Pass ===")
        print("=" * 70)

        # Run vLLM native forward
        print("\n=== Running vLLM native forward ===")
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_native_hidden = ctx.vllm_native_model(ctx.tokens, ctx.positions)
            vllm_native_logits = ctx.vllm_native_model.compute_logits(vllm_native_hidden)
        print(f"vLLM native hidden shape: {vllm_native_hidden.shape}")
        print(f"vLLM native logits shape: {vllm_native_logits.shape}")

        # Run TorchTitan vLLM wrapper forward
        print("\n=== Running TorchTitan vLLM wrapper forward ===")
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            torchtitan_hidden = ctx.torchtitan_vllm_model(ctx.tokens, ctx.positions)
            torchtitan_logits = ctx.torchtitan_vllm_model.compute_logits(torchtitan_hidden)

        # Handle DTensor output
        if isinstance(torchtitan_hidden, DTensor):
            torchtitan_hidden = torchtitan_hidden.full_tensor()
        if isinstance(torchtitan_logits, DTensor):
            torchtitan_logits = torchtitan_logits.full_tensor()

        print(f"TorchTitan hidden shape: {torchtitan_hidden.shape}")
        print(f"TorchTitan logits shape: {torchtitan_logits.shape}")

        # Compare outputs
        print("\n=== Comparing outputs ===")

        # Debug: Compare lm_head weights
        print("\n=== Comparing lm_head weights ===")
        print("vLLM lm_head:", ctx.vllm_native_model.lm_head.weight[:5, :5])
        print("TorchTitan output:", ctx.torchtitan_vllm_model.model.output.weight[:5, :5])

        print(f"vLLM native hidden stats: min={vllm_native_hidden.min():.4f}, "
              f"max={vllm_native_hidden.max():.4f}, mean={vllm_native_hidden.float().mean():.4f}")
        print(f"TorchTitan hidden stats: min={torchtitan_hidden.min():.4f}, "
              f"max={torchtitan_hidden.max():.4f}, mean={torchtitan_hidden.float().mean():.4f}")
        print(f"vLLM native logits stats: min={vllm_native_logits.min():.4f}, "
              f"max={vllm_native_logits.max():.4f}, mean={vllm_native_logits.float().mean():.4f}")
        print(f"TorchTitan logits stats: min={torchtitan_logits.min():.4f}, "
              f"max={torchtitan_logits.max():.4f}, mean={torchtitan_logits.float().mean():.4f}")

        # Verify outputs are close
        torch.testing.assert_close(
            vllm_native_logits,
            torchtitan_logits,
            rtol=1e-3,
            atol=1e-3,
        )
        print("\n=== Test passed! vLLM TorchTitan and vLLM native produce matching outputs ===")

    @torch.inference_mode()
    def test_forward_embedding_and_transformer_layer(self, ctx: TestContext):
        """
        Test forward pass of embedding layer + first transformer layer.

        Compares hidden states after embedding and first transformer layer.
        """
        print("\n" + "=" * 70)
        print("=== TEST: Embedding + Transformer Layer Forward Pass ===")
        print("=" * 70)

        # Enable debug printing
        # enable_qwen3_debug()
        # ctx.torchtitan_vllm_model.model.enable_debug(True)

        # Run vLLM native embedding + first layer
        print("\n--- vLLM Native Model ---")
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_embeddings = ctx.vllm_native_model.model.embed_tokens(ctx.tokens)
            print_tensor_stats("vLLM embedding output", vllm_embeddings)

            residual = None
            hidden_states = vllm_embeddings.clone()
            hidden_states, residual = ctx.vllm_native_model.model.layers[0](
                positions=ctx.positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            vllm_layer0_output_hidden = hidden_states
            vllm_layer0_output_residual = residual
            print_tensor_stats("vLLM Layer 0 hidden_states", hidden_states)
            print_tensor_stats("vLLM Layer 0 residual", residual)

        # Run TorchTitan embedding + first layer
        print("\n--- TorchTitan Model ---")
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            tokens_2d = ctx.tokens.unsqueeze(0)
            positions_2d = ctx.positions.unsqueeze(0)
            tt_embeddings = ctx.torchtitan_vllm_model.model.tok_embeddings(tokens_2d)
            if isinstance(tt_embeddings, DTensor):
                tt_embeddings_local = tt_embeddings.full_tensor()
            else:
                tt_embeddings_local = tt_embeddings
            print_tensor_stats("TorchTitan embedding output", tt_embeddings_local)

            rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
            if isinstance(rope_cache, DTensor):
                rope_cache = rope_cache.to_local()

            h = tt_embeddings.clone()
            h = ctx.torchtitan_vllm_model.model.layers["0"](h, rope_cache, attention_masks=None, positions=positions_2d)
            if isinstance(h, DTensor):
                tt_layer0_output = h.full_tensor()
            else:
                tt_layer0_output = h
            print_tensor_stats("TorchTitan Layer 0 output", tt_layer0_output)

        # Disable debug printing
        disable_qwen3_debug()
        ctx.torchtitan_vllm_model.model.enable_debug(False)

        # Compare outputs
        print("\n--- Comparing outputs ---")
        vllm_embed_flat = vllm_embeddings.view(-1, ctx.model_dim)
        tt_embed_flat = tt_embeddings_local.view(-1, ctx.model_dim)
        embed_output_diff = compare_tensors(
            "Embedding outputs",
            vllm_embed_flat, tt_embed_flat,
            label1="vLLM", label2="TorchTitan"
        )

        # vLLM uses pre-RMSNorm residual pattern
        vllm_combined = vllm_layer0_output_hidden + vllm_layer0_output_residual
        vllm_combined_flat = vllm_combined.view(-1, ctx.model_dim)
        tt_layer0_flat = tt_layer0_output.view(-1, ctx.model_dim)

        layer_output_diff = compare_tensors(
            "Layer 0 outputs (vLLM hidden+residual vs TorchTitan)",
            vllm_combined_flat, tt_layer0_flat,
            label1="vLLM", label2="TorchTitan"
        )

        assert layer_output_diff < 1e-2, f"Layer outputs differ too much! max diff: {layer_output_diff:.6e}"
        print("\n=== Test passed! Embedding and first layer outputs match! ===")

    @torch.inference_mode()
    def test_forward_attention(self, ctx: TestContext):
        """
        Test forward pass of the attention model only.

        This test:
        1. Compares Q, K, V projections between models
        2. Compares Q/K normalization outputs
        3. Compares full attention outputs
        """
        print("\n" + "=" * 70)
        print("=== TEST: Attention Model Forward Pass ===")
        print("=" * 70)

        # Get attention modules
        vllm_attn = ctx.vllm_native_model.model.layers[0].self_attn
        tt_attn = ctx.torchtitan_vllm_model.model.layers["0"].attention
        print(f"vLLM attention type: {type(vllm_attn)}")
        print(f"TorchTitan attention type: {type(tt_attn)}")

        if hasattr(tt_attn, 'inner_attention'):
            print(f"TorchTitan inner_attention type: {type(tt_attn.inner_attention)}")

        # Verify q_norm/k_norm weights are identical
        print("\n--- Checking Q/K norm weights ---")
        print(f"vLLM q_norm type: {type(vllm_attn.q_norm)}")
        print(f"TorchTitan q_norm type: {type(tt_attn.q_norm)}")

        vllm_q_norm_weight = vllm_attn.q_norm.weight.data
        tt_q_norm_weight = tt_attn.q_norm.weight.data
        if isinstance(tt_q_norm_weight, DTensor):
            tt_q_norm_weight = tt_q_norm_weight.full_tensor()
        q_norm_weight_diff = compare_tensors("q_norm.weight", vllm_q_norm_weight, tt_q_norm_weight,
                                              label1="vLLM", label2="TorchTitan")

        vllm_k_norm_weight = vllm_attn.k_norm.weight.data
        tt_k_norm_weight = tt_attn.k_norm.weight.data
        if isinstance(tt_k_norm_weight, DTensor):
            tt_k_norm_weight = tt_k_norm_weight.full_tensor()
        k_norm_weight_diff = compare_tensors("k_norm.weight", vllm_k_norm_weight, tt_k_norm_weight,
                                              label1="vLLM", label2="TorchTitan")

        if q_norm_weight_diff > 1e-5 or k_norm_weight_diff > 1e-5:
            print("\n[ERROR] Q/K norm weights are NOT identical! Check weight copying.")

        # Create a fixed input for attention comparison
        torch.manual_seed(123)
        attn_input = torch.randn(ctx.seq_len, ctx.model_dim, dtype=torch.bfloat16, device="cuda")
        attn_positions = torch.arange(ctx.seq_len, device="cuda")

        print("\n--- Attention input ---")
        print_tensor_stats("attn_input", attn_input)

        # ===== vLLM Attention Forward (step by step) =====
        print("\n--- vLLM Attention Step-by-Step ---")

        # Debug: Print static context keys to see which caches exist
        static_ctx = ctx.vllm_config.compilation_config.static_forward_context
        print(f"Static forward context keys: {list(static_ctx.keys())[:5]}...")  # Show first 5

        # Clear KV cache before vLLM forward
        num_blocks = 30
        for layer_name, layer in static_ctx.items():
            if hasattr(layer, 'kv_cache'):
                kv_cache = torch.zeros(
                    (2, num_blocks, BLOCK_SIZE, ctx.num_kv_heads, ctx.head_dim),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                layer.kv_cache = [kv_cache]

        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            # QKV projection
            vllm_qkv, _ = vllm_attn.qkv_proj(attn_input)
            vllm_q = vllm_qkv[:, :vllm_attn.q_size]
            vllm_k = vllm_qkv[:, vllm_attn.q_size:vllm_attn.q_size + vllm_attn.kv_size]
            vllm_v = vllm_qkv[:, vllm_attn.q_size + vllm_attn.kv_size:]
            print_tensor_stats("vLLM Q (before norm)", vllm_q)
            print_tensor_stats("vLLM K (before norm)", vllm_k)
            print_tensor_stats("vLLM V", vllm_v)

            # Q/K normalization
            q_by_head = vllm_q.view(ctx.seq_len, ctx.num_heads, ctx.head_dim)
            vllm_q_normed = vllm_attn.q_norm(q_by_head).view(ctx.seq_len, -1)
            k_by_head = vllm_k.view(ctx.seq_len, ctx.num_kv_heads, ctx.head_dim)
            vllm_k_normed = vllm_attn.k_norm(k_by_head).view(ctx.seq_len, -1)
            print_tensor_stats("vLLM Q (after norm)", vllm_q_normed)
            print_tensor_stats("vLLM K (after norm)", vllm_k_normed)

            # Full attention forward
            vllm_attn_output = vllm_attn(
                positions=attn_positions,
                hidden_states=attn_input.clone(),
            )
            print_tensor_stats("vLLM attention output", vllm_attn_output)

        # ===== TorchTitan Attention Forward (step by step) =====
        print("\n--- TorchTitan Attention Step-by-Step ---")

        # Enable debug for VLLMAttention
        # tt_attn.inner_attention._debug_enabled = True

        # Clear KV cache before TorchTitan forward
        for layer_name, layer in static_ctx.items():
            if hasattr(layer, 'kv_cache'):
                kv_cache = torch.zeros(
                    (2, num_blocks, BLOCK_SIZE, ctx.num_kv_heads, ctx.head_dim),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                layer.kv_cache = [kv_cache]

        # TorchTitan expects [batch, seq, hidden] format
        tt_attn_input = attn_input.unsqueeze(0)
        tt_attn_positions = attn_positions.unsqueeze(0)

        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            # QKV projection
            tt_q = tt_attn.wq(tt_attn_input)
            tt_k = tt_attn.wk(tt_attn_input)
            tt_v = tt_attn.wv(tt_attn_input)

            if isinstance(tt_q, DTensor):
                tt_q_local = tt_q.full_tensor().squeeze(0)
                tt_k_local = tt_k.full_tensor().squeeze(0)
                tt_v_local = tt_v.full_tensor().squeeze(0)
            else:
                tt_q_local = tt_q.squeeze(0)
                tt_k_local = tt_k.squeeze(0)
                tt_v_local = tt_v.squeeze(0)

            print_tensor_stats("TorchTitan Q (before norm)", tt_q_local)
            print_tensor_stats("TorchTitan K (before norm)", tt_k_local)
            print_tensor_stats("TorchTitan V", tt_v_local)

            # Q/K normalization
            tt_q_by_head = tt_q_local.view(ctx.seq_len, ctx.num_heads, ctx.head_dim)
            tt_q_normed = tt_attn.q_norm(tt_q_by_head).view(ctx.seq_len, -1)
            tt_k_by_head = tt_k_local.view(ctx.seq_len, ctx.num_kv_heads, ctx.head_dim)
            tt_k_normed = tt_attn.k_norm(tt_k_by_head).view(ctx.seq_len, -1)

            if isinstance(tt_q_normed, DTensor):
                tt_q_normed = tt_q_normed.full_tensor()
            if isinstance(tt_k_normed, DTensor):
                tt_k_normed = tt_k_normed.full_tensor()

            print_tensor_stats("TorchTitan Q (after norm)", tt_q_normed)
            print_tensor_stats("TorchTitan K (after norm)", tt_k_normed)

            # Get rope cache
            rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
            if isinstance(rope_cache, DTensor):
                rope_cache = rope_cache.to_local()

            # Full attention forward
            tt_attn_output = tt_attn(
                tt_attn_input.clone(),
                rope_cache,
                attention_masks=None,
                positions=tt_attn_positions,
            )

            if isinstance(tt_attn_output, DTensor):
                tt_attn_output_local = tt_attn_output.full_tensor().squeeze(0)
            else:
                tt_attn_output_local = tt_attn_output.squeeze(0)

            print_tensor_stats("TorchTitan attention output", tt_attn_output_local)

        # Disable debug
        tt_attn.inner_attention._debug_enabled = False

        # ===== Compare intermediate results =====
        print("\n--- Comparing Attention Results ---")
        q_diff = compare_tensors("Q projection (before norm)", vllm_q, tt_q_local, label1="vLLM", label2="TorchTitan")
        k_diff = compare_tensors("K projection (before norm)", vllm_k, tt_k_local, label1="vLLM", label2="TorchTitan")
        v_diff = compare_tensors("V projection", vllm_v, tt_v_local, label1="vLLM", label2="TorchTitan")

        q_norm_diff = compare_tensors("Q (after norm)", vllm_q_normed, tt_q_normed, label1="vLLM", label2="TorchTitan")
        k_norm_diff = compare_tensors("K (after norm)", vllm_k_normed, tt_k_normed, label1="vLLM", label2="TorchTitan")

        # ===== Compare RoPE outputs =====
        print("\n--- Comparing RoPE outputs ---")
        # Get RoPE from vLLM
        vllm_q_rope_input = vllm_q_normed.view(ctx.seq_len, ctx.num_heads, ctx.head_dim)
        vllm_k_rope_input = vllm_k_normed.view(ctx.seq_len, ctx.num_kv_heads, ctx.head_dim)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_q_rope, vllm_k_rope = vllm_attn.rotary_emb(attn_positions, vllm_q_rope_input, vllm_k_rope_input)
        print_tensor_stats("vLLM Q after RoPE", vllm_q_rope)
        print_tensor_stats("vLLM K after RoPE", vllm_k_rope)

        # Get RoPE from TorchTitan
        from torchtitan.models.qwen3.model.model import apply_rotary_emb
        tt_q_rope_input = tt_q_normed.view(1, ctx.seq_len, ctx.num_heads, ctx.head_dim)
        tt_k_rope_input = tt_k_normed.view(1, ctx.seq_len, ctx.num_kv_heads, ctx.head_dim)
        rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
        if isinstance(rope_cache, DTensor):
            rope_cache = rope_cache.to_local()
        tt_positions = attn_positions.unsqueeze(0)
        tt_q_rope, tt_k_rope = apply_rotary_emb(tt_q_rope_input, tt_k_rope_input, rope_cache, tt_positions)
        tt_q_rope = tt_q_rope.squeeze(0)
        tt_k_rope = tt_k_rope.squeeze(0)
        print_tensor_stats("TorchTitan Q after RoPE", tt_q_rope)
        print_tensor_stats("TorchTitan K after RoPE", tt_k_rope)

        # Compare RoPE outputs
        q_rope_diff = compare_tensors("Q after RoPE", vllm_q_rope.view(ctx.seq_len, -1), tt_q_rope.view(ctx.seq_len, -1),
                                       label1="vLLM", label2="TorchTitan")
        k_rope_diff = compare_tensors("K after RoPE", vllm_k_rope.view(ctx.seq_len, -1), tt_k_rope.view(ctx.seq_len, -1),
                                       label1="vLLM", label2="TorchTitan")

        attn_output_diff = compare_tensors("Attention output", vllm_attn_output, tt_attn_output_local, label1="vLLM", label2="TorchTitan")

        print(f"\n=== Attention Comparison Summary ===")
        print(f"Q projection diff: {q_diff:.6e}")
        print(f"K projection diff: {k_diff:.6e}")
        print(f"V projection diff: {v_diff:.6e}")
        print(f"Q after norm diff: {q_norm_diff:.6e}")
        print(f"K after norm diff: {k_norm_diff:.6e}")
        print(f"Q after RoPE diff: {q_rope_diff:.6e}")
        print(f"K after RoPE diff: {k_rope_diff:.6e}")
        print(f"Attention output diff: {attn_output_diff:.6e}")

        # Assert projections match
        assert q_diff < 1e-5, f"Q projections differ! max diff: {q_diff}"
        assert k_diff < 1e-5, f"K projections differ! max diff: {k_diff}"
        assert v_diff < 1e-5, f"V projections differ! max diff: {v_diff}"

        if attn_output_diff > 1e-3:
            print("\n[WARNING] Attention outputs differ significantly!")
            print("First 3 tokens, first 10 dims comparison:")
            for i in range(min(3, ctx.seq_len)):
                print(f"  Token {i}:")
                print(f"    vLLM:       {vllm_attn_output[i, :10].tolist()}")
                print(f"    TorchTitan: {tt_attn_output_local[i, :10].tolist()}")

        # Assert attention outputs match
        assert attn_output_diff < 1e-3, f"Attention outputs differ! max diff: {attn_output_diff:.6e}"
        print("\n=== Test passed! Attention forward completed successfully ===")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
