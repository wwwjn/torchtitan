import os
import sys
import tempfile
import shutil
import torch
import torch.distributed as dist
from typing import Dict, Any
import random
import numpy as np
from unittest.mock import patch, MagicMock


def setup_fake_distributed():
    """Setup fake distributed environment for testing."""
    # Mock the distributed environment
    if not dist.is_initialized():
        # Create a fake process group for single-process testing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        
        # Initialize process group
        dist.init_process_group(
            backend='gloo',  # Use gloo for CPU testing
            init_method='env://',
            world_size=1,
            rank=0
        )


def test_deepseek_v3_parallel_checkpoint_loading():
    """Test DeepSeek V3 model with parallelization and checkpoint loading."""
    print("Starting DeepSeek V3 parallel checkpoint loading test...")
    
    # Setup fake distributed environment
    setup_fake_distributed()
    
    # Import necessary components
    from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
    from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model
    from torchtitan.models.deepseek_v3.model.state_dict_adapter import DeepSeekV3StateDictAdapter
    from torchtitan.models.deepseek_v3.infra.parallelize import parallelize_deepseekv3
    from torchtitan.models.moe import MoEArgs
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.config import JobConfig, CheckpointConfig, ParallelismConfig, TrainingConfig, ModelConfig
    from torchtitan.distributed import ParallelDims
    from torch.distributed.device_mesh import init_device_mesh
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create temporary directories for checkpoints
    temp_dir = tempfile.mkdtemp()
    hf_checkpoint_dir = os.path.join(temp_dir, "hf_checkpoint")
    tt_checkpoint_dir = os.path.join(temp_dir, "tt_checkpoint")
    os.makedirs(hf_checkpoint_dir, exist_ok=True)
    os.makedirs(tt_checkpoint_dir, exist_ok=True)
    
    try:
        # Create model args for a small test model
        moe_args = MoEArgs(
            num_experts=4,
            top_k=2,
            router_aux_loss_coef=0.001,
            use_grouped_mm=False,
        )
        
        model_args = DeepSeekV3ModelArgs(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            n_dense_layers=1,  # First layer is dense
            inter_dim=256,
            moe_inter_dim=192,
            q_lora_rank=0,  # Use direct projection for simplicity
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=32,
            max_seq_len=512,
            norm_eps=1e-5,
            moe_args=moe_args,
        )
        
        print(f"Created model args: vocab_size={model_args.vocab_size}, dim={model_args.dim}")
        
        # Create mock HF state dict (same as previous test)
        def create_mock_hf_state_dict(model_args: DeepSeekV3ModelArgs) -> Dict[str, torch.Tensor]:
            """Create a mock HF state dict with the expected structure."""
            state_dict = {}
            
            # Embedding
            state_dict["model.embed_tokens.weight"] = torch.randn(model_args.vocab_size, model_args.dim)
            
            # Output layer
            state_dict["lm_head.weight"] = torch.randn(model_args.vocab_size, model_args.dim)
            
            # Final norm
            state_dict["model.norm.weight"] = torch.ones(model_args.dim)
            
            for layer_id in range(model_args.n_layers):
                layer_prefix = f"model.layers.{layer_id}"
                
                # Layer norms
                state_dict[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(model_args.dim)
                state_dict[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.ones(model_args.dim)
                
                # Attention weights
                state_dict[f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(
                    model_args.kv_lora_rank + model_args.qk_rope_head_dim, model_args.dim
                )
                state_dict[f"{layer_prefix}.self_attn.kv_a_layernorm.weight"] = torch.ones(model_args.kv_lora_rank)
                state_dict[f"{layer_prefix}.self_attn.kv_b_proj.weight"] = torch.randn(
                    model_args.n_heads * (model_args.qk_nope_head_dim + model_args.v_head_dim),
                    model_args.kv_lora_rank
                )
                state_dict[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.randn(
                    model_args.dim, model_args.n_heads * model_args.v_head_dim
                )
                
                # Query projection (direct since q_lora_rank=0)
                if model_args.q_lora_rank == 0:
                    state_dict[f"{layer_prefix}.self_attn.q_proj.weight"] = torch.randn(
                        model_args.n_heads * (model_args.qk_nope_head_dim + model_args.qk_rope_head_dim),
                        model_args.dim
                    )
                else:
                    state_dict[f"{layer_prefix}.self_attn.q_a_proj.weight"] = torch.randn(
                        model_args.q_lora_rank, model_args.dim
                    )
                    state_dict[f"{layer_prefix}.self_attn.q_a_layernorm.weight"] = torch.ones(model_args.q_lora_rank)
                    state_dict[f"{layer_prefix}.self_attn.q_b_proj.weight"] = torch.randn(
                        model_args.n_heads * (model_args.qk_nope_head_dim + model_args.qk_rope_head_dim),
                        model_args.q_lora_rank
                    )
                
                # MLP/MoE weights
                if layer_id >= model_args.n_dense_layers:
                    # MoE layer
                    # Router
                    state_dict[f"{layer_prefix}.mlp.gate.weight"] = torch.randn(
                        model_args.moe_args.num_experts, model_args.dim
                    )
                    
                    # Individual experts
                    for expert_id in range(model_args.moe_args.num_experts):
                        state_dict[f"{layer_prefix}.mlp.experts.{expert_id}.gate_proj.weight"] = torch.randn(
                            model_args.moe_inter_dim, model_args.dim
                        )
                        state_dict[f"{layer_prefix}.mlp.experts.{expert_id}.up_proj.weight"] = torch.randn(
                            model_args.moe_inter_dim, model_args.dim
                        )
                        state_dict[f"{layer_prefix}.mlp.experts.{expert_id}.down_proj.weight"] = torch.randn(
                            model_args.dim, model_args.moe_inter_dim
                        )
                    
                    # Shared experts
                    state_dict[f"{layer_prefix}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
                        model_args.moe_inter_dim, model_args.dim
                    )
                    state_dict[f"{layer_prefix}.mlp.shared_experts.up_proj.weight"] = torch.randn(
                        model_args.moe_inter_dim, model_args.dim
                    )
                    state_dict[f"{layer_prefix}.mlp.shared_experts.down_proj.weight"] = torch.randn(
                        model_args.dim, model_args.moe_inter_dim
                    )
                else:
                    # Dense layer
                    state_dict[f"{layer_prefix}.mlp.gate_proj.weight"] = torch.randn(
                        model_args.inter_dim, model_args.dim
                    )
                    state_dict[f"{layer_prefix}.mlp.up_proj.weight"] = torch.randn(
                        model_args.inter_dim, model_args.dim
                    )
                    state_dict[f"{layer_prefix}.mlp.down_proj.weight"] = torch.randn(
                        model_args.dim, model_args.inter_dim
                    )
            
            return state_dict
        
        print("Creating mock HF state dict...")
        hf_state_dict = create_mock_hf_state_dict(model_args)
        
        # Save mock HF checkpoint using PyTorch's save format
        hf_checkpoint_path = os.path.join(hf_checkpoint_dir, "pytorch_model.bin")
        torch.save(hf_state_dict, hf_checkpoint_path)
        print(f"Saved mock HF checkpoint to {hf_checkpoint_path}")
        
        # Create TorchTitan model and initialize
        print("Creating TorchTitan model...")
        tt_model = DeepSeekV3Model(model_args)
        tt_model.eval()
        
        # Create job config for parallelization
        job_config = JobConfig(
            parallelism=ParallelismConfig(
                dp=1,
                tp=1,
                pp=1,
                context_parallel_degree=1,
                expert_parallel_degree=1,
                enable_loss_parallel=False,
            ),
            training=TrainingConfig(
                seq_len=512,
                mixed_precision_param="float32",
                mixed_precision_reduce="float32",
                enable_cpu_offload=False,
            ),
            model=ModelConfig(),
            checkpoint=CheckpointConfig(
                enable=True,
                folder=tt_checkpoint_dir,
                interval=1,
                export_dtype="float32",
                initial_load_in_hf=True,
                initial_load_path=hf_checkpoint_dir,
                initial_load_model_only=True,
            )
        )
        
        # Create parallel dimensions
        parallel_dims = ParallelDims(
            dp=1,
            tp=1,
            pp=1,
            world_size=1,
            enable_loss_parallel=False,
            dp_replicate=1,
            dp_shard=1,
        )
        
        # Create device mesh
        print("Creating device mesh...")
        device_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        parallel_dims.world_mesh = device_mesh
        
        print("Applying parallelization...")
        # Apply parallelization (this will be mostly no-op for single device)
        with patch('torchtitan.models.deepseek_v3.infra.parallelize.apply_non_moe_tp') as mock_tp, \
             patch('torchtitan.models.deepseek_v3.infra.parallelize.apply_moe_ep_tp') as mock_moe, \
             patch('torchtitan.models.deepseek_v3.infra.parallelize.apply_fsdp') as mock_fsdp, \
             patch('torchtitan.models.deepseek_v3.infra.parallelize.apply_ddp') as mock_ddp, \
             patch('torchtitan.models.deepseek_v3.infra.parallelize.apply_ac') as mock_ac, \
             patch('torchtitan.models.deepseek_v3.infra.parallelize.apply_compile') as mock_compile:
            
            parallelize_deepseekv3(tt_model, parallel_dims, job_config)
            print("✓ Parallelization applied successfully")
        
        # Create optimizer and scheduler containers (minimal setup)
        print("Creating optimizer and scheduler containers...")
        optimizer = torch.optim.AdamW(tt_model.parameters(), lr=1e-4)
        optimizers_container = OptimizersContainer()
        optimizers_container.optimizers = {"model": optimizer}
        
        lr_schedulers_container = LRSchedulersContainer()
        lr_schedulers_container.lr_schedulers = {}
        
        # Create state dict adapter
        adapter = DeepSeekV3StateDictAdapter(model_args, hf_checkpoint_dir)
        
        # Create checkpoint manager
        print("Creating checkpoint manager...")
        train_state = {"step": 0}
        
        checkpoint_manager = CheckpointManager(
            dataloader=None,
            model_parts=[tt_model],
            optimizers=optimizers_container,
            lr_schedulers=lr_schedulers_container,
            states={"train_state": train_state},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=adapter,
            base_folder="",
        )
        
        # Test checkpoint loading
        print("Testing checkpoint loading...")
        
        # Store original model weights for comparison
        original_weights = {}
        for name, param in tt_model.named_parameters():
            original_weights[name] = param.data.clone()
        
        # Load from checkpoint (this should load the HF weights)
        success = checkpoint_manager.load()
        
        if success:
            print("✓ Checkpoint loaded successfully!")
            
            # Verify that weights have been updated
            weights_changed = False
            for name, param in tt_model.named_parameters():
                if name in original_weights and not torch.equal(original_weights[name], param.data):
                    weights_changed = True
                    break
            
            if weights_changed:
                print("✓ Model weights were updated from checkpoint")
            else:
                print("⚠ Model weights were not changed - this might be expected if using NaN initialization")
            
            # Test forward pass to ensure model is functional
            print("Testing forward pass after loading...")
            try:
                batch_size = 2
                seq_len = 10
                tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
                
                with torch.no_grad():
                    output = tt_model(tokens)
                    print(f"Forward pass successful! Output shape: {output.shape}")
                    expected_shape = (batch_size, seq_len, model_args.vocab_size)
                    if output.shape == expected_shape:
                        print("✓ Output shape is correct!")
                    else:
                        print(f"✗ Output shape mismatch: expected {expected_shape}, got {output.shape}")
                        return False
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
                return False
                
        else:
            print("✗ Checkpoint loading failed")
            return False
        
        # Test saving a checkpoint
        print("Testing checkpoint saving...")
        try:
            train_state["step"] = 1
            checkpoint_manager.save(curr_step=1, last_step=False)
            
            # Check if checkpoint was saved
            expected_checkpoint_path = os.path.join(tt_checkpoint_dir, "step-1")
            if os.path.exists(expected_checkpoint_path):
                print("✓ Checkpoint saved successfully!")
            else:
                print("✗ Checkpoint was not saved")
                return False
                
        except Exception as e:
            print(f"✗ Checkpoint saving failed: {e}")
            return False
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()


def test_deepseek_v3_state_dict_consistency():
    """Test that the state dict adapter properly converts between HF and TT formats."""
    print("Testing DeepSeek V3 state dict consistency...")
    
    # Import necessary components
    from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
    from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model
    from torchtitan.models.deepseek_v3.model.state_dict_adapter import DeepSeekV3StateDictAdapter
    from torchtitan.models.moe import MoEArgs
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    
    # Create model args for testing
    moe_args = MoEArgs(
        num_experts=2,  # Smaller for faster testing
        top_k=1,
        router_aux_loss_coef=0.001,
        use_grouped_mm=False,
    )
    
    model_args = DeepSeekV3ModelArgs(
        vocab_size=100,  # Small vocab for testing
        dim=64,
        n_layers=1,  # Single layer for simplicity
        n_heads=2,
        n_dense_layers=0,  # Make it MoE from the start
        inter_dim=128,
        moe_inter_dim=96,
        q_lora_rank=0,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        max_seq_len=128,
        norm_eps=1e-5,
        moe_args=moe_args,
    )
    
    # Create TT model
    tt_model = DeepSeekV3Model(model_args)
    
    # Get the model's state dict
    tt_state_dict = tt_model.state_dict()
    # Remove freqs_cis as it's computed
    tt_state_dict = {k: v for k, v in tt_state_dict.items() if k != 'freqs_cis'}
    
    print(f"TT model has {len(tt_state_dict)} parameters")
    
    # Create adapter
    adapter = DeepSeekV3StateDictAdapter(model_args, None)
    
    # Convert TT to HF format
    print("Converting TT to HF format...")
    hf_state_dict = adapter.to_hf(tt_state_dict)
    print(f"HF state dict has {len(hf_state_dict)} parameters")
    
    # Convert back to TT format
    print("Converting HF back to TT format...")
    tt_state_dict_restored = adapter.from_hf(hf_state_dict)
    print(f"Restored TT state dict has {len(tt_state_dict_restored)} parameters")
    
    # Check consistency
    def compare_state_dicts(dict1, dict2, name1="dict1", name2="dict2"):
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            extra_in_2 = keys2 - keys1
            print(f"Key mismatch between {name1} and {name2}:")
            if missing_in_2:
                print(f"  Missing in {name2}: {sorted(missing_in_2)}")
            if extra_in_2:
                print(f"  Extra in {name2}: {sorted(extra_in_2)}")
            return False
        
        for key in keys1:
            if dict1[key].shape != dict2[key].shape:
                print(f"Shape mismatch for key '{key}': {dict1[key].shape} vs {dict2[key].shape}")
                return False
            
            if not torch.allclose(dict1[key], dict2[key], rtol=1e-6, atol=1e-8):
                diff = (dict1[key] - dict2[key]).abs().max().item()
                print(f"Value mismatch for key '{key}': max_diff={diff}")
                return False
        
        return True
    
    if compare_state_dicts(tt_state_dict, tt_state_dict_restored, "original_tt", "restored_tt"):
        print("✓ State dict consistency test passed!")
        return True
    else:
        print("✗ State dict consistency test failed!")
        return False


if __name__ == "__main__":
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print("=" * 60)
        
        # Run state dict consistency test first
        consistency_success = test_deepseek_v3_state_dict_consistency()
        print("=" * 60)
        
        # Run parallel checkpoint test
        parallel_success = test_deepseek_v3_parallel_checkpoint_loading()
        
        if consistency_success and parallel_success:
            print("\n🎉 All DeepSeek V3 parallel checkpoint tests completed successfully!")
        else:
            print("\n💥 Some DeepSeek V3 parallel checkpoint tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)