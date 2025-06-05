from contextlib import nullcontext

import torch
import torch.distributed as dist
import verl.utils.torch_functional as verl_F

from engine.base import Engine

from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torchtitan.config_manager import JobConfig

from torchtitan.train import Trainer
from transformers import AutoConfig, AutoModelForCausalLM
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)


default_engine_config = JobConfig()


class TorchTitianEngine(Engine, Trainer):
    def __init__(self, config: JobConfig):
        Trainer.__init__(self, job_config=config)

    def init_model_and_optimizer(self):
        """
        return empty function becuase the model and optimizer are initialized in the parent class initalize function
        """
        return

    def forward_backward_step(self, batch, forward_only=False) -> torch.Tensor:
        """
        return loss of the output
        """
        # follow the implementation in https://github.com/pytorch/torchtitan/pull/1238

        return super().forward_backward_step(batch, forward_only=forward_only)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.config.optim.grad_clip is not None
        grad_norm = self.fsdp_model.clip_grad_norm_(
            max_norm=self.config.optim.grad_clip
        )
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm

    def lr_scheduler_step(self):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        return lr

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
