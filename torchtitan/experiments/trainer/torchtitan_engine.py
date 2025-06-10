from contextlib import nullcontext

import torch
import torch.distributed as dist
from engine.base import Engine

from torchtitan.config_manager import JobConfig

from torchtitan.distributed import ParallelDims, utils as dist_utils

from torchtitan.train import Trainer
from transformers import AutoConfig, AutoModelForCausalLM

default_engine_config = JobConfig()


class TorchTitianEngine(Trainer, Engine):
    def __init__(self, config: JobConfig):
        # Init torchtitan trainer
        super().__init__(config)

    def init_model_and_optimizer(self):
        """
        return empty function becuase the model and optimizer are initialized in the parent class initalize function
        """
        return

    def forward_backward_step(self, batch, forward_only=False) -> torch.Tensor:
        """
        return loss of the forward and backward step for the current batch
        """
        # follow the implementation in https://github.com/pytorch/torchtitan/pull/1238

        input_ids = batch["input_ids"].cuda()
        labels = input_ids[:, 1:].contiguous()
        return super().forward_backward_step(input_dict={"input": batch}, labels=labels)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if self.parallel_dims.pp_enabled else None,
        )
        self.optimizer.step()

    def lr_scheduler_step(self):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        return lr

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
