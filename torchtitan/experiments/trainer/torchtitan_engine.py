# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import torch
import torch.distributed as dist

from tensordict import TensorDict
from torchtitan.config_manager import JobConfig

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.trainer.engine.base import Engine

from torchtitan.train import Trainer

default_engine_config = JobConfig()


class TorchTitianEngine(Trainer, Engine):
    def __init__(self, config: JobConfig):
        # Init torchtitan trainer
        super().__init__(config)

        # Loading pre-trained model checkpoints
        self.checkpointer.load(step=config.checkpoint.load_step)

    def init_model_and_optimizer(self):
        """
        return empty function becuase the model and optimizer are initialized in the parent class initalize function
        """
        return

    def forward_backward_step(
        self, batch: TensorDict, forward_only=False
    ) -> torch.Tensor:
        """
        return loss of the forward and backward step for the current batch
        """
        # follow the implementation in https://github.com/pytorch/torchtitan/pull/1238

        # Fields of TensorDict
        # fields={
        #     attention_mask: Tensor(shape=torch.Size([2, 1024]), device=cuda:0, dtype=torch.int64, is_shared=True),
        #     input_ids: Tensor(shape=torch.Size([2, 1024]), device=cuda:0, dtype=torch.int64, is_shared=True),
        #     loss_mask: Tensor(shape=torch.Size([2, 1024]), device=cuda:0, dtype=torch.int64, is_shared=True),
        #     position_ids: Tensor(shape=torch.Size([2, 1024]), device=cuda:0, dtype=torch.int64, is_shared=True)},
        # batch_size=torch.Size([2]),
        # device=cuda:0,
        # is_shared=True)

        # NOTE(jianiw): For testing, we only use the input_ids tensor for now.
        input_ids = batch.get("input_ids", None)
        if input_ids is None:
            raise KeyError("The 'input_ids' tensor is not found in the TensorDict.")

        # NOTE(jianiw): The label is the next token in the input_ids tensor.
        shift_input = input_ids[:, :-1].contiguous().to(self.device)
        labels = input_ids[:, 1:].contiguous().to(self.device)
        return super().forward_backward_step(
            input_dict={"input": shift_input}, labels=labels
        )

    def optimizer_zero_grad(self):
        self.optimizers.zero_grad()

    def optimizer_step(self):
        dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if self.parallel_dims.pp_enabled else None,
        )
        self.optimizers.step()

    def lr_scheduler_step(self):
        self.lr_schedulers.step()
        lr = None
        for i, scheduler in enumerate(self.lr_schedulers):
            lr = scheduler.get_last_lr()[0]
        # lr = self.lr_schedulers[0].get_last_lr()[0]
        return lr

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
