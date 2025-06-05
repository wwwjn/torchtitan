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
        # model_config = self.config.model
        # system_config = self.config.system
        # optim_config = self.config.optim

        # self._build_model(model_config, system_config)
        # self._build_optimizer(optim_config)

        # for us, do the HF->titan model definition convertion when init the model
        self._convert_model

    def _convert_model(self, model_config, system_config):
        local_model_path = copy_to_local(src=model_config.path, verbose=True)

        # load config first
        hf_config = AutoConfig.from_pretrained(local_model_path)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=hf_config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
            )

        if model_config.use_rmpad or system_config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(
                model=self.model,
                ulysses_sp_size=system_config.ulysses_sequence_parallel_size,
            )

        if model_config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # ???
        cpu_offload = None
        if system_config.param_offload:
            cpu_offload = CPUOffload(offload_params=system_config.param_offload)

    def forward_backward_step(self, batch, forward_only=False):
        # follow the implementation in https://github.com/pytorch/torchtitan/pull/1238

        self.model.train()

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        position_ids = batch["position_ids"].cuda()
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).cuda()

        # Context manager for sequence parallel if needed
        context = nullcontext()
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Standard forward pass without sequence parallel
            labels = input_ids[:, 1:].contiguous()
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)
            loss = loss * loss_mask.to(loss.device)

        valid_token_this_rank = torch.sum(loss_mask)
        dp_size = 1
        loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size
        outputs.loss = loss
        loss.backward()

        return outputs

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
