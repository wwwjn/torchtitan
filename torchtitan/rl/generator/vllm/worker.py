# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

from collections.abc import Callable, Set
from contextlib import nullcontext
from types import SimpleNamespace

import torch.distributed as dist
from vllm import envs
from vllm.config import CUDAGraphMode, get_layers_from_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.math_utils import round_up
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from torchtitan.rl.generator.vllm.linear_attention import (
    TorchTitanLinearAttentionBackend,
)


class TorchTitanCudagraphDispatcher(CudagraphDispatcher):
    """Keep native FULL descriptors distinct for packed linear attention and fused decode.

    Currently installed only for TorchTitan's Attention Gym linear attention
    backends (GDN/KDA).
    vLLM at c6fa1f0 erases the uniform-decode discriminator in FULL mode:
    four one-token decodes collide with one four-token prefill. Attention Gym's
    different numerical paths need distinct keys. Delegate decode keys to the
    native FULL_DECODE_ONLY dispatcher until upstream supports FULL decode
    specialization; native capture, replay and graph ownership stay intact.
    The single-token key also covers fresh requests: initialization is GPU mask
    data, not another dispatch key or a reason to switch to packed arithmetic.

    This costs one additional graph capture per decode-specialized batch size
    and additional memory for the expanded graph pool.
    """

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ) -> None:
        super().initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
        if cudagraph_mode == CUDAGraphMode.FULL:
            self.decode_dispatcher = CudagraphDispatcher(self.vllm_config)
            self.decode_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.FULL_DECODE_ONLY, uniform_decode_query_len
            )
            for mode, descriptors in self.decode_dispatcher.get_capture_descs():
                for descriptor in descriptors:
                    self.add_cudagraph_key(mode, descriptor)

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: Set[CUDAGraphMode] | None = None,
        invalid_modes: Set[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        dispatch = super().dispatch
        if (
            self.keys_initialized
            and self.cudagraph_mode == CUDAGraphMode.FULL
            and uniform_decode
        ):
            dispatch = self.decode_dispatcher.dispatch
        return dispatch(
            num_tokens,
            uniform_decode,
            has_lora,
            num_active_loras,
            valid_modes,
            invalid_modes,
        )


class TorchTitanGPUModelRunner(GPUModelRunner):
    """V1 runner with sequence-parallel padding and linear attention (GDN/KDA) decode specialization."""

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        if any(
            layer.get_attn_backend() is TorchTitanLinearAttentionBackend
            for layer in get_layers_from_vllm_config(
                self.vllm_config, MambaBase
            ).values()
        ):
            assert not self.cudagraph_dispatcher.keys_initialized
            self.cudagraph_dispatcher = TorchTitanCudagraphDispatcher(self.vllm_config)

    def _determine_batch_execution_and_padding(self, num_tokens: int, *args, **kwargs):
        """Tell the model how many of this step's tokens are real, before padding.

        vLLM calls this once before every forward: from ``execute_model`` for a
        real step, and from ``_dummy_run`` for profiling, warm-up, CUDA-graph
        capture and an idle DP rank's dummy batch (whose rows all count as
        real). ``num_tokens`` is the unpadded count; vLLM applies every padding
        source inside this call (TP rounding, CUDA-graph capture size, DP
        equalization). The model turns the count into a padding mask, which keeps
        those rows off Dist-MoE's network; graph replay reads the value published
        for the real step.
        """
        self.get_model().set_num_valid_tokens(num_tokens)
        return super()._determine_batch_execution_and_padding(
            num_tokens, *args, **kwargs
        )

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_dense_sp = self.compilation_config.pass_config.enable_sp and tp_size > 1
        enable_expert_sp = (
            self.vllm_config.parallel_config.enable_expert_parallel and tp_size > 1
        )
        if enable_dense_sp or enable_expert_sp:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens


class _TPAgreedTransferResults:
    """Wraps a KV connector's ``get_transfer_results`` so every TP rank reports a request's KV
    transfer as finished in the same engine step.

    With ``external_launcher`` each TP rank runs its own scheduler, which sees only its own
    worker's transfers (vLLM's multiprocess executor instead merges all workers' into one
    scheduler with ``KVOutputAggregator``). A request whose KV arrived on some TP ranks first
    would be scheduled there a step earlier, and the ranks' batches, and so their TP
    collectives, would diverge. Each step this gathers every TP rank's results and reports a
    request once all ranks finished it, with vLLM's ``KVOutputAggregator``.
    """

    def __init__(
        self,
        get_transfer_results: Callable[[set[str]], KVConnectorTransferResults],
        *,
        group: dist.ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        self._get_transfer_results = get_transfer_results
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._aggregator = KVOutputAggregator(expected_finished_count=world_size)

    def __call__(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        local = self._get_transfer_results(finished_req_ids)
        gathered: list[KVConnectorTransferResults | None] = [None] * self._world_size
        dist.all_gather_object(gathered, local, group=self._group)
        return self.merge(gathered)

    def merge(
        self, gathered: list[KVConnectorTransferResults | None]
    ) -> KVConnectorTransferResults:
        """Merge one step's results from every TP rank (index = TP rank)."""
        outputs = [
            SimpleNamespace(
                kv_connector_output=KVConnectorOutput(
                    finished_sending=results.finished_sending or None,
                    finished_recving=results.finished_recving or None,
                    failed_recving=results.failed_recving,
                )
            )
            for results in gathered
        ]
        merged = self._aggregator.aggregate(outputs, output_rank=self._rank)
        kv_output = merged.kv_connector_output
        return KVConnectorTransferResults(
            finished_sending=set(kv_output.finished_sending or ()),
            finished_recving=set(kv_output.finished_recving or ()),
            failed_recving=set(kv_output.failed_recving),
        )


class TorchTitanGPUWorker(GPUWorker):
    """V1 worker that constructs :class:`TorchTitanGPUModelRunner`."""

    def _maybe_get_memory_pool_context(self, tag: str):
        # vLLM uses CuMem for model weights and the KV cache.
        # TorchTitan only transfers model weights over RDMA, so leave all other
        # allocations on PyTorch's default allocator.
        # The parent method checks enable_cumem_allocator and returns a no-op
        # context when it is disabled.
        if tag == "weights":
            return super()._maybe_get_memory_pool_context(tag)
        return nullcontext()

    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError(
                "TorchTitan's vLLM integration requires the V1 model runner"
            )

        # GPUWorker imports its runner class inside init_device and provides no
        # runner factory. Scope the class substitution to that construction.
        original_runner_cls = vllm_gpu_model_runner.GPUModelRunner
        vllm_gpu_model_runner.GPUModelRunner = TorchTitanGPUModelRunner
        try:
            super().init_device()
        finally:
            vllm_gpu_model_runner.GPUModelRunner = original_runner_cls

    def initialize_from_config(self, kv_cache_config) -> None:
        super().initialize_from_config(kv_cache_config)
        # vLLM creates the KV connector here (prefill/decode generators only).
        tp_group = get_tp_group()
        if has_kv_transfer_group() and tp_group.world_size > 1:
            # Request ids must match across TP ranks for them to agree (see
            # add_kv_transfer_config).
            if not envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
                raise RuntimeError(
                    "KV transfer with TP > 1 needs VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1"
                )
            connector = get_kv_transfer_group()
            connector.get_transfer_results = _TPAgreedTransferResults(
                connector.get_transfer_results,
                group=tp_group.cpu_group,
                rank=tp_group.rank_in_group,
                world_size=tp_group.world_size,
            )
