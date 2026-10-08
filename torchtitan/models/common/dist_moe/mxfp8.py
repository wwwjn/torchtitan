# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 prepared-weight lifecycle for Dist-MoE grouped projections."""

from __future__ import annotations

from dataclasses import dataclass, field

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

from . import _dist_moe as dist_moe
from .routed_experts import DistMoeRoutedExperts


__all__ = ["MXFP8DistMoeRoutedExperts"]


@dataclass(frozen=True, slots=True)
class _DistMoeMXFP8Operands:
    """Independent MXFP8 tensors owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_scale: torch.Tensor

    def prepared(self, source: torch.Tensor) -> dist_moe.PreparedWeight:
        """Return the annex facade consumed by one Dist-MoE invocation."""
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
            fprop_data=self.qdata,
            fprop_scale=self.fprop_scale,
            dgrad_data=self.qdata,
            dgrad_scale=self.dgrad_scale,
        )


@dataclass(frozen=True, slots=True)
class _DynamicPreparedWeightCache:
    """Versioned prepared storage for a weight outside FSDP unshard ownership."""

    logical_weight: torch.Tensor
    storage: torch.Tensor
    version: int
    operands: _DistMoeMXFP8Operands


def _prepare_mxfp8_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeMXFP8Operands | None = None,
) -> _DistMoeMXFP8Operands:
    """Allocate or refill the annex's grouped 32x32 MXFP8 weight operands."""
    prepared_out = None if out is None else out.prepared(weight_EOI)
    prepared = dist_moe.prepare_block_scaled_weight(
        weight_EOI,
        dist_moe.BlockScaledConfig(),
        out=prepared_out,
    )
    if prepared.dgrad_data is not prepared.fprop_data:
        raise RuntimeError("MXFP8 Dist-MoE FPROP and DGRAD must share qdata")
    if prepared.dgrad_scale is None:
        raise RuntimeError("MXFP8 Dist-MoE preparation returned incomplete operands")
    return _DistMoeMXFP8Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        dgrad_scale=prepared.dgrad_scale,
    )


class _DistMoeW13ShardedTensor(_ShardedFSDPTensor):
    """Persistent W13 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(logical_tensor.flatten(1, 2), out)


class _DistMoeW2ShardedTensor(_ShardedFSDPTensor):
    """Persistent W2 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(logical_tensor, out)


def _dynamic_prepared_weight(
    logical_weight: torch.Tensor,
    *,
    gate_up: bool,
    cache: _DynamicPreparedWeightCache | None,
) -> tuple[dist_moe.PreparedWeight, _DynamicPreparedWeightCache]:
    """Prepare or reuse a weight outside an FSDP-managed unshard lifetime."""
    source = logical_weight.flatten(1, 2) if gate_up else logical_weight
    storage = (
        logical_weight._tensor
        if isinstance(logical_weight, _ShardedFSDPTensor)
        else logical_weight
    )
    compute_weight = storage.flatten(1, 2) if gate_up else storage
    version = logical_weight._version
    if (
        cache is None
        or cache.logical_weight is not logical_weight
        or cache.storage is not storage
    ):
        with torch.no_grad():
            operands = _prepare_mxfp8_weight(compute_weight)
    elif cache.version != version:
        with torch.no_grad():
            operands = _prepare_mxfp8_weight(compute_weight, out=cache.operands)
    else:
        operands = cache.operands
    cache = _DynamicPreparedWeightCache(
        logical_weight=logical_weight,
        storage=storage,
        version=version,
        operands=operands,
    )
    return operands.prepared(source), cache


class MXFP8DistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE routed experts using native asynchronous MXFP8 kernels.

    The module preserves the inherited W13/W2 parameter and checkpoint layout,
    while TorchTitan's shared FSDP tensor lifecycle prepares the qdata and scale
    layouts required by Dist-MoE for each unshard lifetime. Without FSDP, the
    same layouts are prepared directly from the live parameters before use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure native MXFP8 execution for Dist-MoE routed experts.

        Args:
            block_scaled_config: Annex MXFP8 kernel policy. Its ``pipeline``
                selects separate staged kernels or the fused Mega pipeline;
                ``fast_math`` enables approximate sigmoid math in fused SwiGLU;
                and ``kernel_config`` is an expert-only CuTe tuning override.
        """

        block_scaled_config: dist_moe.BlockScaledConfig = field(
            default_factory=lambda: dist_moe.BlockScaledConfig()
        )

        def __post_init__(self) -> None:
            """Validate the common expert and MXFP8-specific policies."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if (
                self.block_scaled_config.format
                is not dist_moe.BlockScaledFormat.MXFP8_E4M3
            ):
                raise ValueError(
                    "MXFP8DistMoeRoutedExperts requires the MXFP8_E4M3 format"
                )
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset is not used by MXFP8 Dist-MoE"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_scaled_config = config.block_scaled_config
        self.w13.weight = torch.nn.Parameter(
            _DistMoeW13ShardedTensor(self.w13.weight.data),
            requires_grad=self.w13.weight.requires_grad,
        )
        self.w2.weight = torch.nn.Parameter(
            _DistMoeW2ShardedTensor(self.w2.weight.data),
            requires_grad=self.w2.weight.requires_grad,
        )
        self._w13_dynamic_cache: _DynamicPreparedWeightCache | None = None
        self._w2_dynamic_cache: _DynamicPreparedWeightCache | None = None

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrappers before fully_shard() consumes them.
        super()._parallelize(parallelism_context)
        for proj, wrapper_cls in (
            (self.w13, _DistMoeW13ShardedTensor),
            (self.w2, _DistMoeW2ShardedTensor),
        ):
            if isinstance(proj.weight, wrapper_cls):
                continue
            distributed_weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                wrapper_cls(distributed_weight.data),
                requires_grad=distributed_weight.requires_grad,
            )
            spmd.assert_type_like(wrapped_weight, distributed_weight)
            proj.weight = wrapped_weight

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return prepared W13 and W2 operands for this unshard lifetime."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        if isinstance(w13_E2FD, _UnshardedFSDPTensor):
            w13_operand = w13_E2FD.operands.prepared(w13_EFD)
        else:
            w13_operand, self._w13_dynamic_cache = _dynamic_prepared_weight(
                w13_E2FD,
                gate_up=True,
                cache=self._w13_dynamic_cache,
            )
        if isinstance(w2_EDF, _UnshardedFSDPTensor):
            w2_operand = w2_EDF.operands.prepared(w2_EDF)
        else:
            w2_operand, self._w2_dynamic_cache = _dynamic_prepared_weight(
                w2_EDF,
                gate_up=False,
                cache=self._w2_dynamic_cache,
            )
        return w13_operand, w2_operand
