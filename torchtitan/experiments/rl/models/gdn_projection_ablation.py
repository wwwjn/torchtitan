# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only merged input projections for Qwen3.5 Gated DeltaNet."""

from __future__ import annotations

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torch.distributed.tensor import DTensor, Shard


_MERGED_WEIGHT_CACHE: dict[tuple[int, ...], torch.Tensor] = {}
_PATCHED = False


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _merged_weight(weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
    key = tuple(weight.data_ptr() for weight in weights)
    merged = _MERGED_WEIGHT_CACHE.get(key)
    if merged is None:
        merged = torch.cat(weights, dim=0).contiguous()
        _MERGED_WEIGHT_CACHE[key] = merged
    return merged


def _clear_merged_weight_cache(*_args) -> None:
    _MERGED_WEIGHT_CACHE.clear()


@torch.library.custom_op("torchtitan::merged_gdn_qkvz", mutates_args=())
def _merged_gdn_qkvz(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wz: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, _merged_weight((wq, wk, wv, wz)))


@_merged_gdn_qkvz.register_fake
def _merged_gdn_qkvz_fake(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wz: torch.Tensor,
) -> torch.Tensor:
    output_dim = wq.shape[0] + wk.shape[0] + wv.shape[0] + wz.shape[0]
    return torch.empty(*x.shape[:-1], output_dim, dtype=x.dtype, device=x.device)


@torch.library.custom_op("torchtitan::merged_gdn_ba", mutates_args=())
def _merged_gdn_ba(
    x: torch.Tensor,
    wb: torch.Tensor,
    wa: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, _merged_weight((wb, wa)))


@_merged_gdn_ba.register_fake
def _merged_gdn_ba_fake(
    x: torch.Tensor,
    wb: torch.Tensor,
    wa: torch.Tensor,
) -> torch.Tensor:
    output_dim = wb.shape[0] + wa.shape[0]
    return torch.empty(*x.shape[:-1], output_dim, dtype=x.dtype, device=x.device)


def _wrap_projection_output(
    output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(weight, DTensor):
        return output
    output_dim = output.ndim - 1
    placements = tuple(
        Shard(output_dim) if isinstance(placement, Shard) else placement
        for placement in weight.placements
    )
    return DTensor.from_local(
        output,
        weight.device_mesh,
        placements,
        run_check=False,
    )


def apply_merged_gdn_projections() -> None:
    """Patch Qwen3.5 GDN to merge six input projections into two GEMMs.

    The model keeps its six original parameters and checkpoint keys. Their
    already TP-local weight shards are concatenated once on first use, matching
    vLLM's local ``qkvz`` and ``ba`` projection layout. A load-state hook drops
    the derived cache after RL weight synchronization.
    """
    global _PATCHED
    if _PATCHED:
        return

    from torchtitan.distributed.utils import is_in_batch_invariant_mode
    from torchtitan.models.qwen3_5.gdn import GatedDeltaNet

    original_init = GatedDeltaNet.__init__

    def init(self, config) -> None:
        original_init(self, config)
        _clear_merged_weight_cache()
        self.register_load_state_dict_post_hook(_clear_merged_weight_cache)

    def forward(self, x_TD, attention_masks=None):
        num_tokens = x_TD.shape[0]
        cu_seqlens_host = None
        if attention_masks is not None:
            with spmd.local():
                cu_seqlens = attention_masks.cu_seq_q.clone()
            cu_seqlens_host = attention_masks.cu_seq_q_host
            if cu_seqlens_host is None:
                raise ValueError(
                    "Qwen3.5 Gated DeltaNet varlen requires CPU cu_seqlens "
                    "metadata. Build VarlenMetadata with include_host_offsets=True."
                )
        else:
            cu_seqlens = torch.arange(
                0,
                num_tokens + 1,
                num_tokens,
                dtype=torch.int32,
                device=x_TD.device,
            )
            if is_in_batch_invariant_mode():
                cu_seqlens_host = (0, num_tokens)

        x = _local_tensor(x_TD)
        projection_weights = (
            self.in_proj_q.weight,
            self.in_proj_k.weight,
            self.in_proj_v.weight,
            self.in_proj_z.weight,
        )
        local_projection_weights = tuple(map(_local_tensor, projection_weights))
        qkvz = _merged_gdn_qkvz(x, *local_projection_weights)
        projection_sizes = tuple(weight.shape[0] for weight in local_projection_weights)
        query_TC, key_TC, value_TC, gate_TC = (
            _wrap_projection_output(part, projection_weights[0])
            for part in qkvz.split(projection_sizes, dim=-1)
        )

        gate_weights = (self.in_proj_b.weight, self.in_proj_a.weight)
        local_gate_weights = tuple(map(_local_tensor, gate_weights))
        ba = _merged_gdn_ba(x, *local_gate_weights)
        gate_sizes = tuple(weight.shape[0] for weight in local_gate_weights)
        b_TN, a_TN = (
            _wrap_projection_output(part, gate_weights[0])
            for part in ba.split(gate_sizes, dim=-1)
        )

        output_TNV = self.inner_gated_delta_net(
            query_TC,
            key_TC,
            value_TC,
            a_TN,
            b_TN,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            cu_seqlens_host=cu_seqlens_host,
        )
        gate_TNV = gate_TC.view(num_tokens, -1, self.value_head_dim)
        output_TNV = self.norm(output_TNV, gate_TNV)
        return self.out_proj(output_TNV.reshape(num_tokens, -1))

    GatedDeltaNet.__init__ = init
    GatedDeltaNet.forward = forward
    _PATCHED = True


__all__ = ["apply_merged_gdn_projections"]
