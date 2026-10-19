# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-scoped activation retention preserves the global checkpoint policy."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch_remat as remat

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.experiments.dist_moe_kimi_k3.activation_retention import (
    retain_kda,
    retain_last_kda,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


class _Block(Module):
    def __init__(self):
        super().__init__()
        self.inner_compute = Linear.Config(in_features=32, out_features=32).build()
        self.inner_forwards = 0

    def _inner(self, x_TD: torch.Tensor) -> torch.Tensor:
        self.inner_forwards += 1
        return torch.nn.functional.linear(x_TD, self.inner_compute.weight)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        output_TD = remat.region(
            self._inner,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(x_TD)
        remat.recompute_needs_tensor(output_TD)
        return output_TD.sin()


class _Model(Module):
    def __init__(self, layer_ids: tuple[str, ...] = ("0", "1", "2")):
        super().__init__()
        self.layers = ModuleDict({layer_id: _Block() for layer_id in layer_ids})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        for block in self.layers.values():
            x_TD = block(x_TD)
        return x_TD


def _run(model: _Model, input_TD: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return exact loss, input gradient, and parameter gradients."""
    model.zero_grad(set_to_none=True)
    x_TD = input_TD.detach().clone().requires_grad_(True)
    loss = model(x_TD).square().sum()
    loss.backward()
    return (
        loss.detach(),
        x_TD.grad.detach().clone(),
        *(parameter.grad.detach().clone() for parameter in model.parameters()),
    )


def test_only_selected_block_avoids_recomputation() -> None:
    """One retained region skips replay only in its selected layer, exactly."""
    torch.manual_seed(42)
    reference = _Model()
    actual = deepcopy(reference)
    ActivationRematConfig(
        save_regions=[], block_save_regions={"layers.1": ["inner_attention"]}
    ).apply(actual)
    input_TD = torch.randn(8, 32)
    for actual_tensor, expected in zip(
        _run(actual, input_TD), _run(reference, input_TD), strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected, rtol=0, atol=0)
    assert [block.inner_forwards for block in actual.layers.values()] == [2, 1, 2]


def test_block_globs_preserve_pipeline_layer_fqns() -> None:
    """A PP subset retains original IDs rather than renumbering its local layers."""
    model = _Model(("81", "82", "91"))
    ActivationRematConfig(
        save_regions=[], block_save_regions={"layers.8?": ["inner_attention"]}
    ).apply(model)
    _run(model, torch.randn(8, 32))
    assert [block.inner_forwards for block in model.layers.values()] == [1, 1, 2]


def test_global_recompute_overrides_block_retention() -> None:
    """An explicit recompute pattern still wins over additional block saves."""
    model = _Model()
    ActivationRematConfig(
        save_regions=[],
        recompute_regions=["inner_attention"],
        block_save_regions={"layers.*": ["inner_attention"]},
    ).apply(model)
    _run(model, torch.randn(8, 32))
    assert [block.inner_forwards for block in model.layers.values()] == [2, 2, 2]


@pytest.mark.parametrize(
    "policy", [FullActivationRematConfig, DefaultActivationRematConfig]
)
def test_fixed_checkpoint_presets_reject_block_overrides(policy) -> None:
    """Fixed Full and default remat presets cannot silently acquire custom saves."""
    with pytest.raises(ValueError, match="ActivationRematConfig"):
        policy(block_save_regions={"layers.1": ["inner_attention"]})


def test_last_kda_retention_does_not_change_workload() -> None:
    """The experiment helper retains only the last KDA, without mutating config."""
    config = kimi_k3_debugmodel(seq_len=256)
    config = replace(
        config,
        activation_remat=ActivationRematConfig(
            save_regions=["attention.inner_attention"],
            recompute_regions=["*routed_experts*"],
            block_save_regions={"layers.0": ["feed_forward.w2.*"]},
        ),
    )
    actual = retain_last_kda(config)
    kda_ids = [
        layer_id
        for layer_id, layer in enumerate(config.model.layers)
        if isinstance(layer.attention, KDA.Config)
    ]
    assert actual.model is config.model
    assert actual.parallelism is config.parallelism
    assert actual.training is config.training
    assert actual.dataloader is config.dataloader
    assert actual.dist_moe_runtime is config.dist_moe_runtime
    assert actual.activation_remat.save_regions == ["attention.inner_attention"]
    assert actual.activation_remat.recompute_regions == ["*routed_experts*"]
    assert actual.activation_remat.block_save_regions == {
        "layers.0": ["feed_forward.w2.*"],
        f"layers.{kda_ids[-1]}": ["attention.inner_attention"],
    }
    assert config.activation_remat.block_save_regions == {
        "layers.0": ["feed_forward.w2.*"]
    }


def test_retention_and_shared_overlap_override_compose() -> None:
    """Recipe-time remat retention composes with a shared-expert model override."""
    config = kimi_k3_debugmodel(seq_len=256)
    config.activation_remat = retain_kda(
        FullActivationRematConfig(), block_fqns=["layers.31"]
    )
    overrides = OverrideConfig(
        imports=[
            "torchtitan_recipes.overrides.shared_expert_overlap.shared_expert_overlap",
        ]
    )
    apply_overrides(overrides, config)
    assert type(config.activation_remat) is ActivationRematConfig
    assert config.activation_remat.block_save_regions == {
        "layers.31": ["attention.inner_attention"]
    }
    assert all(
        layer.feed_forward.overlap_shared_experts
        for layer in config.model.layers
        if isinstance(layer.feed_forward, MoE.Config)
        and layer.feed_forward.shared_experts is not None
    )


def test_retention_override_is_idempotent() -> None:
    """Repeated selection does not duplicate the retained region or mutate input."""
    baseline = FullActivationRematConfig()
    once = retain_kda(baseline, block_fqns=["layers.91"])
    twice = retain_kda(once, block_fqns=["layers.91"])
    assert once == twice
    assert baseline.block_save_regions == {}
    with pytest.raises(ValueError, match="at least one block FQN"):
        retain_kda(baseline, block_fqns=[])
