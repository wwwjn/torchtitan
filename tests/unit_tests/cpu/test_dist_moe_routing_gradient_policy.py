# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts for opt-in BF16 routing gradients at the framework boundary."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from torchtitan.models.common.dist_moe import DistMoeRuntime


def _runtime(enabled: bool) -> DistMoeRuntime:
    """Construct only policy state; no process group or CUDA allocation is needed."""
    runtime = object.__new__(DistMoeRuntime)
    runtime.config = DistMoeRuntime.Config(routing_gradient_via_dgrad=enabled)
    return runtime


def _module() -> SimpleNamespace:
    """Describe the expert fields consumed by backend configuration resolution."""
    return SimpleNamespace(
        hidden_dim=128,
        intermediate_dim=128,
        top_k=2,
        num_experts=4,
        activation="situglu",
        situglu_beta=4.0,
        situglu_linear_beta=25.0,
        block_scaled_config=None,
        bf16_grouped_gemm_preset=None,
    )


def _resolve(runtime: DistMoeRuntime):
    """Resolve a backend contract without building runtime resources."""
    return runtime._resolve_context_config(
        _module(),
        max_num_local_input_tokens=32,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_forwards_explicit_routing_policy(enabled):
    """The opt-in bool is forwarded only when requested; other capacities survive."""

    def factory(**kwargs):
        return SimpleNamespace(**kwargs)

    factory.__dataclass_fields__ = {
        "max_num_local_input_tokens": None,
        "routing_gradient_via_dgrad": None,
    }
    with patch(
        "torchtitan.models.common.dist_moe.runtime.dist_moe.Config", new=factory
    ):
        config = _resolve(_runtime(enabled))
    assert config.max_num_local_input_tokens == 32
    assert config.num_activation_slots == 2
    assert config.max_moe_layers_per_activation_slot == 3
    assert config.activation == "situglu"
    assert config.situglu_beta == 4.0
    assert getattr(config, "routing_gradient_via_dgrad", False) is enabled
    assert ("routing_gradient_via_dgrad" in vars(config)) is enabled


def test_old_backend_cannot_silently_ignore_opt_in():
    """Unsupported packages reject the opt-in while default-off remains compatible."""

    def factory(**kwargs):
        return SimpleNamespace(**kwargs)

    factory.__dataclass_fields__ = {"max_num_local_input_tokens": None}
    with patch(
        "torchtitan.models.common.dist_moe.runtime.dist_moe.Config", new=factory
    ):
        with pytest.raises(ValueError, match="Installed Dist-MoE does not support"):
            _resolve(_runtime(True))
        assert "routing_gradient_via_dgrad" not in vars(_resolve(_runtime(False)))


@pytest.mark.parametrize("value", ["enabled", 1, None])
def test_runtime_rejects_non_boolean_routing_policy(value):
    """String or numeric configuration cannot accidentally enable changed arithmetic."""
    with pytest.raises(TypeError, match="must be a bool"):
        DistMoeRuntime.Config(routing_gradient_via_dgrad=value)
