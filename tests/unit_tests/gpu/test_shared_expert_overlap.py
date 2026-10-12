# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact numerical checks for shared-expert CUDA stream scheduling."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.activation import Sigmoid, SiTUGLU
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan.protocols.module import Module, ModuleDict
from torchtitan_recipes.overrides.shared_expert_overlap import shared_expert_overlap


def _moe_config(*, latent: bool) -> MoE.Config:
    """Build shared and learned-router branches with a reduced SiTU-GLU FFN."""
    dim, expert_dim, hidden_dim, num_experts = 128, 64 if latent else 128, 128, 4
    values = dict(
        num_experts=num_experts,
        load_balance_coeff=None,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=HiMidLoLinear.Config(in_features=dim, out_features=num_experts),
            score_func=Sigmoid.Config(),
            top_k=2,
            route_norm=True,
        ),
        routed_experts=RoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=num_experts,
                in_features=expert_dim,
                out_features=hidden_dim,
                num_linears=2,
            ),
            w2=GroupedLinear.Config(
                group_size=num_experts,
                in_features=hidden_dim,
                out_features=expert_dim,
            ),
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=num_experts, top_k=2
            ),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
        shared_experts=FeedForward.Config(
            w13=Linear.Config(in_features=dim, out_features=hidden_dim, num_linears=2),
            w2=Linear.Config(in_features=hidden_dim, out_features=dim),
            activation_fn=SiTUGLU.Config(beta=4.0, linear_beta=25.0),
        ),
    )
    if latent:
        return KimiLatentMoE.Config(
            **values,
            routed_down=Linear.Config(in_features=dim, out_features=expert_dim),
            routed_norm=RMSNorm.Config(normalized_shape=expert_dim),
            routed_up=Linear.Config(in_features=expert_dim, out_features=dim),
        )
    return MoE.Config(**values)


class _Block(Module):
    def __init__(self, config: MoE.Config):
        super().__init__()
        self.moe = config.build()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.moe(x_TD)


class _Model(Module):
    def __init__(self, config: MoE.Config):
        super().__init__()
        self.layers = ModuleDict({"0": _Block(config)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_overlap_preserves_exact_training(
    latent: bool, checkpoint: bool
) -> None:
    """Twelve deterministic updates preserve exact outputs, norms and gradients."""
    torch.manual_seed(42)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        config = _moe_config(latent=latent)
        serial = _Model(config).cuda().bfloat16()
        overlapped = (
            _Model(replace(config, overlap_shared_experts=True)).cuda().bfloat16()
        )
        # GroupedLinear and RMSNorm parameters are initialized via init_states.
        with torch.no_grad():
            for parameter in serial.parameters():
                parameter.uniform_(-0.05, 0.05)
        overlapped.load_state_dict(deepcopy(serial.state_dict()))
        if checkpoint:
            FullAC.Config().build().apply(serial)
            FullAC.Config().build().apply(overlapped)
        for _ in range(12):
            input_TD = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
            results = []
            for model in (serial, overlapped):
                model.zero_grad(set_to_none=True)
                x_TD = input_TD.detach().clone().requires_grad_(True)
                output_TD = model(x_TD)
                loss = output_TD.float().square().mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                results.append((output_TD.detach(), loss.detach(), norm, x_TD.grad))
                del output_TD, loss
            torch.cuda.synchronize()
            for actual, expected in zip(results[1], results[0], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(
                overlapped.parameters(), serial.parameters(), strict=True
            ):
                assert actual.grad is not None and expected.grad is not None
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
                with torch.no_grad():
                    actual.add_(actual.grad, alpha=-0.01)
                    expected.add_(expected.grad, alpha=-0.01)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_overlap_cuda_graph_replay(checkpoint: bool) -> None:
    """A fork/join captured graph matches eager forward and every parameter grad."""
    torch.manual_seed(42)
    model = _Model(replace(_moe_config(latent=True), overlap_shared_experts=True))
    model = model.cuda().bfloat16()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    if checkpoint:
        FullAC.Config().build().apply(model)
    x_TD = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            x_TD.grad = None
            model(x_TD).float().square().mean().backward()
    torch.cuda.current_stream().wait_stream(warmup)
    model.zero_grad(set_to_none=True)
    x_TD.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output_TD = model(x_TD)
        output_TD.float().square().mean().backward()
    graph.replay()
    torch.cuda.synchronize()
    actual_output = output_TD.detach().clone()
    actual_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    x_TD.grad = None
    expected_output = model(x_TD)
    expected_output.float().square().mean().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    for actual, parameter in zip(actual_grads, model.parameters(), strict=True):
        torch.testing.assert_close(actual, parameter.grad, rtol=0, atol=0)


def test_shared_overlap_rejects_cpu_inputs() -> None:
    """An explicitly enabled CUDA scheduling policy never silently falls back."""
    model = _Model(replace(_moe_config(latent=True), overlap_shared_experts=True))
    with pytest.raises(ValueError, match="requires CUDA inputs"):
        model(torch.randn(32, 128))


def test_shared_overlap_requires_shared_experts() -> None:
    """Reject an overlap policy with no branch to execute on the side stream."""
    with pytest.raises(ValueError, match="requires shared experts"):
        replace(
            _moe_config(latent=False), shared_experts=None, overlap_shared_experts=True
        )


@pytest.mark.parametrize("latent", [False, True])
def test_shared_overlap_override_preserves_layout(latent: bool) -> None:
    """The registered opt-in preserves ordinary and latent expert/router configs."""
    config = _moe_config(latent=latent)
    actual = shared_expert_overlap(config)
    assert type(actual) is type(config)
    assert actual.overlap_shared_experts
    assert not config.overlap_shared_experts
    assert actual.routed_experts is config.routed_experts
    assert actual.shared_experts is config.shared_experts
    assert actual.router is config.router
    unshared = replace(config, shared_experts=None)
    assert shared_expert_overlap(unshared) is unshared
