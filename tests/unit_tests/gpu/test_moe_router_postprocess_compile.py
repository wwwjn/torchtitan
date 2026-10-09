# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import QuantileBalancedTopKRouter


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRouterPostprocessLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_compiled_matches_eager(self):
        config = QuantileBalancedTopKRouter.Config(
            num_experts=64,
            top_k=8,
            gate=HiMidLoLinear.Config(
                in_features=32,
                out_features=64,
                backward_mode="hi_mid_lo",
                bias=False,
            ),
            score_func=Sigmoid.Config(),
            route_norm=True,
            route_scale=2.5,
            num_bins=32,
        )
        with torch.device("cuda"):
            router = config.build()

        generator = torch.Generator(device="cuda").manual_seed(0)
        scores_TE = torch.rand(
            256,
            config.num_experts,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
        topk_expert_ids_TK = torch.topk(
            scores_TE.detach(), config.top_k, dim=-1, sorted=False
        ).indices
        grad_TK = torch.randn(
            256,
            config.top_k,
            device="cuda",
            generator=generator,
        )

        def run():
            scores = scores_TE.detach().clone().requires_grad_(True)
            weights_TK, routing_map_TE = router._postprocess_route(
                scores,
                topk_expert_ids_TK,
            )
            grad_scores_TE = torch.autograd.grad(weights_TK, scores, grad_TK)[0]
            return weights_TK, routing_map_TE, grad_scores_TE

        apply_local_compile([])
        eager = run()
        apply_local_compile(["router_postprocess"])
        compiled = run()

        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
