# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.attention.mla import materialize_mla_kv
from torchtitan.models.kimi_k3 import build_model_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMLAKVLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_compiled_matches_eager_bitwise(self):
        torch.manual_seed(0)
        layers = build_model_config("debugmodel", attn_backend="flex").layers
        attention_config = next(
            layer.attention for layer in layers if layer.attention is not None
        )
        kv_width = attention_config.qk_nope_head_dim + attention_config.v_head_dim
        q_THK = torch.empty(
            64,
            attention_config.n_heads,
            attention_config.qk_nope_head_dim + attention_config.qk_rope_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        kv_THC = torch.randn(
            64,
            attention_config.n_heads,
            kv_width,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_rope_TK = torch.randn(
            64,
            attention_config.qk_rope_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def run():
            inputs = [
                kv_THC.clone().requires_grad_(),
                k_rope_TK.clone().requires_grad_(),
            ]
            k_THK, v_THV = materialize_mla_kv(q_THK, *inputs)
            generator = torch.Generator(device="cuda").manual_seed(1)
            grad_k_THK = torch.randn(
                k_THK.shape,
                dtype=k_THK.dtype,
                device="cuda",
                generator=generator,
            )
            grad_v_THV = torch.randn(
                v_THV.shape,
                dtype=v_THV.dtype,
                device="cuda",
                generator=generator,
            )
            grads = torch.autograd.grad(
                (k_THK, v_THV), inputs, (grad_k_THK, grad_v_THV)
            )
            return (k_THK, v_THV, *grads)

        apply_local_compile([])
        eager = run()
        apply_local_compile(["mla_kv"])
        compiled = run()

        # This region only moves data and sums the repeated rope gradients.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
