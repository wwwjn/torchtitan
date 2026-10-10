# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for explicit local KDA chunk-kernel options."""

from unittest.mock import patch

import pytest
import torch
from attn_gym.linear import KernelOptions

from torchtitan.models.common.attention.kda import KDAKernel


@pytest.mark.parametrize("backend", [None, "fused", "cudnn"])
def test_chunk_kernel_options_preserve_auto_and_allow_explicit_backend(backend):
    options = None if backend is None else KernelOptions(backend=backend)
    config = KDAKernel.Config(kernel_options=options)
    kernel = config.build()
    q = torch.empty(1, 8, 1, 128)
    v = torch.empty_like(q)
    gate = torch.empty_like(q)
    beta = torch.empty(1, 8, 1)
    selected = KernelOptions(backend="cudnn")
    output = torch.empty_like(v)

    with (
        patch.object(kernel, "prepare_inputs", return_value=(q, q, gate, beta)),
        patch(
            "torchtitan.models.common.attention.kda.chunk_kernel_options",
            return_value=selected,
        ) as auto_select,
        patch(
            "torchtitan.models.common.attention.kda.chunk_kda",
            return_value=(output, None),
        ) as chunk,
    ):
        result = kernel(q, q, v, gate, beta, torch.empty(1), torch.empty(1, 128))

    assert result is output
    assert chunk.call_args.kwargs["kernel_options"] == (
        selected if options is None else options
    )
    if options is None:
        auto_select.assert_called_once_with(q)
    else:
        auto_select.assert_not_called()


def test_default_keeps_automatic_selection():
    assert KDAKernel.Config().kernel_options is None


def test_explicit_options_preserve_additional_scheduling_fields():
    options = KernelOptions(backend="fused", schedule="persistent")
    kernel = KDAKernel.Config(kernel_options=options).build()
    assert kernel.kernel_options == options
