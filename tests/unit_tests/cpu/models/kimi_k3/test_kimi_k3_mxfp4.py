# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
from pathlib import Path

import pytest
import torch

from torchtitan.models.kimi_k3.mxfp4 import decode_mxfp4, packed_mxfp4_weights

_REFERENCE = Path(__file__).parents[4] / "assets" / "mxfp4-reference.json"


def test_decode_matches_compressed_tensors_reference():
    # Bytes and values serialized by compressed-tensors: all 16 E2M1 codes at
    # several E8M0 scales.
    reference = json.loads(_REFERENCE.read_text())
    packed = torch.tensor(reference["weight_packed"], dtype=torch.uint8)
    scales = torch.tensor(reference["weight_scale"], dtype=torch.uint8)
    expected = torch.tensor(reference["dequantized"], dtype=torch.float32)
    block_size = packed.shape[-1] * 2 // scales.shape[-1]

    assert torch.equal(
        decode_mxfp4(packed, scales, block_size, torch.float32), expected
    )
    assert torch.equal(
        decode_mxfp4(packed, scales, block_size, torch.bfloat16),
        expected.to(torch.bfloat16),
    )


def _write_checkpoint(path: Path, weight_map: dict[str, str]) -> None:
    quantization = {
        "format": "mxfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": {"group_size": 32}}},
    }
    (path / "config.json").write_text(
        json.dumps({"text_config": {"quantization_config": quantization}})
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_packed_weights_come_from_index_pairs():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        _write_checkpoint(
            path,
            {
                "experts.0.w1.weight_packed": "a.safetensors",
                "experts.0.w1.weight_scale": "a.safetensors",
                "self_attn.q_proj.weight": "a.safetensors",
            },
        )
        assert packed_mxfp4_weights(str(path)) == (
            frozenset({"experts.0.w1.weight"}),
            32,
        )


def test_packed_weight_without_scale_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        _write_checkpoint(path, {"experts.0.w1.weight_packed": "a.safetensors"})
        with pytest.raises(ValueError, match="no matching scale"):
            packed_mxfp4_weights(str(path))
