# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decode the released Kimi K3 checkpoint's MXFP4 routed experts while loading."""

import json
from pathlib import Path

import torch

# FP4 E2M1 code -> value; bit 3 is the sign.
_E2M1_VALUES = (
    *(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    *(-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
)
_E8M0_BIAS = 127
# E8M0 byte 255 encodes NaN (the whole block decodes to NaN).
_E8M0_NAN = 255


def decode_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize compressed-tensors ``mxfp4-pack-quantized`` weights.

    ``packed`` (uint8) holds two E2M1 codes per byte, the even element in the low
    nibble; ``scales`` (uint8) holds one E8M0 exponent, ``2 ** (e - 127)`` (255 is
    NaN), per ``block_size`` elements along the last dimension.
    """
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    values = lut[codes.long()].unflatten(-1, (-1, block_size))
    group_scales = torch.exp2(scales.to(torch.float32) - _E8M0_BIAS)
    group_scales = group_scales.masked_fill(scales == _E8M0_NAN, float("nan"))
    return (values * group_scales.unsqueeze(-1)).flatten(-2).to(target_dtype)


def packed_mxfp4_weights(path: str) -> tuple[frozenset[str], int]:
    """Return the HF weight names stored as MXFP4 pairs and their block size.

    A weight ``X.weight`` is packed when the index has both ``X.weight_packed``
    and ``X.weight_scale``; the block size is the quantization config's group size.
    """
    config = json.loads((Path(path) / "config.json").read_text())
    quantization = config.get("text_config", config).get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = config.get("quantization_config")
    if (
        not isinstance(quantization, dict)
        or quantization.get("format") != "mxfp4-pack-quantized"
    ):
        raise ValueError(f"{path}: expected an mxfp4-pack-quantized checkpoint")
    (group,) = quantization["config_groups"].values()
    block_size = group["weights"]["group_size"]
    weight_map = json.loads((Path(path) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    targets = set()
    for name in weight_map:
        if name.endswith(".weight_packed"):
            stem = name.removesuffix("_packed")
            if f"{stem}_scale" not in weight_map:
                raise ValueError(f"{path}: {name} has no matching scale tensor")
            targets.add(stem)
    return frozenset(targets), block_size
