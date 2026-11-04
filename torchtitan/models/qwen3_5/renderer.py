# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 renderer whose images are references to image files, not pixels.

The library's ``Qwen35Renderer`` expands each image part into
``<|vision_start|>`` + N x ``<|image_pad|>`` + ``<|vision_end|>`` and runs the
Hugging Face image processor inside ``render()`` to find N and produce
``pixel_values``. In RL, rendering runs on the rollout controller, so that
would decode and resize every image on the controller's event loop and carry
megabytes of pixels through every RPC.

This renderer gets N from `ImageLoader.lookup` instead (see ``image_loader.py``),
which computes the grid from the image file's header without decoding it. Per
image it returns the content key as its ``mm_hashes`` entry and
``image_grid_thw``; ``pixel_values`` stays ``None``, and the generator and the
trainer decode the file with the same preprocessing, so both see identical
pixels. Token ids equal the library renderer's whenever the loader uses the
Hugging Face processor's pixel budget (the defaults of ``ImageSpec``).

Image parts must name a file, e.g. ``{"type": "image", "path": ...}``. Select it
with ``CustomRendererConfig(import_path=..., images=...)``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import model_validator
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from torchtitan.components.data.multimodal.image_loader import ImageLoader


class Qwen35ImageRefRendererConfig(Qwen35RendererConfig):
    """Qwen3.5 renderer config plus the image loader it reads grids with."""

    # pyrefly: ignore [bad-override]
    name: Literal["qwen3.5_image_ref"] = "qwen3.5_image_ref"
    # image_cache_max is inherited and unused: it bounds the HF-processor cache
    # of the `_process_image` this renderer replaces.
    _internal_fields = frozenset({"image_cache_max", "images"})

    images: ImageLoader.Config
    """Reads the images' grids, with the model's ``ImageSpec``. In RL the generator
    and the trainer decode the image files with the same spec."""

    @model_validator(mode="after")
    def _reject_image_cache_max(self) -> "Qwen35ImageRefRendererConfig":
        if "image_cache_max" in self.model_fields_set:
            raise ValueError(
                "image_cache_max bounds the HF-processor cache, which "
                f"{type(self).__name__} never uses; remove it."
            )
        return self


class Qwen35ImageRefRenderer(Qwen35Renderer):
    """``Qwen35Renderer`` that resolves image parts with an `ImageLoader`.

    Example:

        renderer = Qwen35ImageRefRenderer(
            tokenizer,
            Qwen35ImageRefRendererConfig(images=ImageLoader.Config()),
        )
        rendered = renderer.render(
            [{"role": "user", "content": [
                {"type": "image", "path": "/data/slake/imgs/xmlab1/source.jpg"},
                {"type": "text", "text": "Is this a CT?"},
            ]}],
            add_generation_prompt=True,
        )
        rendered.multi_modal_data.mm_hashes["image"]  # [content key]
    """

    config_class = Qwen35ImageRefRendererConfig

    def __init__(self, tokenizer: Any, config: Qwen35ImageRefRendererConfig) -> None:
        super().__init__(tokenizer, config)
        self._images = config.images.build()

    # pyrefly: ignore [bad-override]
    def _process_image(self, part: dict[str, Any]):
        """Return ``(None, {"pixel_values": None, "image_grid_thw"}, num_image_tokens, key)``.

        The grid and placeholder count come from the file header; no pixels are decoded.
        """
        path = part.get("path")
        if path is None:
            raise ValueError(
                f"{type(self).__name__} needs image parts with a 'path' to an image "
                f"file, got keys {sorted(part)}."
            )
        image = self._images.lookup(path)
        out = {"pixel_values": None, "image_grid_thw": np.array([image.grid_thw])}
        return None, out, self._images.num_tokens(image), image.key
