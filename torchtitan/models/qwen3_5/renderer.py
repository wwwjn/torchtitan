# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 renderer whose images come from a preprocessed `ImageCache`.

The library's ``Qwen35Renderer`` expands each image part into
``<|vision_start|>`` + N x ``<|image_pad|>`` + ``<|vision_end|>`` and runs the
Hugging Face image processor inside ``render()`` to find N and produce
``pixel_values``. In RL, rendering runs on the rollout controller, so that
would decode and resize every image on the controller's event loop and carry
megabytes of pixels through every RPC.

This renderer reads N from the cache index instead (the grid is known from
the image size, so no pixels are touched) and returns, per image, the cache
key as its ``mm_hashes`` entry and ``image_grid_thw``. ``pixel_values`` stays
``None``: the generator and the trainer load patches by key from the same
cache, so both see identical pixels. Token ids equal the library renderer's
whenever the cache was built with the Hugging Face processor's pixel budget
(the defaults of ``ImageCacheSpec``).

Image parts must name a cached file, e.g. ``{"type": "image", "path": ...}``.
Select it with ``CustomRendererConfig(import_path=..., image_cache_dir=...)``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import model_validator
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from torchtitan.components.data.multimodal.image_cache import ImageCache


class Qwen35ImageCacheRendererConfig(Qwen35RendererConfig):
    """Qwen3.5 renderer config plus the image cache it reads."""

    # pyrefly: ignore [bad-override]
    name: Literal["qwen3.5_image_cache"] = "qwen3.5_image_cache"
    # image_cache_max is inherited and unused: it bounds the HF-processor cache
    # of the `_process_image` this renderer replaces.
    _internal_fields = frozenset({"image_cache_max", "image_cache_dir"})

    image_cache_dir: str
    """Directory written by `build_image_cache` from every image the dataset can
    show, with the model's preprocessing (`ImageCacheSpec`). In RL the generator
    and the trainer read the same directory, so it must be readable at this path
    on every host that renders, generates or trains."""

    @model_validator(mode="after")
    def _reject_image_cache_max(self) -> "Qwen35ImageCacheRendererConfig":
        if "image_cache_max" in self.model_fields_set:
            raise ValueError(
                "image_cache_max bounds the HF-processor cache, which "
                f"{type(self).__name__} never uses; remove it."
            )
        return self


class Qwen35ImageCacheRenderer(Qwen35Renderer):
    """``Qwen35Renderer`` that resolves image parts through an `ImageCache`.

    Example:

        renderer = Qwen35ImageCacheRenderer(
            tokenizer,
            Qwen35ImageCacheRendererConfig(image_cache_dir="/data/slake/image_cache"),
        )
        rendered = renderer.render(
            [{"role": "user", "content": [
                {"type": "image", "path": "/data/slake/imgs/xmlab1/source.jpg"},
                {"type": "text", "text": "Is this a CT?"},
            ]}],
            add_generation_prompt=True,
        )
        rendered.multi_modal_data.mm_hashes["image"]  # [cache key]
    """

    config_class = Qwen35ImageCacheRendererConfig

    def __init__(self, tokenizer: Any, config: Qwen35ImageCacheRendererConfig) -> None:
        super().__init__(tokenizer, config)
        self._images = ImageCache(config.image_cache_dir)

    # pyrefly: ignore [bad-override]
    def _process_image(self, part: dict[str, Any]):
        """Return ``(None, {"pixel_values": None, "image_grid_thw"}, num_image_tokens, key)``.

        The grid and placeholder count come from the cache index; no pixels are read.
        """
        path = part.get("path")
        if path is None:
            raise ValueError(
                f"{type(self).__name__} needs image parts with a 'path' to a cached "
                f"file, got keys {sorted(part)}."
            )
        image = self._images.lookup(path)
        out = {"pixel_values": None, "image_grid_thw": np.array([image.grid_thw])}
        return None, out, self._images.num_tokens(image), image.key
