# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image files for image prompts, decoded where the pixels are used.

RL renders, generates and trains on the same images many times (every sample
of a GRPO group, generator and trainer). Rendering needs only each image's
patch grid, for its placeholder count; the generator and the batcher need its
pixels. So prompts carry each image's file bytes (JPEG, PNG, ...), not pixels
(see `ImageInfo`), and the processes that need pixels decode them:

- `ImageLoader.lookup` (the renderer): reads the file, hashes it into the key,
  and computes the grid from the image header, without decoding the pixels.
- `ImageLoader.load_pixel_values` (the generator and the batcher): decodes and
  resizes the file bytes.

A file is usually much smaller than its resized pixels (SLAKE median: 53 KB vs 0.79 MB;
a large photo the resize scales down gains less), and the processes that decode it need
no access to the file system the dataset is on.

Loading runs ``decode_and_resize_image`` (``process_image`` before
normalization), so ``load_pixel_values`` returns exactly the patches
``process_image`` + ``vision_to_patches`` produce. The key is a hash of the
file bytes, so the samples that show one image share it.
"""

from __future__ import annotations

import functools
import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from PIL import Image

from torchtitan.components.data.multimodal.utils.image import (
    decode_and_resize_image,
    normalize_image,
    resize_to_pixel_budget,
    vision_to_patches,
)
from torchtitan.config.configurable import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class ImageSpec:
    """The model's image preprocessing; defaults match Qwen3-VL / Qwen3.5.

    The arguments of ``process_image`` + ``vision_to_patches`` for the model's
    vision encoder.
    """

    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    """Patches merged per side into one placeholder token (``spatial_merge_size``)."""
    min_pixels: int = 65536
    """Lower bound on H * W after the resize."""
    max_pixels: int = 16777216
    """Upper bound on H * W after the resize."""
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass(frozen=True, kw_only=True, slots=True)
class ImageInfo:
    """One image of a prompt: content key, file, raw patch grid and file bytes."""

    key: str
    """Hash of the file bytes."""
    path: str
    """The file `ImageLoader.lookup` read."""
    grid_thw: tuple[int, int, int]
    """Raw patch grid ``(T, H, W)`` before spatial merging."""
    data: bytes = field(repr=False, compare=False)
    """The file's bytes; loading decodes them. Not compared: ``key`` is their hash."""


class ImageLoader(Configurable):
    """Looks up image files (key and grid, no decode) and decodes their bytes.

    The file must be readable at the prompt's path where the prompt is rendered.

    Example:

        images = ImageLoader.Config().build()
        image = images.lookup("/data/slake/imgs/xmlab1/source.jpg")
        images.num_tokens(image)  # placeholder tokens after spatial merging
        pixel_values, grid_thw = images.load_pixel_values([image.data])
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        spec: ImageSpec = field(default_factory=ImageSpec)
        """The model's preprocessing; the defaults are Qwen3-VL / Qwen3.5's."""

    def __init__(self, config: Config) -> None:
        self.spec = config.spec

    def lookup(self, path: str) -> ImageInfo:
        """Return ``path``'s key and patch grid, without decoding the image."""
        with open(path, "rb") as f:
            data = f.read()
        # The header gives the stored size; decode_image applies no EXIF rotation either.
        width, height = Image.open(io.BytesIO(data)).size
        return ImageInfo(
            key=_content_key(data),
            path=path,
            grid_thw=_grid_thw(height, width, self.spec),
            data=data,
        )

    def num_tokens(self, image: ImageInfo) -> int:
        """Placeholder tokens of ``image`` after ``merge_size x merge_size`` merging."""
        t, h, w = image.grid_thw
        return t * h * w // self.spec.merge_size**2

    def load_pixel_values(
        self, files: Sequence[bytes]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the float32 patches and ``grid_thw (n, 3)`` of image ``files``.

        Patches are ``(num_patches, C * temporal_patch_size * patch_size**2)``.
        Patches are in block order, concatenated in ``files`` order, matching
        ``MultiModalCollator`` and the Qwen-style vision encoders.
        """
        spec = self.spec
        patches, grids = [], []
        for data in files:
            image_patches, grid = vision_to_patches(
                normalize_image(
                    _decode_and_resize(data, spec), spec.image_mean, spec.image_std
                ),
                spec.patch_size,
                spec.temporal_patch_size,
                spec.merge_size,
            )
            patches.append(image_patches)
            grids.append(grid)
        return torch.cat(patches), torch.stack(grids)


@functools.lru_cache(maxsize=64)
def read_image_file(path: str, key: str) -> bytes:
    """Return the bytes of the image file at ``path``, which must hash to ``key``.

    The renderer read them in `ImageLoader.lookup`, but the renderers library keeps only
    ``pixel_values`` and the grid of what `_process_image` returns, so `to_prompt_images`
    reads the file again. Cached: the samples of a GRPO group render together in one
    process, so they share one bytes object, and a message that holds several of them
    pickles the bytes once.
    """
    with open(path, "rb") as f:
        data = f.read()
    if _content_key(data) != key:
        raise ValueError(
            f"{path!r} changed after it was rendered: its content key is "
            f"{_content_key(data)}, the prompt has {key}."
        )
    return data


def _content_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _decode_and_resize(data: bytes, spec: ImageSpec) -> torch.Tensor:
    return decode_and_resize_image(
        data,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
    )


def _grid_thw(height: int, width: int, spec: ImageSpec) -> tuple[int, int, int]:
    """Patch grid of a ``height x width`` image after `_decode_and_resize`."""
    resize_h, resize_w, pad_h, pad_w = resize_to_pixel_budget(
        height,
        width,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
    )
    return (
        1,
        (resize_h + pad_h) // spec.patch_size,
        (resize_w + pad_w) // spec.patch_size,
    )
