# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preprocessed image cache: decode and resize each image once, patchify on load.

RL renders, generates and trains on the same images many times (every sample
of a GRPO group, every epoch, generator and trainer). Decoding and resizing on
each use would put JPEG decode and bicubic resize on the rollout controller's
event loop and ship megabytes of ``pixel_values`` through every RPC. Instead a
dataset's images are processed once, offline, into resized ``uint8`` arrays
keyed by a content hash. Callers pass the key; the generator and the batcher
load the uint8 images, and the generator and the trainer patchify them on the
GPU.

The stored array is ``decode_and_resize_image``'s output (``process_image``
before normalization), so ``patchify(cache.load_images(keys), cache.spec)``
returns exactly the patches ``process_image`` + ``vision_to_patches`` produce
for the cache's `ImageCacheSpec` (e.g. Qwen3.5's pixel budget and block order,
or Muse Spark's merged-patch grid and raster order), at ~8x fewer stored bytes than float32 patches (uint8, no temporal-patch
duplication). Loading takes ~0.6 ms per image; normalizing and patchifying on
the CPU ~20 ms on one thread, and on the GPU it gives the same values.

An image's key is a hash of its file bytes, so `ImageCache.lookup` finds an
image by content wherever the dataset is mounted, without decoding it.

Layout of a cache directory::

    index.json        spec, {key: grid_thw}
    <key>.npy         resized uint8 image, (C, H, W)

Build one with::

    python -m torchtitan.components.data.multimodal.image_cache \\
        --cache-dir /data/slake/image_cache --image-root /data/slake/imgs \\
        --name-glob source.jpg [--spec muse_spark]
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import functools
import hashlib
import json
import logging
import multiprocessing
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from torchtitan.components.data.multimodal.utils.image import (
    decode_and_resize_image,
    normalize_image,
    resize_to_merged_patch_grid,
    resize_to_pixel_budget,
    vision_to_patches,
)
from torchtitan.observability.logging import init_logger

logger = logging.getLogger(__name__)

_INDEX = "index.json"
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True, kw_only=True, slots=True)
class ImageCacheSpec:
    """Preprocessing that produced a cache; defaults match Qwen3-VL / Qwen3.5.

    The arguments of ``process_image`` + ``vision_to_patches`` for the model's
    vision encoder. It must match the model's vision encoder and the SFT data
    pipeline of the same model, so the generator and the trainer see the pixels
    the model was trained on.
    """

    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    """Patches merged per side into one placeholder token (``spatial_merge_size``)."""
    resize: Literal["pixel_budget", "merged_patch_grid"] = "pixel_budget"
    """``resize_to_pixel_budget`` (bounded by ``min_pixels``/``max_pixels``) or
    ``resize_to_merged_patch_grid`` (bounded by ``max_image_tokens``)."""
    min_pixels: int = 65536
    """Lower bound on H * W after the resize."""
    max_pixels: int = 16777216
    """Upper bound on H * W after the resize."""
    max_image_tokens: int = 4096
    patch_order: Literal["block", "raster"] = "block"
    """Patch sequence layout the vision encoder expects (see ``vision_to_patches``)."""
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_json(cls, spec: dict) -> ImageCacheSpec:
        """Read a spec from ``index.json``; fields added later take their defaults."""
        spec = dict(spec)
        spec["image_mean"] = tuple(spec["image_mean"])
        spec["image_std"] = tuple(spec["image_std"])
        return cls(**spec)


IMAGE_CACHE_SPECS = {
    "qwen3_5": ImageCacheSpec(),
    # Muse Spark's SFT pipeline: `torchtitan_recipes.models.muse_spark._muse_spark_mm_dataloader`.
    "muse_spark": ImageCacheSpec(
        patch_size=14, resize="merged_patch_grid", patch_order="raster"
    ),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class CachedImage:
    """One cached image: its content key and raw patch grid ``(T, H, W)``."""

    key: str
    grid_thw: tuple[int, int, int]


class ImageCache:
    """Read-only view of a cache directory written by `build_image_cache`.

    Example:

        cache = ImageCache("/data/slake/image_cache")
        image = cache.lookup("/data/slake/imgs/xmlab1/source.jpg")
        cache.num_tokens(image)  # placeholder tokens after spatial merging
        pixel_values = patchify(cache.load_images([image.key]), cache.spec)
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, _INDEX)) as f:
            index = json.load(f)
        self.spec = ImageCacheSpec.from_json(index["spec"])
        self._grids: dict[str, tuple[int, int, int]] = {
            key: (t, h, w) for key, (t, h, w) in index["images"].items()
        }

    def lookup(self, path: str) -> CachedImage:
        """Return the cached image whose file content equals ``path``'s.

        Reads and hashes the file (no decode).
        """
        with open(path, "rb") as f:
            key = _content_key(f.read())
        if key not in self._grids:
            raise KeyError(
                f"{path!r} is not in the image cache at {self.cache_dir!r}; "
                "add it with build_image_cache."
            )
        return CachedImage(key=key, grid_thw=self._grids[key])

    def get(self, key: str) -> CachedImage:
        """Return the cached image whose content key is ``key``."""
        try:
            grid_thw = self._grids[key]
        except KeyError:
            raise KeyError(
                f"Image {key!r} is not in the image cache at {self.cache_dir!r}."
            ) from None
        return CachedImage(key=key, grid_thw=grid_thw)

    def __iter__(self) -> Iterator[CachedImage]:
        for key, grid_thw in self._grids.items():
            yield CachedImage(key=key, grid_thw=grid_thw)

    def num_tokens(self, image: CachedImage) -> int:
        """Placeholder tokens of ``image`` after ``merge_size x merge_size`` merging."""
        t, h, w = image.grid_thw
        return t * h * w // self.spec.merge_size**2

    def load_images(self, keys: Sequence[str]) -> list[torch.Tensor]:
        """Return the resized uint8 images ``(C, H, W)`` of ``keys``, unnormalized."""
        return [
            torch.from_numpy(np.load(os.path.join(self.cache_dir, f"{key}.npy")))
            for key in keys
        ]


def patchify(images: Sequence[torch.Tensor], spec: ImageCacheSpec) -> torch.Tensor:
    """Normalize resized uint8 images and cut them into concatenated patches.

    These are ``normalize_image`` and ``vision_to_patches``. They run on the images'
    device and are elementwise float32 ops plus reshapes, so the CPU and the GPU
    give bitwise-equal patches.
    """
    patches = []
    for image in images:
        image_patches, _ = vision_to_patches(
            normalize_image(image, spec.image_mean, spec.image_std),
            spec.patch_size,
            spec.temporal_patch_size,
            spec.merge_size,
            patch_order=spec.patch_order,
        )
        patches.append(image_patches)
    return torch.cat(patches)


def patchify_flat(
    flat_images: torch.Tensor, grid_thw: list[list[int]], spec: ImageCacheSpec
) -> torch.Tensor:
    """`patchify` for `ImageCache.load_images` outputs flattened and concatenated in
    ``grid_thw`` order.

    Cached images are ``(3, h * patch_size, w * patch_size)`` with ``t == 1`` (see
    `_process_one`).
    """
    assert all(t == 1 for t, _, _ in grid_thw), grid_thw
    shapes = [(3, h * spec.patch_size, w * spec.patch_size) for _, h, w in grid_thw]
    flat = flat_images.split([c * h * w for c, h, w in shapes])
    return patchify(
        [image.view(shape) for image, shape in zip(flat, shapes, strict=True)], spec
    )


def _content_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _process_one(
    path: str, cache_dir: str, spec: ImageCacheSpec
) -> tuple[str, tuple[int, int, int]]:
    with open(path, "rb") as f:
        data = f.read()
    key = _content_key(data)
    resize_fn = (
        resize_to_pixel_budget
        if spec.resize == "pixel_budget"
        else functools.partial(
            resize_to_merged_patch_grid, max_image_tokens=spec.max_image_tokens
        )
    )
    image = decode_and_resize_image(
        data,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
        resize_fn=resize_fn,
    )
    _, height, width = image.shape
    # Always (re)write: a .npy left by a build with another spec must not survive.
    out = os.path.join(cache_dir, f"{key}.npy")
    tmp = f"{out}.{os.getpid()}.tmp.npy"
    np.save(tmp, image.numpy())
    os.replace(tmp, out)
    return key, (1, height // spec.patch_size, width // spec.patch_size)


def build_image_cache(
    image_paths: Iterable[str],
    cache_dir: str,
    spec: ImageCacheSpec = ImageCacheSpec(),
    *,
    num_workers: int = 8,
) -> ImageCache:
    """Decode, resize and store every image in ``image_paths`` under ``cache_dir``.

    Identical files share one entry. An existing cache must use the same spec;
    the images are added to it. ``num_workers=1`` processes them in this process.
    """
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, _INDEX)
    index = {"spec": dataclasses.asdict(spec), "images": {}}
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        if ImageCacheSpec.from_json(existing["spec"]) != spec:
            raise ValueError(
                f"{cache_dir!r} was built with {existing['spec']}, not {index['spec']}; "
                "use a new cache_dir for another spec."
            )
        index["images"] = existing["images"]
    paths = sorted(set(image_paths))
    args = (paths, [cache_dir] * len(paths), [spec] * len(paths))
    if num_workers == 1:
        results = list(map(_process_one, *args))
    else:
        # forkserver: the caller may be multi-threaded (torch), where fork can deadlock.
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("forkserver"),
        ) as pool:
            results = list(pool.map(_process_one, *args, chunksize=16))
    for key, grid_thw in results:
        index["images"][key] = list(grid_thw)
    tmp = f"{index_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, index_path)
    return ImageCache(cache_dir)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--image-root", required=True, help="Directory searched recursively for images."
    )
    parser.add_argument(
        "--spec",
        choices=sorted(IMAGE_CACHE_SPECS),
        default="qwen3_5",
        help="Preprocessing of the model the cache is for.",
    )
    parser.add_argument("--max-pixels", type=int, help="Override the spec's budget.")
    parser.add_argument("--min-pixels", type=int, help="Override the spec's budget.")
    parser.add_argument(
        "--name-glob", default="*", help="Only cache files whose name matches."
    )
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    init_logger()
    paths = [
        os.path.join(root, name)
        for root, _, names in os.walk(args.image_root)
        for name in names
        if name.lower().endswith(_IMAGE_SUFFIXES)
        and fnmatch.fnmatch(name, args.name_glob)
    ]
    spec = IMAGE_CACHE_SPECS[args.spec]
    overrides = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    spec = dataclasses.replace(
        spec, **{name: value for name, value in overrides.items() if value is not None}
    )
    build_image_cache(paths, args.cache_dir, spec, num_workers=args.num_workers)
    logger.info(f"Cached {len(paths)} images in {args.cache_dir}")


if __name__ == "__main__":
    _main()
