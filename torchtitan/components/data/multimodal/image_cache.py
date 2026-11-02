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
keyed by a content hash. Callers pass the key; only the processes that need
pixels (generator, batcher) load and patchify them.

The stored array is ``decode_and_resize_image``'s output (``process_image``
before normalization), so ``ImageCache.load_pixel_values`` returns exactly the
patches ``process_image`` + ``vision_to_patches`` produce, at ~8x fewer stored
bytes than float32 patches (uint8, no temporal-patch duplication). Normalizing
and patchifying on the CPU costs ~20 ms per image on one thread; callers on a
hot path load the uint8 images (``load_images``, ~0.6 ms) and run `patchify` on
the GPU instead, which gives the same values.

An image's key is a hash of its file bytes, so `ImageCache.lookup` finds an
image by content wherever the dataset is mounted, without decoding it.

Layout of a cache directory::

    index.json        spec, {key: grid_thw}
    <key>.npy         resized uint8 image, (C, H, W)

Build one with::

    python -m torchtitan.components.data.multimodal.image_cache \\
        --cache-dir /data/slake/image_cache --image-root /data/slake/imgs \\
        --name-glob source.jpg
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import logging
import multiprocessing
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from torchtitan.components.data.multimodal.utils.image import (
    decode_and_resize_image,
    normalize_image,
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
        pixel_values, grid_thw = cache.load_pixel_values([image.key])
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, _INDEX)) as f:
            index = json.load(f)
        spec = index["spec"]
        spec["image_mean"] = tuple(spec["image_mean"])
        spec["image_std"] = tuple(spec["image_std"])
        self.spec = ImageCacheSpec(**spec)
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

    def load_pixel_values(
        self, keys: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 patches and ``grid_thw (n, 3)``.

        Patches are ``(num_patches, C * temporal_patch_size * patch_size**2)``.
        Patches are in block order, concatenated in ``keys`` order, matching
        ``MultiModalCollator`` and the Qwen-style vision encoders.
        """
        grid_thw = torch.tensor([self._grids[key] for key in keys], dtype=torch.int64)
        return patchify(self.load_images(keys), self.spec), grid_thw

    def load_images(self, keys: Sequence[str]) -> list[torch.Tensor]:
        """Return the resized uint8 images ``(C, H, W)`` of ``keys``, unnormalized."""
        return [
            torch.from_numpy(np.load(os.path.join(self.cache_dir, f"{key}.npy")))
            for key in keys
        ]


def patchify(images: Sequence[torch.Tensor], spec: ImageCacheSpec) -> torch.Tensor:
    """Normalize resized uint8 images and cut them into concatenated block-order patches.

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
        )
        patches.append(image_patches)
    return torch.cat(patches)


def _content_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _process_one(
    path: str, cache_dir: str, spec: ImageCacheSpec
) -> tuple[str, tuple[int, int, int]]:
    with open(path, "rb") as f:
        data = f.read()
    key = _content_key(data)
    image = decode_and_resize_image(
        data,
        patch_size=spec.patch_size,
        merge_size=spec.merge_size,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
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
        if existing["spec"] != json.loads(json.dumps(index["spec"])):
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
    defaults = ImageCacheSpec()
    parser.add_argument("--max-pixels", type=int, default=defaults.max_pixels)
    parser.add_argument("--min-pixels", type=int, default=defaults.min_pixels)
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
    build_image_cache(
        paths,
        args.cache_dir,
        ImageCacheSpec(min_pixels=args.min_pixels, max_pixels=args.max_pixels),
        num_workers=args.num_workers,
    )
    logger.info(f"Cached {len(paths)} images in {args.cache_dir}")


if __name__ == "__main__":
    _main()
