# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from torchtitan.components.data.multimodal.image_cache import (
    build_image_cache,
    ImageCacheSpec,
)
from torchtitan.components.data.multimodal.utils.image import (
    process_image,
    vision_to_patches,
)


@pytest.fixture
def image_paths(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for name, size in (
        ("a.png", (300, 200)),
        ("b.jpg", (64, 48)),
        ("c.png", (40, 900)),
    ):
        path = tmp_path / "imgs" / name
        path.parent.mkdir(exist_ok=True)
        pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


def test_cached_patches_match_process_image(tmp_path, image_paths) -> None:
    spec = ImageCacheSpec(min_pixels=32 * 32, max_pixels=256 * 256)
    cache = build_image_cache(image_paths, str(tmp_path / "cache"), spec, num_workers=1)

    images = [cache.lookup(path) for path in image_paths]
    pixel_values, grid_thw = cache.load_pixel_values([image.key for image in images])

    expected = [
        vision_to_patches(
            process_image(path, min_pixels=spec.min_pixels, max_pixels=spec.max_pixels),
            spec.patch_size,
            spec.temporal_patch_size,
            spec.merge_size,
        )
        for path in image_paths
    ]
    assert torch.equal(pixel_values, torch.cat([patches for patches, _ in expected]))
    assert torch.equal(grid_thw, torch.stack([grid for _, grid in expected]))
    for image, (patches, grid) in zip(images, expected):
        assert image.grid_thw == tuple(grid.tolist())
        assert cache.num_tokens(image) == patches.shape[0] // spec.merge_size**2


def test_identical_files_share_a_key_and_cache_extends(tmp_path, image_paths) -> None:
    copy = str(tmp_path / "imgs" / "a_copy.png")
    shutil.copy(image_paths[0], copy)
    cache_dir = str(tmp_path / "cache")
    build_image_cache(image_paths[:1], cache_dir, num_workers=1)
    cache = build_image_cache([copy, *image_paths[1:]], cache_dir, num_workers=1)
    assert cache.lookup(copy).key == cache.lookup(image_paths[0]).key
    assert len({cache.lookup(p).key for p in image_paths}) == 3

    with pytest.raises(ValueError, match="was built with"):
        build_image_cache(image_paths, cache_dir, ImageCacheSpec(max_pixels=1 << 16))


def test_lookup_of_an_uncached_path_raises(tmp_path, image_paths) -> None:
    cache = build_image_cache(image_paths[:1], str(tmp_path / "cache"), num_workers=1)
    with pytest.raises(KeyError, match="not in the image cache"):
        cache.lookup(image_paths[1])


def test_lookup_finds_an_image_by_content_from_any_path(
    tmp_path, image_paths, monkeypatch
) -> None:
    cache = build_image_cache(image_paths, str(tmp_path / "cache"), num_workers=1)
    moved = tmp_path / "elsewhere" / "renamed.png"
    moved.parent.mkdir()
    shutil.copy(image_paths[0], moved)
    monkeypatch.chdir(moved.parent)
    assert cache.lookup("renamed.png") == cache.lookup(image_paths[0])
