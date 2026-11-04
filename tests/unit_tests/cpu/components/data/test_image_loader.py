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

from torchtitan.components.data.multimodal.image_loader import (
    decode_flat,
    ImageLoader,
    ImageSpec,
    patchify,
    read_image_file,
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


def test_patches_and_header_grids_match_process_image(image_paths) -> None:
    spec = ImageSpec(min_pixels=32 * 32, max_pixels=256 * 256)
    loader = ImageLoader.Config(spec=spec).build()

    images = [loader.lookup(path) for path in image_paths]
    pixel_values = patchify(loader.load_images([image.data for image in images]), spec)
    grid_thw = torch.tensor([image.grid_thw for image in images])

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
    # The grids `lookup` read from the headers are the decoded images' grids.
    for image, (patches, grid) in zip(images, expected):
        assert image.grid_thw == tuple(grid.tolist())
        assert loader.num_tokens(image) == patches.shape[0] // spec.merge_size**2


def test_lookup_keys_an_image_by_content_from_any_path(
    tmp_path, image_paths, monkeypatch
) -> None:
    loader = ImageLoader.Config().build()
    moved = tmp_path / "elsewhere" / "renamed.png"
    moved.parent.mkdir()
    shutil.copy(image_paths[0], moved)
    monkeypatch.chdir(moved.parent)
    found, original = loader.lookup("renamed.png"), loader.lookup(image_paths[0])
    assert (found.key, found.grid_thw) == (original.key, original.grid_thw)
    assert len({loader.lookup(path).key for path in image_paths}) == 3


def test_read_image_file_shares_the_bytes_and_rejects_a_changed_file(
    image_paths,
) -> None:
    loader = ImageLoader.Config().build()
    image = loader.lookup(image_paths[0])
    data = read_image_file(image.path, image.key)
    assert data == image.data
    # One object for every sample of a group, so a message holding them pickles it once.
    assert read_image_file(image.path, image.key) is data
    rendered = loader.lookup(image_paths[2])
    shutil.copy(image_paths[1], image_paths[2])
    with pytest.raises(ValueError, match="changed after it was rendered"):
        read_image_file(rendered.path, rendered.key)


def test_decode_flat_matches_load_images_and_checks_grids(image_paths) -> None:
    loader = ImageLoader.Config().build()
    images = [loader.lookup(path) for path in image_paths]
    files = torch.frombuffer(
        bytearray(b"".join(image.data for image in images)), dtype=torch.uint8
    )
    sizes = [len(image.data) for image in images]
    grids = [list(image.grid_thw) for image in images]
    expected = torch.cat(
        [
            image.flatten()
            for image in loader.load_images([image.data for image in images])
        ]
    )
    assert torch.equal(decode_flat(files, sizes, grids, loader.spec), expected)
    grids[1][1] += 1
    with pytest.raises(ValueError, match="decodes to grid"):
        decode_flat(files, sizes, grids, loader.spec)


def test_largest_grid() -> None:
    loader = ImageLoader.Config(
        spec=ImageSpec(min_pixels=32 * 32, max_pixels=256 * 256)
    ).build()
    # 256 x 256 pixels in 32-pixel merged tokens: at most 8 x 8 = 64 tokens.
    assert loader.largest_grid(10_000) == (1, 16, 16)
    assert loader.largest_grid(20) == (1, 8, 8)
    with pytest.raises(ValueError, match="fits"):
        loader.largest_grid(0)
