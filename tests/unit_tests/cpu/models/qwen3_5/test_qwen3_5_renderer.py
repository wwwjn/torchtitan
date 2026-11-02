# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The image-cache Qwen3.5 renderer renders the same tokens as the library's.

Needs a Qwen3.5 tokenizer directory in ``QWEN3_5_TOKENIZER``; skipped when unset.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from renderers import create_renderer, CustomRendererConfig, Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from torchtitan.components.data.multimodal.image_cache import (
    build_image_cache,
    patchify,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.models.qwen3_5.renderer import (
    Qwen35ImageCacheRenderer,
    Qwen35ImageCacheRendererConfig,
)

from tests.utils import qwen3_5_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return qwen3_5_tokenizer()


@pytest.fixture
def image_paths(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (300, 200)), ("b.png", (640, 512))):
        path = tmp_path / name
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


def _hf_image_processor():
    """The Hugging Face processor Qwen3.5 checkpoints ship (preprocessor_config.json)."""
    transformers = pytest.importorskip("transformers")
    return transformers.Qwen2VLImageProcessorFast(
        size={"longest_edge": 16777216, "shortest_edge": 65536},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


def _messages(image_paths):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_paths[0]},
                {"type": "text", "text": "Compare this image"},
                {"type": "image", "path": image_paths[1]},
                {"type": "text", "text": "with this one."},
            ],
        }
    ]


def test_matches_library_renderer(tokenizer, tmp_path, image_paths) -> None:
    cache = build_image_cache(image_paths, str(tmp_path / "cache"), num_workers=1)
    hf_processor = _hf_image_processor()
    library = Qwen35Renderer(
        tokenizer,
        Qwen35RendererConfig(enable_thinking=False),
        processor=SimpleNamespace(image_processor=hf_processor),
    )
    cached = Qwen35ImageCacheRenderer(
        tokenizer,
        Qwen35ImageCacheRendererConfig(
            enable_thinking=False, image_cache_dir=str(tmp_path / "cache")
        ),
    )

    expected = library.render(_messages(image_paths), add_generation_prompt=True)
    actual = cached.render(_messages(image_paths), add_generation_prompt=True)

    assert actual.token_ids == expected.token_ids
    assert actual.message_indices == expected.message_indices
    expected_mm, actual_mm = expected.multi_modal_data, actual.multi_modal_data
    assert actual_mm.mm_placeholders == expected_mm.mm_placeholders
    assert actual_mm.mm_hashes["image"] == [cache.lookup(p).key for p in image_paths]
    for actual_item, expected_item in zip(
        actual_mm.mm_items["image"], expected_mm.mm_items["image"]
    ):
        assert actual_item["pixel_values"] is None
        np.testing.assert_array_equal(
            actual_item["image_grid_thw"], expected_item["image_grid_thw"]
        )

    # The cache's patches are what the HF processor feeds the encoder (equal up to
    # float32 rounding of the normalization).
    pixel_values = patchify(cache.load_images(actual_mm.mm_hashes["image"]), cache.spec)
    hf_pixel_values = np.concatenate(
        [item["pixel_values"] for item in expected_mm.mm_items["image"]]
    )
    torch.testing.assert_close(pixel_values, torch.from_numpy(hf_pixel_values))


def test_builds_through_custom_renderer_config(
    tokenizer, tmp_path, image_paths
) -> None:
    cache = build_image_cache(image_paths, str(tmp_path / "cache"), num_workers=1)
    config = from_renderers(
        CustomRendererConfig(
            import_path="torchtitan.models.qwen3_5.renderer.Qwen35ImageCacheRenderer",
            enable_thinking=False,
            image_cache_dir=str(tmp_path / "cache"),
        )
    )
    renderer = create_renderer(tokenizer, config.get_renderers_config())
    assert isinstance(renderer, Qwen35ImageCacheRenderer)
    rendered = renderer.render(_messages(image_paths), add_generation_prompt=True)
    assert rendered.multi_modal_data.mm_hashes["image"] == [
        cache.lookup(path).key for path in image_paths
    ]


def test_rejects_image_cache_max(tmp_path, image_paths) -> None:
    build_image_cache(image_paths, str(tmp_path / "cache"), num_workers=1)
    with pytest.raises(ValueError, match="image_cache_max"):
        Qwen35ImageCacheRendererConfig(
            image_cache_dir=str(tmp_path / "cache"), image_cache_max=1
        )


def test_rejects_image_parts_without_a_path(tokenizer, tmp_path, image_paths) -> None:
    build_image_cache(image_paths, str(tmp_path / "cache"), num_workers=1)
    renderer = Qwen35ImageCacheRenderer(
        tokenizer,
        Qwen35ImageCacheRendererConfig(image_cache_dir=str(tmp_path / "cache")),
    )
    message = {"role": "user", "content": [{"type": "image", "url": "http://x/y.png"}]}
    with pytest.raises(ValueError, match="needs image parts with a 'path'"):
        renderer.render([message], add_generation_prompt=True)
