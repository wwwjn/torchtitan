# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from PIL import Image

from torchtitan.components.data.multimodal.image_cache import (
    build_image_cache,
    ImageCacheSpec,
    patchify,
)


def test_patchify_on_gpu_matches_cpu(tmp_path) -> None:
    # The generator patchifies on the GPU; the trainer side must see the same values.
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (300, 200)), ("b.png", (40, 900))):
        path = tmp_path / name
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    spec = ImageCacheSpec(min_pixels=32 * 32, max_pixels=256 * 256)
    cache = build_image_cache(paths, str(tmp_path / "cache"), spec, num_workers=1)
    images = cache.load_images([image.key for image in cache])

    on_gpu = patchify([image.cuda() for image in images], spec)

    assert torch.equal(on_gpu.cpu(), patchify(images, spec))
