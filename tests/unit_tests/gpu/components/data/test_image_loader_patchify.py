# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from PIL import Image

from torchtitan.components.data.multimodal.image_loader import (
    ImageLoader,
    ImageSpec,
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
    spec = ImageSpec(min_pixels=32 * 32, max_pixels=256 * 256)
    loader = ImageLoader.Config(spec=spec).build()
    images = loader.load_images([loader.lookup(path).data for path in paths])

    on_gpu = patchify([image.cuda() for image in images], spec)

    assert torch.equal(on_gpu.cpu(), patchify(images, spec))
