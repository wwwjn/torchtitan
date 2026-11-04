# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for sending prompt image files to the generator as RDMA buffers
(`Controller.Config.rdma_min_image_bytes`), with fakes and once over real Monarch (TCP)."""

import asyncio
import dataclasses
from types import SimpleNamespace

import cloudpickle
import pytest
from monarch.actor import Actor, endpoint, this_host

from torchtitan.rl.actors.monarch.rdma_prompt_image import (
    MonarchRDMAPromptImage,
    register_rdma_prompt_image,
)
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.rdma_prompt_image import (
    RDMAPromptImage,
    read_prompt_images,
)
from torchtitan.rl.rollout.types import PromptImage


def _image(key: str, data: bytes) -> PromptImage:
    return PromptImage(
        key=key, path=f"{key}.png", grid_thw=(1, 4, 4), offset=1, length=4, data=data
    )


@dataclasses.dataclass
class _FakeRDMAPromptImage:
    """Stands in for `MonarchRDMAPromptImage`: keeps the image and records the drop."""

    image: PromptImage
    dropped: bool = False

    async def read(self) -> PromptImage:
        return self.image

    async def drop(self) -> None:
        self.dropped = True


class _FakeRouter:
    def __init__(self, error: Exception | None = None) -> None:
        self.images = None
        self.error = error
        self.started = asyncio.Event()
        self.block = False
        self.generate = SimpleNamespace(call_one=self._generate)

    async def _generate(self, prompt_token_ids, *, images, **kwargs):
        self.images = images
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return "completion"


def _generate_fn(router: _FakeRouter, rdma_min_image_bytes: int | None, register=None):
    controller = SimpleNamespace(
        generator_router=router,
        config=SimpleNamespace(rdma_min_image_bytes=rdma_min_image_bytes),
        actor_system=SimpleNamespace(
            register_rdma_prompt_image=register or _FakeRDMAPromptImage
        ),
    )
    return Controller._make_generate_fn(controller, metrics_prefix="generator")


def test_generate_fn_sends_large_files_as_rdma_buffers_and_drops_them() -> None:
    small, large = _image("small", b"123"), _image("large", b"12345678")
    router = _FakeRouter()

    result = asyncio.run(
        _generate_fn(router, rdma_min_image_bytes=4)(
            [1], request_id="r", group_id=0, images=(small, large)
        )
    )

    assert result == "completion"
    sent_small, sent_large = router.images
    assert sent_small is small
    assert isinstance(sent_large, RDMAPromptImage) and sent_large.image is large
    # Rank 0 has read the buffer once `generate` returns.
    assert sent_large.dropped


def test_generate_fn_drops_the_buffers_when_generate_fails() -> None:
    router = _FakeRouter(error=RuntimeError("generator died"))

    with pytest.raises(RuntimeError, match="generator died"):
        asyncio.run(
            _generate_fn(router, rdma_min_image_bytes=1)(
                [1], request_id="r", group_id=0, images=(_image("a", b"12"),)
            )
        )

    [sent] = router.images
    assert sent.dropped


def test_generate_fn_drops_the_registered_buffers_when_a_registration_fails() -> None:
    registered = []

    def register(image: PromptImage) -> _FakeRDMAPromptImage:
        if image.key == "b":
            raise RuntimeError("no RDMA backend")
        registered.append(_FakeRDMAPromptImage(image))
        return registered[-1]

    router = _FakeRouter()
    with pytest.raises(RuntimeError, match="no RDMA backend"):
        asyncio.run(
            _generate_fn(router, rdma_min_image_bytes=1, register=register)(
                [1],
                request_id="r",
                group_id=0,
                images=(_image("a", b"12"), _image("b", b"34")),
            )
        )

    [first] = registered
    assert first.dropped and router.images is None


def test_generate_fn_drops_the_buffers_when_cancelled() -> None:
    router = _FakeRouter()
    router.block = True

    async def run() -> None:
        call = asyncio.create_task(
            _generate_fn(router, rdma_min_image_bytes=1)(
                [1], request_id="r", group_id=0, images=(_image("a", b"12"),)
            )
        )
        await router.started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(run())

    [sent] = router.images
    assert sent.dropped


def test_generate_fn_with_the_monarch_registration_pickles() -> None:
    # Monarch pickles the GenerateFn into every rollout worker call.
    controller = SimpleNamespace(
        generator_router=None,
        config=SimpleNamespace(rdma_min_image_bytes=1024),
        actor_system=SimpleNamespace(
            register_rdma_prompt_image=register_rdma_prompt_image
        ),
    )
    generate_fn = Controller._make_generate_fn(controller, metrics_prefix="generator")

    assert callable(cloudpickle.loads(cloudpickle.dumps(generate_fn)))


def test_read_prompt_images_reads_only_rdma_images() -> None:
    plain, remote = _image("a", b"12"), _image("b", b"34")

    images = asyncio.run(read_prompt_images([plain, _FakeRDMAPromptImage(remote)]))

    assert images == (plain, remote)
    assert [image.data for image in images] == [b"12", b"34"]


class _GeneratorRank0(Actor):
    """Reads prompt images as generator rank 0 does."""

    @endpoint
    async def read(self, images: list[MonarchRDMAPromptImage]) -> list[PromptImage]:
        return list(await read_prompt_images(images))


def test_monarch_rdma_prompt_image_round_trips_the_file_bytes(monkeypatch) -> None:
    # Real Monarch RDMA over TCP (works on any CI host; the generator process inherits the env var).
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    images = [_image("a", bytes(range(256)) * 4), _image("b", b"\xff\xd8\xff")]

    async def run() -> list[PromptImage]:
        mesh = this_host().spawn_procs(per_host={"procs": 1})
        rdma_images = [register_rdma_prompt_image(image) for image in images]
        try:
            # The handles carry no file bytes; the generator process reads them.
            assert all(rdma_image.image.data == b"" for rdma_image in rdma_images)
            rank0 = mesh.spawn("rank0", _GeneratorRank0)
            return await rank0.read.call_one(rdma_images)
        finally:
            await asyncio.gather(*(rdma_image.drop() for rdma_image in rdma_images))
            await mesh.stop()

    read = asyncio.run(run())

    assert read == images
    assert [image.data for image in read] == [image.data for image in images]
