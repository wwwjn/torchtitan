# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.rl.torch_transport_rdma`: the registered-buffer pool, range reads,
and the setup probe.

Most tests use `LoopbackTransport`, an in-process stand-in for torch transport. The tests marked
`requires_nixl` use the real NIXL backend (two transports in this process) and skip without it;
locally, run them with the nixl overlay (reports/19_rl_data_plane/14, section "Environment").
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from torchtitan.rl.torch_transport_rdma import (
    buffer_size_class,
    check_rdma_probes,
    make_rdma_probe,
    RDMAByteBuffers,
    RegisteredBufferPool,
)

MiB = 1 << 20


@dataclass(frozen=True)
class _LoopbackRemoteBuffer:
    key: int
    offset: int
    length: int

    def slice(self, offset: int, length: int) -> _LoopbackRemoteBuffer:
        assert 0 <= offset and offset + length <= self.length
        return replace(self, offset=self.offset + offset, length=length)


@dataclass(frozen=True)
class _LoopbackView:
    tensor: torch.Tensor
    offset: int
    length: int


class _LoopbackMemory:
    def __init__(self, key: int, tensor: torch.Tensor) -> None:
        self.key = key
        self.tensor = tensor

    def to_remote_buffer(self) -> _LoopbackRemoteBuffer:
        return _LoopbackRemoteBuffer(self.key, 0, self.tensor.numel())

    def to_mutable_view(self, offset: int, length: int) -> _LoopbackView:
        return _LoopbackView(self.tensor, offset, length)


class _DoneWork:
    def wait(self) -> None:
        pass


class LoopbackTransport:
    """In-process stand-in for a torch transport: registrations of every `LoopbackTransport`
    sharing `registry` are readable by the others, and each call is counted.

    `fail_reads` makes every read raise, as a host whose RDMA transfers fail."""

    def __init__(self, registry: dict[int, torch.Tensor], name: str) -> None:
        self.registry = registry
        self.name = name
        self.num_registrations = 0
        self.num_unregistrations = 0
        self.read_ranges: list[tuple[int, int]] = []
        self.connected: list[bytes] = []
        self.fail_reads = False

    def bind(self) -> bytes:
        return self.name.encode()

    def connect(self, peer_url: bytes) -> int:
        self.connected.append(peer_url)
        return 0

    def register_memory(self, tensor: torch.Tensor) -> _LoopbackMemory:
        key = len(self.registry)
        self.registry[key] = tensor
        self.num_registrations += 1
        return _LoopbackMemory(key, tensor)

    def unregister_memory(self, memory: _LoopbackMemory) -> None:
        self.num_unregistrations += 1
        del self.registry[memory.key]

    def read(
        self, local: _LoopbackView, remote: _LoopbackRemoteBuffer, *, async_op: bool
    ) -> _DoneWork:
        assert async_op and local.length == remote.length
        if self.fail_reads:
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
        source = self.registry[remote.key][
            remote.offset : remote.offset + remote.length
        ]
        local.tensor[local.offset : local.offset + local.length] = source
        self.read_ranges.append((remote.offset, remote.length))
        return _DoneWork()


def loopback_buffers() -> tuple[RDMAByteBuffers, RDMAByteBuffers]:
    """An owner's and a reader's `RDMAByteBuffers` over `LoopbackTransport`s that see each
    other's registrations."""
    registry: dict[int, torch.Tensor] = {}
    return (
        RDMAByteBuffers(LoopbackTransport(registry, "owner")),
        RDMAByteBuffers(LoopbackTransport(registry, "reader")),
    )


def test_buffer_size_class_rounds_up_by_at_most_a_quarter() -> None:
    assert buffer_size_class(1) == 64 * 1024
    assert buffer_size_class(64 * 1024) == 64 * 1024
    assert buffer_size_class(64 * 1024 + 1) == 80 * 1024
    assert buffer_size_class(16 * MiB) == 16 * MiB
    assert buffer_size_class(17 * MiB) == 20 * MiB
    assert buffer_size_class(66 * MiB) == 80 * MiB
    for num_bytes in range(64 * 1024, 64 * MiB, 997 * 1024 + 13):
        size = buffer_size_class(num_bytes)
        assert num_bytes <= size <= num_bytes * 1.25


def test_a_released_buffer_is_reused_for_an_equal_or_smaller_request() -> None:
    transport = LoopbackTransport({}, "owner")
    pool = RegisteredBufferPool(transport)

    first = pool.acquire(5 * MiB)
    pool.release(first)

    assert pool.acquire(5 * MiB) == first
    pool.release(first)
    assert pool.acquire(1 * MiB) == first
    assert transport.num_registrations == 1
    assert pool.registered_bytes == 5 * MiB


def test_a_request_no_free_buffer_fits_registers_one_of_its_size_class() -> None:
    transport = LoopbackTransport({}, "owner")
    pool = RegisteredBufferPool(transport)
    small = pool.acquire(4 * MiB)
    pool.release(small)

    large = pool.acquire(17 * MiB)

    assert large != small
    assert pool.buffer(large).tensor.numel() == 20 * MiB
    assert transport.num_registrations == 2
    assert pool.registered_bytes == 24 * MiB
    assert pool.free_bytes == 4 * MiB


def test_the_pool_takes_the_smallest_free_buffer_that_fits() -> None:
    pool = RegisteredBufferPool(LoopbackTransport({}, "owner"))
    indices = {size: pool.acquire(size * MiB) for size in (4, 16, 8)}
    for index in indices.values():
        pool.release(index)

    assert pool.acquire(5 * MiB) == indices[8]
    assert pool.acquire(5 * MiB) == indices[16]
    assert pool.acquire(4 * MiB) == indices[4]


def test_released_buffers_stay_registered() -> None:
    """Buffers go back to the free list, never to the transport: a reader may hold their
    metadata, and NIXL would match a later buffer at the same address to it."""
    owner, _ = loopback_buffers()
    transport = owner._transport
    for size in (4 * MiB, 16 * MiB, 4 * MiB, 1 * MiB, 16 * MiB):
        handles = [owner.allocate(size)[1] for _ in range(3)]
        for handle in handles:
            owner.release(handle)

    assert transport.num_unregistrations == 0
    assert transport.num_registrations == 6  # 3 x 4 MiB, then 3 x 16 MiB
    assert owner.pool.free_bytes == owner.pool.registered_bytes


def test_a_buffer_released_twice_asserts() -> None:
    pool = RegisteredBufferPool(LoopbackTransport({}, "owner"))
    index = pool.acquire(MiB)
    pool.release(index)
    with pytest.raises(AssertionError, match="released twice"):
        pool.release(index)


def test_read_returns_each_range_and_connects_once_per_owner() -> None:
    owner, reader = loopback_buffers()
    tensor, handle = owner.allocate(1000)
    tensor.copy_(torch.arange(1000) % 251)
    other_tensor, other_handle = owner.allocate(10)
    other_tensor.fill_(7)

    chunks = reader.read([(handle, [(0, 4), (500, 3)]), (other_handle, [(2, 5)])])
    reader.read([(handle, [(999, 1)])])

    assert [[chunk.tolist() for chunk in request] for request in chunks] == [
        [[0, 1, 2, 3], [249, 250, 0]],
        [[7] * 5],
    ]
    assert reader._transport.connected == [b"owner"]
    # The staging buffer went back to the reader's pool.
    assert reader.pool.free_bytes == reader.pool.registered_bytes


def test_a_torch_transport_that_fails_to_start_raises_value_error_naming_the_host(
    monkeypatch,
) -> None:
    def new_transport(name, device):
        raise ModuleNotFoundError("No module named 'nixl'")

    monkeypatch.setattr("torch.distributed._transport.new_transport", new_transport)

    with pytest.raises(ValueError) as error:
        RDMAByteBuffers()

    message = str(error.value)
    assert socket.gethostname() in message
    assert "No module named 'nixl'" in message
    assert "rdma_min_group_bytes=None" in message


def test_a_probe_read_that_fails_raises_value_error_naming_both_hosts() -> None:
    owner, reader = loopback_buffers()
    probe = replace(make_rdma_probe(owner), host="rollout-host-3")
    check_rdma_probes(reader, [probe])
    reader._transport.fail_reads = True

    with pytest.raises(ValueError) as error:
        check_rdma_probes(reader, [probe])

    message = str(error.value)
    assert "from rollout-host-3: RuntimeError: NIXL_ERR_REMOTE_DISCONNECT" in message
    assert socket.gethostname() in message
    assert "rdma_min_group_bytes=None" in message


def test_a_probe_that_reads_wrong_bytes_raises_value_error() -> None:
    owner, reader = loopback_buffers()
    probe = make_rdma_probe(owner)
    owner.pool.buffer(probe.buffer.buffer_index).tensor[1500] += 1

    with pytest.raises(ValueError, match=f"from {socket.gethostname()}: wrong bytes"):
        check_rdma_probes(reader, [probe])


def _nixl_available() -> bool:
    try:
        from torch.distributed._transport.nixl import NIXLTransport
    except ImportError:
        return False
    return NIXLTransport.supported()


requires_nixl = pytest.mark.skipif(
    not _nixl_available(), reason="needs torch transport with the nixl package"
)


def _stamp(tensor: torch.Tensor, value: int) -> None:
    tensor.copy_((torch.arange(tensor.numel()) + value) % 251)


@requires_nixl
def test_nixl_reads_slices_after_freeing_small_buffers_for_larger_ones() -> None:
    """The stale-section case: a NIXL reader keeps each buffer it read as its owner registered
    it. Freeing 4 MiB buffers and registering 16 MiB ones over the freed addresses made reads of
    the second half fail with NIXL_ERR_NOT_FOUND. The pool keeps the 4 MiB buffers registered,
    so the 16 MiB ones land elsewhere and every slice reads the right bytes."""
    owner, reader = RDMAByteBuffers(), RDMAByteBuffers()
    small = []
    for value in range(16):
        tensor, handle = owner.allocate(4 * MiB)
        _stamp(tensor, value)
        small.append(handle)
    for value, handle in enumerate(small):
        [[chunk]] = reader.read([(handle, [(0, 4 * MiB)])])
        np.testing.assert_array_equal(chunk, (np.arange(4 * MiB) + value) % 251)
    for handle in small:
        owner.release(handle)

    large = []
    for value in range(16):
        tensor, handle = owner.allocate(16 * MiB)
        _stamp(tensor, 100 + value)
        large.append(handle)

    assert owner.pool.num_registrations == 32
    expected_all = np.arange(16 * MiB)
    for value, handle in enumerate(large):
        [[second_half, inside]] = reader.read(
            [(handle, [(8 * MiB, 8 * MiB), (4 * MiB + 64, 64)])]
        )
        expected = (expected_all + 100 + value) % 251
        np.testing.assert_array_equal(second_half, expected[8 * MiB :])
        np.testing.assert_array_equal(inside, expected[4 * MiB + 64 : 4 * MiB + 128])


@requires_nixl
def test_nixl_reuses_large_buffers_for_smaller_groups_without_registering() -> None:
    owner, reader = RDMAByteBuffers(), RDMAByteBuffers()
    large = [owner.allocate(16 * MiB)[1] for _ in range(16)]
    for handle in large:
        reader.read([(handle, [(0, 16)])])
        owner.release(handle)

    small = []
    for value in range(16):
        tensor, handle = owner.allocate(4 * MiB)
        _stamp(tensor, value)
        small.append(handle)

    assert owner.pool.num_registrations == 16
    for value, handle in enumerate(small):
        [[second_half]] = reader.read([(handle, [(2 * MiB, 2 * MiB)])])
        np.testing.assert_array_equal(
            second_half, (np.arange(2 * MiB, 4 * MiB) + value) % 251
        )


@requires_nixl
def test_nixl_probe_passes_between_two_transports() -> None:
    owner, reader = RDMAByteBuffers(), RDMAByteBuffers()
    check_rdma_probes(reader, [make_rdma_probe(owner)])
